import argparse
import json
import os
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from tqdm import tqdm

from ..utils.utils import convert_to_torch_dtype, get_tokenizer


CHOICE_LETTERS: Tuple[str, ...] = ("A", "B", "C", "D")


def parse_tasks(tasks: str) -> List[str]:
    parsed = [task.strip().lower() for task in tasks.split(",") if task.strip()]
    valid_tasks = {"gsm8k", "mmlu"}
    invalid = [task for task in parsed if task not in valid_tasks]
    if invalid:
        raise ValueError(f"Unsupported tasks: {invalid}. Expected a subset of {sorted(valid_tasks)}.")
    if not parsed:
        raise ValueError("At least one task must be specified.")
    return parsed


def maybe_limit_dataset(dataset, max_samples: int):
    if max_samples is None or max_samples < 0:
        return dataset
    return dataset.select(range(min(max_samples, len(dataset))))


def load_dataset_from_repo(dataset_name: str, split: str, max_samples: int):
    from ..datasets.utils import blending_datasets

    dataset = blending_datasets(dataset_name, None, dataset_split=split, is_rank_0=True)
    return maybe_limit_dataset(dataset, max_samples)


def maybe_apply_chat_template(prompt: str, tokenizer, args) -> str:
    if not args.apply_chat_template:
        return prompt
    if args.tokenizer_chat_template:
        tokenizer.chat_template = args.tokenizer_chat_template
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def extract_gsm8k_answer(text: str) -> str:
    if text is None:
        return ""
    match = re.search(r"####\s*([^\n]+)", text)
    if match:
        candidate = match.group(1).strip()
    else:
        numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
        candidate = numbers[-1] if numbers else ""
    return candidate.replace(",", "").strip().rstrip(".")


def build_gsm8k_prompt(question: str) -> str:
    return (
        "Solve the following grade-school math problem. "
        "Show your reasoning, and end the final answer with '#### <answer>'.\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def evaluate_gsm8k(args, tokenizer) -> Dict[str, float]:
    from vllm import LLM, SamplingParams

    dataset = load_dataset_from_repo(args.gsm8k_dataset, args.gsm8k_split, args.gsm8k_max_samples)
    llm_kwargs = {
        "model": args.model_path,
        "tensor_parallel_size": args.gsm8k_tensor_parallel_size,
        "dtype": "bfloat16" if args.param_dtype == "bf16" else "float16",
        "trust_remote_code": True,
        "gpu_memory_utilization": args.gsm8k_gpu_memory_utilization,
        "seed": args.seed,
        "disable_log_stats": True,
    }
    if args.gsm8k_max_model_len is not None:
        llm_kwargs["max_model_len"] = args.gsm8k_max_model_len
    if args.gsm8k_enforce_eager:
        llm_kwargs["enforce_eager"] = True

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        temperature=args.gsm8k_temperature,
        top_p=args.gsm8k_top_p,
        max_tokens=args.gsm8k_max_new_tokens,
        skip_special_tokens=False,
    )

    correct = 0
    total = 0
    predictions = []

    progress = tqdm(range(0, len(dataset), args.gsm8k_batch_size), desc="Evaluating GSM8K")
    for start in progress:
        end = min(start + args.gsm8k_batch_size, len(dataset))
        batch = [dataset[idx] for idx in range(start, end)]
        prompts = [
            maybe_apply_chat_template(build_gsm8k_prompt(example[args.gsm8k_question_key]), tokenizer, args)
            for example in batch
        ]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

        for example, output in zip(batch, outputs):
            generated_text = output.outputs[0].text
            predicted_answer = extract_gsm8k_answer(generated_text)
            expected_answer = extract_gsm8k_answer(example[args.gsm8k_answer_key])
            is_correct = predicted_answer == expected_answer
            correct += int(is_correct)
            total += 1
            if len(predictions) < args.num_preview_examples:
                predictions.append(
                    {
                        "question": example[args.gsm8k_question_key],
                        "prediction": generated_text,
                        "predicted_answer": predicted_answer,
                        "expected_answer": expected_answer,
                        "correct": is_correct,
                    }
                )

        if total > 0:
            progress.set_postfix({"accuracy": correct / total})

    results = {
        "dataset": args.gsm8k_dataset,
        "split": args.gsm8k_split,
        "num_examples": total,
        "accuracy": correct / total if total else 0.0,
        "preview_examples": predictions,
    }
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def resolve_choice_tokenization(tokenizer) -> Dict[str, object]:
    variants = [
        {
            "answer_suffix": "Answer:",
            "choice_strings": [f" {letter}" for letter in CHOICE_LETTERS],
            "label": "leading_space",
        },
        {
            "answer_suffix": "Answer: ",
            "choice_strings": list(CHOICE_LETTERS),
            "label": "no_leading_space",
        },
    ]

    fallback = None
    for variant in variants:
        encoded = [tokenizer.encode(choice, add_special_tokens=False) for choice in variant["choice_strings"]]
        first_token_ids = [token_ids[0] for token_ids in encoded if token_ids]
        if len(first_token_ids) != len(CHOICE_LETTERS):
            continue
        if len(set(first_token_ids)) != len(CHOICE_LETTERS):
            continue
        candidate = {
            "answer_suffix": variant["answer_suffix"],
            "choice_strings": variant["choice_strings"],
            "token_ids": first_token_ids,
            "single_token_choices": all(len(token_ids) == 1 for token_ids in encoded),
            "variant": variant["label"],
        }
        if candidate["single_token_choices"]:
            return candidate
        if fallback is None:
            fallback = candidate

    if fallback is not None:
        return fallback

    raise ValueError("Could not find unique first-token ids for MMLU answer choices A/B/C/D.")


def build_mmlu_prompt(question: str, choices: Sequence[str], answer_suffix: str) -> str:
    formatted_choices = "\n".join(f"{letter}. {choice}" for letter, choice in zip(CHOICE_LETTERS, choices))
    return (
        "Answer the following multiple-choice question by selecting the correct option.\n\n"
        f"Question: {question}\n"
        f"{formatted_choices}\n"
        f"{answer_suffix}"
    )


def normalize_mmlu_answer(answer) -> int:
    if isinstance(answer, int):
        if 0 <= answer < len(CHOICE_LETTERS):
            return answer
        raise ValueError(f"Integer answer out of range: {answer}")

    answer_str = str(answer).strip()
    if answer_str in CHOICE_LETTERS:
        return CHOICE_LETTERS.index(answer_str)
    if answer_str.isdigit():
        numeric = int(answer_str)
        if 0 <= numeric < len(CHOICE_LETTERS):
            return numeric
        if 1 <= numeric <= len(CHOICE_LETTERS):
            return numeric - 1
    raise ValueError(f"Unsupported MMLU answer format: {answer!r}")


def chunked_indices(total_size: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, total_size, batch_size):
        yield start, min(start + batch_size, total_size)


def evaluate_mmlu(args, tokenizer) -> Dict[str, object]:
    from transformers import AutoModelForCausalLM

    dataset = load_dataset_from_repo(args.mmlu_dataset, args.mmlu_split, args.mmlu_max_samples)

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": convert_to_torch_dtype(args.param_dtype),
    }
    if args.mmlu_device_map not in (None, "", "none"):
        model_kwargs["device_map"] = args.mmlu_device_map
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    if "device_map" not in model_kwargs:
        model = model.to(args.mmlu_device)
    model.eval()
    model_device = next(model.parameters()).device

    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    choice_config = resolve_choice_tokenization(tokenizer)
    choice_token_ids = torch.tensor(choice_config["token_ids"], dtype=torch.long, device=model_device)

    subject_totals: Dict[str, int] = defaultdict(int)
    subject_correct: Dict[str, int] = defaultdict(int)
    total = 0
    correct = 0
    preview_examples = []

    progress = tqdm(list(chunked_indices(len(dataset), args.mmlu_batch_size)), desc="Evaluating MMLU")
    for start, end in progress:
        batch = [dataset[idx] for idx in range(start, end)]
        prompts = []
        gold_indices = []
        subjects = []
        for example in batch:
            choices = example[args.mmlu_choices_key]
            if len(choices) != len(CHOICE_LETTERS):
                raise ValueError(f"Expected exactly 4 choices for MMLU, got {len(choices)}")
            prompt = build_mmlu_prompt(
                question=example[args.mmlu_question_key],
                choices=choices,
                answer_suffix=choice_config["answer_suffix"],
            )
            prompt = maybe_apply_chat_template(prompt, tokenizer, args)
            prompts.append(prompt)
            gold_indices.append(normalize_mmlu_answer(example[args.mmlu_answer_key]))
            subjects.append(str(example[args.mmlu_subject_key]))

        tokenized = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=args.mmlu_max_len,
            return_tensors="pt",
            add_special_tokens=False,
        )
        attention_mask = tokenized["attention_mask"]
        input_ids = tokenized["input_ids"].to(model_device)
        attention_mask_device = attention_mask.to(model_device)

        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask_device).logits.float()

        last_token_indices = attention_mask.sum(dim=1).to(logits.device) - 1
        batch_indices = torch.arange(logits.shape[0], device=logits.device)
        next_token_logits = logits[batch_indices, last_token_indices, :]
        option_logits = next_token_logits.index_select(dim=-1, index=choice_token_ids)
        predicted_indices = option_logits.argmax(dim=-1).tolist()

        for example, subject, gold_idx, pred_idx, prompt, logits_row in zip(
            batch, subjects, gold_indices, predicted_indices, prompts, option_logits.tolist()
        ):
            is_correct = pred_idx == gold_idx
            total += 1
            correct += int(is_correct)
            subject_totals[subject] += 1
            subject_correct[subject] += int(is_correct)

            if len(preview_examples) < args.num_preview_examples:
                preview_examples.append(
                    {
                        "subject": subject,
                        "question": example[args.mmlu_question_key],
                        "prompt": prompt,
                        "gold_choice": CHOICE_LETTERS[gold_idx],
                        "predicted_choice": CHOICE_LETTERS[pred_idx],
                        "choice_logits": {letter: value for letter, value in zip(CHOICE_LETTERS, logits_row)},
                        "correct": is_correct,
                    }
                )

        if total > 0:
            progress.set_postfix({"macro_acc_pending": True, "overall_acc": correct / total})

    subject_accuracies = {
        subject: subject_correct[subject] / subject_totals[subject] for subject in sorted(subject_totals.keys())
    }
    macro_accuracy = (
        sum(subject_accuracies.values()) / len(subject_accuracies) if subject_accuracies else 0.0
    )

    results = {
        "dataset": args.mmlu_dataset,
        "split": args.mmlu_split,
        "num_examples": total,
        "num_subjects": len(subject_accuracies),
        "overall_accuracy": correct / total if total else 0.0,
        "macro_accuracy": macro_accuracy,
        "subject_accuracies": subject_accuracies,
        "choice_tokenization": choice_config,
        "preview_examples": preview_examples,
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone evaluation entrypoint for GSM8K and MMLU.")

    parser.add_argument("--model_path", type=str, required=True, help="HF model id or local save_pretrained directory.")
    parser.add_argument("--tasks", type=str, default="gsm8k,mmlu", help="Comma-separated tasks: gsm8k,mmlu")
    parser.add_argument("--save_path", type=str, default=None, help="Optional path to save a JSON summary.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--param_dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--apply_chat_template", action="store_true", default=False)
    parser.add_argument("--tokenizer_chat_template", type=str, default=None)
    parser.add_argument("--num_preview_examples", type=int, default=3)

    parser.add_argument("--gsm8k_dataset", type=str, default="openai/gsm8k#main")
    parser.add_argument("--gsm8k_split", type=str, default="test")
    parser.add_argument("--gsm8k_question_key", type=str, default="question")
    parser.add_argument("--gsm8k_answer_key", type=str, default="answer")
    parser.add_argument("--gsm8k_max_samples", type=int, default=-1)
    parser.add_argument("--gsm8k_batch_size", type=int, default=32)
    parser.add_argument("--gsm8k_max_new_tokens", type=int, default=512)
    parser.add_argument("--gsm8k_temperature", type=float, default=0.0)
    parser.add_argument("--gsm8k_top_p", type=float, default=1.0)
    parser.add_argument("--gsm8k_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gsm8k_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--gsm8k_max_model_len", type=int, default=None)
    parser.add_argument("--gsm8k_enforce_eager", action="store_true", default=False)

    parser.add_argument("--mmlu_dataset", type=str, default="cais/mmlu#all")
    parser.add_argument("--mmlu_split", type=str, default="validation")
    parser.add_argument("--mmlu_question_key", type=str, default="question")
    parser.add_argument("--mmlu_choices_key", type=str, default="choices")
    parser.add_argument("--mmlu_answer_key", type=str, default="answer")
    parser.add_argument("--mmlu_subject_key", type=str, default="subject")
    parser.add_argument("--mmlu_max_samples", type=int, default=-1)
    parser.add_argument("--mmlu_batch_size", type=int, default=16)
    parser.add_argument("--mmlu_max_len", type=int, default=2048)
    parser.add_argument("--mmlu_device_map", type=str, default="none")
    parser.add_argument("--mmlu_device", type=str, default="cuda")

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    tasks = parse_tasks(args.tasks)

    tokenizer = get_tokenizer(args.model_path, model=None, padding_side="right")

    results: Dict[str, object] = {
        "model_path": args.model_path,
        "tasks": tasks,
        "seed": args.seed,
        "param_dtype": args.param_dtype,
    }

    if "gsm8k" in tasks:
        results["gsm8k"] = evaluate_gsm8k(args, tokenizer)

    if "mmlu" in tasks:
        results["mmlu"] = evaluate_mmlu(args, tokenizer)

    print(json.dumps(results, indent=2, ensure_ascii=False))

    if args.save_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
        with open(args.save_path, "w", encoding="utf-8") as fout:
            json.dump(results, fout, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
