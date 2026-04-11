from typing import Callable

import torch
from torch.utils.data import Dataset

from openrlhf.utils.utils import zero_pad_sequences


def preprocess_sft_sample(
    data,
    input_key="input",
    output_key="output",
    prompt_template=None,
    apply_chat_template=None,
):
    if apply_chat_template:
        if output_key is not None:
            prompt_message = data[input_key]
            response_message = data[output_key]

            if isinstance(prompt_message, str) and isinstance(response_message, str):
                prompt_message = [{"role": "user", "content": prompt_message}]
                response_message = [{"role": "assistant", "content": response_message}]

            prompt = apply_chat_template(prompt_message, tokenize=False, add_generation_prompt=True)
            response = apply_chat_template(prompt_message + response_message, tokenize=False)[len(prompt) :]
        else:
            prompt = apply_chat_template(data[input_key][:-1], tokenize=False, add_generation_prompt=True)
            response = apply_chat_template(data[input_key], tokenize=False)[len(prompt) :]
    else:
        prompt = data[input_key]
        if prompt_template:
            prompt = prompt_template.format(prompt)
        response = data[output_key]

    return prompt, response


class SFTDataset(Dataset):
    """Assistant-only supervised fine-tuning dataset."""

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        input_key="input",
        output_key="output",
        prompt_template=None,
        apply_chat_template=False,
        tokenizer_chat_template=None,
        multiturn=False,
        num_processors=8,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.multiturn = multiturn
        self.input_key = input_key
        self.output_key = output_key
        self.prompt_template = prompt_template

        self.apply_chat_template = None
        if apply_chat_template:
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template
            self.apply_chat_template = self.tokenizer.apply_chat_template

        processed_dataset = dataset.map(
            self.process_data,
            remove_columns=dataset.column_names,
            num_proc=num_processors,
        )
        processed_dataset = processed_dataset.filter(lambda x: x["prompt"] is not None)

        self.prompts = processed_dataset["prompt"]
        self.responses = processed_dataset["response"]
        self.prompt_ids_lens = processed_dataset["prompt_ids_len"]
        self.response_ranges = processed_dataset["response_ranges"] if self.multiturn else None

    def process_data(self, data):
        if self.multiturn and self.output_key:
            data[self.input_key].append(data[self.output_key])
            data[self.output_key] = None

        response_ranges = None
        if self.multiturn:
            assert (
                not self.output_key or not data[self.output_key]
            ), "Put the whole conversation in input_key and leave output_key unset for multiturn mode."
            response_ranges = []
            for idx, message in enumerate(data[self.input_key]):
                if message["role"] != "assistant":
                    continue

                prompt = self.apply_chat_template(data[self.input_key][:idx], tokenize=False, add_generation_prompt=True)
                response = self.apply_chat_template(data[self.input_key][: idx + 1], tokenize=False)[len(prompt) :]

                start_idx = (
                    self.tokenizer(
                        prompt,
                        max_length=self.max_length,
                        padding=False,
                        truncation=True,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )["attention_mask"]
                    .int()
                    .sum()
                    .item()
                )
                end_idx = (
                    start_idx
                    + self.tokenizer(
                        response,
                        max_length=self.max_length,
                        padding=False,
                        truncation=True,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )["attention_mask"]
                    .int()
                    .sum()
                    .item()
                    - 1
                )
                response_ranges.append((start_idx, end_idx))

        prompt, response = preprocess_sft_sample(
            data,
            input_key=self.input_key,
            output_key=self.output_key,
            prompt_template=self.prompt_template,
            apply_chat_template=self.apply_chat_template,
        )

        prompt_token = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_ids_len = prompt_token["attention_mask"].int().sum().item()
        if not prompt or not response or prompt_ids_len >= self.max_length - 2:
            prompt = None

        return {
            "prompt": prompt,
            "response": response,
            "prompt_ids_len": prompt_ids_len,
            "response_ranges": response_ranges,
        }

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        response = self.responses[idx]

        text = (prompt + response).rstrip("\n")
        if not text.endswith(self.tokenizer.eos_token):
            text += " " + self.tokenizer.eos_token

        input_token = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = input_token["input_ids"]
        attention_mask = input_token["attention_mask"]
        loss_mask = self.get_loss_mask(input_ids, idx)

        input_ids[0][-1] = self.tokenizer.eos_token_id
        attention_mask[0][-1] = True
        return input_ids, attention_mask, loss_mask

    def get_loss_mask(self, input_ids, idx):
        loss_mask = torch.zeros_like(input_ids, dtype=torch.float32)
        if not self.multiturn:
            prompt_ids_len = self.prompt_ids_lens[idx]
            loss_mask[0, prompt_ids_len - 1 : -1] = 1
        else:
            for start_idx, end_idx in self.response_ranges[idx]:
                loss_mask[0, start_idx - 1 : end_idx] = 1
        return loss_mask

    def collate_fn(self, item_list):
        input_ids = []
        attention_masks = []
        loss_masks = []
        for input_id, attention_mask, loss_mask in item_list:
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            loss_masks.append(loss_mask)

        input_ids = zero_pad_sequences(input_ids, "right", self.tokenizer.pad_token_id)
        attention_masks = zero_pad_sequences(attention_masks, "right")
        loss_masks = zero_pad_sequences(loss_masks, "right")
        return input_ids, attention_masks, loss_masks
