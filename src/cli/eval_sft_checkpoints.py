import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, List


CKPT_RE = re.compile(r"global_step(\d+)_hf$")


def parse_tasks(tasks: str) -> List[str]:
    parsed = [task.strip().lower() for task in tasks.split(",") if task.strip()]
    valid = {"gsm8k", "mmlu"}
    invalid = [task for task in parsed if task not in valid]
    if invalid:
        raise ValueError(f"Unsupported tasks: {invalid}. Expected a subset of {sorted(valid)}.")
    if not parsed:
        raise ValueError("At least one task must be specified.")
    return parsed


def discover_visible_gpus() -> List[str]:
    env_value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_value:
        return [gpu.strip() for gpu in env_value.split(",") if gpu.strip()]

    try:
        import torch

        count = torch.cuda.device_count()
        if count > 0:
            return [str(i) for i in range(count)]
    except Exception:
        pass

    return ["0"]


def discover_checkpoints(ckpt_path: Path, save_path: Path, include_final: bool) -> List[Dict[str, str]]:
    jobs = []
    if ckpt_path.exists():
        for child in ckpt_path.iterdir():
            if not child.is_dir():
                continue
            match = CKPT_RE.fullmatch(child.name)
            if not match:
                continue
            jobs.append({"label": f"step{int(match.group(1)):04d}", "model_path": str(child), "sort_key": int(match.group(1))})

    jobs.sort(key=lambda item: item["sort_key"])

    if include_final and save_path.exists():
        jobs.append({"label": "final", "model_path": str(save_path), "sort_key": 10**12})

    return jobs


def run_eval(repo_root: Path, model_path: str, save_path: Path, task: str, gpu_id: str, args) -> Dict[str, object]:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = save_path.with_suffix(".log")

    cmd = [
        sys.executable,
        "-m",
        "src.cli.eval_model",
        "--model_path",
        model_path,
        "--tasks",
        task,
        "--save_path",
        str(save_path),
    ]

    if task == "gsm8k":
        cmd += [
            "--gsm8k_batch_size",
            str(args.gsm8k_batch_size),
            "--gsm8k_tensor_parallel_size",
            str(args.gsm8k_tensor_parallel_size),
            "--gsm8k_gpu_memory_utilization",
            str(args.gsm8k_gpu_memory_utilization),
        ]
        if args.gsm8k_max_model_len is not None:
            cmd += ["--gsm8k_max_model_len", str(args.gsm8k_max_model_len)]
        if args.gsm8k_enforce_eager:
            cmd.append("--gsm8k_enforce_eager")
    else:
        cmd += [
            "--mmlu_batch_size",
            str(args.mmlu_batch_size),
            "--mmlu_device_map",
            "none",
            "--mmlu_device",
            "cuda",
        ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    with open(log_path, "w", encoding="utf-8") as log_file:
        subprocess.run(
            cmd,
            cwd=repo_root,
            env=env,
            check=True,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )

    with open(save_path, "r", encoding="utf-8") as fin:
        return json.load(fin)


def worker(gpu_id: str, job_queue: Queue, repo_root: Path, output_dir: Path, args, summary: List[Dict[str, object]]) -> None:
    while True:
        try:
            job = job_queue.get_nowait()
        except Empty:
            return

        result_row: Dict[str, object] = {
            "label": job["label"],
            "model_path": job["model_path"],
            "sort_key": job["sort_key"],
        }
        ckpt_output_dir = output_dir / job["label"]

        try:
            if "gsm8k" in args.tasks:
                gsm8k_save_path = ckpt_output_dir / "gsm8k.json"
                gsm8k_result = run_eval(repo_root, job["model_path"], gsm8k_save_path, "gsm8k", gpu_id, args)
                result_row["gsm8k_accuracy"] = gsm8k_result["gsm8k"]["accuracy"]
                result_row["gsm8k_json"] = str(gsm8k_save_path)

            if "mmlu" in args.tasks:
                mmlu_save_path = ckpt_output_dir / "mmlu.json"
                mmlu_result = run_eval(repo_root, job["model_path"], mmlu_save_path, "mmlu", gpu_id, args)
                result_row["mmlu_overall_accuracy"] = mmlu_result["mmlu"]["overall_accuracy"]
                result_row["mmlu_macro_accuracy"] = mmlu_result["mmlu"]["macro_accuracy"]
                result_row["mmlu_json"] = str(mmlu_save_path)

            result_row["status"] = "ok"
        except subprocess.CalledProcessError as exc:
            result_row["status"] = "failed"
            result_row["returncode"] = exc.returncode
        finally:
            summary.append(result_row)
            job_queue.task_done()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate all saved SFT HF checkpoints.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Directory that contains global_step*_hf checkpoints.")
    parser.add_argument("--save_path", type=str, required=True, help="Final save_pretrained SFT model directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store per-checkpoint eval JSON/log files.")
    parser.add_argument("--tasks", type=str, default="gsm8k,mmlu")
    parser.add_argument("--include_final", action="store_true", default=False)
    parser.add_argument("--gpus", type=str, default=None, help="Optional comma-separated GPU ids. Defaults to all visible GPUs.")
    parser.add_argument("--gsm8k_batch_size", type=int, default=32)
    parser.add_argument("--gsm8k_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gsm8k_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--gsm8k_max_model_len", type=int, default=None)
    parser.add_argument("--gsm8k_enforce_eager", action="store_true", default=False)
    parser.add_argument("--mmlu_batch_size", type=int, default=16)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args.tasks = parse_tasks(args.tasks)

    repo_root = Path(__file__).resolve().parents[2]
    ckpt_path = Path(args.ckpt_path).resolve()
    save_path = Path(args.save_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()] if args.gpus else discover_visible_gpus()
    jobs = discover_checkpoints(ckpt_path, save_path, include_final=args.include_final)
    if not jobs:
        raise FileNotFoundError(f"No HF checkpoints found under {ckpt_path} and no final model under {save_path}.")

    job_queue: Queue = Queue()
    for job in jobs:
        job_queue.put(job)

    summary: List[Dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        for gpu_id in gpu_ids:
            executor.submit(worker, gpu_id, job_queue, repo_root, output_dir, args, summary)

    summary.sort(key=lambda row: row["sort_key"])
    summary_path = output_dir / "summary.json"
    serializable_results = []
    for row in summary:
        serializable_row = dict(row)
        serializable_row.pop("sort_key", None)
        serializable_results.append(serializable_row)
    with open(summary_path, "w", encoding="utf-8") as fout:
        json.dump(
            {
                "tasks": args.tasks,
                "ckpt_path": str(ckpt_path),
                "save_path": str(save_path),
                "output_dir": str(output_dir),
                "gpus": gpu_ids,
                "results": serializable_results,
            },
            fout,
            indent=2,
            ensure_ascii=False,
        )
    print(json.dumps({"summary_path": str(summary_path), "results": serializable_results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
