"""Logging utilities for the trimmed OpenRLHF tree."""

import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict

_FORMAT = "%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"


class NewLineFormatter(logging.Formatter):
    """Adds logging prefix to newlines to align multi-line messages."""

    def __init__(self, fmt, datefmt=None):
        logging.Formatter.__init__(self, fmt, datefmt)

    def format(self, record):
        msg = logging.Formatter.format(self, record)
        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\r\n" + parts[0])
        return msg


_root_logger = logging.getLogger("openrlhf")
_default_handler = None


def _setup_logger():
    _root_logger.setLevel(logging.DEBUG)
    global _default_handler
    if _default_handler is None:
        _default_handler = logging.StreamHandler(sys.stdout)
        _default_handler.flush = sys.stdout.flush  # type: ignore
        _default_handler.setLevel(logging.INFO)
        _root_logger.addHandler(_default_handler)
    fmt = NewLineFormatter(_FORMAT, datefmt=_DATE_FORMAT)
    _default_handler.setFormatter(fmt)
    _root_logger.propagate = False


_setup_logger()


def init_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(_default_handler)
    logger.propagate = False
    return logger


def _to_serializable(value: Any):
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


class JsonlLogger:
    """Append train/eval metrics to a local jsonl file."""

    def __init__(self, log_path: str) -> None:
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(self, split: str, step: int, metrics: Dict[str, Any]) -> None:
        record = {
            "time": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "split": split,
            "step": step,
            "metrics": _to_serializable(metrics),
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_train(self, global_step: int, logs_dict: Dict[str, Any]) -> None:
        self.log("train", global_step, logs_dict)

    def log_eval(self, global_step: int, logs_dict: Dict[str, Any]) -> None:
        self.log("eval", global_step, logs_dict)

    def close(self) -> None:
        return None


class WandbLogger:
    """Handle wandb setup and training-time logging."""

    def __init__(self, args) -> None:
        import wandb

        if not wandb.api.api_key:
            wandb.login(key=args.use_wandb)
        wandb.init(
            entity=args.wandb_org,
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_run_name,
            config=args.__dict__,
            reinit=True,
        )

        wandb.define_metric("train/global_step")
        wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
        wandb.define_metric("eval/epoch")
        wandb.define_metric("eval/*", step_metric="eval/epoch", step_sync=True)
        self.handle = wandb
        self.samples_table = wandb.Table(columns=["global_step", "text", "reward"])

    def log_train(self, global_step: int, logs_dict: Dict[str, Any]) -> None:
        logs_dict = dict(logs_dict)

        generated_samples = logs_dict.pop("generated_samples", None)
        if generated_samples:
            new_table = self.handle.Table(columns=self.samples_table.columns, data=self.samples_table.data)
            new_table.add_data(global_step, *generated_samples)
            self.samples_table = new_table
            self.handle.log({"train/generated_samples": new_table})

        metrics = {k: v for k, v in logs_dict.items() if v is not None}
        logs = {"train/%s" % k: v for k, v in {**metrics, "global_step": global_step}.items()}
        self.handle.log(logs)

    def log_eval(self, global_step: int, logs_dict: Dict[str, Any]) -> None:
        logs_dict = dict(logs_dict)
        metrics = {k: v for k, v in logs_dict.items() if v is not None}
        logs = {"eval/%s" % k: v for k, v in {**metrics, "global_step": global_step}.items()}
        self.handle.log(logs)

    def close(self) -> None:
        self.handle.finish()


class TensorboardLogger:
    """Handle tensorboard setup and training-time logging."""

    def __init__(self, args) -> None:
        from torch.utils.tensorboard import SummaryWriter

        os.makedirs(args.use_tensorboard, exist_ok=True)
        log_dir = os.path.join(args.use_tensorboard, args.wandb_run_name)
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_train(self, global_step: int, logs_dict: Dict[str, Any]) -> None:
        generated_samples = logs_dict.get("generated_samples")
        for k, v in logs_dict.items():
            if k == "generated_samples" and v is not None:
                text, reward = generated_samples
                formatted_text = f"Sample:\\n{text}\\n\\nReward: {reward:.4f}"
                self.writer.add_text("train/generated_samples", formatted_text, global_step)
            elif v is not None:
                self.writer.add_scalar(f"train/{k}", v, global_step)

    def log_eval(self, global_step: int, logs_dict: Dict[str, Any]) -> None:
        for k, v in logs_dict.items():
            self.writer.add_scalar(f"eval/{k}", v, global_step)

    def close(self) -> None:
        self.writer.close()
