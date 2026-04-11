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
