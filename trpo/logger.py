"""
Minimal TensorBoard scalar logger.

Trimmed port of the `Logger` class in `Deep-RL-intro-main/utils.py`. Only the
scalar methods are kept — MVP PPO doesn't need video/figure logging.
"""
import os
from typing import Mapping

from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir: str) -> None:
        self._log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        print("########################")
        print(f"logging outputs to {log_dir}")
        print("########################")
        self._writer = SummaryWriter(log_dir, flush_secs=1, max_queue=1)

    def log_scalar(self, name: str, value: float, step: int) -> None:
        self._writer.add_scalar(name, value, step)

    def log_scalars(self, scalars: Mapping[str, float], step: int) -> None:
        for name, value in scalars.items():
            self._writer.add_scalar(name, value, step)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()
