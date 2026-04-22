import os
import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        self.scalars = {}

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)
        if tag not in self.scalars:
            self.scalars[tag] = []
        self.scalars[tag].append({"step": step, "value": value})

    def log_image(self, tag: str, tensor, step: int):
        self.writer.add_image(tag, tensor, step)

    def log_dict(self, d: dict, step: int, prefix: str = ""):
        for k, v in d.items():
            if isinstance(v, (int, float)):
                self.log_scalar(f"{prefix}/{k}" if prefix else k, v, step)

    def save_config(self, config: dict):
        with open(os.path.join(self.log_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()
