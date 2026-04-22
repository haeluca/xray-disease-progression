import os
import torch
from pathlib import Path


def save_checkpoint(state: dict, path: str, is_best: bool = False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(os.path.dirname(path), "best.pt")
        torch.save(state, best_path)


def load_checkpoint(path: str, model, optimizer=None, device="cpu"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    return state.get("epoch", 0), state.get("global_step", 0)


def find_best_checkpoint(checkpoint_dir: str) -> str:
    checkpoint_dir = Path(checkpoint_dir)
    if (checkpoint_dir / "best.pt").exists():
        return str(checkpoint_dir / "best.pt")

    checkpoints = sorted(checkpoint_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if checkpoints:
        return str(checkpoints[0])

    raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
