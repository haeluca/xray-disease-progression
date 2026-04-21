import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
from data.transforms import get_transforms

class PairedXrayDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.transforms = get_transforms()
        self.patient_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        self.samples = []

        for patient_dir in self.patient_dirs:
            t0_path = patient_dir / "t0.png"
            t1_path = patient_dir / "t1.png"
            if t0_path.exists() and t1_path.exists():
                self.samples.append((t0_path, t1_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        t0_path, t1_path = self.samples[idx]

        x_t0 = Image.open(t0_path).convert("L")
        x_t1 = Image.open(t1_path).convert("L")

        x_t0 = self.transforms(x_t0)
        x_t1 = self.transforms(x_t1)

        return {"x_t0": x_t0, "x_t1": x_t1}
