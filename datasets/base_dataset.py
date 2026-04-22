import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path


class BaseDataset(Dataset):
    def __init__(self, split_csv, metadata_csv, roi_dir, transforms=None, image_size=256):
        self.roi_dir = Path(roi_dir)
        self.transforms = transforms
        self.image_size = image_size

        self.split_df = pd.read_csv(split_csv)
        self.metadata_df = pd.read_csv(metadata_csv)

        self.split_patients = set(self.split_df["patient_id"].unique())
        self.metadata_df = self.metadata_df[
            self.metadata_df["patient_id"].isin(self.split_patients)
        ]

        self.samples = []
        for _, row in self.metadata_df.iterrows():
            img_path = self.roi_dir / f"{row['patient_id']}_{row['side']}.png"
            if img_path.exists():
                self.samples.append((img_path, row))

    def _load_image(self, path):
        img = Image.open(path).convert("L")
        return img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, metadata = self.samples[idx]
        img = self._load_image(img_path)

        if self.transforms:
            img = self.transforms(img)

        return {
            "image": img,
            "patient_id": metadata["patient_id"],
            "side": metadata["side"],
        }
