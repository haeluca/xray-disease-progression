import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path


class BaseDataset(Dataset):
    def __init__(self, split_csv, metadata_csv, roi_dir, transforms=None, image_size=256, num_features=0, feature_schema=None):
        self.roi_dir = Path(roi_dir)
        self.transforms = transforms
        self.image_size = image_size
        self.feature_schema = feature_schema
        self.num_features = len(feature_schema) if feature_schema is not None else num_features

        self.split_df = pd.read_csv(split_csv)
        self.metadata_df = pd.read_csv(metadata_csv)

        self.split_patients = set(self.split_df["patient_id"].unique())
        self.metadata_df = self.metadata_df[
            self.metadata_df["patient_id"].isin(self.split_patients)
        ]

        if feature_schema is not None:
            self.feature_cols = [feat["name"] for feat in feature_schema]
        else:
            available = [c for c in self.metadata_df.columns if c.startswith("feature_")]
            self.feature_cols = available[: self.num_features]

        self.samples = []
        for _, row in self.metadata_df.iterrows():
            img_path = self.roi_dir / f"{row['patient_id']}_{row['side']}.png"
            if img_path.exists():
                self.samples.append((img_path, row))

    def _load_image(self, path):
        return Image.open(path).convert("L")

    def _feature_tensor(self, metadata):
        values = []
        for i in range(self.num_features):
            if i < len(self.feature_cols) and self.feature_cols[i] in metadata:
                values.append(float(metadata[self.feature_cols[i]]))
            else:
                values.append(0.0)
        return torch.tensor(values, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, metadata = self.samples[idx]
        img = self._load_image(img_path)

        if self.transforms:
            img = self.transforms(img)

        return {
            "image": img,
            "features": self._feature_tensor(metadata),
            "patient_id": metadata["patient_id"],
            "side": metadata["side"],
        }
