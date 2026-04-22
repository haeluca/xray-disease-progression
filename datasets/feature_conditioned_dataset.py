import torch
import numpy as np
from .base_dataset import BaseDataset


class FeatureConditionedDataset(BaseDataset):
    def __init__(
        self,
        split_csv,
        metadata_csv,
        roi_dir,
        num_features,
        transforms=None,
        image_size=256,
        randomize_target=False,
    ):
        super().__init__(split_csv, metadata_csv, roi_dir, transforms, image_size)
        self.num_features = num_features
        self.randomize_target = randomize_target

        feature_cols = [c for c in self.metadata_df.columns if c.startswith("feature_")]
        self.feature_cols = feature_cols[: num_features] if feature_cols else []

        while len(self.feature_cols) < num_features:
            self.feature_cols.append(f"feature_{len(self.feature_cols)}")

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        img = item["image"]
        metadata = self.samples[idx][1]

        source_features = []
        for col in self.feature_cols:
            if col in metadata:
                source_features.append(float(metadata[col]))
            else:
                source_features.append(0.0)

        source_features = torch.tensor(source_features, dtype=torch.float32)

        if self.randomize_target:
            target_features = torch.rand(self.num_features, dtype=torch.float32)
        else:
            target_features = source_features.clone()

        return {
            "image": img,
            "source_features": source_features,
            "target_features": target_features,
            "patient_id": metadata["patient_id"],
        }
