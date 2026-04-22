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
        feature_schema=None,
    ):
        super().__init__(
            split_csv,
            metadata_csv,
            roi_dir,
            transforms,
            image_size,
            num_features=num_features,
            feature_schema=feature_schema,
        )
        self.randomize_target = randomize_target

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
