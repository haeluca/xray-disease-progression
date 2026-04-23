"""
Dataset for Project A: feature-conditioned X-ray synthesis.

Each sample provides a single image together with source and target feature
vectors. When randomize_target=True the target vector is sampled uniformly —
this is used during training to teach the generator to reach arbitrary feature
states, not just reproduce the input.
"""

import torch
import numpy as np
from .base_dataset import BaseDataset


class FeatureConditionedDataset(BaseDataset):
    """
    Single-image dataset with source and target OA feature vectors.

    Args:
        split_csv:        Path to the split CSV (columns: patient_id, filename).
        metadata_csv:     Path to metadata CSV containing per-image feature values.
        roi_dir:          Directory containing normalised ROI PNGs.
        num_features:     Number of feature dimensions (overridden by feature_schema length).
        transforms:       Torchvision transforms applied to the loaded image.
        image_size:       Resize target for the loaded image.
        randomize_target: If True, sample target_features uniformly (training augmentation).
        feature_schema:   Feature definition list used to select and order metadata columns.

    Returns per sample:
        image:           (1, H, W) normalised grayscale tensor.
        source_features: (num_features,) float tensor from metadata.
        target_features: (num_features,) float tensor — same as source or randomised.
        patient_id:      Patient identifier string.
    """

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
