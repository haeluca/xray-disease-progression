"""
Dataset for Project A: feature-conditioned X-ray synthesis.

Each sample provides a single image together with its real OA feature vector.
The image is the training target; the feature vector is the condition signal.
The model learns to generate images whose features match the requested vector,
not to reconstruct a specific source image.
"""

import torch
from .base_dataset import BaseDataset


class FeatureConditionedDataset(BaseDataset):
    """
    Single-image dataset with ground-truth OA feature vector as condition.

    Each sample returns a real image and the features actually present in that
    image. During DDPM training the model receives pure noise + this feature
    vector and must denoise toward the real image — teaching it to generate
    images with the requested disease state.

    Args:
        split_csv:     Path to the split CSV (columns: patient_id, filename).
        metadata_csv:  Path to metadata CSV containing per-image feature values.
        roi_dir:       Directory containing normalised ROI PNGs.
        num_features:  Number of feature dimensions (overridden by feature_schema length).
        transforms:    Torchvision transforms applied to the loaded image.
        image_size:    Resize target for the loaded image.
        feature_schema: Feature definition list used to select and order metadata columns.

    Returns per sample:
        target:          (1, H, W) normalised grayscale tensor.
        target_features: (num_features,) float tensor — real features of that image.
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

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        img = item["image"]
        metadata = self.samples[idx][1]

        target_features = []
        for col in self.feature_cols:
            if col in metadata:
                target_features.append(float(metadata[col]))
            else:
                target_features.append(0.0)

        target_features = torch.tensor(target_features, dtype=torch.float32)

        return {
            "target": img,
            "target_features": target_features,
            "patient_id": metadata["patient_id"],
        }
