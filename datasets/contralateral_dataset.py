import torch
import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class ContralateralDataset(Dataset):
    def __init__(
        self,
        split_csv,
        contralateral_pairs_csv,
        roi_dir,
        num_features,
        transforms=None,
        image_size=256,
        feature_schema=None,
    ):
        self.roi_dir = Path(roi_dir)
        self.transforms = transforms
        self.image_size = image_size
        self.feature_schema = feature_schema
        self.num_features = len(feature_schema) if feature_schema is not None else num_features

        split_df = pd.read_csv(split_csv)
        split_patients = set(split_df["patient_id"].unique())

        pairs_df = pd.read_csv(contralateral_pairs_csv)
        self.pairs = pairs_df[pairs_df["patient_id"].isin(split_patients)].reset_index(drop=True)

        if feature_schema is not None:
            self.delta_cols = [f"{feat['name']}_delta" for feat in feature_schema]
        else:
            self.delta_cols = [c for c in pairs_df.columns if c.endswith("_delta")][: self.num_features]

    def _load_image(self, path):
        img = Image.open(path).convert("L")
        return img

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs.iloc[idx]

        less_affected_path = self.roi_dir / f"{pair['patient_id']}_{pair['less_affected_side']}.png"
        more_affected_path = self.roi_dir / f"{pair['patient_id']}_{pair['more_affected_side']}.png"

        source_img = self._load_image(less_affected_path)
        target_img = self._load_image(more_affected_path)

        if self.transforms:
            source_img = self.transforms(source_img)
            target_img = self.transforms(target_img)

        feature_delta = []
        for col in self.delta_cols:
            if col in pair:
                feature_delta.append(float(pair[col]))
            else:
                feature_delta.append(0.0)
        feature_delta = torch.tensor(feature_delta, dtype=torch.float32)

        return {
            "source": source_img,
            "target": target_img,
            "side": 0 if pair["less_affected_side"] == "L" else 1,
            "feature_delta": feature_delta,
            "patient_id": pair["patient_id"],
        }
