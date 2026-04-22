import csv
import argparse
import random
from pathlib import Path
from PIL import Image

from utils.feature_schema import DEFAULT_FEATURE_SCHEMA


def _mock_value(feat, rng, base, jitter_scale):
    if feat["type"] == "ordinal":
        n = feat["num_classes"]
        target = base * n
        target += rng.uniform(-jitter_scale, jitter_scale) * n
        return int(max(0, min(n - 1, round(target))))
    else:
        return round(max(0.0, min(1.0, base + rng.uniform(-jitter_scale, jitter_scale))), 4)


def prepare_metadata(raw_dir, output_path, feature_schema=None, mock_features=False, seed=0):
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if feature_schema is None:
        feature_schema = DEFAULT_FEATURE_SCHEMA

    feature_cols = [feat["name"] for feat in feature_schema]
    rng = random.Random(seed)

    with open(output_path, 'w', newline='') as f:
        fieldnames = ['patient_id', 'image_id', 'side', 'filepath', 'qc_pass'] + feature_cols
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for patient_dir in sorted(raw_dir.iterdir()):
            if not patient_dir.is_dir():
                continue

            patient_id = patient_dir.name
            patient_base = [rng.random() for _ in feature_schema]

            for img_file in sorted(patient_dir.glob("*.png")):
                try:
                    img = Image.open(img_file)
                    extrema = img.getextrema()
                    qc_pass = extrema[1] > 10
                except Exception:
                    qc_pass = False

                side = 'L' if 'left' in img_file.name.lower() else 'R'
                image_id = img_file.stem

                row = {
                    'patient_id': patient_id,
                    'image_id': image_id,
                    'side': side,
                    'filepath': str(img_file),
                    'qc_pass': qc_pass,
                }

                jitter = 0.1 if side == 'R' else 0.05
                for i, feat in enumerate(feature_schema):
                    if mock_features:
                        row[feat["name"]] = _mock_value(feat, rng, patient_base[i], jitter)
                    else:
                        row[feat["name"]] = 0

                writer.writerow(row)

    print(f"Metadata saved to {output_path} (features={feature_cols}, mock_features={mock_features})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/metadata.csv")
    parser.add_argument("--mock_features", action="store_true")
    args = parser.parse_args()

    prepare_metadata(args.raw_dir, args.output, mock_features=args.mock_features)
