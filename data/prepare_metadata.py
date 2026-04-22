import os
import csv
import argparse
from pathlib import Path
from PIL import Image


def prepare_metadata(raw_dir, output_path, feature_cols=None):
    raw_dir = Path(raw_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if feature_cols is None:
        feature_cols = []

    with open(output_path, 'w', newline='') as f:
        fieldnames = ['patient_id', 'image_id', 'side', 'filepath', 'qc_pass'] + feature_cols
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for patient_dir in sorted(raw_dir.iterdir()):
            if not patient_dir.is_dir():
                continue

            patient_id = patient_dir.name

            for img_file in sorted(patient_dir.glob("*.png")):
                try:
                    img = Image.open(img_file)
                    img_array = img.getdata()

                    qc_pass = len(img_array) > 0 and max(img_array) > 10

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

                for col in feature_cols:
                    row[col] = 0

                writer.writerow(row)

    print(f"Metadata saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, required=True, help="Path to raw image directory")
    parser.add_argument("--output", type=str, default="data/metadata.csv", help="Output metadata CSV path")
    args = parser.parse_args()

    prepare_metadata(args.raw_dir, args.output)
