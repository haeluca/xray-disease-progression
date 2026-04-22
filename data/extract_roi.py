import os
import csv
import argparse
from pathlib import Path
from PIL import Image


def extract_roi(metadata_path, roi_dir, crop_size=256):
    roi_dir = Path(roi_dir)
    roi_dir.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in rows:
        if row.get('qc_pass', 'True') == 'False':
            continue

        filepath = row['filepath']
        if not os.path.exists(filepath):
            continue

        patient_id = row['patient_id']
        side = row['side']

        img = Image.open(filepath).convert('L')
        img = img.resize((crop_size, crop_size), Image.BILINEAR)

        output_name = f"{patient_id}_{side}.png"
        output_path = roi_dir / output_name
        img.save(output_path)

    print(f"ROI crops saved to {roi_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, required=True, help="Path to metadata CSV")
    parser.add_argument("--output", type=str, default="data/processed/roi", help="Output ROI directory")
    args = parser.parse_args()

    extract_roi(args.metadata, args.output)
