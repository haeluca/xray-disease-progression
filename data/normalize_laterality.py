import csv
import argparse
from pathlib import Path
from PIL import Image


def normalize_laterality(roi_dir, output_dir):
    roi_dir = Path(roi_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_file in roi_dir.glob("*.png"):
        patient_id, side = img_file.stem.rsplit('_', 1)

        img = Image.open(img_file).convert('L')

        if side == 'L':
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        output_path = output_dir / img_file.name
        img.save(output_path)

    print(f"Normalized images saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi_dir", type=str, required=True, help="Path to ROI directory")
    parser.add_argument("--output", type=str, default="data/processed/normalized", help="Output directory")
    args = parser.parse_args()

    normalize_laterality(args.roi_dir, args.output)
