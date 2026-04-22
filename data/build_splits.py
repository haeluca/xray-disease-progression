import csv
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split


def build_splits(metadata_path, output_dir, train_ratio=0.7, val_ratio=0.15):
    metadata_path = Path(metadata_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    patients = sorted(set(row['patient_id'] for row in rows))

    train_patients, temp_patients = train_test_split(
        patients, train_size=train_ratio, random_state=42
    )

    test_ratio = 1.0 - train_ratio - val_ratio
    val_size = val_ratio / (val_ratio + test_ratio)
    val_patients, test_patients = train_test_split(
        temp_patients, train_size=val_size, random_state=42
    )

    for split_name, patient_list in [
        ('train', train_patients),
        ('val', val_patients),
        ('test', test_patients),
    ]:
        output_path = output_dir / f"{split_name}.csv"
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['patient_id'])
            writer.writeheader()
            for patient_id in patient_list:
                writer.writerow({'patient_id': patient_id})

        print(f"Wrote {len(patient_list)} patients to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, required=True, help="Path to metadata CSV")
    parser.add_argument("--output", type=str, default="data/splits", help="Output splits directory")
    args = parser.parse_args()

    build_splits(args.metadata, args.output)
