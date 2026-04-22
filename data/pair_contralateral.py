import csv
import argparse
from pathlib import Path

from utils.feature_schema import DEFAULT_FEATURE_SCHEMA


def pair_contralateral(metadata_path, output_path, feature_schema=None):
    metadata_path = Path(metadata_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if feature_schema is None:
        feature_schema = DEFAULT_FEATURE_SCHEMA

    feature_names = [feat["name"] for feat in feature_schema]

    with open(metadata_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    patients_data = {}
    for row in rows:
        patients_data.setdefault(row['patient_id'], {})[row['side']] = row

    with open(output_path, 'w', newline='') as f:
        fieldnames = ['patient_id', 'less_affected_side', 'more_affected_side']
        fieldnames += [f'{name}_delta' for name in feature_names]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for patient_id, sides_data in sorted(patients_data.items()):
            left_data = sides_data.get('L')
            right_data = sides_data.get('R')
            if left_data is None or right_data is None:
                continue

            left_sev = sum(float(left_data.get(n, 0)) for n in feature_names)
            right_sev = sum(float(right_data.get(n, 0)) for n in feature_names)

            if left_sev > right_sev:
                less, more = right_data, left_data
                less_side, more_side = 'R', 'L'
            else:
                less, more = left_data, right_data
                less_side, more_side = 'L', 'R'

            row_out = {
                'patient_id': patient_id,
                'less_affected_side': less_side,
                'more_affected_side': more_side,
            }
            for name in feature_names:
                row_out[f'{name}_delta'] = float(more.get(name, 0)) - float(less.get(name, 0))
            writer.writerow(row_out)

    print(f"Contralateral pairs saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--output", type=str, default="data/contralateral_pairs.csv")
    args = parser.parse_args()
    pair_contralateral(args.metadata, args.output)
