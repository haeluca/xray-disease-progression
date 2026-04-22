import csv
import argparse
from pathlib import Path


def pair_contralateral(metadata_path, output_path, num_features=5):
    metadata_path = Path(metadata_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    patients_data = {}
    for row in rows:
        patient_id = row['patient_id']
        if patient_id not in patients_data:
            patients_data[patient_id] = {}

        side = row['side']
        patients_data[patient_id][side] = row

    with open(output_path, 'w', newline='') as f:
        fieldnames = ['patient_id', 'less_affected_side', 'more_affected_side']
        fieldnames += [f'feature_{i}_delta' for i in range(num_features)]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for patient_id, sides_data in sorted(patients_data.items()):
            if len(sides_data) < 2:
                continue

            left_data = sides_data.get('L')
            right_data = sides_data.get('R')

            if left_data is None or right_data is None:
                continue

            less_affected_severity = sum(
                float(left_data.get(f'feature_{i}', 0)) for i in range(num_features)
            )
            more_affected_severity = sum(
                float(right_data.get(f'feature_{i}', 0)) for i in range(num_features)
            )

            if less_affected_severity > more_affected_severity:
                less_affected_data = right_data
                more_affected_data = left_data
                less_affected_side = 'R'
                more_affected_side = 'L'
            else:
                less_affected_data = left_data
                more_affected_data = right_data
                less_affected_side = 'L'
                more_affected_side = 'R'

            row_out = {
                'patient_id': patient_id,
                'less_affected_side': less_affected_side,
                'more_affected_side': more_affected_side,
            }

            for i in range(num_features):
                delta = float(more_affected_data.get(f'feature_{i}', 0)) - float(
                    less_affected_data.get(f'feature_{i}', 0)
                )
                row_out[f'feature_{i}_delta'] = delta

            writer.writerow(row_out)

    print(f"Contralateral pairs saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, required=True, help="Path to metadata CSV")
    parser.add_argument("--output", type=str, default="data/contralateral_pairs.csv")
    args = parser.parse_args()

    pair_contralateral(args.metadata, args.output)
