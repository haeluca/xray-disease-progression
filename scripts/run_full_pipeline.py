import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def run(cmd, skip_on_error=False):
    print(f"\n{'=' * 70}\n>> {' '.join(cmd)}\n{'=' * 70}")
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + existing if existing else "")
    result = subprocess.run(cmd, cwd=ROOT, env=env)
    if result.returncode != 0:
        msg = f"Command failed with exit code {result.returncode}: {' '.join(cmd)}"
        if skip_on_error:
            print(f"[WARN] {msg} — continuing")
        else:
            print(f"[ERROR] {msg}")
            sys.exit(result.returncode)


def data_pipeline(args):
    py = sys.executable

    prep_cmd = [py, "data/prepare_metadata.py", "--raw_dir", args.raw_dir, "--output", "data/metadata.csv"]
    if args.mock_features:
        prep_cmd.append("--mock_features")
    run(prep_cmd)

    run([py, "data/extract_roi.py", "--metadata", "data/metadata.csv", "--output", "data/processed/roi"])
    run([py, "data/normalize_laterality.py", "--roi_dir", "data/processed/roi", "--output", "data/processed/normalized"])
    run([py, "data/build_splits.py", "--metadata", "data/metadata.csv", "--output", "data/splits"])
    run([py, "data/pair_contralateral.py", "--metadata", "data/metadata.csv", "--output", "data/contralateral_pairs.csv"])


def train_stage(script, config, stage, device):
    py = sys.executable
    run([py, f"scripts/{script}", "--config", config, "--stage", stage, "--device", device])


def main():
    parser = argparse.ArgumentParser(description="Run the full CMC I OA pipeline end-to-end")
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--mock_features", action="store_true", help="Fill metadata with random feature values (smoke testing)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip_data", action="store_true", help="Skip the data-preparation steps")
    parser.add_argument("--skip_classifier", action="store_true")
    parser.add_argument("--skip_project_a", action="store_true")
    parser.add_argument("--skip_project_b", action="store_true")
    parser.add_argument("--skip_baselines", action="store_true", help="Skip VAE / Pix2Pix baselines")
    parser.add_argument("--skip_test", action="store_true", help="Skip held-out test stage")
    args = parser.parse_args()

    if not args.skip_data:
        print("\n### [1/4] Data preparation")
        data_pipeline(args)

    if not args.skip_classifier:
        print("\n### [2/4] Classifier")
        train_stage("run_feature_conditioned.py", "configs/classifier.yaml", "classifier", args.device)

    if not args.skip_project_a:
        print("\n### [3/4] Project A (feature-conditioned)")
        if not args.skip_baselines:
            train_stage("run_feature_conditioned.py", "configs/project_a.yaml", "baseline", args.device)
        train_stage("run_feature_conditioned.py", "configs/project_a.yaml", "main", args.device)
        if not args.skip_test:
            train_stage("run_feature_conditioned.py", "configs/project_a.yaml", "test", args.device)

    if not args.skip_project_b:
        print("\n### [4/4] Project B (contralateral)")
        if not args.skip_baselines:
            train_stage("run_contralateral.py", "configs/project_b.yaml", "baseline", args.device)
        train_stage("run_contralateral.py", "configs/project_b.yaml", "main", args.device)
        if not args.skip_test:
            train_stage("run_contralateral.py", "configs/project_b.yaml", "test", args.device)

    print("\n### Pipeline complete.")


if __name__ == "__main__":
    main()
