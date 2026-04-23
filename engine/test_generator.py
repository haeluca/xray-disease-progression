"""
Held-out evaluation engine for trained generators.

Loads the best checkpoint from the configured checkpoint directory, runs all
batches of the test split, computes image quality (SSIM/PSNR/L1) and optionally
feature fidelity (using the frozen classifier for Project A), then writes
per-batch metrics to outputs/<project>/test_results.csv and prints averages.
"""

import csv
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.transforms import get_val_transforms
from datasets.feature_conditioned_dataset import FeatureConditionedDataset
from datasets.contralateral_dataset import ContralateralDataset
from metrics.image_metrics import evaluate_image_quality
from metrics.feature_metrics import evaluate_feature_fidelity
from utils.checkpoint import load_checkpoint, find_best_checkpoint


def _build_test_loader(config, project, feature_schema):
    """Build a non-shuffled test DataLoader for project 'a' or 'b'."""
    tf = get_val_transforms(config["data"]["image_size"])
    num_features = len(feature_schema)

    if project == "a":
        ds = FeatureConditionedDataset(
            split_csv=config["data"]["test_split"],
            metadata_csv=config["data"]["metadata_path"],
            roi_dir=config["data"]["roi_dir"],
            num_features=num_features,
            transforms=tf,
            image_size=config["data"]["image_size"],
            randomize_target=False,
            feature_schema=feature_schema,
        )
    else:
        ds = ContralateralDataset(
            split_csv=config["data"]["test_split"],
            contralateral_pairs_csv=config["data"]["contralateral_pairs"],
            roi_dir=config["data"]["roi_dir"],
            num_features=num_features,
            transforms=tf,
            image_size=config["data"]["image_size"],
            feature_schema=feature_schema,
        )

    return DataLoader(ds, batch_size=config["training"]["batch_size"], shuffle=False,
                      num_workers=config["training"].get("num_workers", 0))


def test_generator(config, model, project, objective, device="cpu",
                   classifier=None, feature_schema=None):
    """
    Run held-out evaluation and write per-batch metrics to outputs/test_results.csv.

    project:   "a" | "b"
    objective: "ddpm" | "vae" | "pix2pix"
    classifier: optional frozen ClassifierBackbone for feature fidelity; skipped if None
    """
    output_dir = Path(config["paths"].get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "test_results.csv"

    # load best checkpoint
    ckpt_dir = config["paths"]["checkpoint_dir"]
    try:
        ckpt_path = find_best_checkpoint(ckpt_dir)
        load_checkpoint(ckpt_path, model, device=device)
        print(f"Loaded checkpoint: {ckpt_path}")
    except FileNotFoundError as e:
        print(f"Warning: {e}. Evaluating with current model weights.")

    model = model.to(device)
    model.eval()

    if classifier is not None:
        classifier = classifier.to(device)
        classifier.eval()

    loader = _build_test_loader(config, project, feature_schema)

    agg = {}
    rows = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Test evaluation")):
            if project == "a":
                image = batch["image"].to(device)
                target_features = batch["target_features"].to(device)
                cond_img = image
                cond_vec = target_features
                target = image
            else:
                source = batch["source"].to(device)
                target = batch["target"].to(device)
                delta = batch["feature_delta"].to(device)
                cond_img = source
                cond_vec = delta
                target_features = delta

            # generate
            if objective == "ddpm":
                shape = target.shape
                generated = model.sample(cond_img, shape, condition_vector=cond_vec)
            elif objective == "vae":
                generated, _, _ = model(target, cond_vec)
            else:
                generated = model(cond_img)

            # image quality
            img_metrics = evaluate_image_quality(generated, target)

            # feature fidelity (only if classifier provided and project a)
            fid_metrics = {}
            if classifier is not None and project == "a":
                fid_metrics = evaluate_feature_fidelity(
                    generated, target_features, classifier,
                    feature_schema=feature_schema, device=device
                )

            row = {"batch": batch_idx, **img_metrics, **fid_metrics}
            rows.append(row)

            for k, v in {**img_metrics, **fid_metrics}.items():
                agg.setdefault(k, []).append(v)

    # write CSV
    fieldnames = list(rows[0].keys()) if rows else ["batch"]
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # print summary
    print(f"\n=== Test Results ({len(rows)} batches) ===")
    for k, vs in agg.items():
        finite = [v for v in vs if v == v]  # drop NaN
        mean = sum(finite) / len(finite) if finite else float("nan")
        print(f"  {k}: {mean:.4f}")
    print(f"Results saved to {results_path}")

    return {k: sum(vs) / len(vs) for k, vs in agg.items() if vs}
