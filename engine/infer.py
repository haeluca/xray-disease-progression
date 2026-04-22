"""
Batch inference / sampling from a trained generator.

Usage (Project A — DDPM, condition vector from CSV):
    python engine/infer.py \
        --config configs/project_a.yaml \
        --checkpoint checkpoints/project_a/best.pt \
        --project a --objective ddpm \
        --output outputs/samples_a/

Usage (Project B — DDPM, source images from test split):
    python engine/infer.py \
        --config configs/project_b.yaml \
        --checkpoint checkpoints/project_b/best.pt \
        --project b --objective ddpm \
        --output outputs/samples_b/
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.transforms import get_val_transforms
from datasets.feature_conditioned_dataset import FeatureConditionedDataset
from datasets.contralateral_dataset import ContralateralDataset
from models.diffusion_unet import DDPM, DiffusionUNet
from models.vae_baseline import ConditionalVAE
from models.pix2pix_baseline import Pix2PixGenerator
from utils.checkpoint import load_checkpoint
from utils.feature_schema import DEFAULT_FEATURE_SCHEMA


def _tensor_to_pil(tensor):
    img = tensor.squeeze(0).cpu()
    img = (img * 0.5 + 0.5).clamp(0, 1)
    img = (img * 255).byte()
    if img.shape[0] == 1:
        return Image.fromarray(img.squeeze(0).numpy(), mode="L")
    return Image.fromarray(img.permute(1, 2, 0).numpy())


def _build_loader(config, project, feature_schema, split="test"):
    tf = get_val_transforms(config["data"]["image_size"])
    num_features = len(feature_schema)
    split_csv = config["data"][f"{split}_split"]

    if project == "a":
        ds = FeatureConditionedDataset(
            split_csv=split_csv,
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
            split_csv=split_csv,
            contralateral_pairs_csv=config["data"]["contralateral_pairs"],
            roi_dir=config["data"]["roi_dir"],
            num_features=num_features,
            transforms=tf,
            image_size=config["data"]["image_size"],
            feature_schema=feature_schema,
        )

    return DataLoader(ds, batch_size=1, shuffle=False,
                      num_workers=config["training"].get("num_workers", 0))


def _build_model(config, objective, num_features, device):
    if objective == "ddpm":
        unet = DiffusionUNet(in_channels=2, out_channels=1, condition_dim=num_features)
        model = DDPM(
            unet,
            T=config["model"]["T"],
            beta_start=float(config["model"]["beta_start"]),
            beta_end=float(config["model"]["beta_end"]),
            device=device,
        )
    elif objective == "vae":
        model = ConditionalVAE(
            image_channels=1,
            latent_dim=32,
            condition_dim=num_features,
            image_size=config["data"]["image_size"],
        )
    elif objective == "pix2pix":
        model = Pix2PixGenerator(input_channels=1, output_channels=1)
    else:
        raise ValueError(f"Unknown objective: {objective}")
    return model


def run_infer(config, checkpoint_path, project, objective, output_dir, device, split="test", max_samples=None):
    feature_schema = DEFAULT_FEATURE_SCHEMA
    num_features = len(feature_schema)

    model = _build_model(config, objective, num_features, device)
    load_checkpoint(checkpoint_path, model, device=device)
    model = model.to(device)
    model.eval()
    print(f"Loaded {objective} model from {checkpoint_path}")

    loader = _build_loader(config, project, feature_schema, split=split)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="Generating")):
            if max_samples is not None and saved >= max_samples:
                break

            if project == "a":
                image = batch["image"].to(device)
                cond_vec = batch["target_features"].to(device)
                cond_img = image
                real = image
            else:
                source = batch["source"].to(device)
                target = batch["target"].to(device)
                cond_vec = batch["feature_delta"].to(device)
                cond_img = source
                real = target

            if objective == "ddpm":
                generated = model.sample(cond_img, cond_img.shape, condition_vector=cond_vec)
            elif objective == "vae":
                generated, _, _ = model(real, cond_vec)
            else:
                generated = model(cond_img)

            _tensor_to_pil(generated).save(output_dir / f"{i:04d}_generated.png")
            _tensor_to_pil(real).save(output_dir / f"{i:04d}_real.png")
            if project == "b":
                _tensor_to_pil(cond_img).save(output_dir / f"{i:04d}_source.png")
            saved += 1

    print(f"Saved {saved} sample(s) to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--project", required=True, choices=["a", "b"])
    parser.add_argument("--objective", required=True, choices=["ddpm", "vae", "pix2pix"])
    parser.add_argument("--output", default="outputs/samples")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_infer(config, args.checkpoint, args.project, args.objective,
              args.output, args.device, split=args.split, max_samples=args.max_samples)
