"""
Feature traversal panels: vary one feature at a time while holding others fixed.

Project A generates from pure noise conditioned on a feature vector, so traversals
require no source image — each panel column is a fresh sample at a different feature value.

Usage:
    python viz/make_feature_traversals.py \
        --config configs/project_a.yaml \
        --checkpoint checkpoints/project_a/best.pt \
        --output outputs/traversals/
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml
import numpy as np
from PIL import Image, ImageDraw

from models.diffusion_unet import DDPM, DiffusionUNet
from utils.checkpoint import load_checkpoint
from utils.feature_schema import DEFAULT_FEATURE_SCHEMA


def _tensor_to_pil(tensor):
    img = tensor.squeeze(0).squeeze(0).cpu()
    img = (img * 0.5 + 0.5).clamp(0, 1)
    return Image.fromarray((img.numpy() * 255).astype(np.uint8), mode="L")


def _make_condition_tensor(feature_schema, feature_values):
    """Build a (1, num_features) float tensor from a dict {name: value}."""
    vec = [float(feature_values.get(feat["name"], 0.0)) for feat in feature_schema]
    return torch.tensor([vec], dtype=torch.float32)


def _traversal_values(feat):
    """Return the list of values to sweep for one feature."""
    if feat["type"] == "ordinal":
        return list(range(feat["num_classes"]))
    else:
        return [round(v, 2) for v in np.linspace(0.0, 1.0, 5).tolist()]


def make_traversals(config, checkpoint_path, output_dir, device,
                    feature_schema=None, base_features=None):
    """
    For each feature in feature_schema, sweep its values while holding all others at
    base_features (defaults to zeros). Save one labeled panel PNG per feature.

    Project A uses feature-only conditioning — no source image is needed.
    Each image in the panel is sampled independently from noise conditioned on the
    requested feature vector.

    Args:
        config:          Parsed project_a.yaml config dict.
        checkpoint_path: Path to the trained DDPM checkpoint.
        output_dir:      Directory to write traversal panel PNGs.
        device:          Torch device string.
        feature_schema:  Feature definition list; defaults to DEFAULT_FEATURE_SCHEMA.
        base_features:   Dict {feature_name: value} for non-swept features; defaults to zeros.
    """
    if feature_schema is None:
        feature_schema = DEFAULT_FEATURE_SCHEMA

    num_features = len(feature_schema)
    image_size = config["data"]["image_size"]

    in_ch = config["model"]["generator"].get("in_channels", 1)
    unet = DiffusionUNet(in_channels=in_ch, out_channels=1, condition_dim=num_features)
    model = DDPM(
        unet,
        T=config["model"]["T"],
        beta_start=float(config["model"]["beta_start"]),
        beta_end=float(config["model"]["beta_end"]),
        device=device,
    )
    load_checkpoint(checkpoint_path, model, device=device)
    model = model.to(device)
    model.eval()
    print(f"Loaded model from {checkpoint_path}")

    if base_features is None:
        base_features = {feat["name"]: 0.0 for feat in feature_schema}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_shape = (1, 1, image_size, image_size)

    with torch.no_grad():
        for feat in feature_schema:
            values = _traversal_values(feat)
            frames = []

            for val in values:
                feats = {**base_features, feat["name"]: val}
                cond_vec = _make_condition_tensor(feature_schema, feats).to(device)
                # Generate from noise — no conditioning image for Project A
                generated = model.sample(sample_shape, condition_vector=cond_vec)
                frames.append((val, _tensor_to_pil(generated)))

            # Build panel: images side by side with value labels above each column
            cell_w, cell_h = image_size, image_size + 20
            panel = Image.new("L", (cell_w * len(frames), cell_h), color=200)
            draw = ImageDraw.Draw(panel)

            for col, (val, img) in enumerate(frames):
                x = col * cell_w
                panel.paste(img.resize((image_size, image_size)), (x, 20))
                draw.text((x + 2, 2), f"{feat['name']}={val}", fill=0)

            out_path = output_dir / f"traversal_{feat['name']}.png"
            panel.save(out_path)
            print(f"Saved {out_path}")

    print(f"All traversal panels saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/traversals")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    make_traversals(config, args.checkpoint, args.output, args.device)
