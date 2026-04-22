"""
Feature traversal panels: vary one feature at a time while holding others fixed.

Usage:
    python viz/make_feature_traversals.py \
        --config configs/project_a.yaml \
        --checkpoint checkpoints/project_a/best.pt \
        --source_image data/processed/normalized/P001_R.png \
        --output outputs/traversals/
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import yaml
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from models.diffusion_unet import DDPM, DiffusionUNet
from utils.checkpoint import load_checkpoint
from utils.feature_schema import DEFAULT_FEATURE_SCHEMA
from datasets.transforms import get_val_transforms


def _load_image(path, image_size, device):
    tf = get_val_transforms(image_size)
    img = Image.open(path).convert("L")
    tensor = tf(img).unsqueeze(0).to(device)
    return tensor


def _tensor_to_pil(tensor):
    img = tensor.squeeze(0).squeeze(0).cpu()
    img = (img * 0.5 + 0.5).clamp(0, 1)
    return Image.fromarray((img.numpy() * 255).astype(np.uint8), mode="L")


def _make_label_tensor(feature_schema, feature_values):
    """Build a (1, num_features) float tensor from a dict {name: value}."""
    vec = []
    for feat in feature_schema:
        vec.append(float(feature_values.get(feat["name"], 0.0)))
    return torch.tensor([vec], dtype=torch.float32)


def _traversal_values(feat):
    """Return list of values to sweep for one feature."""
    if feat["type"] == "ordinal":
        return list(range(feat["num_classes"]))
    else:
        return [round(v, 2) for v in np.linspace(0.0, 1.0, 5).tolist()]


def make_traversals(config, checkpoint_path, source_image_path, output_dir, device,
                    feature_schema=None, base_features=None):
    """
    For each feature in feature_schema, sweep its values while holding all others
    at base_features (defaults to zeros). Save one panel PNG per feature.

    source_image_path: path to a real image used as the conditioning image.
    """
    if feature_schema is None:
        feature_schema = DEFAULT_FEATURE_SCHEMA

    num_features = len(feature_schema)
    image_size = config["data"]["image_size"]

    unet = DiffusionUNet(in_channels=2, out_channels=1, condition_dim=num_features)
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

    source = _load_image(source_image_path, image_size, device)

    if base_features is None:
        base_features = {feat["name"]: 0.0 for feat in feature_schema}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for feat in feature_schema:
            values = _traversal_values(feat)
            frames = []

            for val in values:
                feats = {**base_features, feat["name"]: val}
                cond_vec = _make_label_tensor(feature_schema, feats).to(device)
                generated = model.sample(source, source.shape, condition_vector=cond_vec)
                pil = _tensor_to_pil(generated)
                frames.append((val, pil))

            # build panel: images side by side with labels
            cell_w, cell_h = image_size, image_size + 20
            panel = Image.new("L", (cell_w * len(frames), cell_h), color=200)
            draw = ImageDraw.Draw(panel)

            for col, (val, img) in enumerate(frames):
                x = col * cell_w
                panel.paste(img.resize((image_size, image_size)), (x, 20))
                label = f"{feat['name']}={val}"
                draw.text((x + 2, 2), label, fill=0)

            out_path = output_dir / f"traversal_{feat['name']}.png"
            panel.save(out_path)
            print(f"Saved {out_path}")

    print(f"All traversal panels saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source_image", required=True,
                        help="Path to a real image used as the conditioning source")
    parser.add_argument("--output", default="outputs/traversals")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    make_traversals(config, args.checkpoint, args.source_image, args.output, args.device)
