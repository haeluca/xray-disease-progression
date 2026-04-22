import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml
import torch

from utils.reproducibility import set_seed, capture_env
from models.classifier_backbone import ClassifierBackbone, DEFAULT_FEATURE_SCHEMA
from models.diffusion_unet import DDPM, DiffusionUNet
from models.vae_baseline import ConditionalVAE
from engine.train_classifier import train_classifier
from engine.train_generator import train_generator


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_stage(stage, config, device):
    set_seed(42)
    feature_schema = DEFAULT_FEATURE_SCHEMA
    num_features = len(feature_schema)

    if stage == "classifier":
        print("Training classifier on real images...")
        classifier = ClassifierBackbone(feature_schema=feature_schema)
        train_classifier(config, classifier, device=device)

    elif stage == "baseline":
        print("Training baseline VAE...")
        model = ConditionalVAE(
            image_channels=1,
            latent_dim=32,
            condition_dim=num_features,
            image_size=config["data"]["image_size"],
        )
        print(f"VAE parameters: {sum(p.numel() for p in model.parameters())}")
        train_generator(config, model, project="a", objective="vae", device=device, feature_schema=feature_schema)

    elif stage == "main":
        print("Training main conditional diffusion model...")
        unet = DiffusionUNet(
            in_channels=2,
            out_channels=1,
            condition_dim=num_features,
        )
        model = DDPM(
            unet,
            T=config["model"]["T"],
            beta_start=float(config["model"]["beta_start"]),
            beta_end=float(config["model"]["beta_end"]),
            device=device,
        )
        print(f"DDPM parameters: {sum(p.numel() for p in model.parameters())}")
        train_generator(config, model, project="a", objective="ddpm", device=device, feature_schema=feature_schema)

    elif stage == "test":
        print("Running held-out evaluation...")
        print("(test stage not yet implemented)")

    else:
        raise ValueError(f"Unknown stage: {stage}")

    print(f"Environment: {capture_env()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--stage",
        type=str,
        choices=["classifier", "baseline", "main", "test"],
        required=True,
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    config = load_config(args.config)
    run_stage(args.stage, config, args.device)
