import argparse
import yaml
import torch
from pathlib import Path

from utils.reproducibility import set_seed, capture_env
from models.classifier_backbone import ClassifierBackbone
from models.diffusion_unet import DDPM, DiffusionUNet
from models.vae_baseline import ConditionalVAE
from engine.train_classifier import train_classifier


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_stage(stage, config, device):
    set_seed(42)

    if stage == "classifier":
        print("Training classifier on real images...")
        classifier = ClassifierBackbone(num_features=config["model"]["num_features"])
        train_classifier(config, classifier, device=device)

    elif stage == "baseline":
        print("Training baseline VAE...")
        num_features = config["data"]["num_features"]
        model = ConditionalVAE(
            image_channels=1,
            latent_dim=32,
            condition_dim=num_features,
        )
        model = model.to(device)
        print(f"Baseline VAE initialized with {sum(p.numel() for p in model.parameters())} parameters")

    elif stage == "main":
        print("Training main conditional diffusion model...")
        num_features = config["data"]["num_features"]
        unet = DiffusionUNet(
            in_channels=1,
            out_channels=1,
            condition_dim=num_features,
        )
        model = DDPM(
            unet,
            T=config["model"]["T"],
            beta_start=config["model"]["beta_start"],
            beta_end=config["model"]["beta_end"],
            device=device,
        )
        model = model.to(device)
        print(f"DDPM initialized with {sum(p.numel() for p in model.parameters())} parameters")

    elif stage == "test":
        print("Running held-out evaluation...")

    else:
        raise ValueError(f"Unknown stage: {stage}")

    print(f"Environment: {capture_env()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["classifier", "baseline", "main", "test"],
        required=True,
        help="Training stage",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    config = load_config(args.config)
    run_stage(args.stage, config, args.device)
