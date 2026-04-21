import torch
import os
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

from config import DEVICE, IMAGE_SIZE, MODEL_SAVE_PATH
from models.unet import UNet
from models.diffusion import DDPM

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
    img = Image.open(image_path).convert("L")
    return transform(img).unsqueeze(0)

def tensor_to_image(tensor):
    tensor = tensor.squeeze(0).cpu()
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
    return transforms.ToPILImage()(tensor)

def sample(baseline_xray_path, checkpoint_path, output_path):
    # Load model
    unet = UNet(in_channels=2, out_channels=1).to(DEVICE)
    model = DDPM(unet).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    # Load conditioning X-ray
    x_t0 = load_image(baseline_xray_path).to(DEVICE)

    # Generate sample
    with torch.no_grad():
        x_t1_gen = model.sample(x_t0, shape=x_t0.shape)

    # Save output
    img = tensor_to_image(x_t1_gen)
    img.save(output_path)
    print(f"Generated X-ray saved to {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic future X-rays using DDPM")
    parser.add_argument("--baseline", type=str, required=True, help="Path to baseline X-ray image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="generated_xray.png", help="Output path for generated image")

    args = parser.parse_args()

    if not os.path.exists(args.baseline):
        raise FileNotFoundError(f"Baseline X-ray not found: {args.baseline}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    sample(args.baseline, args.checkpoint, args.output)
