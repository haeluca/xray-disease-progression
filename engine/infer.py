import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms


def save_image(tensor, path):
    tensor = tensor.squeeze(0).cpu()
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
    img = transforms.ToPILImage()(tensor)
    img.save(path)


def generate_samples(model, condition_images, condition_vectors=None, device="cuda", output_dir=None):
    model = model.to(device)
    model.eval()

    output_dir = Path(output_dir) if output_dir else Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        condition_images = condition_images.to(device)

        if condition_vectors is not None:
            condition_vectors = condition_vectors.to(device)

        generated = model.sample(
            x_condition=condition_images,
            shape=condition_images.shape,
            condition_vector=condition_vectors,
        )

    return generated
