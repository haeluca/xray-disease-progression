import torch
import torchvision.transforms as transforms
from PIL import Image
import os


def create_validation_grid(generated_images, real_images=None, conditions=None, output_path="validation_grid.png", grid_size=4):
    B = generated_images.shape[0]
    num_images = min(B, grid_size * grid_size)

    generated_images = generated_images[:num_images]

    num_cols = 1
    if real_images is not None:
        num_cols += 1
    if conditions is not None:
        num_cols += 1

    cell_h, cell_w = 256, 256
    grid_h = grid_size * cell_h
    grid_w = num_cols * grid_size * cell_w

    grid = Image.new('L', (grid_w, grid_h), 255)

    for idx in range(num_images):
        row = idx // grid_size
        col = idx % grid_size

        gen_img = generated_images[idx].squeeze(0).cpu()
        gen_img = (gen_img * 0.5 + 0.5).clamp(0, 1)
        gen_img = transforms.ToPILImage()(gen_img)

        x = col * cell_w
        y = row * cell_h
        grid.paste(gen_img, (x, y))

        if real_images is not None:
            real_img = real_images[idx].squeeze(0).cpu()
            real_img = (real_img * 0.5 + 0.5).clamp(0, 1)
            real_img = transforms.ToPILImage()(real_img)

            x = (grid_size + col) * cell_w
            grid.paste(real_img, (x, y))

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    grid.save(output_path)
    print(f"Validation grid saved to {output_path}")
