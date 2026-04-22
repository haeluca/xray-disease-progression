import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw


def generate_mock_xray(seed, size=512, add_ellipse=True):
    rng = np.random.default_rng(seed)
    base = rng.normal(loc=80, scale=15, size=(size, size))

    yy, xx = np.mgrid[0:size, 0:size]
    cy, cx = size // 2, size // 2
    bone_mask = ((xx - cx) ** 2 / (size * 0.15) ** 2 + (yy - cy) ** 2 / (size * 0.35) ** 2) < 1
    base[bone_mask] += rng.uniform(80, 140)

    joint_y = int(size * 0.45 + rng.integers(-20, 20))
    joint_mask = (np.abs(yy - joint_y) < 8) & bone_mask
    base[joint_mask] -= 50

    base += rng.normal(0, 10, size=base.shape)
    base = np.clip(base, 0, 255).astype(np.uint8)

    img = Image.fromarray(base, mode='L')

    if add_ellipse:
        draw = ImageDraw.Draw(img)
        ox = cx + int(rng.integers(-30, 30))
        oy = cy + int(rng.integers(-30, 30))
        r = rng.integers(10, 25)
        draw.ellipse([ox - r, oy - r, ox + r, oy + r], outline=200, width=2)

    return img


def main(out_dir, num_patients=20, images_per_patient=2):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in range(num_patients):
        pid = f"P{p:03d}"
        patient_dir = out_dir / pid
        patient_dir.mkdir(exist_ok=True)

        for side in ['left', 'right']:
            seed = p * 100 + (0 if side == 'left' else 1)
            img = generate_mock_xray(seed=seed)
            img.save(patient_dir / f"{pid}_{side}.png")

    print(f"Generated {num_patients} patients (x2 sides) in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data/raw")
    parser.add_argument("--num_patients", type=int, default=20)
    args = parser.parse_args()
    main(args.out_dir, args.num_patients)
