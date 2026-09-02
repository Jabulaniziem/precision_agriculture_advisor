"""
generate_leaf_dataset.py

Generates a SIMULATED leaf-image dataset (healthy vs diseased) for the
Precision Agriculture project's CNN disease-detection component.

Real labelled datasets (e.g. PlantVillage on Kaggle, ~50,000 images)
are the standard choice for this task, but require an internet
download. This script generates synthetic leaf images with the same
underlying visual logic - healthy leaves are mostly uniform green,
diseased leaves have brown/yellow lesion spots and discoloration - so
the CNN pipeline, training, and evaluation code below is fully working
and can be pointed at the real PlantVillage dataset later by simply
swapping the data-loading step.

State clearly in your documentation that these images are simulated
for demonstration, and that PlantVillage (Kaggle) is the intended
real-world data source referenced in your Data section.
"""

import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

RANDOM_SEED = 42
IMG_SIZE = 128
N_PER_CLASS_TRAIN = 600
N_PER_CLASS_VAL = 120
OUT_DIR = "leaf_dataset"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


def make_leaf_base(size):
    """Draws a rough leaf-shaped green blob with vein texture."""
    img = Image.new("RGB", (size, size), (235, 245, 230))
    draw = ImageDraw.Draw(img)

    base_green = (
        random.randint(40, 80),
        random.randint(110, 160),
        random.randint(30, 60),
    )
    cx, cy = size // 2, size // 2
    rx, ry = size * 0.38, size * 0.46
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=base_green)

    # Vein line
    draw.line([(cx, cy - ry * 0.9), (cx, cy + ry * 0.9)], fill=(20, 70, 20), width=2)
    for i in range(-3, 4):
        if i == 0:
            continue
        y = cy + i * ry * 0.22
        draw.line([(cx, y), (cx + i * rx * 0.35, y - ry * 0.15)], fill=(20, 70, 20), width=1)

    return img, base_green, (cx, cy, rx, ry)


def add_disease_spots(img, leaf_geom, severity):
    """Adds brown/yellow lesion spots proportional to severity (0-1)."""
    draw = ImageDraw.Draw(img)
    cx, cy, rx, ry = leaf_geom
    n_spots = int(3 + severity * 25)

    for _ in range(n_spots):
        angle = random.uniform(0, 2 * np.pi)
        dist = random.uniform(0, 0.85)
        sx = cx + np.cos(angle) * rx * dist
        sy = cy + np.sin(angle) * ry * dist
        r = random.uniform(2, 4 + severity * 6)

        spot_color = random.choice([
            (120, 80, 30),   # brown lesion
            (170, 150, 40),  # yellow/chlorotic
            (90, 60, 20),    # dark necrotic
        ])
        draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=spot_color)

    return img


def generate_image(diseased: bool):
    img, base_green, geom = make_leaf_base(IMG_SIZE)
    if diseased:
        severity = random.uniform(0.4, 1.0)
        img = add_disease_spots(img, geom, severity)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))

    # Small amount of noise/lighting variation for realism
    arr = np.array(img).astype(np.int16)
    arr += np.random.randint(-8, 8, arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def build_split(split_name: str, n_per_class: int):
    for label in ["healthy", "diseased"]:
        out_path = os.path.join(OUT_DIR, split_name, label)
        os.makedirs(out_path, exist_ok=True)
        for i in range(n_per_class):
            img = generate_image(diseased=(label == "diseased"))
            img.save(os.path.join(out_path, f"{label}_{i:04d}.png"))
    print(f"{split_name}: {n_per_class} healthy + {n_per_class} diseased images -> "
          f"{os.path.join(OUT_DIR, split_name)}")


def main():
    build_split("train", N_PER_CLASS_TRAIN)
    build_split("val", N_PER_CLASS_VAL)
    print("\nDone. Folder structure:")
    print(f"  {OUT_DIR}/train/healthy, {OUT_DIR}/train/diseased")
    print(f"  {OUT_DIR}/val/healthy, {OUT_DIR}/val/diseased")


if __name__ == "__main__":
    main()
