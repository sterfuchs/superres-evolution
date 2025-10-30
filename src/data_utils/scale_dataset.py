"""
make_scaled_dataset.py

Downscale all images in a directory by a given factor using bicubic interpolation.

Usage:
    python src/data_utils/make_scaled_dataset.py \
        --input data/raw/DIV2K_train_HR \
        --output data/raw/DIV2K_train_LR_bicubic_X4 \
        --scale 4
"""

import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def downscale_images(input_dir: Path, output_dir: Path, scale: int):
    """
    Downscale all images in input_dir by the given scale factor using bicubic interpolation.

    Args:
        input_dir (Path): Path to the folder containing high-resolution images.
        output_dir (Path): Where to save the downscaled images.
        scale (int): Downscale factor (e.g., 2, 3, 4).
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(list(input_dir.glob("*.[pj][pn]g")))  # matches .jpg/.png
    if not img_files:
        raise FileNotFoundError(f"No .jpg/.png images found in {input_dir}")

    print(f"Downscaling {len(img_files)} images by ×{scale}...")
    for img_path in tqdm(img_files, ncols=80):
        with Image.open(img_path) as img:
            w, h = img.size
            new_size = (w // scale, h // scale)
            img_resized = img.resize(new_size, resample=Image.BICUBIC)
            img_resized.save(output_dir / img_path.name)

    print(f"Saved downscaled images to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Downscale a directory of images.")
    parser.add_argument("--input", type=str, required=True, help="Path to HR image folder")
    parser.add_argument("--output", type=str, required=True, help="Path to save LR images")
    parser.add_argument("--scale", type=int, required=True, help="Downscale factor (e.g., 2, 3, 4)")
    args = parser.parse_args()

    downscale_images(args.input, args.output, args.scale)


if __name__ == "__main__":
    main()