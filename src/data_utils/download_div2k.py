import os
import requests, zipfile, io
from pathlib import Path
from PIL import Image

DIV2K_URLS = {
    "train_HR": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "valid_HR": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
}

def download_and_extract(url, dest):
    print(f"Downloading {url}...")
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(dest)
    print("Done.")

def prepare_div2k(base="data/raw/DIV2K", scale=2):
    Path(base).mkdir(parents=True, exist_ok=True)
    for k, url in DIV2K_URLS.items():
        download_and_extract(url, base)
    # Optional: create LR versions
    hr_dir = Path(base) / "DIV2K_train_HR"
    lr_dir = Path(base) / f"DIV2K_train_LR_bicubic_X{scale}"
    lr_dir.mkdir(exist_ok=True)
    for img_path in hr_dir.glob("*.png"):
        img = Image.open(img_path)
        w, h = img.size
        img.resize((w // scale, h // scale), Image.BICUBIC).save(
            lr_dir / img_path.name
        )

if __name__ == "__main__":
    prepare_div2k()
