from pathlib import Path
from skimage import io
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

DATASETS = {
    "Set12": ROOT / "data" / "Set12",
    "BSD68": ROOT / "data" / "BSD68",
}

def check_dataset(name, path):
    files = sorted(list(path.glob("*.png")) + list(path.glob("*.jpg")) + list(path.glob("*.bmp")))

    print(f"\n{name}")
    print("-" * 40)
    print(f"Path: {path}")
    print(f"Images: {len(files)}")

    if len(files) == 0:
        print("ERROR: No image files found.")
        return

    for f in files[:5]:
        img = io.imread(f, as_gray=True)
        print(f"{f.name:20s} shape={img.shape}, dtype={img.dtype}, range=({img.min():.3f}, {img.max():.3f})")

    if name == "Set12" and len(files) != 12:
        print("WARNING: Set12 should contain 12 images.")

    if name == "BSD68" and len(files) != 68:
        print("WARNING: BSD68 should contain 68 images.")

if __name__ == "__main__":
    for name, path in DATASETS.items():
        check_dataset(name, path)