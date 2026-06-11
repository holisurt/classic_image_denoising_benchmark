"""
Stage 2 — Hyperparameter Study for Classic Image Denoising Benchmark

Run from the repository root, for example:

python scripts/hyperparameter_study.py --dataset set12 --image 01 --sigma 25
python scripts/hyperparameter_study.py --dataset set12 --image 01 --sigma 25 --run-ksvd

Outputs:
results/hyperparams/<dataset>_<image>_sigma<sigma>_hyperparams_raw.csv
results/hyperparams/<dataset>_<image>_sigma<sigma>_hyperparams_summary.csv
results/hyperparams/figures/*.png
"""

import sys
import time
import argparse
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, color, img_as_ubyte

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from noise import add_gaussian_noise
from filters import gaussian_filter
from metrics import psnr, ssim

try:
    from nlm import nlm_denoise_fast
    HAS_NLM = True
except Exception as e:
    print(f"[WARN] Cannot import NLM: {e}")
    HAS_NLM = False

try:
    from bm3d_wrapper import bm3d_denoise
    HAS_BM3D = True
except Exception as e:
    print(f"[WARN] Cannot import BM3D: {e}")
    HAS_BM3D = False

try:
    from ksvd_denoising import ksvd_denoise
    HAS_KSVD = True
except Exception as e:
    print(f"[WARN] Cannot import K-SVD: {e}")
    HAS_KSVD = False


def read_grayscale_uint8(path):
    img = io.imread(path)
    if img.ndim == 3:
        img = color.rgb2gray(img)
        img = img_as_ubyte(img)
    else:
        if img.dtype != np.uint8:
            img = img_as_ubyte(img)
    return img.astype(np.uint8)


def get_dataset_folder(dataset):
    dataset = dataset.lower()
    if dataset == "set12":
        return ROOT / "data" / "Set12"
    if dataset == "bsd68":
        return ROOT / "data" / "BSD68"
    raise ValueError("dataset must be set12 or bsd68")


def find_image_path(dataset, image_name):
    folder = get_dataset_folder(dataset)
    candidates = []
    for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
        candidates.append(folder / f"{image_name}{ext}")
        if str(image_name).isdigit():
            candidates.append(folder / f"{int(image_name):02d}{ext}")

    for path in candidates:
        if path.exists():
            return path

    available = sorted([p.stem for p in folder.glob("*") if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]])
    raise FileNotFoundError(
        f"Image '{image_name}' not found in {folder}. "
        f"Available examples: {available[:20]}"
    )


def run_method(clean, noisy, method_name, param_dict):
    t0 = time.perf_counter()

    if method_name == "Gaussian":
        denoised = gaussian_filter(
            noisy,
            kernel_size=int(param_dict["kernel_size"]),
            sigma=float(param_dict["sigma_filter"]),
        )

    elif method_name == "NLM":
        denoised = nlm_denoise_fast(
            noisy,
            patch_size=int(param_dict["patch_size"]),
            search_size=int(param_dict["search_size"]),
            h=float(param_dict["h"]),
        )

    elif method_name == "BM3D":
        denoised = bm3d_denoise(
            noisy,
            sigma_psd=float(param_dict["sigma_psd"]),
        )

    elif method_name == "K-SVD":
        denoised, _, _ = ksvd_denoise(
            noisy,
            patch_size=int(param_dict["patch_size"]),
            dict_size=int(param_dict["dict_size"]),
            sparsity=int(param_dict["sparsity"]),
            iterations=int(param_dict["iterations"]),
        )

    else:
        raise ValueError(f"Unknown method: {method_name}")

    runtime_ms = (time.perf_counter() - t0) * 1000.0
    denoised = np.clip(denoised, 0, 255).astype(np.uint8)

    return {
        "psnr": psnr(clean, denoised),
        "ssim": ssim(clean, denoised),
        "runtime_ms": runtime_ms,
    }


def make_experiments(sigma, run_ksvd=False):
    experiments = []

    for kernel_size, sigma_filter in product([3, 5, 7, 9], [0.5, 1.0, 1.5, 2.0, 3.0]):
        experiments.append({
            "method": "Gaussian",
            "params": {
                "kernel_size": kernel_size,
                "sigma_filter": sigma_filter,
            }
        })

    if HAS_NLM:
        for patch_size, search_size, h_factor in product([3, 5, 7], [11, 21, 31], [0.6, 0.8, 1.0, 1.2, 1.5]):
            experiments.append({
                "method": "NLM",
                "params": {
                    "patch_size": patch_size,
                    "search_size": search_size,
                    "h": sigma * h_factor,
                    "h_factor": h_factor,
                }
            })

    if HAS_BM3D:
        for sigma_factor in [0.5, 0.75, 1.0, 1.25, 1.5]:
            experiments.append({
                "method": "BM3D",
                "params": {
                    "sigma_psd": sigma * sigma_factor,
                    "sigma_factor": sigma_factor,
                }
            })

    if run_ksvd and HAS_KSVD:
        for patch_size, dict_size, sparsity, iterations in product([6, 8], [64, 128], [2, 3, 4], [3, 5]):
            experiments.append({
                "method": "K-SVD",
                "params": {
                    "patch_size": patch_size,
                    "dict_size": dict_size,
                    "sparsity": sparsity,
                    "iterations": iterations,
                }
            })

    return experiments


def run_hyperparameter_study(dataset, image_name, sigma, seed, run_ksvd=False):
    image_path = find_image_path(dataset, image_name)
    clean = read_grayscale_uint8(image_path)
    noisy = add_gaussian_noise(clean, sigma=sigma, seed=seed)

    print("=" * 70)
    print("STAGE 2 — HYPERPARAMETER STUDY")
    print("=" * 70)
    print(f"Dataset     : {dataset}")
    print(f"Image       : {image_path.name}")
    print(f"Sigma       : {sigma}")
    print(f"Seed        : {seed}")
    print(f"Noisy PSNR  : {psnr(clean, noisy):.2f} dB")
    print(f"Noisy SSIM  : {ssim(clean, noisy):.4f}")
    print("=" * 70)

    experiments = make_experiments(sigma=sigma, run_ksvd=run_ksvd)
    records = []

    for idx, exp in enumerate(experiments, start=1):
        method = exp["method"]
        params = exp["params"]
        print(f"[{idx}/{len(experiments)}] {method} | {params}", end="")

        try:
            result = run_method(clean, noisy, method, params)
            record = {
                "dataset": dataset,
                "image": image_path.stem,
                "sigma": sigma,
                "seed": seed,
                "method": method,
                "psnr": result["psnr"],
                "ssim": result["ssim"],
                "runtime_ms": result["runtime_ms"],
            }
            record.update(params)
            records.append(record)

            print(
                f" -> PSNR={result['psnr']:.2f}, "
                f"SSIM={result['ssim']:.4f}, "
                f"time={result['runtime_ms']:.1f} ms"
            )

        except Exception as e:
            record = {
                "dataset": dataset,
                "image": image_path.stem,
                "sigma": sigma,
                "seed": seed,
                "method": method,
                "psnr": np.nan,
                "ssim": np.nan,
                "runtime_ms": np.nan,
                "error": str(e),
            }
            record.update(params)
            records.append(record)
            print(f" -> ERROR: {e}")

    df = pd.DataFrame(records)

    out_dir = ROOT / "results" / "hyperparams"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    raw_path = out_dir / f"{dataset}_{image_path.stem}_sigma{sigma}_hyperparams_raw.csv"
    top5_path = out_dir / f"{dataset}_{image_path.stem}_sigma{sigma}_hyperparams_top5.csv"

    df.to_csv(raw_path, index=False)

    top5 = (
        df.dropna(subset=["psnr"])
          .sort_values(["method", "psnr"], ascending=[True, False])
          .groupby("method", as_index=False)
          .head(5)
    )
    top5.to_csv(top5_path, index=False)

    print()
    print(f"Saved raw   : {raw_path}")
    print(f"Saved top 5 : {top5_path}")

    return df, raw_path, top5_path


def plot_best_psnr_by_method(df, output_path):
    best = (
        df.dropna(subset=["psnr"])
          .sort_values("psnr", ascending=False)
          .groupby("method", as_index=False)
          .first()
          .sort_values("psnr", ascending=False)
    )

    plt.figure(figsize=(8, 4))
    plt.bar(best["method"], best["psnr"])
    plt.ylabel("Best PSNR (dB)")
    plt.title("Best Hyperparameter Result by Method")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {output_path}")


def plot_psnr_runtime(df, output_path):
    valid = df.dropna(subset=["psnr", "runtime_ms"])

    plt.figure(figsize=(7, 5))
    for method in valid["method"].unique():
        sub = valid[valid["method"] == method]
        plt.scatter(sub["runtime_ms"], sub["psnr"], label=method)

    plt.xscale("log")
    plt.xlabel("Runtime (ms, log scale)")
    plt.ylabel("PSNR (dB)")
    plt.title("PSNR vs Runtime Trade-off")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["set12", "bsd68"], default="set12")
    parser.add_argument("--image", default="01", help="Image stem, e.g. 01 or test001")
    parser.add_argument("--sigma", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-ksvd", action="store_true")
    args = parser.parse_args()

    df, raw_path, top5_path = run_hyperparameter_study(
        dataset=args.dataset,
        image_name=args.image,
        sigma=args.sigma,
        seed=args.seed,
        run_ksvd=args.run_ksvd,
    )

    fig_dir = ROOT / "results" / "hyperparams" / "figures"
    image_stem = find_image_path(args.dataset, args.image).stem

    plot_best_psnr_by_method(
        df,
        fig_dir / f"{args.dataset}_{image_stem}_sigma{args.sigma}_best_psnr.png",
    )

    plot_psnr_runtime(
        df,
        fig_dir / f"{args.dataset}_{image_stem}_sigma{args.sigma}_psnr_runtime.png",
    )


if __name__ == "__main__":
    main()
