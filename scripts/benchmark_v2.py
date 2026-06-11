"""
Benchmark v2:
- Uses real Set12 / BSD68 folders
- Fixed seed for reproducible AWGN
- Saves raw result per image
- Saves mean/std summary
"""

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io, color, img_as_ubyte

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from noise import add_gaussian_noise
from filters import gaussian_filter, median_filter
from metrics import psnr, ssim


def read_grayscale_uint8(path):
    img = io.imread(path)

    if img.ndim == 3:
        img = color.rgb2gray(img)
        img = img_as_ubyte(img)
    else:
        if img.dtype != np.uint8:
            img = img_as_ubyte(img)

    return img.astype(np.uint8)


def load_image_folder(folder):
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Dataset folder not found: {folder}")

    files = sorted(
        list(folder.glob("*.png"))
        + list(folder.glob("*.jpg"))
        + list(folder.glob("*.jpeg"))
        + list(folder.glob("*.bmp"))
    )

    if len(files) == 0:
        raise RuntimeError(f"No images found in: {folder}")

    images = {}
    for f in files:
        images[f.stem] = read_grayscale_uint8(f)

    return images


def load_dataset(dataset_name):
    dataset_name = dataset_name.lower()

    if dataset_name == "set12":
        path = ROOT / "data" / "Set12"
    elif dataset_name == "bsd68":
        path = ROOT / "data" / "BSD68"
    else:
        raise ValueError("dataset must be 'set12' or 'bsd68'")

    images = load_image_folder(path)
    return images, path


def load_methods(skip_slow=False):
    methods = {}

    methods["Gaussian"] = lambda img, sigma: gaussian_filter(
        img, kernel_size=5, sigma=1.5
    )

    methods["Median"] = lambda img, sigma: median_filter(
        img, kernel_size=3
    )

    try:
        from nlm import nlm_denoise_fast
        methods["NLM"] = lambda img, sigma: nlm_denoise_fast(
            img, patch_size=5, search_size=21, h=sigma
        )
        print("Loaded: NLM")
    except Exception as e:
        print(f"Skipped NLM: {e}")

    try:
        from bm3d_wrapper import bm3d_denoise
        methods["BM3D"] = lambda img, sigma: bm3d_denoise(
            img, sigma_psd=sigma
        )
        print("Loaded: BM3D")
    except Exception as e:
        print(f"Skipped BM3D: {e}")

    if not skip_slow:
        try:
            from ksvd_denoising import ksvd_denoise

            def run_ksvd(img, sigma):
                result, _, _ = ksvd_denoise(
                    img,
                    dict_size=128,
                    patch_size=8,
                    iterations=5,
                    sparsity=3,
                )
                return result

            methods["KSVD"] = run_ksvd
            print("Loaded: KSVD")
        except Exception as e:
            print(f"Skipped KSVD: {e}")
    else:
        print("Skipped: KSVD")

    return methods


def benchmark_one_method(clean, noisy, method_fn, sigma):
    t0 = time.perf_counter()
    denoised = method_fn(noisy, sigma)
    runtime_ms = (time.perf_counter() - t0) * 1000

    denoised = np.clip(denoised, 0, 255).astype(np.uint8)

    return {
        "psnr": psnr(clean, denoised),
        "ssim": ssim(clean, denoised),
        "runtime_ms": runtime_ms,
    }


def run_benchmark(dataset, sigmas, seed, skip_slow):
    images, dataset_path = load_dataset(dataset)
    methods = load_methods(skip_slow=skip_slow)

    print()
    print("=" * 60)
    print("BENCHMARK V2")
    print("=" * 60)
    print(f"Dataset      : {dataset}")
    print(f"Dataset path : {dataset_path}")
    print(f"Images       : {len(images)}")
    print(f"Sigmas       : {sigmas}")
    print(f"Seed         : {seed}")
    print(f"Methods      : {list(methods.keys())}")
    print("=" * 60)

    records = []

    total = len(images) * len(sigmas) * (len(methods) + 1)
    done = 0

    for image_name, clean in images.items():
        for sigma in sigmas:
            noise_seed = seed + sigma * 1000 + abs(hash(image_name)) % 1000
            noisy = add_gaussian_noise(clean, sigma=sigma, seed=noise_seed)

            noisy_record = {
                "dataset": dataset,
                "image": image_name,
                "sigma": sigma,
                "method": "Noisy",
                "psnr": psnr(clean, noisy),
                "ssim": ssim(clean, noisy),
                "runtime_ms": 0.0,
                "seed": noise_seed,
            }
            records.append(noisy_record)

            done += 1
            print(f"[{done}/{total}] {image_name} | Noisy | sigma={sigma}")

            for method_name, method_fn in methods.items():
                done += 1
                print(f"[{done}/{total}] {image_name} | {method_name} | sigma={sigma}", end="")

                try:
                    result = benchmark_one_method(clean, noisy, method_fn, sigma)
                    record = {
                        "dataset": dataset,
                        "image": image_name,
                        "sigma": sigma,
                        "method": method_name,
                        "psnr": result["psnr"],
                        "ssim": result["ssim"],
                        "runtime_ms": result["runtime_ms"],
                        "seed": noise_seed,
                    }
                    records.append(record)

                    print(
                        f" -> PSNR={result['psnr']:.2f}, "
                        f"SSIM={result['ssim']:.4f}, "
                        f"time={result['runtime_ms']:.1f} ms"
                    )

                except Exception as e:
                    print(f" -> ERROR: {e}")
                    records.append({
                        "dataset": dataset,
                        "image": image_name,
                        "sigma": sigma,
                        "method": method_name,
                        "psnr": np.nan,
                        "ssim": np.nan,
                        "runtime_ms": np.nan,
                        "seed": noise_seed,
                    })

    return pd.DataFrame(records)


def make_summary(df):
    return (
        df
        .groupby(["dataset", "method", "sigma"])
        .agg(
            psnr_mean=("psnr", "mean"),
            psnr_std=("psnr", "std"),
            ssim_mean=("ssim", "mean"),
            ssim_std=("ssim", "std"),
            runtime_mean=("runtime_ms", "mean"),
            runtime_std=("runtime_ms", "std"),
            n_images=("image", "count"),
        )
        .reset_index()
        .round({
            "psnr_mean": 4,
            "psnr_std": 4,
            "ssim_mean": 6,
            "ssim_std": 6,
            "runtime_mean": 2,
            "runtime_std": 2,
        })
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["set12", "bsd68"], required=True)
    parser.add_argument("--sigmas", nargs="+", type=int, default=[15, 25, 50])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-slow", action="store_true")
    parser.add_argument("--output-dir", default="results")

    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = run_benchmark(
        dataset=args.dataset,
        sigmas=args.sigmas,
        seed=args.seed,
        skip_slow=args.skip_slow,
    )

    raw_path = output_dir / f"benchmark_v2_{args.dataset}_raw.csv"
    summary_path = output_dir / f"benchmark_v2_{args.dataset}_summary.csv"

    df.to_csv(raw_path, index=False)
    make_summary(df).to_csv(summary_path, index=False)

    print()
    print(f"Saved raw     : {raw_path}")
    print(f"Saved summary : {summary_path}")

    print()
    print("Average PSNR:")
    pivot = (
        make_summary(df)
        .pivot(index="method", columns="sigma", values="psnr_mean")
    )
    print(pivot)


if __name__ == "__main__":
    main()