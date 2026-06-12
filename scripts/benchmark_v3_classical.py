"""
Benchmark V3: fair classical denoising baseline.

Run from repo root, for example:
    python scripts/benchmark_v3_classical.py --dataset-name set12 --data-dir "Data/Gray test/Set12" --sigmas 15 25 50

Design choices:
- Loads real image files from a dataset directory. No fake Set12 fallback.
- Converts all images to grayscale float [0, 1].
- Generates AWGN once per image/sigma/seed and reuses the same noisy input for every method.
- Records status/error per method so long runs do not silently corrupt the CSV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from skimage import color, io
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.classical_v3 import (  # noqa: E402
    MethodSpec,
    ensure_float01,
    float01_to_uint8,
    get_method_registry,
)

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_SIGMAS = [5, 10, 15, 20, 25, 30, 35, 40, 50, 75]
DEFAULT_METHODS = ["gaussian_scipy", "median_scipy", "nlm_fast", "nlm_ipol", "bm3d_tau"]


def stable_int_seed(base_seed: int, dataset_name: str, image_name: str, sigma: float) -> int:
    text = f"{base_seed}|{dataset_name}|{image_name}|{float(sigma):.6f}"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % (2**32 - 1)


def add_awgn_float01(clean: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    sigma_norm = float(sigma) / 255.0
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma_norm, size=clean.shape).astype(np.float32)
    return np.clip(clean.astype(np.float32) + noise, 0.0, 1.0).astype(np.float32)


def image_to_gray_float01(path: Path) -> np.ndarray:
    arr = io.imread(str(path))

    # Drop alpha if present.
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]

    if arr.ndim == 3:
        arr_float = ensure_float01(arr)
        arr_gray = color.rgb2gray(arr_float)
        return ensure_float01(arr_gray)

    if arr.ndim != 2:
        raise ValueError(f"Unsupported image shape {arr.shape} for {path}")

    return ensure_float01(arr)


def list_image_files(data_dir: Path, recursive: bool = False) -> List[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")
    pattern = "**/*" if recursive else "*"
    files = [p for p in data_dir.glob(pattern) if p.is_file() and p.suffix.lower() in VALID_EXTS]
    return sorted(files, key=lambda p: p.name.lower())


def load_dataset(data_dir: Path, recursive: bool = False, image_limit: int | None = None) -> Dict[str, np.ndarray]:
    files = list_image_files(data_dir, recursive=recursive)
    if image_limit is not None:
        files = files[: int(image_limit)]
    if not files:
        raise RuntimeError(f"No image files found in {data_dir}. Supported: {sorted(VALID_EXTS)}")

    images: Dict[str, np.ndarray] = {}
    for path in files:
        key = path.stem
        if key in images:
            key = path.relative_to(data_dir).as_posix().replace("/", "__").rsplit(".", 1)[0]
        images[key] = image_to_gray_float01(path)
    return images


def compute_metrics(clean: np.ndarray, result: np.ndarray) -> Tuple[float, float]:
    clean = ensure_float01(clean)
    result = ensure_float01(result)
    psnr = peak_signal_noise_ratio(clean, result, data_range=1.0)
    # Explicit data_range is important for float images.
    ssim = structural_similarity(clean, result, data_range=1.0)
    return float(psnr), float(ssim)


def run_method(method: MethodSpec, noisy: np.ndarray, sigma: float, trials: int, warmup: int) -> Tuple[np.ndarray, float]:
    # Warmup is useful for implementations with first-call overhead.
    for _ in range(max(0, warmup)):
        _ = method.fn(noisy, sigma)

    runtimes: List[float] = []
    result = None
    for _ in range(max(1, trials)):
        t0 = time.perf_counter()
        result = method.fn(noisy, sigma)
        runtimes.append((time.perf_counter() - t0) * 1000.0)

    assert result is not None
    return ensure_float01(result), float(np.mean(runtimes))


def save_debug_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    io.imsave(str(path), float01_to_uint8(image), check_contrast=False)


def write_partial(df_records: List[dict], output_raw: Path) -> None:
    output_raw.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(df_records).to_csv(output_raw, index=False)


def benchmark(args: argparse.Namespace) -> pd.DataFrame:
    data_dir = Path(args.data_dir)
    images = load_dataset(data_dir, recursive=args.recursive, image_limit=args.image_limit)

    registry = get_method_registry(include_ksvd=args.include_ksvd)
    selected = args.methods if args.methods else DEFAULT_METHODS
    unknown = [m for m in selected if m not in registry]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Available: {sorted(registry)}")
    methods = {key: registry[key] for key in selected}

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_raw = output_dir / f"benchmark_v3_{args.dataset_name}_raw.csv"

    metadata = {
        "dataset_name": args.dataset_name,
        "data_dir": str(data_dir),
        "num_images": len(images),
        "sigmas": args.sigmas,
        "methods": [methods[k].name for k in methods],
        "seed": args.seed,
        "trials": args.trials,
        "warmup": args.warmup,
        "python": sys.version,
        "platform": platform.platform(),
    }
    (output_dir / f"benchmark_v3_{args.dataset_name}_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    records: List[dict] = []
    total = len(images) * len(args.sigmas) * (len(methods) + 1)
    done = 0

    print(f"\nBenchmark V3: {args.dataset_name}")
    print(f"  data_dir: {data_dir}")
    print(f"  images:   {len(images)}")
    print(f"  sigmas:   {args.sigmas}")
    print(f"  methods:  {[methods[k].name for k in methods]}")
    print(f"  total rows: {total}\n")

    for image_name, clean in images.items():
        height, width = clean.shape[:2]
        for sigma in args.sigmas:
            per_case_seed = stable_int_seed(args.seed, args.dataset_name, image_name, sigma)
            noisy = add_awgn_float01(clean, sigma=sigma, seed=per_case_seed)

            noisy_psnr, noisy_ssim = compute_metrics(clean, noisy)
            records.append(
                {
                    "dataset": args.dataset_name,
                    "image": image_name,
                    "height": height,
                    "width": width,
                    "sigma": sigma,
                    "seed": per_case_seed,
                    "method": "Noisy",
                    "implementation": "clean + AWGN, clipped to [0,1]",
                    "psnr": round(noisy_psnr, 6),
                    "ssim": round(noisy_ssim, 8),
                    "runtime_ms": 0.0,
                    "status": "ok",
                    "error": "",
                    "notes": "same noisy image reused for all methods",
                }
            )
            done += 1
            print(f"[{done:>4}/{total}] {image_name} | Noisy | sigma={sigma} -> PSNR={noisy_psnr:.3f}")

            if args.save_images and sigma in args.save_image_sigmas:
                save_debug_image(output_dir / "images" / args.dataset_name / f"{image_name}_sigma{sigma}_noisy.png", noisy)

            for method_key, method in methods.items():
                done += 1
                print(f"[{done:>4}/{total}] {image_name} | {method.name} | sigma={sigma}", end="", flush=True)
                try:
                    result, runtime_ms = run_method(method, noisy, float(sigma), trials=args.trials, warmup=args.warmup)
                    psnr, ssim = compute_metrics(clean, result)
                    status = "ok"
                    error = ""
                    print(f" -> PSNR={psnr:.3f}, SSIM={ssim:.4f}, {runtime_ms:.1f} ms")

                    if args.save_images and sigma in args.save_image_sigmas:
                        safe_method = method.name.replace("/", "_").replace(" ", "_")
                        save_debug_image(
                            output_dir / "images" / args.dataset_name / f"{image_name}_sigma{sigma}_{safe_method}.png",
                            result,
                        )
                except Exception as exc:
                    psnr = np.nan
                    ssim = np.nan
                    runtime_ms = np.nan
                    status = "failed"
                    error = repr(exc)
                    print(f" -> FAILED: {error}")

                records.append(
                    {
                        "dataset": args.dataset_name,
                        "image": image_name,
                        "height": height,
                        "width": width,
                        "sigma": sigma,
                        "seed": per_case_seed,
                        "method": method.name,
                        "implementation": method.implementation,
                        "psnr": None if np.isnan(psnr) else round(float(psnr), 6),
                        "ssim": None if np.isnan(ssim) else round(float(ssim), 8),
                        "runtime_ms": None if np.isnan(runtime_ms) else round(float(runtime_ms), 3),
                        "status": status,
                        "error": error,
                        "notes": method.notes,
                    }
                )

                if args.write_every > 0 and len(records) % args.write_every == 0:
                    write_partial(records, output_raw)

    df = pd.DataFrame(records)
    df.to_csv(output_raw, index=False)
    print(f"\nSaved raw results: {output_raw}")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V3 classical image-denoising benchmark")
    parser.add_argument("--dataset-name", required=True, help="Name stored in CSV, e.g. set12 or bsd68")
    parser.add_argument("--data-dir", required=True, help="Directory containing real test images")
    parser.add_argument("--sigmas", nargs="+", type=int, default=DEFAULT_SIGMAS)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/v3")
    parser.add_argument("--trials", type=int, default=1, help="Repeated timing trials per method")
    parser.add_argument("--warmup", type=int, default=0, help="Warmup calls before timed trials")
    parser.add_argument("--image-limit", type=int, default=None, help="Optional limit for smoke tests")
    parser.add_argument("--recursive", action="store_true", help="Search images recursively")
    parser.add_argument("--include-ksvd", action="store_true", help="Enable optional existing K-SVD bridge")
    parser.add_argument("--save-images", action="store_true", help="Save noisy and denoised sample outputs")
    parser.add_argument("--save-image-sigmas", nargs="+", type=int, default=[15, 25, 50])
    parser.add_argument("--write-every", type=int, default=20, help="Write partial CSV every N records")
    return parser.parse_args()


if __name__ == "__main__":
    benchmark(parse_args())
