"""
scripts/benchmark.py — Benchmark toàn diện 5 phương pháp denoising

Cấu trúc project:
    image-denoising/
    ├── src/           ← core modules
    ├── scripts/       ← file này nằm ở đây
    ├── notebooks/
    └── results/       ← output sẽ được ghi vào đây

Chạy từ root của project:
    python scripts/benchmark.py
    python scripts/benchmark.py --dataset bsd68 --sigmas 15 25 50
    python scripts/benchmark.py --skip-slow
"""

import sys, os, time
from pathlib import Path
import numpy as np
import pandas as pd
from skimage import data as skdata
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import warnings
warnings.filterwarnings('ignore')

# scripts/ nằm trong root/scripts/ → root = parent của scripts/
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

# Đổi working directory về root để results/ path đúng
os.chdir(ROOT)

from noise import add_gaussian_noise
from filters import gaussian_filter, median_filter
from metrics import psnr, ssim


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

def load_set12():
    """
    Set12: 12 ảnh grayscale chuẩn trong xử lý ảnh.
    Dùng ảnh từ scikit-image khi chưa có BSD68.
    """
    imgs = {
        'camera'    : skdata.camera(),
        'astronaut' : skdata.astronaut()[:,:,0],
        'coins'     : skdata.coins(),
        'clock'     : skdata.clock(),
        'page'      : skdata.page(),
        'horse'     : skdata.horse().astype(np.uint8) * 255,
        'chelsea'   : skdata.chelsea()[:,:,0],
        'coffee'    : skdata.coffee()[:,:,0],
        'moon'      : skdata.moon(),
        'text'      : skdata.text(),
        'brick'     : skdata.brick(),
        'grass'     : skdata.grass(),
    }
    return imgs


def load_bsd68_subset(bsd68_path=None):
    """
    Tải BSD68 subset.
    Nếu không có file, dùng Set12 thay thế.

    Để dùng BSD68 thực, download tại:
    https://github.com/cszn/FFDNet/tree/master/testsets

    Đặt vào: data/bsd68/
    """
    if bsd68_path and Path(bsd68_path).exists():
        import cv2
        imgs = {}
        fnames = sorted(Path(bsd68_path).glob('*.png'))[:20]   # lấy 20 ảnh
        for f in fnames:
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                imgs[f.stem] = img
        print(f"  Loaded {len(imgs)} BSD68 images from {bsd68_path}")
        return imgs
    else:
        print("  BSD68 not found — using Set12 (scikit-image) instead")
        return load_set12()


# ─────────────────────────────────────────────
# Methods
# ─────────────────────────────────────────────

def load_methods():
    """
    Load tất cả 5 denoising methods.
    Trả về dict: name → callable(noisy_img, sigma)
    """
    methods = {}

    # 1. Gaussian Filter
    methods['Gaussian'] = lambda img, sigma: gaussian_filter(img, kernel_size=5, sigma=1.5)

    # 2. Median Filter
    methods['Median'] = lambda img, sigma: median_filter(img, kernel_size=3)

    # 3. NLM
    try:
        from nlm import nlm_denoise_fast
        methods['NLM'] = lambda img, sigma: nlm_denoise_fast(img, patch_size=5,
                                                              search_size=21, h=sigma)
        print("  ✓ NLM loaded")
    except ImportError:
        print("  ✗ NLM not available")

    # 4. BM3D
    try:
        from bm3d_wrapper import bm3d_denoise
        methods['BM3D'] = lambda img, sigma: bm3d_denoise(img, sigma_psd=sigma)
        print("  ✓ BM3D loaded")
    except ImportError:
        print("  ✗ BM3D not available (pip install bm3d)")

    # 5. K-SVD
    try:
        from ksvd_denoising import ksvd_denoise
        def run_ksvd(img, sigma):
            result, _, _ = ksvd_denoise(img, dict_size=128,
                                        patch_size=8, iterations=5, sparsity=3)
            return result
        methods['KSVD'] = run_ksvd
        print("  ✓ K-SVD loaded")
    except ImportError:
        print("  ✗ K-SVD not available")

    return methods


# ─────────────────────────────────────────────
# Benchmark Core
# ─────────────────────────────────────────────

def benchmark_single(img, method_fn, sigma, n_trials=1):
    """
    Benchmark 1 ảnh × 1 method × 1 sigma.

    Returns: {'psnr', 'ssim', 'runtime_ms'}
    """
    noisy = add_gaussian_noise(img, sigma=sigma)

    # Đo thời gian chạy
    runtimes = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        result = method_fn(noisy, sigma)
        runtimes.append((time.perf_counter() - t0) * 1000)   # ms

    result = np.clip(result, 0, 255).astype(np.uint8)

    return {
        'psnr'       : round(psnr(img, result), 4),
        'ssim'       : round(ssim(img, result), 6),
        'runtime_ms' : round(np.mean(runtimes), 1),
    }


def run_benchmark(dataset='set12', sigmas=(15, 25, 50),
                  bsd68_path='data/bsd68',
                  skip_slow=False, verbose=True):
    """
    Chạy benchmark đầy đủ.

    Parameters:
    -----------
    dataset    : 'set12' | 'bsd68'
    sigmas     : tuple of noise levels to test
    skip_slow  : skip K-SVD (rất chậm)
    verbose    : print progress

    Returns:
    --------
    df : DataFrame với columns:
         image, method, sigma, psnr, ssim, runtime_ms
    """
    # Load dataset
    print(f"\nLoading {dataset.upper()} dataset...")
    if dataset == 'bsd68':
        images = load_bsd68_subset(bsd68_path)
    else:
        images = load_set12()

    # Load methods
    print("\nLoading denoising methods...")
    methods = load_methods()

    if skip_slow:
        methods.pop('KSVD', None)
        print("  (K-SVD skipped)")

    print(f"\nRunning benchmark:")
    print(f"  Dataset:  {len(images)} images")
    print(f"  Methods:  {list(methods.keys())}")
    print(f"  Sigmas:   {sigmas}")
    print(f"  Total:    {len(images) * len(methods) * len(sigmas)} runs")

    records = []
    total = len(images) * len(methods) * len(sigmas)
    done = 0

    for img_name, img in images.items():
        img = img.astype(np.uint8)

        for sigma in sigmas:
            # Baseline: noisy image
            noisy = add_gaussian_noise(img, sigma=sigma)
            records.append({
                'image'      : img_name,
                'method'     : 'Noisy',
                'sigma'      : sigma,
                'psnr'       : round(psnr(img, noisy), 4),
                'ssim'       : round(ssim(img, noisy), 6),
                'runtime_ms' : 0,
            })

            for method_name, method_fn in methods.items():
                done += 1
                if verbose:
                    print(f"  [{done}/{total}] {img_name} | {method_name} | σ={sigma}",
                          end='', flush=True)

                try:
                    res = benchmark_single(img, method_fn, sigma)
                    records.append({
                        'image'  : img_name,
                        'method' : method_name,
                        'sigma'  : sigma,
                        **res,
                    })
                    if verbose:
                        print(f"  → PSNR={res['psnr']:.2f} dB  ({res['runtime_ms']:.0f}ms)")

                except Exception as e:
                    print(f"\n  ERROR: {e}")
                    records.append({
                        'image'      : img_name,
                        'method'     : method_name,
                        'sigma'      : sigma,
                        'psnr'       : None,
                        'ssim'       : None,
                        'runtime_ms' : None,
                    })

    df = pd.DataFrame(records)
    return df


# ─────────────────────────────────────────────
# Summary Tables
# ─────────────────────────────────────────────

def make_summary(df):
    """
    Tạo bảng tóm tắt: mean PSNR/SSIM/runtime theo method × sigma.
    """
    summary = (df
               .groupby(['method', 'sigma'])
               .agg(
                   psnr_mean    = ('psnr',       'mean'),
                   psnr_std     = ('psnr',       'std'),
                   ssim_mean    = ('ssim',       'mean'),
                   runtime_mean = ('runtime_ms', 'mean'),
               )
               .round({'psnr_mean': 2, 'psnr_std': 2,
                       'ssim_mean': 4, 'runtime_mean': 1})
               .reset_index())
    return summary


def print_pivot(df, metric='psnr_mean'):
    """
    In bảng pivot: method × sigma.
    """
    summary = make_summary(df)
    pivot = summary.pivot(index='method', columns='sigma', values=metric)

    # Sắp xếp theo thứ tự quan trọng
    order = ['Noisy', 'Gaussian', 'Median', 'NLM', 'BM3D', 'KSVD']
    order = [m for m in order if m in pivot.index]
    pivot = pivot.reindex(order)

    return pivot


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Image Denoising Benchmark')
    parser.add_argument('--dataset', default='set12',
                        choices=['set12', 'bsd68'],
                        help='Dataset to use')
    parser.add_argument('--sigmas', nargs='+', type=int,
                        default=[15, 25, 50],
                        help='Noise levels')
    parser.add_argument('--skip-slow', action='store_true',
                        help='Skip K-SVD (slow)')
    parser.add_argument('--bsd68-path', default='data/bsd68',
                        help='Path to BSD68 images')
    parser.add_argument('--output-dir', default='results',
                        help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Run benchmark
    df = run_benchmark(
        dataset   = args.dataset,
        sigmas    = tuple(args.sigmas),
        bsd68_path = args.bsd68_path,
        skip_slow  = args.skip_slow,
    )

    # Save raw results
    raw_path = f"{args.output_dir}/benchmark_raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"\n✓ Raw results saved: {raw_path}")

    # Save summary
    summary = make_summary(df)
    summary_path = f"{args.output_dir}/benchmark_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"✓ Summary saved: {summary_path}")

    # Print pivots
    print("\n\n" + "="*60)
    print("BENCHMARK RESULTS — Average PSNR (dB)")
    print("="*60)
    print(print_pivot(df, 'psnr_mean').to_string())

    print("\n" + "="*60)
    print("BENCHMARK RESULTS — Average SSIM")
    print("="*60)
    print(print_pivot(df, 'ssim_mean').to_string())

    print("\n" + "="*60)
    print("BENCHMARK RESULTS — Runtime (ms/image)")
    print("="*60)
    print(print_pivot(df, 'runtime_mean').to_string())
