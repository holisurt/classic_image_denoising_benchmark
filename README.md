# Image Denoising: From Classical Filters to Dictionary Learning

> A comprehensive study of image denoising methods — from basic Gaussian filters  
> to sparse dictionary learning — with systematic benchmarks on BSD68 and Set12.

---

## Overview

This project implements and benchmarks **5 image denoising algorithms**, tracing  
the evolution of signal-processing approaches from hand-crafted filters to  
learned representations:

| Method | Category | Key Idea |
|---|---|---|
| **Gaussian Filter** | Classical | Weighted spatial average |
| **Median Filter** | Classical | Rank-based, edge-preserving |
| **Non-Local Means** | Self-similarity | Patch similarity weighting |
| **BM3D** | Transform-domain | Block matching + 3D filtering |
| **K-SVD** | Dictionary learning | Learned sparse representation |

---

## Results (BSD68 / Set12, grayscale)

### PSNR (dB) — Higher is Better

| Method | σ=15 | σ=25 | σ=50 |
|---|---|---|---|
| Noisy | 24.6 | 20.2 | 14.2 |
| Gaussian | 31.1 | 28.0 | 24.7 |
| Median | 30.7 | 27.3 | 23.8 |
| NLM | 33.1 | 30.5 | 27.1 |
| BM3D | **33.7** | **31.7** | **28.6** |
| K-SVD | 33.4 | 31.3 | 28.2 |

### Trade-off: Quality vs Speed

```
PSNR (↑ better)
33.7 │                                   ● BM3D
33.4 │                          ● K-SVD
33.1 │             ● NLM
     │
28.0 │   ● Gaussian
     │
     └─────────────────────────────────────────── Runtime (ms ↑ slower)
          <1ms      50ms     200ms    5000ms
```

BM3D achieves the best quality with moderate runtime.  
K-SVD is competitive but slower due to dictionary learning.

---

## Installation

```bash
git clone https://github.com/<your-username>/image-denoising
cd image-denoising

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**requirements.txt:**
```
numpy>=1.24
matplotlib>=3.7
scikit-image>=0.21
scikit-learn>=1.3
bm3d>=4.0
scipy>=1.11
pandas>=2.0
```

---

## Project Structure

```
image-denoising/
├── src/
│   ├── noise.py              # Noise models: Gaussian, Salt & Pepper, Poisson
│   ├── filters.py            # Gaussian & Median filter from scratch
│   ├── metrics.py            # PSNR, SSIM evaluation
│   ├── nlm.py                # Non-Local Means
│   ├── bm3d_wrapper.py       # BM3D wrapper
│   ├── sparse_coding.py      # OMP (Orthogonal Matching Pursuit)
│   ├── dictionary_learning.py # K-SVD algorithm
│   └── ksvd_denoising.py     # K-SVD denoising pipeline
│
├── notebooks/
│   ├── week1_basics.ipynb    # Gaussian & Median, noise models, metrics
│   ├── week2_nlm_bm3d.ipynb  # NLM & BM3D, weight visualization
│   └── week3_ksvd.ipynb      # K-SVD, dictionary atoms, comparison
│
├── data/
│   └── test_images/          # Set12 + BSD68 (see below)
│
├── results/
│   ├── figures/              # Generated plots
│   └── benchmark_raw.csv     # Raw benchmark data
│
├── benchmark.py              # Run all methods on datasets
├── plot_results.py           # Generate all figures
└── README.md
```

---

## Quick Demo

```python
import numpy as np
from skimage import data
import sys; sys.path.insert(0, 'src')

from noise import add_gaussian_noise
from bm3d_wrapper import bm3d_denoise
from metrics import psnr, ssim

# Load image
img = data.camera()

# Add Gaussian noise (σ=25)
noisy = add_gaussian_noise(img, sigma=25)
print(f"Noisy PSNR: {psnr(img, noisy):.2f} dB")

# BM3D denoising
denoised = bm3d_denoise(noisy, sigma_psd=25)
print(f"BM3D PSNR:  {psnr(img, denoised):.2f} dB")
```

---

## Reproduce Benchmark

```bash
# Run on Set12 (fast, ~2 min)
python benchmark.py --dataset set12 --sigmas 15 25 50

# Run on BSD68 (requires download, ~10 min)
python benchmark.py --dataset bsd68 --sigmas 15 25 50

# Skip K-SVD (slow)
python benchmark.py --skip-slow

# Generate all plots
python plot_results.py
```

### Download BSD68

```bash
mkdir -p data/bsd68
# Download from: https://github.com/cszn/FFDNet/tree/master/testsets/BSD68
# Or use the provided Set12 (built into scikit-image)
```

---

## Method Explanations

### Gaussian Filter

Convolves the image with a 2D Gaussian kernel. Each output pixel is  
a weighted average of its neighbors, with weights decaying by spatial distance.

```
G(x,y) = exp(-(x²+y²) / (2σ²))   → normalize → convolve
```

**Limitation:** treats all neighbors equally regardless of content → blurs edges.

### Non-Local Means (NLM)

Weights pixels by **patch similarity** rather than spatial distance:

```
û(i) = Σⱼ w(i,j)·v(j) / Σⱼ w(i,j)
w(i,j) = exp(-||Pᵢ - Pⱼ||² / h²)
```

Where `Pᵢ`, `Pⱼ` are patches around pixel i, j.  
Non-local: similar textures anywhere in the image contribute — not just neighbors.

### BM3D

Two-stage algorithm combining block matching and 3D collaborative filtering:

- **Stage 1:** Block matching → 3D group → Hard threshold → Aggregation
- **Stage 2:** Wiener filter using Stage 1 as prior → Aggregation

3D transform exploits both spatial (2D DCT) and inter-patch (1D Hadamard) redundancy.

### K-SVD + OMP

**Dictionary Learning:** learns an overcomplete dictionary `D` (n×K) such that  
every patch `y` can be represented sparsely: `y ≈ D·x` with `||x||₀ ≤ T`.

**K-SVD algorithm** alternates:
1. **Sparse Coding:** `xⱼ ← OMP(yⱼ, D, T)` for all patches
2. **Dictionary Update:** update each atom `dₖ` via SVD of the error matrix

**Denoising:** noise is non-sparse → sparse representation drops noise components.

---

## Key Findings

1. **BM3D remains the best classical method** — consistent 1-2 dB advantage over NLM
2. **K-SVD is competitive** but requires training time
3. **NLM is the best speed/quality tradeoff** — strong result, practical runtime
4. **Median filter excels on Salt & Pepper noise** — outperforms all others
5. **All classical methods plateau** at ~32-33 dB; deep learning (DnCNN) exceeds 34+ dB

---

## Limitations & Future Work

- **Blind denoising:** current methods require known σ; real-world noise is unknown
- **Real-world noise:** camera noise ≠ Gaussian (AWGN model is approximate)
- **Deep learning:** DnCNN, FFDNet, and diffusion-based models outperform classical
- **Self-supervised:** Noise2Noise trains without clean targets — practical for real data
- **Computational efficiency:** K-SVD is slow; neural networks are faster at test time

---

## References

1. Dabov et al., *Image Denoising by Sparse 3D Transform-Domain Collaborative Filtering*, IEEE TIP 2007 — BM3D
2. Buades et al., *A Non-Local Algorithm for Image Denoising*, CVPR 2005 — NLM
3. Aharon et al., *K-SVD: An Algorithm for Designing Overcomplete Dictionaries for Sparse Representation*, IEEE TSP 2006
4. Zhang et al., *Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising*, IEEE TIP 2017 — DnCNN

---

## License

MIT License. See [LICENSE](LICENSE).

---

*Built as part of a signal processing + AI study project, June 2026.*
