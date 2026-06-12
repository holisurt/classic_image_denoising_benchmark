"""
classical_v3.py

V3 classical denoising baselines for AWGN grayscale benchmark.
All functions accept and return float images in [0, 1].
The noise sigma argument is kept in the usual image-denoising scale [0, 255].

Methods included:
- Gaussian_SciPy: optimized spatial Gaussian filter
- Median_SciPy: optimized rank median filter
- NLM_Skimage_Fast: skimage optimized NLM, h = 0.8 * sigma
- NLM_IPOL_Like: skimage original-mode NLM with IPOL-like parameter schedule
- BM3D_TAU: PyPI bm3d wrapper from Tampere/TAU implementation
- KSVD_Repo: optional bridge to the existing repo K-SVD implementation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
from scipy.ndimage import gaussian_filter as scipy_gaussian_filter
from scipy.ndimage import median_filter as scipy_median_filter
from skimage.restoration import denoise_nl_means


Array = np.ndarray


@dataclass(frozen=True)
class MethodSpec:
    name: str
    implementation: str
    fn: Callable[[Array, float], Array]
    notes: str = ""


def ensure_float01(image: Array) -> Array:
    """Return a float32 grayscale image clipped to [0, 1]."""
    arr = np.asarray(image)
    if arr.dtype == np.bool_:
        arr = arr.astype(np.float32)
    elif np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / np.iinfo(arr.dtype).max
    else:
        arr = arr.astype(np.float32)
        if arr.size and arr.max() > 1.5:
            arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def float01_to_uint8(image: Array) -> Array:
    return np.clip(np.rint(np.asarray(image) * 255.0), 0, 255).astype(np.uint8)


def gaussian_scipy(noisy: Array, sigma_noise: float, sigma_filter: float = 1.5) -> Array:
    """Optimized Gaussian filter baseline. sigma_noise is unused by design."""
    x = ensure_float01(noisy)
    y = scipy_gaussian_filter(x, sigma=sigma_filter, mode="reflect")
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def median_scipy(noisy: Array, sigma_noise: float, kernel_size: int = 3) -> Array:
    """Optimized median filter baseline. sigma_noise is unused by design."""
    x = ensure_float01(noisy)
    y = scipy_median_filter(x, size=kernel_size, mode="reflect")
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def nlm_skimage_fast(
    noisy: Array,
    sigma_noise: float,
    patch_size: int = 5,
    search_size: int = 21,
    h_factor: float = 0.8,
) -> Array:
    """
    Fast NLM baseline using skimage.restoration.denoise_nl_means.

    The benchmark passes the known AWGN sigma rather than estimating sigma from
    the noisy image. This makes the protocol explicit and reproducible.
    """
    x = ensure_float01(noisy)
    sigma_norm = float(sigma_noise) / 255.0
    y = denoise_nl_means(
        x,
        patch_size=patch_size,
        patch_distance=search_size // 2,
        h=h_factor * sigma_norm,
        sigma=sigma_norm,
        fast_mode=True,
        preserve_range=True,
        channel_axis=None,
    )
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def get_nlm_ipol_params(sigma_noise: float) -> tuple[int, int, float]:
    """
    IPOL-like grayscale NLM parameter schedule.

    Returns:
        patch_size, search_size, h_factor

    The search_size values are the full research block sizes from IPOL
    (21x21 or 35x35). skimage receives patch_distance = search_size // 2.
    """
    sigma = float(sigma_noise)
    if sigma <= 15:
        return 3, 21, 0.40
    if sigma <= 30:
        return 5, 21, 0.40
    if sigma <= 45:
        return 7, 35, 0.35
    if sigma <= 75:
        return 9, 35, 0.35
    return 11, 35, 0.30


def nlm_ipol_like(noisy: Array, sigma_noise: float) -> Array:
    """
    NLM variant closer to the patchwise IPOL parameter table.

    This uses skimage's classic/original mode (fast_mode=False) to avoid the
    fast integral-image approximation. It can be very slow on BSD68.
    """
    x = ensure_float01(noisy)
    sigma_norm = float(sigma_noise) / 255.0
    patch_size, search_size, h_factor = get_nlm_ipol_params(sigma_noise)
    y = denoise_nl_means(
        x,
        patch_size=patch_size,
        patch_distance=search_size // 2,
        h=h_factor * sigma_norm,
        sigma=sigma_norm,
        fast_mode=False,
        preserve_range=True,
        channel_axis=None,
    )
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def bm3d_tau(noisy: Array, sigma_noise: float) -> Array:
    """BM3D baseline using the PyPI bm3d package. Input/output are float [0, 1]."""
    try:
        import bm3d  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError("BM3D is not installed. Run: pip install bm3d") from exc

    x = ensure_float01(noisy)
    sigma_norm = float(sigma_noise) / 255.0
    y = bm3d.bm3d(x, sigma_psd=sigma_norm)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def ksvd_repo_bridge(noisy: Array, sigma_noise: float) -> Array:
    """
    Optional bridge to the existing repo K-SVD code.

    The existing implementation works with uint8 images, so this function converts
    float [0, 1] -> uint8 -> float [0, 1]. K-SVD is slow and is not part of the
    BM3D/NLM paper-reproduction comparison by default.
    """
    try:
        from ksvd_denoising import ksvd_denoise  # type: ignore
    except Exception as exc:  # pragma: no cover - repo dependent
        raise ImportError("Could not import existing ksvd_denoising.py from src/") from exc

    noisy_u8 = float01_to_uint8(noisy)
    result_u8, _, _ = ksvd_denoise(
        noisy_u8,
        dict_size=128,
        patch_size=8,
        iterations=5,
        sparsity=3,
    )
    return ensure_float01(result_u8)


def get_method_registry(include_ksvd: bool = False) -> Dict[str, MethodSpec]:
    methods: Dict[str, MethodSpec] = {
        "gaussian_scipy": MethodSpec(
            name="Gaussian_SciPy",
            implementation="scipy.ndimage.gaussian_filter(sigma_filter=1.5, reflect)",
            fn=gaussian_scipy,
            notes="Optimized replacement for Python-loop Gaussian filter.",
        ),
        "median_scipy": MethodSpec(
            name="Median_SciPy",
            implementation="scipy.ndimage.median_filter(size=3, reflect)",
            fn=median_scipy,
            notes="Optimized replacement for Python-loop median filter.",
        ),
        "nlm_fast": MethodSpec(
            name="NLM_Skimage_Fast",
            implementation="skimage.denoise_nl_means(fast_mode=True, h=0.8*sigma, sigma=known_sigma)",
            fn=nlm_skimage_fast,
            notes="Fast engineering baseline; not the original patchwise NLM runtime.",
        ),
        "nlm_ipol": MethodSpec(
            name="NLM_IPOL_Like",
            implementation="skimage.denoise_nl_means(fast_mode=False) with IPOL grayscale parameter schedule",
            fn=nlm_ipol_like,
            notes="Closer to IPOL patchwise parameters; can be very slow.",
        ),
        "bm3d_tau": MethodSpec(
            name="BM3D_TAU",
            implementation="bm3d.bm3d(noisy_float01, sigma_psd=sigma/255)",
            fn=bm3d_tau,
            notes="Reference classical upper baseline for AWGN.",
        ),
    }
    if include_ksvd:
        methods["ksvd_repo"] = MethodSpec(
            name="KSVD_Repo",
            implementation="existing repo ksvd_denoise(dict_size=128, patch_size=8, iterations=5, sparsity=3)",
            fn=ksvd_repo_bridge,
            notes="Optional slow dictionary-learning baseline.",
        )
    return methods
