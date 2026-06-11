import numpy as np

def add_gaussian_noise(image, sigma=25, seed=None):
    """
    Add AWGN noise to image.

    Parameters
    ----------
    image : np.ndarray
        Clean image, uint8 range [0, 255].
    sigma : float
        Noise standard deviation.
    seed : int or None
        Random seed for reproducible benchmark.
    """
    image_float = image.astype(np.float64)

    if seed is None:
        noise = np.random.normal(loc=0, scale=sigma, size=image_float.shape)
    else:
        rng = np.random.default_rng(seed)
        noise = rng.normal(loc=0, scale=sigma, size=image_float.shape)

    noisy = image_float + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def add_salt_pepper_noise(image, prob=0.05, seed=None):
    noisy = image.copy()

    if seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)

    salt_mask = rng.random(image.shape) < prob / 2
    pepper_mask = rng.random(image.shape) < prob / 2

    noisy[salt_mask] = 255
    noisy[pepper_mask] = 0

    return noisy