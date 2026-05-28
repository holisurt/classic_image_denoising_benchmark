import numpy as np

def psnr(original, denoised):
    original = original.astype(np.float64)
    denoised = denoised.astype(np.float64)
    
    mse = np.mean((original - denoised) ** 2)
    
    if mse == 0:
        return float('inf')  # Hai ảnh giống hệt nhau
    
    max_pixel = 255.0
    return 10 * np.log10(max_pixel ** 2 / mse)


def ssim(original, denoised, window_size=11, sigma=1.5):
    from skimage.metrics import structural_similarity
    
    # Nếu ảnh grayscale
    if original.ndim == 2:
        return structural_similarity(original, denoised, data_range=255)
    
    # Nếu ảnh màu (RGB), tính SSIM từng channel rồi average
    ssim_channels = []
    for c in range(original.shape[2]):
        s = structural_similarity(
            original[:, :, c], 
            denoised[:, :, c], 
            data_range=255
        )
        ssim_channels.append(s)
    
    return np.mean(ssim_channels)