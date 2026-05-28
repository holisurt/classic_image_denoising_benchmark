import numpy as np
import bm3d

def bm3d_denoise(image, sigma_psd=25):
    """
    BM3D denoising dùng thư viện chuẩn.
    
    Tham số:
        sigma_psd: ước lượng standard deviation của noise.
                   Nên truyền đúng sigma dùng để add noise.
                   Nếu không biết, dùng estimate_sigma().
    
    Lưu ý: bm3d nhận input trong [0, 1], không phải [0, 255]
    """
    image_float = image.astype(np.float64) / 255.0
    sigma_norm  = sigma_psd / 255.0          # normalize sigma cùng tỉ lệ
    
    denoised = bm3d.bm3d(image_float, sigma_psd=sigma_norm)
    
    return np.clip(denoised * 255, 0, 255).astype(np.uint8)


def bm3d_denoise_color(image_rgb, sigma_psd=25):
    """
    BM3D cho ảnh màu — dùng BM3D-RGB, xử lý color correlation.
    """
    image_float = image_rgb.astype(np.float64) / 255.0
    sigma_norm  = sigma_psd / 255.0
    
    # profile='np' = normal profile, cân bằng tốc độ và chất lượng
    denoised = bm3d.bm3d_rgb(image_float, sigma_psd=sigma_norm)
    
    return np.clip(denoised * 255, 0, 255).astype(np.uint8)