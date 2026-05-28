import numpy as np

def add_gaussian_noise(image, sigma=25):
    image_float = image.astype(np.float64) 
    """Đổi sang float để tránh tràn số khi cộng noise vì ảnh được lưu duới dạng uint8 (0-255). Nếu cộng trực tiếp sẽ bị tràn số và kết quả không đúng."""
    
    # Tạo noise từ phân phối N(0, sigma^2)
    noise = np.random.normal(loc=0, scale=sigma, size=image_float.shape)
    
    noisy = image_float + noise
    
    # Clip về [0, 255] và convert về uint8
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def add_salt_pepper_noise(image, prob=0.05):
    noisy = image.copy()
    
    # Salt: set pixel thành 255
    salt_mask = np.random.random(image.shape) < prob / 2
    noisy[salt_mask] = 255
    
    # Pepper: set pixel thành 0
    pepper_mask = np.random.random(image.shape) < prob / 2
    noisy[pepper_mask] = 0
    
    return noisy