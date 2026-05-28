import numpy as np

def gaussian_kernel_2d(size, sigma):
    # Tạo grid tọa độ từ -size//2 đến size//2
    k = size // 2  # bán kính
    x, y = np.mgrid[-k:k+1, -k:k+1]
    
    # Tính giá trị Gaussian tại mỗi điểm
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    # Normalize để tổng = 1 (quan trọng! giữ nguyên độ sáng)
    kernel = kernel / kernel.sum()
    
    return kernel


def convolve2d_manual(image, kernel):
    H, W = image.shape
    kH, kW = kernel.shape
    
    pad_h = kH // 2
    pad_w = kW // 2
    
    # Zero-padding để output cùng size với input
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    # mode='reflect' tốt hơn zero-padding ở biên ảnh
    
    output = np.zeros_like(image, dtype=np.float64)
    
    for i in range(H):
        for j in range(W):
            # Lấy vùng ảnh cùng size với kernel
            region = padded[i:i+kH, j:j+kW]
            # Element-wise multiply rồi sum = dot product
            output[i, j] = np.sum(region * kernel)
    
    return np.clip(output, 0, 255).astype(np.uint8)


def gaussian_filter(image, kernel_size=5, sigma=1.0):
    kernel = gaussian_kernel_2d(kernel_size, sigma)
    
    if image.ndim == 2:
        return convolve2d_manual(image, kernel)
    
    # Ảnh màu: áp từng channel độc lập
    result = np.zeros_like(image)
    for c in range(image.shape[2]):
        result[:, :, c] = convolve2d_manual(image[:, :, c], kernel)
    return result


def median_filter(image, kernel_size=3):
    H, W = image.shape if image.ndim == 2 else image.shape[:2]
    k = kernel_size // 2
    
    if image.ndim == 2:
        padded = np.pad(image, k, mode='reflect')
        output = np.zeros_like(image)
        for i in range(H):
            for j in range(W):
                region = padded[i:i+kernel_size, j:j+kernel_size]
                output[i, j] = np.median(region)
        return output
    
    result = np.zeros_like(image)
    for c in range(image.shape[2]):
        result[:, :, c] = median_filter(image[:, :, c], kernel_size)
    return result