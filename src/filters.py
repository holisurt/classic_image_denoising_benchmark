import numpy as np

def gaussian_kernel_2d(size, sigma):
    """
    Tạo Gaussian kernel 2D từ scratch.
    
    Đây là phần cốt lõi — hiểu rõ cái này sẽ giúp bạn 
    hiểu BM3D và các filter nâng cao hơn sau.
    
    Công thức: G(x,y) = (1/(2*pi*sigma^2)) * exp(-(x^2+y^2)/(2*sigma^2))
    """
    # Tạo grid tọa độ từ -size//2 đến size//2
    k = size // 2  # bán kính
    x, y = np.mgrid[-k:k+1, -k:k+1]
    
    # Tính giá trị Gaussian tại mỗi điểm
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    # Normalize để tổng = 1 (quan trọng! giữ nguyên độ sáng)
    kernel = kernel / kernel.sum()
    
    return kernel


def convolve2d_manual(image, kernel):
    """
    Thực hiện 2D convolution thủ công (không dùng scipy/cv2).
    
    Đây là bản naive O(n^2 * k^2) — chậm nhưng rõ ràng về logic.
    Sau này bạn sẽ học FFT-based convolution để tăng tốc.
    
    image:  2D array (H, W)
    kernel: 2D array (kH, kW)
    """
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
    """
    Gaussian filter hoàn chỉnh.
    
    Hoạt động: mỗi pixel được thay bằng weighted average
    của các pixel lân cận, weight theo phân phối Gaussian.
    Pixel gần trung tâm có weight cao hơn pixel xa.
    
    Ưu: làm mờ đều, bảo toàn cấu trúc chung
    Nhược: làm nhòe cạnh (edges)
    """
    kernel = gaussian_kernel_2d(kernel_size, sigma)
    
    if image.ndim == 2:
        return convolve2d_manual(image, kernel)
    
    # Ảnh màu: áp từng channel độc lập
    result = np.zeros_like(image)
    for c in range(image.shape[2]):
        result[:, :, c] = convolve2d_manual(image[:, :, c], kernel)
    return result


def median_filter(image, kernel_size=3):
    """
    Median filter từ scratch.
    
    Khác với Gaussian: không tính trung bình có trọng số
    mà lấy giá trị TRUNG VỊ của vùng lân cận.
    
    Tại sao tốt cho Salt & Pepper? 
    Vì giá trị 0 hoặc 255 cực đoan sẽ không bao giờ
    là median của một vùng bình thường.
    """
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