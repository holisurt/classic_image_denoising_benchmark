import numpy as np
from skimage.util import view_as_windows

def nlm_denoise(image, patch_size=5, search_size=21, h=10):
    """
    Non-Local Means denoising từ scratch.
    
    Tham số:
        patch_size  : kích thước patch để so sánh (thường 5 hoặc 7)
        search_size : vùng tìm kiếm patch tương đồng (thường 21)
        h           : filter strength — lớn hơn = smooth hơn nhưng mất detail
    
    Ý tưởng:
        Mỗi pixel i được tính lại bằng trung bình có trọng số
        của TẤT CẢ pixel j trong vùng search_size×search_size,
        với weight = exp(-||patch_i - patch_j||^2 / h^2)
    """
    image = image.astype(np.float64)
    H, W = image.shape
    
    half_patch  = patch_size  // 2
    half_search = search_size // 2
    
    # Padding để xử lý biên
    # reflect padding tốt hơn zero vì không tạo artifact ở biên
    pad = half_patch + half_search
    padded = np.pad(image, pad, mode='reflect')
    
    output = np.zeros_like(image)
    
    for i in range(H):
        for j in range(W):
            # Tọa độ trong ảnh đã pad
            pi = i + pad
            pj = j + pad
            
            # Lấy patch của pixel đang xét (query patch)
            patch_i = padded[pi - half_patch : pi + half_patch + 1,
                             pj - half_patch : pj + half_patch + 1]
            
            weights  = []
            values   = []
            
            # Duyệt qua tất cả pixel trong vùng search
            for si in range(i - half_search, i + half_search + 1):
                for sj in range(j - half_search, j + half_search + 1):
                    
                    # Clamp về trong ảnh (tránh ra ngoài)
                    si_c = np.clip(si, 0, H - 1)
                    sj_c = np.clip(sj, 0, W - 1)
                    
                    # Tọa độ trong padded
                    psi = si_c + pad
                    psj = sj_c + pad
                    
                    # Lấy patch của pixel candidate
                    patch_j = padded[psi - half_patch : psi + half_patch + 1,
                                    psj - half_patch : psj + half_patch + 1]
                    
                    # Tính khoảng cách L2 bình phương giữa 2 patch
                    dist_sq = np.sum((patch_i - patch_j) ** 2)
                    
                    # Tính weight — đây chính là "Gaussian trên patch space"
                    w = np.exp(-dist_sq / (h ** 2))
                    
                    weights.append(w)
                    values.append(image[si_c, sj_c])
            
            weights = np.array(weights)
            values  = np.array(values)
            
            # Normalize weights rồi tính trung bình có trọng số
            output[i, j] = np.sum(weights * values) / np.sum(weights)
    
    return np.clip(output, 0, 255).astype(np.uint8)


def nlm_denoise_fast(image, patch_size=5, search_size=21, h=10):
    """
    Phiên bản nhanh hơn dùng scikit-image.
    Dùng cái này để benchmark, dùng nlm_denoise() để hiểu lý thuyết.
    """
    from skimage.restoration import denoise_nl_means, estimate_sigma
    
    image_float = image.astype(np.float64) / 255.0
    
    # Ước lượng sigma của noise tự động
    sigma_est = estimate_sigma(image_float)
    
    denoised = denoise_nl_means(
        image_float,
        h            = 1.15 * sigma_est,  # filter strength
        patch_size   = patch_size,
        patch_distance = search_size // 2,
        fast_mode    = True
    )
    
    return np.clip(denoised * 255, 0, 255).astype(np.uint8)