"""
K-SVD for Image Denoising

Pipeline:
  1. Extract patches từ ảnh noisy
  2. Learn dictionary using K-SVD
  3. Mỗi patch được biểu diễn sparse: patch ≈ D × x
  4. Reconstruct: denoised_patch = D × x (noise bị cắt)
  5. Aggregate overlapping patches
"""

import numpy as np


def extract_patches(image, patch_size=8, stride=1):
    """
    Extract patches từ ảnh.
    
    Parameters:
    -----------
    image : numpy.ndarray, shape (H, W)
        Ảnh input
        
    patch_size : int, default 8
        Kích thước patch (8 → 8×8)
        
    stride : int, default 1
        Stride khi lấy patch
        stride=1 → fully overlapping (nhiều patch, nhưng slow)
        stride>1 → non-overlapping (ít patch, nhanh)
    
    Returns:
    --------
    patches : numpy.ndarray, shape (patch_size², n_patches)
        Patches đã flatten thành columns
        
    patch_indices : list of tuple (i, j)
        Tọa độ (row, col) của top-left corner của mỗi patch
    
    Examples:
    ---------
    >>> img = np.random.randn(32, 32)
    >>> patches, indices = extract_patches(img, patch_size=8, stride=1)
    >>> patches.shape
    (64, 625)  # 8²=64, (32-8+1)²=625
    """
    H, W = image.shape
    patches = []
    indices = []
    
    # Lấy patch với stride
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            # Lấy patch 8×8
            patch = image[i:i+patch_size, j:j+patch_size]
            
            # Flatten thành vector
            patch_flat = patch.flatten()
            
            patches.append(patch_flat)
            indices.append((i, j))
    
    # Convert to (n_dim, n_patches)
    patches = np.array(patches).T  # Transpose: (patch_size², n_patches)
    
    return patches, indices


def reconstruct_from_patches(patches_flat, patch_indices, image_shape,
                              patch_size=8, stride=1):
    """
    Reconstruct ảnh từ patches (với overlap averaging).
    
    Khi stride=1, patches overlap → ta average các vùng overlap.
    
    Parameters:
    -----------
    patches_flat : numpy.ndarray, shape (patch_size², n_patches)
        Patches đã flatten
        
    patch_indices : list of tuple (i, j)
        Tọa độ của mỗi patch
        
    image_shape : tuple (H, W)
        Hình dáng ảnh output
        
    patch_size : int
        Kích thước patch
        
    stride : int
        (không dùng để reconstruct, chỉ documentation)
    
    Returns:
    --------
    image : numpy.ndarray, shape (H, W)
        Reconstructed image
    
    Method:
    -------
    Accumulator approach:
      1. Tạo output array (H, W)
      2. Tạo count array để tracking overlap
      3. Đặt mỗi patch vào vị trí của nó
      4. Average các vùng overlap
    """
    H, W = image_shape
    image = np.zeros((H, W))
    count = np.zeros((H, W))
    
    # Untuk mỗi patch
    for idx, (i, j) in enumerate(patch_indices):
        # Lấy patch từ patches_flat
        patch_vec = patches_flat[:, idx]
        
        # Reshape thành 2D
        patch = patch_vec.reshape(patch_size, patch_size)
        
        # Accumulate vào image
        image[i:i+patch_size, j:j+patch_size] += patch
        
        # Tracking jumlah patch yang overlap di region ini
        count[i:i+patch_size, j:j+patch_size] += 1
    
    # Average overlapping regions
    # Tránh division by zero
    image = np.divide(image, count, where=count > 0, out=image)
    
    return image


def ksvd_denoise(image, dict_size=256, patch_size=8,
                 iterations=10, sparsity=5):
    """
    Denoising sử dụng K-SVD Dictionary Learning.
    
    ╔════════════════════════════════════════════════════════════╗
    ║ PIPELINE:                                                  ║
    ║                                                            ║
    ║ 1. Input: noisy image                                      ║
    ║                                                            ║
    ║ 2. Extract patches từ ảnh                                  ║
    ║                                                            ║
    ║ 3. K-SVD: learn dictionary D từ patches                   ║
    ║    → Y ≈ D × X (sparse)                                   ║
    ║                                                            ║
    ║ 4. Reconstruct patches:                                    ║
    ║    denoised_patches = D × X                                ║
    ║    (noise bị cắt vì sparse representation)                 ║
    ║                                                            ║
    ║ 5. Aggregate patches → denoised image                      ║
    ║    (average overlapping regions)                           ║
    ║                                                            ║
    ║ 6. Output: denoised image                                  ║
    ╚════════════════════════════════════════════════════════════╝
    
    Parameters:
    -----------
    image : numpy.ndarray, shape (H, W), dtype uint8
        Ảnh grayscale noisy [0, 255]
        
    dict_size : int, default 256
        Kích thước dictionary
        Lớn hơn = linh hoạt hơn nhưng chậm
        
    patch_size : int, default 8
        Kích thước patch (8×8 hay 16×16)
        Lớn hơn = context hơn nhưng slow
        
    iterations : int, default 10
        Iterations của K-SVD
        
    sparsity : int, default 5
        Sparsity trong OMP
        Nhỏ hơn = sparse hơn → denoise strong
        Lớn hơn = flexible hơn → fit tốt hơn
    
    Returns:
    --------
    denoised : numpy.ndarray, shape (H, W), dtype uint8
        Ảnh denoised
        
    D : numpy.ndarray, shape (patch_size², dict_size)
        Learned dictionary (có thể visualize)
        
    X : numpy.ndarray, shape (dict_size, n_patches)
        Sparse codes
    
    Time Complexity:
    ----------------
    Dominated by K-SVD:
    O(iterations × n_patches × (sparsity × k² + k × n_dim))
    
    Ví dụ: ảnh 256×256, patch_size=8, dict_size=256, iterations=5:
    → ~few minutes on CPU
    
    Cách tốc độ:
    - Giảm iterations (5-10 đủ)
    - Giảm dict_size (128-256)
    - Tăng stride (stride=2 → 4x nhanh hơn)
    - Giảm sparsity (3-5)
    """
    from dictionary_learning import ksvd
    
    # ===== PREPROCESSING =====
    print("Step 1: Normalization")
    image_float = image.astype(np.float64)
    
    # ===== PATCH EXTRACTION =====
    print("Step 2: Extracting patches...")
    Y, patch_indices = extract_patches(image_float, patch_size=patch_size, stride=1)
    
    print(f"  Extracted {Y.shape[1]} patches of size {patch_size}×{patch_size}")
    print(f"  Total patch dimension: {Y.shape[0]}")
    
    # ===== DICTIONARY LEARNING =====
    print("Step 3: Learning dictionary with K-SVD...")
    D, X = ksvd(Y, dict_size=dict_size, iterations=iterations,
                sparsity=sparsity, verbose=True)
    
    # ===== PATCH RECONSTRUCTION =====
    print("Step 4: Reconstructing patches...")
    Y_denoised = D @ X
    
    # ===== IMAGE RECONSTRUCTION =====
    print("Step 5: Assembling image...")
    denoised = reconstruct_from_patches(Y_denoised, patch_indices,
                                        image.shape, patch_size=patch_size, stride=1)
    
    # ===== POSTPROCESSING =====
    print("Step 6: Post-processing...")
    denoised = np.clip(denoised, 0, 255).astype(np.uint8)
    
    print("✓ Denoising completed")
    
    return denoised, D, X


def visualize_atoms(D, patch_size=8, n_atoms_display=64):
    """
    Visualize learned dictionary atoms.
    
    Dùng trong notebook để xem atoms.
    
    Parameters:
    -----------
    D : (patch_size², dict_size)
    patch_size : int
    n_atoms_display : int
        Số atoms để visualize (default 64 = 8×8 grid)
    
    Returns:
    --------
    grid : (patch_size*n_rows, patch_size*n_cols) array
        Atom visualization grid
    """
    n_atoms = min(D.shape[1], n_atoms_display)
    n_rows = int(np.sqrt(n_atoms))
    n_cols = int(np.ceil(n_atoms / n_rows))
    
    # Normalize atoms để visualization
    D_norm = D / (np.abs(D).max(axis=0) + 1e-10)
    
    # Create grid
    grid = np.zeros((n_rows * patch_size, n_cols * patch_size))
    
    for idx in range(n_atoms):
        atom = D_norm[:, idx].reshape(patch_size, patch_size)
        
        # Normalize atom để visualization
        atom = (atom - atom.min()) / (atom.max() - atom.min() + 1e-10)
        
        row = idx // n_cols
        col = idx % n_cols
        
        grid[row*patch_size:(row+1)*patch_size,
             col*patch_size:(col+1)*patch_size] = atom
    
    return grid


# ===== TESTING =====
if __name__ == "__main__":
    from skimage import data
    from skimage.metrics import peak_signal_noise_ratio
    
    print("Testing K-SVD Denoising")
    print("=" * 50)
    
    # Load image
    img = data.camera()
    
    # Add noise
    sigma = 25
    noise = np.random.normal(0, sigma, img.shape)
    img_noisy = np.clip(img + noise, 0, 255).astype(np.uint8)
    
    print(f"Original PSNR: {peak_signal_noise_ratio(img, img_noisy):.2f} dB")
    
    # Denoise
    print("\nRunning K-SVD...")
    denoised, D, X = ksvd_denoise(
        img_noisy,
        dict_size=128,   # nhỏ để test nhanh
        patch_size=8,
        iterations=3,    # nhỏ để test nhanh
        sparsity=3
    )
    
    psnr = peak_signal_noise_ratio(img, denoised)
    print(f"\nResult PSNR: {psnr:.2f} dB")
    
    # Visualize atoms
    print("\nVisualizing atoms...")
    grid = visualize_atoms(D, patch_size=8, n_atoms_display=64)
    print(f"Atom grid shape: {grid.shape}")
    
    print("\n✓ Test completed")
