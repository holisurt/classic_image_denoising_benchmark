# Tuần 3: K-SVD & Dictionary Learning — Lý Thuyết + Code Chi Tiết

---

## Mục lục
1. [Tại sao cần K-SVD?](#tại-sao-cần-k-svd)
2. [Khái niệm cốt lõi: Sparse Representation](#khái-niệm-cốt-lõi)
3. [OMP — Orthogonal Matching Pursuit](#omp--orthogonal-matching-pursuit)
4. [Dictionary Learning — K-SVD Algorithm](#dictionary-learning)
5. [Implement K-SVD](#implement-k-svd)
6. [Code chi tiết + Giải thích](#code-chi-tiết)
7. [Notebook tuần 3](#notebook-tuần-3)

---

## Tại sao cần K-SVD?

### Vấn đề của NLM/BM3D

Tuần 2 bạn học:
- **NLM**: tìm patch giống nhau, lấy trung bình
- **BM3D**: tìm patch + transform 3D + threshold

Sự cố chung: **phụ thuộc vào dữ liệu hiện tại**
- NLM phải search toàn ảnh → chậm
- BM3D cần tìm patch match tốt → nếu patch không có match → kém

### Ý tưởng K-SVD

Thay vì **tìm patch giống nhau trong ảnh**, hãy **học một "từ điển"** của các pattern phổ biến:

```
TRƯỚC (NLM/BM3D):
  Ảnh input → tìm patch giống nhau trực tiếp → denoised ảnh

SAU (K-SVD):
  1. Học dictionary từ ảnh sạch hoặc noisy
  2. Mỗi patch được biểu diễn = tổ hợp sparse của các "từ" trong dictionary
  3. Denoising = tìm sparse representation tốt nhất
```

**Analogies thực tế:**

Giả sử ảnh là một cuốn sách:
- **NLM**: "Để hiểu đoạn này, tôi sẽ tìm các đoạn giống nhau khác trong sách"
- **K-SVD**: "Tôi biết bộ từ vựng thường dùng trong sách này, tôi sẽ dùng bộ từ vựng đó để hiểu/sửa sai"

Lợi thế:
1. **Không phải search online** → nhanh hơn NLM
2. **Khai thác structure** → chất lượng tốt hơn NLM single-pixel
3. **Có thể học adaptive** → mỗi ảnh có dictionary khác

---

## Khái Niệm Cốt Lõi: Sparse Representation

### Định nghĩa

Cho một patch **y** (vector kích thước n) và một dictionary **D** (matrix n × k):

```
y ≈ D × x

trong đó:
  y : patch input (n × 1) — một patch từ ảnh
  D : dictionary (n × k) — k "từ" (atoms)
  x : sparse code (k × 1) — hệ số của từng từ
  
Ý nghĩa:
  patch y được biểu diễn = tổ hợp của k từ trong dictionary
  nhưng chỉ một vài từ có hệ số khác 0 (sparse)
```

### Ví dụ cụ thể

```
Giả sử patch 8×8 được flatten thành vector 64 chiều:
y = [100, 105, 110, ..., 80]ᵀ  (64 × 1)

Dictionary có 256 atoms (từ):
D = [d₁, d₂, ..., d₂₅₆]  (64 × 256)

Sparse code (chỉ 3 từ khác 0):
x = [0, 0, ..., 2.5, 0, ..., -1.2, 0, ..., 3.1, 0, ...]ᵀ  (256 × 1)
     ^sparse: chỉ 3 phần tử ≠ 0

Nó có nghĩa:
y ≈ 2.5 × d₃₂ + (-1.2) × d₈₅ + 3.1 × d₁₉₂
```

### Tại sao "sparse" lại tốt?

1. **Noise không thưa** — noise là random, có hệ số cao ở tất cả atoms
2. **Signal thưa** — pattern tự nhiên chỉ cần một vài atoms
3. **Denoising** — ngắt cắt các hệ số nhỏ (nhiễu)

```
Ảnh sạch: y_clean ≈ D × x_sparse (chỉ 3-5 hệ số khác 0)
Noise:    n ~ Gaussian (tất cả hệ số khác 0)

Tổng cộng:
y_noisy = y_clean + n
        ≈ D × x_sparse + n

Denoising:
1. Tìm sparse code x cho y_noisy
2. Reconstruct: y_denoised = D × x
   → cắt bỏ noise (hệ số nhỏ)
```

---

## OMP — Orthogonal Matching Pursuit

### Bài toán

Cho y và D, tìm x sao cho:

```
minimize: ||y - D×x||₂²
subject to: ||x||₀ ≤ T  (T = số hệ số khác 0 tối đa)
```

Nghĩa là: tìm biểu diễn của y dùng tối đa T atoms sao cho error nhỏ nhất.

Bài toán này là **NP-hard** (không có algorithm optimal đa thức). OMP là **greedy approximation**.

### Thuật toán OMP

```
Input: y (patch), D (dictionary), T (sparsity — số term max)
Output: x (sparse code)

1. Initialize:
   r₀ ← y           (residual = error hiện tại)
   A ← {} (empty)   (support = tập atom được chọn)
   
2. For t = 1 to T:
   
   a. Find best atom:
      i_t ← argmax_i |⟨r_{t-1}, d_i⟩|
      (tìm atom d_i có dot product lớn nhất với residual)
   
   b. Add to support:
      A ← A ∪ {i_t}
   
   c. Solve least square:
      x_A ← argmin_x ||y - D_A × x||₂²
      (fit các atom trong A tối ưu nhất)
   
   d. Update residual:
      r_t ← y - D_A × x_A
      
   e. Check convergence:
      if ||r_t||₂ < ε:
         break
   
3. Return: x (vector full, các phần tử không trong A = 0)
```

### Ví dụ trực quan

```
Bước 0: y = [100, 50, 30]ᵀ, r₀ = y

Dictionary: D = [d₁, d₂, d₃] = [[1, 0.5, 0.2], [0, 1, 0.1], [0, 0, 1]]

Bước 1:
  |⟨r₀, d₁⟩| = |1×100 + 0.5×50 + 0.2×30| = |100 + 25 + 6| = 131 ← MAX
  |⟨r₀, d₂⟩| = |0×100 + 1×50 + 0.1×30| = |50 + 3| = 53
  |⟨r₀, d₃⟩| = |0×100 + 0×50 + 1×30| = 30
  
  Chọn d₁ → A = {1}
  Fit: x₁ = (d₁ᵀ×d₁)⁻¹ × d₁ᵀ × y
  ...update residual...

Bước 2:
  Tìm best atom từ {d₂, d₃} có dot product lớn nhất với r₁
  ...
```

### Tại sao OMP tốt?

- **Greedy**: mỗi bước chọn atom best current → nhanh
- **Orthogonal update**: fit lại tất cả atom (không chỉ mới) → chính xác hơn matching pursuit
- **Convergence**: error giảm sau mỗi bước
- **Complexity**: O(nkT) với n=patch size, k=dict size, T=sparsity

---

## Dictionary Learning — K-SVD Algorithm

### Bài toán

Cho tập patch {y₁, y₂, ..., yₘ}, tìm dictionary D và sparse codes X sao cho:

```
minimize: Σ_i ||y_i - D×x_i||₂² + λ||x_i||₀
over D, X

Tức là:
  - Fit tốt: ||y_i - D×x_i||₂² nhỏ
  - Sparse: ||x_i||₀ nhỏ (số hệ số ≠ 0 ít)
  - λ: tradeoff parameter
```

### K-SVD Algorithm

```
Input: patches {y₁, ..., yₘ}, dictionary size K, iterations I
Output: dictionary D (n × K), sparse codes X (K × m)

Initialize:
  D ← random normalized (n × K)

For iteration i = 1 to I:
  
  ===== SPARSE CODING STEP =====
  For j = 1 to m:
    x_j ← OMP(y_j, D, T=T_spar)  (tìm sparse code cho patch j)
  
  ===== DICTIONARY UPDATE STEP =====
  For k = 1 to K:
    
    1. Find patches using atom d_k:
       ω_k ← {j : x_j[k] ≠ 0}  (patch nào dùng atom k)
    
    2. If ω_k empty: skip atom k
    
    3. Compute error matrix (bỏ contribution của atom k):
       E_k = [y_j - D×x_j + d_k×x_j[k] for j in ω_k]
           = [y_j - Σ_{l≠k} d_l×x_j[l] for j in ω_k]
    
    4. SVD decomposition:
       E_k ≈ U × Σ × Vᵀ  (SVD)
       d_k ← U[:, 0]     (left singular vector của largest singular value)
       x_j[k] ← Σ[0,0] × V[j, 0]  (update coefficient)
```

### Tại sao gọi "K-SVD"?

Vì dictionary update dùng **SVD decomposition**. Cách làm này tối ưu hơn simple gradient descent.

### Flow của K-SVD

```
                ┌─────────────────────────────────────┐
                │   Initialize Dictionary D randomly  │
                └────────────┬────────────────────────┘
                             │
            ┌────────────────▼────────────────┐
            │  SPARSE CODING STEP (OMP)       │
            │  For each patch y_j:            │
            │    Find sparse code x_j         │
            │    s.t. y_j ≈ D × x_j          │
            └────────────┬─────────────────────┘
                         │
            ┌────────────▼────────────────┐
            │  DICTIONARY UPDATE (SVD)    │
            │  For each atom d_k:         │
            │    Recompute d_k using SVD  │
            │    Update coefficients      │
            └────────────┬─────────────────┘
                         │
            ┌────────────▼────────────────┐
            │  Converged?                 │
            │  (error < threshold)        │
            └──┬────────────────────┬─────┘
               │ No                 │ Yes
               │                    │
               └──────────┬─────────┘
                          ▼
                    Return D, X
```

### Hyperparameters của K-SVD

```
K          : dictionary size (kích thước dictionary)
             thường K = 256 (overcompete: K > n patch_size)
             lớn hơn = flexible hơn nhưng chậm

T_spar     : sparsity trong OMP (số term max)
             thường T_spar = 5-10
             lớn hơn = fit tốt hơn nhưng chậm

λ          : regularization weight
             thường λ = 0.001 - 0.01
             lớn hơn = ưu tiên sparse hơn fit

iterations : số lần lặp K-SVD
             thường 10-50 lần
```

---

## Implement K-SVD

### Pseudocode đơn giản

```python
def ksvd(patches, dict_size=256, iterations=10, sparsity=5):
    """
    K-SVD dictionary learning
    """
    n_atoms = patches.shape[0]  # patch dimension
    m_patches = patches.shape[1]  # number of patches
    
    # Initialize D randomly
    D = np.random.randn(n_atoms, dict_size)
    D = D / np.linalg.norm(D, axis=0)  # normalize columns
    
    for iter in range(iterations):
        
        # === SPARSE CODING ===
        X = np.zeros((dict_size, m_patches))
        for j in range(m_patches):
            x_j = omp(patches[:, j], D, sparsity)
            X[:, j] = x_j
        
        # === DICTIONARY UPDATE ===
        for k in range(dict_size):
            
            # Find patches using atom k
            omega_k = np.where(X[k, :] != 0)[0]
            
            if len(omega_k) == 0:
                continue
            
            # Error without atom k
            d_k = D[:, k]
            errors = patches[:, omega_k] - D @ X[:, omega_k]
            errors = errors + np.outer(d_k, X[k, omega_k])
            
            # SVD update
            U, S, Vt = np.linalg.svd(errors, full_matrices=False)
            D[:, k] = U[:, 0]
            X[k, omega_k] = S[0] * Vt[0, :]
    
    return D, X
```

---

## Code Chi Tiết

### File: `src/sparse_coding.py`

```python
import numpy as np

def omp(y, D, sparsity):
    """
    Orthogonal Matching Pursuit — tìm sparse representation.
    
    Parameters:
    -----------
    y : (n,) array
        Patch input
    
    D : (n, k) array
        Dictionary (k atoms)
    
    sparsity : int
        Số hệ số max khác 0
    
    Returns:
    --------
    x : (k,) array
        Sparse code
    """
    n_atoms = D.shape[1]
    x = np.zeros(n_atoms)
    
    # Residual
    r = y.copy()
    
    # Support (atoms được chọn)
    support = []
    
    for t in range(sparsity):
        
        # Find best atom
        scores = np.abs(D.T @ r)  # |⟨r, d_i⟩|
        i_best = np.argmax(scores)
        
        support.append(i_best)
        
        # Solve least square with atoms in support
        D_support = D[:, support]
        x_support = np.linalg.lstsq(D_support, y, rcond=None)[0]
        
        # Update residual
        r = y - D_support @ x_support
        
        # Convergence check
        if np.linalg.norm(r) < 1e-6:
            break
    
    # Construct full sparse vector
    x[support] = x_support
    return x


def batch_omp(Y, D, sparsity):
    """
    OMP cho batch patch.
    
    Parameters:
    -----------
    Y : (n, m) array
        m patches
    
    D : (n, k) array
        Dictionary
    
    sparsity : int
    
    Returns:
    --------
    X : (k, m) array
        Sparse codes
    """
    m = Y.shape[1]
    X = np.zeros((D.shape[1], m))
    
    for j in range(m):
        X[:, j] = omp(Y[:, j], D, sparsity)
    
    return X
```

### File: `src/dictionary_learning.py`

```python
import numpy as np
from sparse_coding import batch_omp

def ksvd(Y, dict_size=256, iterations=10, sparsity=5, verbose=True):
    """
    K-SVD Dictionary Learning.
    
    ╔═══════════════════════════════════════════════════╗
    ║ ALGORITHM:                                        ║
    ║                                                   ║
    ║ 1. Initialize D randomly                          ║
    ║                                                   ║
    ║ repeat iterations times:                          ║
    ║   2. Sparse Coding: find X s.t. Y ≈ D × X        ║
    ║      using OMP                                    ║
    ║                                                   ║
    ║   3. Dictionary Update: update each atom d_k      ║
    ║      using SVD on error matrix                    ║
    ║                                                   ║
    ║ 4. Return D, X                                    ║
    ╚═══════════════════════════════════════════════════╝
    
    Parameters:
    -----------
    Y : (n, m) array
        m patches mỗi cái n-dimensional
    
    dict_size : int
        Kích thước dictionary (số atoms)
        Thường dict_size > n (overcomplete)
    
    iterations : int
        Số lần lặp K-SVD
    
    sparsity : int
        Số hệ số max ≠ 0 trong OMP
    
    verbose : bool
        In progress
    
    Returns:
    --------
    D : (n, dict_size) array
        Học được dictionary
    
    X : (dict_size, m) array
        Sparse codes của tất cả patches
    """
    n, m = Y.shape
    
    # === INITIALIZATION ===
    # Dictionary khởi tạo random
    D = np.random.randn(n, dict_size)
    
    # Normalize columns (mỗi atom có norm = 1)
    D = D / np.linalg.norm(D, axis=0, keepdims=True)
    
    if verbose:
        print(f"K-SVD: {m} patches, dict_size={dict_size}, sparsity={sparsity}")
    
    # === MAIN LOOP ===
    for iteration in range(iterations):
        
        # ===== SPARSE CODING STEP =====
        if verbose:
            print(f"  Iter {iteration+1}/{iterations}: Sparse coding...")
        
        X = batch_omp(Y, D, sparsity)
        
        # ===== DICTIONARY UPDATE STEP =====
        if verbose:
            print(f"  Iter {iteration+1}/{iterations}: Dictionary update...")
        
        for k in range(dict_size):
            
            # Find patches using atom k
            # ω_k = {j : X[k, j] ≠ 0}
            omega_k = np.where(X[k, :] != 0)[0]
            
            # Nếu atom k không được dùng, bỏ qua
            if len(omega_k) == 0:
                continue
            
            # === Compute error matrix ===
            # E_k = Y_ωk - Σ_{l≠k} D_l × X_l[ω_k]
            # 
            # Cách 1 (directional):
            d_k_old = D[:, k]
            
            # Lấy các patch dùng atom k
            Y_omega = Y[:, omega_k]
            
            # Reconstruct bằng các atoms khác (không dùng k)
            D_others = np.delete(D, k, axis=1)
            X_others = np.delete(X, k, axis=0)
            
            # Error do bỏ qua atom k
            error = Y_omega - D_others @ X_others[:, omega_k]
            
            # Cách 2 (cleaner):
            # error = Y_omega - (D @ X[:, omega_k] - d_k_old × X[k, omega_k])
            error = Y_omega - D @ X[:, omega_k] + np.outer(d_k_old, X[k, omega_k])
            
            # === SVD Decomposition ===
            U, S, Vt = np.linalg.svd(error, full_matrices=False)
            
            # Update atom (left singular vector)
            D[:, k] = U[:, 0]
            
            # Update coefficient (right singular vector × singular value)
            X[k, omega_k] = S[0] * Vt[0, :]
        
        # === Normalize dictionary ===
        D = D / np.linalg.norm(D, axis=0, keepdims=True)
        
        # Compute reconstruction error
        if verbose:
            recon_error = np.linalg.norm(Y - D @ X) / np.linalg.norm(Y)
            print(f"       Reconstruction error: {recon_error:.6f}")
    
    return D, X


def ksvd_with_validation(Y_train, Y_val, dict_size=256, iterations=10, 
                         sparsity=5, verbose=True):
    """
    K-SVD với validation set để monitor overfitting.
    """
    D, X_train = ksvd(Y_train, dict_size, iterations, sparsity, verbose=False)
    
    if verbose:
        print("\nK-SVD Training with Validation:")
    
    for iteration in range(iterations):
        
        # Sparse coding
        X_train = batch_omp(Y_train, D, sparsity)
        X_val = batch_omp(Y_val, D, sparsity)
        
        # Dictionary update
        for k in range(dict_size):
            omega_k = np.where(X_train[k, :] != 0)[0]
            if len(omega_k) == 0:
                continue
            
            d_k_old = D[:, k]
            error = Y_train[:, omega_k] - D @ X_train[:, omega_k] + \
                    np.outer(d_k_old, X_train[k, omega_k])
            
            U, S, Vt = np.linalg.svd(error, full_matrices=False)
            D[:, k] = U[:, 0]
            X_train[k, omega_k] = S[0] * Vt[0, :]
        
        D = D / np.linalg.norm(D, axis=0, keepdims=True)
        
        # Evaluate
        train_error = np.linalg.norm(Y_train - D @ X_train) / np.linalg.norm(Y_train)
        val_error = np.linalg.norm(Y_val - D @ X_val) / np.linalg.norm(Y_val)
        
        if verbose and (iteration+1) % 2 == 0:
            print(f"  Iter {iteration+1}: Train error={train_error:.6f}, "
                  f"Val error={val_error:.6f}")
    
    return D, X_train
```

### File: `src/ksvd_denoising.py`

```python
import numpy as np
from dictionary_learning import ksvd
from sparse_coding import omp

def extract_patches(image, patch_size=8, stride=1):
    """
    Extract patches từ ảnh.
    
    Parameters:
    -----------
    image : (H, W) array
    patch_size : int, default 8
    stride : int, default 1 (1 = no overlap, stride > 1 = overlapping)
    
    Returns:
    --------
    patches : (patch_size², n_patches) array
    patch_indices : list of (i, j) tọa độ của patch
    """
    H, W = image.shape
    patches = []
    indices = []
    
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            patch = image[i:i+patch_size, j:j+patch_size].flatten()
            patches.append(patch)
            indices.append((i, j))
    
    return np.array(patches).T, indices  # (patch_size², n_patches)


def reconstruct_from_patches(patches_flat, patch_indices, image_shape, 
                              patch_size=8, stride=1):
    """
    Reconstruct ảnh từ patches (với overlap averaging).
    
    Parameters:
    -----------
    patches_flat : (patch_size², n_patches) array
    patch_indices : list of (i, j)
    image_shape : (H, W)
    
    Returns:
    --------
    image : (H, W) array
    """
    H, W = image_shape
    image = np.zeros((H, W))
    count = np.zeros((H, W))
    
    for idx, (i, j) in enumerate(patch_indices):
        patch = patches_flat[:, idx].reshape(patch_size, patch_size)
        image[i:i+patch_size, j:j+patch_size] += patch
        count[i:i+patch_size, j:j+patch_size] += 1
    
    # Average overlapping regions
    image = np.divide(image, count, where=count>0, out=image)
    return image


def ksvd_denoise(image, dict_size=256, patch_size=8, 
                 iterations=10, sparsity=5):
    """
    Denoising sử dụng K-SVD Dictionary Learning.
    
    Ý tưởng:
    1. Extract patches từ ảnh noisy
    2. Learn dictionary D từ patches
    3. Mỗi patch được biểu diễn sparse: patch ≈ D × x
    4. Reconstruct: denoised_patch = D × x
       (noise được cắt vì sparse)
    
    Parameters:
    -----------
    image : (H, W) uint8 array
    dict_size : int
    patch_size : int
    iterations : int
    sparsity : int
    
    Returns:
    --------
    denoised : (H, W) uint8 array
    D : dictionary
    X : sparse codes
    """
    # Normalize input
    image_float = image.astype(np.float64)
    
    # Extract patches
    print("Extracting patches...")
    Y, patch_indices = extract_patches(image_float, patch_size=patch_size, stride=1)
    
    print(f"Got {Y.shape[1]} patches of size {patch_size}×{patch_size}")
    
    # Learn dictionary
    print("Learning dictionary with K-SVD...")
    D, X = ksvd(Y, dict_size=dict_size, iterations=iterations, 
                sparsity=sparsity, verbose=True)
    
    # Reconstruct patches
    print("Reconstructing patches...")
    Y_denoised = D @ X
    
    # Reconstruct image
    print("Assembling image...")
    denoised = reconstruct_from_patches(Y_denoised, patch_indices, 
                                        image.shape, patch_size=patch_size, stride=1)
    
    # Clip and convert
    denoised = np.clip(denoised, 0, 255).astype(np.uint8)
    
    return denoised, D, X
```

---

## Notebook Tuần 3

### Cell 0: Setup

```python
import sys
from pathlib import Path

notebook_dir = Path.cwd()
src_path = notebook_dir.parent / 'src' if (notebook_dir.parent / 'src').exists() \
           else notebook_dir / 'src'
sys.path.insert(0, str(src_path))

print(f"✓ Added to path: {src_path}")
```

### Cell 1: Import

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import data
import pandas as pd

from noise import add_gaussian_noise
from filters import gaussian_filter
from metrics import psnr, ssim
from nlm import nlm_denoise_fast
from bm3d_wrapper import bm3d_denoise
from ksvd_denoising import ksvd_denoise
from sparse_coding import omp

img = data.camera().astype(np.uint8)
print(f"✓ Ảnh: {img.shape}, range [{img.min()}, {img.max()}]")
```

### Cell 2: Hiểu OMP qua ví dụ

```python
# Tạo dictionary đơn giản
np.random.seed(42)
D_simple = np.random.randn(5, 3)  # 5D, 3 atoms
D_simple = D_simple / np.linalg.norm(D_simple, axis=0)

# Tạo patch
y = np.array([1.0, 0.5, -0.2, 0.1, 0.3])

# OMP với sparsity=2
x = omp(y, D_simple, sparsity=2)

print("Dictionary atoms:")
for i in range(3):
    print(f"  d_{i} = {D_simple[:, i]}")

print(f"\nInput y: {y}")
print(f"\nSparse code x: {x}")
print(f"Non-zero entries: {np.where(x != 0)[0]}")
print(f"Reconstruction error: {np.linalg.norm(y - D_simple @ x):.6f}")

# Visualize
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(range(len(x)), x, color=['red' if xi == 0 else 'blue' for xi in x])
ax.set_xlabel('Atom index')
ax.set_ylabel('Coefficient')
ax.set_title('Sparse Code from OMP (blue = non-zero)')
ax.grid(True, alpha=0.3)
plt.show()
```

### Cell 3: K-SVD Learning trên toy data

```python
# Tạo patches từ ảnh nhỏ
crop = img[50:150, 50:150]
crop_noisy = add_gaussian_noise(crop, sigma=25)

from ksvd_denoising import extract_patches
Y_noisy, indices = extract_patches(crop_noisy.astype(float), patch_size=8, stride=2)

print(f"Extracted {Y_noisy.shape[1]} patches from 100×100 image")

# Learn dictionary
from dictionary_learning import ksvd
D, X = ksvd(Y_noisy, dict_size=64, iterations=10, sparsity=3, verbose=True)

print(f"\nDictionary shape: {D.shape}")
print(f"Sparse codes shape: {X.shape}")

# Visualize learned atoms
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
for idx in range(16):
    ax = axes[idx // 4, idx % 4]
    atom = D[:, idx].reshape(8, 8)
    ax.imshow(atom, cmap='gray')
    ax.set_title(f"Atom {idx}")
    ax.axis('off')

plt.suptitle("Learned Dictionary Atoms (8×8)", fontsize=12)
plt.tight_layout()
plt.savefig('../results/week3_atoms.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Cell 4: K-SVD Denoising

```python
# Denoise ảnh lớn (có thể mất vài phút)
sigma = 25
noisy = add_gaussian_noise(img, sigma=sigma)

print(f"PSNR noisy: {psnr(img, noisy):.2f} dB")

print("\nRunning K-SVD denoising...")
# Chú ý: K-SVD chậm, dùng patch_size nhỏ và dict_size nhỏ để test nhanh
ksvd_result, D_full, X_full = ksvd_denoise(
    noisy, 
    dict_size=128,      # nhỏ hơn (thường 256) để nhanh
    patch_size=8, 
    iterations=5,       # nhỏ hơn (thường 10) để nhanh
    sparsity=3
)

print(f"PSNR K-SVD: {psnr(img, ksvd_result):.2f} dB")
```

### Cell 5: So sánh tất cả phương pháp

```python
sigma = 25
noisy = add_gaussian_noise(img, sigma=sigma)

results = {}
methods = {
    'Gaussian': lambda n: gaussian_filter(n, 5, 1.5),
    'NLM': lambda n: nlm_denoise_fast(n, h=sigma),
    'BM3D': lambda n: bm3d_denoise(n, sigma_psd=sigma),
    # K-SVD bị comment vì chậm — đã chạy ở cell 4
}

print("Running denoising methods...")
for name, fn in methods.items():
    print(f"  {name}...", end='')
    result = fn(noisy)
    results[name] = {
        'image': result,
        'psnr': psnr(img, result),
        'ssim': ssim(img, result)
    }
    print(f" PSNR={results[name]['psnr']:.2f}")

# Add K-SVD từ cell trước
if 'ksvd_result' in locals():
    results['K-SVD'] = {
        'image': ksvd_result,
        'psnr': psnr(img, ksvd_result),
        'ssim': ssim(img, ksvd_result)
    }

# Visualize
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Row 1
axes[0, 0].imshow(img, cmap='gray'); axes[0, 0].set_title("Original")
axes[0, 1].imshow(noisy, cmap='gray')
axes[0, 1].set_title(f"Noisy (σ={sigma})\nPSNR={psnr(img, noisy):.2f} dB")
axes[0, 2].imshow(results['Gaussian']['image'], cmap='gray')
axes[0, 2].set_title(f"Gaussian\nPSNR={results['Gaussian']['psnr']:.2f} dB")

# Row 2
axes[1, 0].imshow(results['NLM']['image'], cmap='gray')
axes[1, 0].set_title(f"NLM\nPSNR={results['NLM']['psnr']:.2f} dB")
axes[1, 1].imshow(results['BM3D']['image'], cmap='gray')
axes[1, 1].set_title(f"BM3D\nPSNR={results['BM3D']['psnr']:.2f} dB")

if 'K-SVD' in results:
    axes[1, 2].imshow(results['K-SVD']['image'], cmap='gray')
    axes[1, 2].set_title(f"K-SVD\nPSNR={results['K-SVD']['psnr']:.2f} dB")
else:
    axes[1, 2].text(0.5, 0.5, "K-SVD not run\n(see Cell 4)", 
                    ha='center', va='center', transform=axes[1, 2].transAxes)
    axes[1, 2].set_title("K-SVD")
    axes[1, 2].axis('off')

for ax in axes.flat:
    ax.axis('off')

plt.suptitle(f"Tuần 3: All Methods Comparison (σ={sigma})", fontsize=14)
plt.tight_layout()
plt.savefig('../results/week3_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Cell 6: Benchmark Table

```python
# Summary table
summary = []
for method, data in results.items():
    summary.append({
        'Method': method,
        'PSNR (dB)': round(data['psnr'], 2),
        'SSIM': round(data['ssim'], 4)
    })

df = pd.DataFrame(summary)
df = df.sort_values('PSNR (dB)', ascending=False).reset_index(drop=True)

print("\n" + "="*50)
print(f"BENCHMARK (σ={sigma})")
print("="*50)
print(df.to_string(index=False))

df.to_csv('../results/week3_benchmark.csv', index=False)
```

### Cell 7: Sparsity Analysis

```python
# Phân tích sparsity của X
if 'X_full' in locals():
    sparsity_count = np.sum(X_full != 0, axis=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram sparsity
    axes[0].hist(sparsity_count, bins=20, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(sparsity_count), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(sparsity_count):.2f}')
    axes[0].set_xlabel('Non-zero coefficients per patch')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Sparsity Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Sorted sparsity
    sorted_sparsity = np.sort(sparsity_count)
    axes[1].plot(sorted_sparsity, 'b-')
    axes[1].fill_between(range(len(sorted_sparsity)), sorted_sparsity, alpha=0.3)
    axes[1].set_xlabel('Patch index (sorted)')
    axes[1].set_ylabel('Non-zero coefficients')
    axes[1].set_title('Sparsity Sorted')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Average sparsity: {np.mean(sparsity_count):.2f} non-zero coefficients per patch")
```

---

## So sánh 5 phương pháp

| Method | Tuần | Ý tưởng | Ưu điểm | Nhược điểm |
|---|---|---|---|---|
| **Gaussian** | 1 | Filter cố định | Nhanh, đơn giản | Blur cạnh |
| **NLM** | 2 | Tìm patch giống + average | Bảo toàn cạnh | Chậm (search online) |
| **BM3D** | 2 | Block matching + 3D transform + threshold | SOTA truyền thống | Phức tạp, chậm |
| **K-SVD** | 3 | Học dictionary + sparse represent | Khai thác structure | Chậm train |
| **DnCNN** | 5 | Học filter bằng CNN | End-to-end | Cần data, GPU |

---

## Câu hỏi tự kiểm tra

1. **OMP vs Thresholding:** Tại sao OMP (greedy) tốt hơn chỉ threshold small coefficients?
2. **Dictionary update:** Giải thích tại sao dùng SVD mà không dùng gradient descent?
3. **K-SVD vs BM3D:** K-SVD cần train tách biệt, BM3D không cần. Lợi thế/nhược điểm?
4. **Sparsity parameter:** Nếu sparsity = 1 (rất sparse) vs = 20 (không sparse), cái nào denoise tốt hơn?

---

## Chuẩn bị Tuần 4 & 5

Tuần 3 bạn đã hiểu:
- Sparse representation: mỗi signal = tổ hợp sparse của basis
- Dictionary learning: học basis tối ưu cho dữ liệu
- K-SVD: thuật toán học dictionary hiệu quả

**Tuần 4** sẽ là "cầu nối":
- So sánh toàn diện KSVD vs BM3D vs NLM
- Viết báo cáo technical
- Chuẩn bị GitHub repo

**Tuần 5** là jump sang deep learning:
- DnCNN: end-to-end CNN để denoising
- So sánh traditional (KSVD/BM3D) vs deep learning
- Viết "Future Work" hướng tới research

Timeline cả 5 tuần thể hiện evolution: **signal processing → sparse model → deep learning**
