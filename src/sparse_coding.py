"""
Sparse Coding: OMP (Orthogonal Matching Pursuit)

File này implement thuật toán tìm sparse representation.
"""

import numpy as np


def omp(y, D, sparsity):
    """
    Orthogonal Matching Pursuit — tìm sparse representation.
    
    ╔══════════════════════════════════════════════════════════════╗
    ║ ALGORITHM:                                                   ║
    ║                                                              ║
    ║ Input: signal y, dictionary D, max sparsity T               ║
    ║ Output: sparse code x (s.t. y ≈ D×x, ||x||₀ ≤ T)          ║
    ║                                                              ║
    ║ 1. Initialize: r₀ ← y (residual), A ← {} (support)         ║
    ║                                                              ║
    ║ 2. For t = 1 to T:                                           ║
    ║    a. Find best atom: i_t ← argmax_i |⟨r_{t-1}, d_i⟩|     ║
    ║    b. Add to support: A ← A ∪ {i_t}                        ║
    ║    c. Solve LS: x_A ← argmin_x ||y - D_A × x||²           ║
    ║    d. Update residual: r_t ← y - D_A × x_A                ║
    ║    e. Check convergence: if ||r_t|| < ε: break             ║
    ║                                                              ║
    ║ 3. Return: x (vector full, x[A] ≠ 0, x[not A] = 0)        ║
    ╚══════════════════════════════════════════════════════════════╝
    
    Parameters:
    -----------
    y : numpy.ndarray, shape (n,)
        Input signal/patch (n-dimensional)
        
    D : numpy.ndarray, shape (n, k)
        Dictionary (k atoms)
        Thường chuẩn hóa: mỗi cột có norm = 1
        
    sparsity : int
        Số hệ số max khác 0 (T parameter)
        Thường: sparsity = 3-10
        Lớn hơn = fit tốt hơn nhưng lâu
    
    Returns:
    --------
    x : numpy.ndarray, shape (k,)
        Sparse code
        Chỉ sparsity phần tử ≠ 0, phần còn lại = 0
    
    Complexity:
    -----------
    O(n × k × sparsity + k² × sparsity)
    
    Ví dụ:
    ------
    y = [100, 50, 30]  (3D signal)
    D = [[1, 0], [0, 1], [0.5, 0.5]]  (3×2 dictionary)
    sparsity = 2
    
    output: x = [100, 30] hoặc [95, 28] (tùy init)
    
    Meaning: y ≈ 100 * d₁ + 30 * d₂
    """
    # ===== INITIALIZATION =====
    n_atoms = D.shape[1]
    x = np.zeros(n_atoms)
    
    # Residual = error hiện tại (ban đầu = signal gốc)
    r = y.copy()
    
    # Support = tập index các atom được chọn
    support = []
    
    # ===== MAIN LOOP =====
    for t in range(sparsity):
        
        # === STEP 1: Find best atom ===
        # Tìm atom d_i có dot product lớn nhất với residual
        # |⟨r, d_i⟩| = |D.T @ r|
        
        inner_products = D.T @ r  # shape (k,)
        scores = np.abs(inner_products)  # shape (k,)
        i_best = np.argmax(scores)
        
        # === STEP 2: Add to support ===
        support.append(i_best)
        
        # === STEP 3: Solve least square ===
        # Recompute x cho tất cả atoms trong support
        # min_x ||y - D_support × x||²
        
        D_support = D[:, support]  # n × |support|
        
        # Least square: x_support = (D_support^T × D_support)^-1 × D_support^T × y
        # Nhưng dùng lstsq cho numerical stability
        x_support = np.linalg.lstsq(D_support, y, rcond=None)[0]
        
        # === STEP 4: Update residual ===
        r = y - D_support @ x_support
        
        # === STEP 5: Convergence check ===
        residual_norm = np.linalg.norm(r)
        if residual_norm < 1e-6:
            break
    
    # ===== CONSTRUCT SPARSE VECTOR =====
    # Gán giá trị x_support vào vị trí support
    x[support] = x_support
    
    return x


def omp_with_error(y, D, sparsity):
    """
    OMP nhưng trả về thêm reconstruction error.
    
    Returns:
    --------
    x : sparse code
    error : reconstruction error ||y - D×x||
    """
    x = omp(y, D, sparsity)
    error = np.linalg.norm(y - D @ x)
    return x, error


def batch_omp(Y, D, sparsity, verbose=False):
    """
    OMP cho batch patches (vectorized version).
    
    Parameters:
    -----------
    Y : numpy.ndarray, shape (n, m)
        m patches (columns)
    
    D : numpy.ndarray, shape (n, k)
        Dictionary
    
    sparsity : int
    
    verbose : bool
    
    Returns:
    --------
    X : numpy.ndarray, shape (k, m)
        Sparse codes (columns)
    
    Time: Σ_j OMP(Y[:, j]) ~ m × O(sparsity)
    """
    m = Y.shape[1]
    k = D.shape[1]
    X = np.zeros((k, m))
    
    for j in range(m):
        if verbose and j % max(1, m//10) == 0:
            print(f"  OMP: {j}/{m}", end='\r')
        
        X[:, j] = omp(Y[:, j], D, sparsity)
    
    if verbose:
        print(f"  OMP: {m}/{m} done")
    
    return X


def omp_multi_signal(Y, D, sparsity_level):
    """
    Alternative name for batch_omp (hỗ trợ tương thích).
    """
    return batch_omp(Y, D, sparsity_level)


# ===== TESTING =====
if __name__ == "__main__":
    print("Testing OMP")
    print("=" * 50)
    
    # Create random dictionary
    np.random.seed(42)
    n, k = 10, 5
    D = np.random.randn(n, k)
    D = D / np.linalg.norm(D, axis=0)  # normalize
    
    # Create sparse signal
    y_true = np.zeros(k)
    y_true[[1, 3]] = [2.0, -1.5]  # sparse
    y = D @ y_true + 0.1 * np.random.randn(n)  # add noise
    
    # Run OMP
    x = omp(y, D, sparsity=3)
    
    print(f"True sparse code: {y_true}")
    print(f"OMP result:       {x}")
    print(f"Reconstruction error: {np.linalg.norm(y - D @ x):.6f}")
    print(f"Sparsity (non-zero): {np.sum(x != 0)}")
    
    # Batch test
    Y = np.random.randn(n, 100)
    X = batch_omp(Y, D, sparsity=2, verbose=True)
    
    print(f"\nBatch OMP:")
    print(f"  Input shape: {Y.shape}")
    print(f"  Output shape: {X.shape}")
    print(f"  Average sparsity: {np.mean(np.sum(X != 0, axis=0)):.2f}")
