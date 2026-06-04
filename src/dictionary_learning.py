"""
Dictionary Learning: K-SVD Algorithm

K-SVD = K-Singular Value Decomposition
Thuật toán học dictionary tối ưu cho sparse representation.
"""

import numpy as np


def ksvd(Y, dict_size=256, iterations=10, sparsity=5, verbose=True):
    """
    K-SVD Dictionary Learning Algorithm.
    
    ╔═══════════════════════════════════════════════════════════════╗
    ║ ALGORITHM OVERVIEW:                                           ║
    ║                                                               ║
    ║ Input:  Y = tập m patches (columns), dict_size, iterations   ║
    ║ Output: D = dictionary (n × dict_size), X = sparse codes     ║
    ║                                                               ║
    ║ Initialize D randomly (normalized)                            ║
    ║                                                               ║
    ║ For iter = 1 to iterations:                                   ║
    ║                                                               ║
    ║   [SPARSE CODING STEP]                                        ║
    ║   For j = 1 to m:                                             ║
    ║     x_j ← OMP(y_j, D, sparsity)                              ║
    ║                                                               ║
    ║   [DICTIONARY UPDATE STEP]                                    ║
    ║   For k = 1 to dict_size:                                     ║
    ║     ω_k ← {j : X[k, j] ≠ 0}  (patch dùng atom k)            ║
    ║     E_k ← Y_ω - (D @ X - d_k ⊗ X[k,:]) |_ω                 ║
    ║     [U, S, V] ← SVD(E_k)                                      ║
    ║     d_k ← U[:, 0]              (left singular vector)         ║
    ║     X[k, ω] ← S[0] × V[ω, 0]  (update coeff)                 ║
    ║   Normalize D                                                 ║
    ║                                                               ║
    ║ Return D, X                                                   ║
    ╚═══════════════════════════════════════════════════════════════╝
    
    Mathematical Foundation:
    ------------------------
    Dictionary Learning Problem:
      min_{D,X} ||Y - D×X||_F² + λ||X||_0
    
    Cách giải: alternating optimization
      1. Fix D → optimize X (via OMP)
      2. Fix X → optimize D (via SVD)
      
    Vì bài toán này là non-convex nhưng alternating giúp hội tụ tốt.
    
    Parameters:
    -----------
    Y : numpy.ndarray, shape (n, m)
        m patches, mỗi cái n-dimensional
        Thường: patches flattened từ ảnh
        
    dict_size : int, default 256
        Kích thước dictionary (số atoms)
        Thường > n (overcomplete dictionary)
        Lớn hơn = linh hoạt hơn nhưng chậm hơn
        
    iterations : int, default 10
        Số lần lặp K-SVD
        Thường 10-50, cân bằng accuracy vs speed
        
    sparsity : int, default 5
        Số hệ số max ≠ 0 trong OMP
        Thường 3-10
        
    verbose : bool, default True
        In progress
    
    Returns:
    --------
    D : numpy.ndarray, shape (n, dict_size)
        Learned dictionary
        
    X : numpy.ndarray, shape (dict_size, m)
        Sparse codes của tất cả patches
    
    Examples:
    ---------
    >>> import numpy as np
    >>> Y = np.random.randn(64, 1000)  # 1000 patches of size 64
    >>> D, X = ksvd(Y, dict_size=256, iterations=10, sparsity=5)
    >>> # Mỗi patch được biểu diễn: y_j ≈ D @ X[:, j]
    >>> recon = D @ X
    >>> error = np.linalg.norm(Y - recon)
    """
    from sparse_coding import batch_omp
    
    n, m = Y.shape
    
    # ===== INITIALIZATION =====
    if verbose:
        print(f"K-SVD: {m} patches of dim {n}, dict_size={dict_size}")
        print(f"       sparsity={sparsity}, iterations={iterations}")
    
    # Initialize dictionary with random Gaussian
    D = np.random.randn(n, dict_size)
    
    # Normalize columns (mỗi atom = unit norm)
    D = D / np.linalg.norm(D, axis=0, keepdims=True)
    
    # ===== MAIN ITERATION =====
    for iteration in range(iterations):
        if verbose:
            print(f"\nIteration {iteration + 1}/{iterations}")
        
        # ===== SPARSE CODING STEP =====
        if verbose:
            print(f"  Sparse coding (OMP)...")
        
        X = batch_omp(Y, D, sparsity, verbose=verbose)
        
        # ===== DICTIONARY UPDATE STEP =====
        if verbose:
            print(f"  Dictionary update (SVD)...")
        
        # Duyệt từng atom
        for k in range(dict_size):
            
            # Find patches using atom k: ω_k = {j : X[k, j] ≠ 0}
            omega_k = np.where(X[k, :] != 0)[0]
            
            # Nếu atom k không được dùng → skip
            if len(omega_k) == 0:
                continue
            
            # === Compute error matrix ===
            # E_k = Y_ωk - D @ X[:, ωk] + d_k ⊗ X[k, ωk]
            #
            # Ý nghĩa:
            # - Y_ωk: actual patches using atom k
            # - D @ X[:, ωk]: reconstruction bằng tất cả atoms
            # - Trừ đi reconstruction → lỗi
            # - Nhưng cộng lại d_k ⊗ X[k, ωk]: "khôi phục" contribution của atom k
            #   → lỗi DO THIẾU atom k (nếu cập nhật nó)
            
            d_k_old = D[:, k]  # atom hiện tại
            
            # Cách 1: explicit
            Y_omega = Y[:, omega_k]  # patches dùng atom k
            recon_all = D @ X[:, omega_k]  # reconstruction hiện tại
            error = Y_omega - recon_all + np.outer(d_k_old, X[k, omega_k])
            
            # === SVD Decomposition ===
            # U, S, V = SVD(E_k) có dạng E_k ≈ U @ diag(S) @ V.T
            U, S, Vt = np.linalg.svd(error, full_matrices=False)
            
            # === Update atom ===
            # d_k ← U[:, 0] (left singular vector của largest singular value)
            D[:, k] = U[:, 0]
            
            # === Update coefficient ===
            # x_j[k] ← S[0] × V[j, 0]
            # S[0] = largest singular value
            # V[j, 0] = jth entry của right singular vector
            X[k, omega_k] = S[0] * Vt[0, :]
        
        # === Normalize dictionary ===
        # Đảm bảo mỗi atom có norm = 1 (ổn định numerical)
        D = D / np.linalg.norm(D, axis=0, keepdims=True)
        
        # === Compute reconstruction error ===
        if verbose:
            recon = D @ X
            recon_error = np.linalg.norm(Y - recon) / np.linalg.norm(Y)
            print(f"       Relative error: {recon_error:.6f}")
    
    if verbose:
        print(f"\n✓ K-SVD completed")
    
    return D, X


def ksvd_with_validation(Y_train, Y_val, dict_size=256, iterations=10,
                         sparsity=5, verbose=True):
    """
    K-SVD với validation set để monitor overfitting.
    
    Útil để chọn số iterations tối ưu.
    
    Parameters:
    -----------
    Y_train : (n, m_train)
    Y_val : (n, m_val)
    ... (các tham số khác giống ksvd)
    
    Returns:
    --------
    D : dictionary
    X_train : sparse codes trên train
    
    Side effect: in validation error
    """
    from sparse_coding import batch_omp
    
    n, m_train = Y_train.shape
    m_val = Y_val.shape[1]
    
    # Initialize
    D = np.random.randn(n, dict_size)
    D = D / np.linalg.norm(D, axis=0, keepdims=True)
    
    if verbose:
        print(f"K-SVD with validation:")
        print(f"  Train: {m_train} patches, Val: {m_val} patches")
    
    best_val_error = float('inf')
    best_D = D.copy()
    
    for iteration in range(iterations):
        
        # Sparse coding
        X_train = batch_omp(Y_train, D, sparsity, verbose=False)
        X_val = batch_omp(Y_val, D, sparsity, verbose=False)
        
        # Dictionary update (trên train data)
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
        
        if val_error < best_val_error:
            best_val_error = val_error
            best_D = D.copy()
        
        if verbose and (iteration + 1) % max(1, iterations // 5) == 0:
            print(f"  Iter {iteration+1}: Train={train_error:.6f}, Val={val_error:.6f}")
    
    return best_D, X_train


# ===== TESTING =====
if __name__ == "__main__":
    from sparse_coding import batch_omp
    
    print("Testing K-SVD")
    print("=" * 50)
    
    # Create synthetic data
    np.random.seed(42)
    n_samples, n_atoms = 1000, 64
    
    # True dictionary
    D_true = np.random.randn(10, n_atoms)
    D_true = D_true / np.linalg.norm(D_true, axis=0)
    
    # True sparse codes
    X_true = np.random.randn(n_atoms, n_samples)
    # Make sparse (set 90% to 0)
    X_true[np.random.rand(*X_true.shape) < 0.9] = 0
    
    # Generate signals
    Y = D_true @ X_true + 0.01 * np.random.randn(10, n_samples)
    
    print(f"Data shape: {Y.shape}")
    print(f"Average sparsity: {np.mean(np.sum(X_true != 0, axis=0)):.2f} non-zero")
    
    # Learn dictionary
    print("\nRunning K-SVD...")
    D_learned, X_learned = ksvd(Y, dict_size=n_atoms, iterations=5, sparsity=5)
    
    # Evaluate
    recon_error = np.linalg.norm(Y - D_learned @ X_learned) / np.linalg.norm(Y)
    print(f"\nReconstruction error: {recon_error:.6f}")
    print(f"Learned sparsity: {np.mean(np.sum(X_learned != 0, axis=0)):.2f} non-zero")
