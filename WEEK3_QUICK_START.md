# Tuần 3: K-SVD Dictionary Learning — Quick Start

---

## 📚 Files bạn cần

Tôi vừa tạo các file sau cho bạn:

### File Tài Liệu
- **`WEEK3_FULL_GUIDE.md`** — Lý thuyết chi tiết (25KB)
  - Phần 1: Tại sao cần K-SVD?
  - Phần 2: Sparse representation & OMP
  - Phần 3: K-SVD algorithm
  - Phần 4: Code chi tiết
  - Phần 5: Notebook thực hành

### File Code Python
1. **`sparse_coding.py`** — OMP implementation
   - `omp(y, D, sparsity)` — tìm sparse code
   - `batch_omp(Y, D, sparsity)` — batch version
   
2. **`dictionary_learning.py`** — K-SVD implementation
   - `ksvd(Y, dict_size, iterations, sparsity)` — learn dictionary
   - `ksvd_with_validation()` — với validation set

3. **`ksvd_denoising.py`** — Wrapper cho denoising
   - `ksvd_denoise(image, ...)` — denoise ảnh
   - `visualize_atoms(D)` — visualize atoms
   - `extract_patches()`, `reconstruct_from_patches()`

### File Notebook
- **`week3_notebook.ipynb`** — Complete notebook với 9 cells
  - Cell 2: OMP example
  - Cell 3: K-SVD learning
  - Cell 4: K-SVD denoising (optional)
  - Cell 5-9: Comparison & analysis

---

## 🚀 Cách Chạy

### Bước 1: Copy files vào project

Nếu cấu trúc project của bạn là:
```
image-denoising/
├── notebooks/
│   └── week2_nlm_bm3d.ipynb
├── src/
│   ├── noise.py
│   ├── filters.py
│   ├── metrics.py
│   ├── nlm.py
│   └── bm3d_wrapper.py
├── data/
└── results/
```

Thì copy 3 file code vào `src/`:
```bash
cp sparse_coding.py     src/
cp dictionary_learning.py src/
cp ksvd_denoising.py    src/
```

Copy notebook vào `notebooks/`:
```bash
cp week3_notebook.ipynb notebooks/
```

### Bước 2: Chạy notebook

Mở `notebooks/week3_notebook.ipynb` trong VS Code (Jupyter):
1. Cell 0: Setup path
2. Cell 1: Import
3. Cell 2: OMP example (nhanh)
4. Cell 3: K-SVD learning toy data (nhanh)
5. Cell 4: K-SVD denoising (CHẬM — 5-10 phút, có thể skip)
6. Cell 5-7: So sánh tất cả methods
7. Cell 8-9: Analysis

---

## 💡 Khái Niệm Chính Tuần 3

### 1. Sparse Representation
```
y ≈ D × x

y  : patch (vector)
D  : dictionary (n × k atoms)
x  : sparse code (chỉ 3-5 phần tử ≠ 0)
```

**Ý nghĩa:** patch được biểu diễn = tổ hợp sparse của k "từ"

### 2. OMP (Orthogonal Matching Pursuit)
```
Input: y (patch), D (dictionary), sparsity T
Output: x (s.t. y ≈ D×x, chỉ T hệ số ≠ 0)

Thuật toán greedy:
  Loop T lần:
    1. Tìm atom d_i có dot product lớn nhất với residual
    2. Thêm vào support set
    3. Recompute sparse code với atoms trong support
    4. Update residual
```

Độ phức tạp: O(nkT)

### 3. K-SVD (K-Singular Value Decomposition)
```
2 bước lặp:
  1. Sparse Coding: X ← OMP(Y, D, sparsity)
  2. Dictionary Update: D[k] ← update bằng SVD
```

Tối ưu: fix D → tìm X (via OMP), fix X → tìm D (via SVD)

---

## ⏱️ Thời Gian Chạy

| Phần | Thời Gian | Ghi Chú |
|---|---|---|
| Cell 0-1: Setup | <1s | |
| Cell 2: OMP | <1s | toy example |
| Cell 3: K-SVD learning | 2-5s | synthetic data |
| Cell 4: K-SVD denoising | 5-10 min | **CHẬM** — optional |
| Cell 5-7: Comparison | 1-2 min | Gaussian, NLM, BM3D |
| Cell 8-9: Analysis | <1s | |
| **TOTAL** | **~10-15 phút** | K-SVD denoising là chai cổ |

**Tip:** Skip Cell 4 nếu muốn chạy nhanh (bạn vẫn có NLM, BM3D, Gaussian)

---

## 🎯 Mục Tiêu Tuần 3

✅ **Hiểu được:**
- Sparse representation & tại sao hữu dụng
- OMP algorithm (greedy sparse coding)
- K-SVD (learning + sparse coding)
- So sánh 4 methods: Gaussian, NLM, BM3D, K-SVD

✅ **Có được:**
- Working K-SVD implementation
- Benchmark table so sánh tất cả
- Notebook hoàn chỉnh cho portfolio

---

## 📊 Kết Quả Mong Đợi (σ=25)

| Method | PSNR (dB) | SSIM | Speed |
|---|---|---|---|
| Noisy | ~20 | 0.35 | — |
| Gaussian | 28.0 | 0.78 | Instant |
| NLM | 30.5 | 0.82 | Fast |
| BM3D | **31.5** | **0.84** | Slow |
| K-SVD | ~31.0 | ~0.83 | **Slowest** |

**Insight:** 
- K-SVD competitive với BM3D (learned model vs hand-crafted)
- K-SVD có thể tối ưu hơn qua tuning hyperparameters
- Traditional methods (BM3D) vẫn strongest trước deep learning

---

## 🔧 Troubleshooting

### Lỗi: `ModuleNotFoundError: No module named 'sparse_coding'`
→ Kiểm tra files có trong `src/` không? Path setup đúng không?

### Lỗi: `numpy shape mismatch` ở OMP
→ Kiểm tra input Y shape (n_dim × n_patches)

### Chậm ở Cell 4
→ K-SVD denoising là O(iterations × n_patches × (k² + complexity))
→ Cách tốc độ: giảm dict_size (128 thay 256), giảm iterations (5 thay 10)

---

## 📖 Chi Tiết Lý Thuyết

Xem `WEEK3_FULL_GUIDE.md` để hiểu:
- OMP algorithm từng bước
- K-SVD mathematical derivation
- Tại sao SVD optimal cho dictionary update
- Comparision với NLM/BM3D

---

## 🚀 Chuẩn Bị Tuần 4-5

**Tuần 4:**
- Viết báo cáo technical comparing tất cả 4 methods
- Chuẩn bị GitHub repo (README, structure, examples)
- Plot nice comparison charts

**Tuần 5:**
- DnCNN: end-to-end CNN cho denoising
- So sánh traditional (K-SVD) vs deep learning
- Hướng research: blind denoising, real-world noise, etc.

---

## 💬 Câu Hỏi?

Nếu bạn bị stuck ở bất kỳ part nào:
1. Check `WEEK3_FULL_GUIDE.md` section liên quan
2. Run code cell by cell, debug từng dòng
3. Ask me to explain lý thuyết chi tiết

Good luck! 🎯
