# Benchmark V3 — Classical Image Denoising Baseline

Bản V3 này được thiết kế để thay thế benchmark cũ khi cần so sánh công bằng hơn giữa Gaussian, Median, NLM, BM3D và chuẩn bị bước sang AI denoising.

## 1. Điểm sửa quan trọng so với benchmark cũ

- Không dùng Set12 giả từ `skimage.data`. Muốn so với literature thì phải dùng đúng thư mục ảnh Set12/BSD68.
- Noise được sinh **một lần duy nhất** cho mỗi `(image, sigma, seed)`, sau đó mọi method dùng chung noisy image.
- Gaussian và Median dùng SciPy optimized implementation, không dùng Python loop.
- NLM có 2 bản:
  - `NLM_Skimage_Fast`: nhanh, dùng `fast_mode=True`, `h=0.8*sigma`.
  - `NLM_IPOL_Like`: gần bảng tham số IPOL hơn, dùng `fast_mode=False`, rất chậm hơn.
- BM3D dùng `bm3d.bm3d(noisy_float01, sigma_psd=sigma/255)`.
- CSV có `status` và `error` để không mất dấu các run bị fail.

## 2. Copy vào repo

Copy các thư mục/file trong gói này vào root repo:

```text
classic_image_denoising_benchmark/
  src/baselines/classical_v3.py
  src/baselines/__init__.py
  scripts/benchmark_v3_classical.py
  scripts/summarize_v3_results.py
  scripts/make_v3_report_figures.py
  references/literature_bm3d_dncnn.csv
  references/literature_notes.md
  requirements_v3.txt
  README_V3_BENCHMARK.md
```

## 3. Cài dependency

```powershell
pip install -r requirements.txt
pip install -r requirements_v3.txt
```

Nếu `bm3d` lỗi trên Windows, thử:

```powershell
pip install --upgrade pip setuptools wheel
pip install bm3d
```

## 4. Chuẩn bị dataset

Nên đặt ảnh thật theo cấu trúc:

```text
Data/Gray test/
  Set12/
    01.png
    ...
  BSD68/
    001.png
    ...
```

Tên thư mục không bắt buộc. Bạn có thể truyền đường dẫn bằng `--data-dir`.

## 5. Smoke test trước

Chạy 3 ảnh, 2 mức sigma trước để kiểm tra môi trường:

```powershell
python scripts/benchmark_v3_classical.py `
  --dataset-name set12 `
  --data-dir "Data/Gray test/Set12" `
  --image-limit 3 `
  --sigmas 15 25 `
  --methods gaussian_scipy median_scipy nlm_fast bm3d_tau `
  --output-dir results/v3_smoke
```

## 6. Chạy Set12 đầy đủ

```powershell
python scripts/benchmark_v3_classical.py `
  --dataset-name set12 `
  --data-dir "Data/Gray test/Set12" `
  --sigmas 5 10 15 20 25 30 35 40 50 75 `
  --methods gaussian_scipy median_scipy nlm_fast nlm_ipol bm3d_tau `
  --seed 42 `
  --trials 1 `
  --output-dir results/v3
```

## 7. Chạy BSD68 đầy đủ

Bản có `nlm_ipol` có thể rất lâu. Nên chạy bản nhanh trước:

```powershell
python scripts/benchmark_v3_classical.py `
  --dataset-name bsd68 `
  --data-dir "Data/Gray test/BSD68" `
  --sigmas 5 10 15 20 25 30 35 40 50 75 `
  --methods gaussian_scipy median_scipy nlm_fast bm3d_tau `
  --seed 42 `
  --trials 1 `
  --output-dir results/v3
```

Sau đó nếu còn thời gian, chạy riêng NLM IPOL-like ở 3 sigma literature:

```powershell
python scripts/benchmark_v3_classical.py `
  --dataset-name bsd68_nlm_ipol `
  --data-dir "Data/Gray test/BSD68" `
  --sigmas 15 25 50 `
  --methods nlm_ipol `
  --seed 42 `
  --trials 1 `
  --output-dir results/v3
```

## 8. Tổng hợp kết quả

```powershell
python scripts/summarize_v3_results.py `
  --raw results/v3/benchmark_v3_set12_raw.csv `
  --output-dir results/v3/summary_set12 `
  --literature references/literature_bm3d_dncnn.csv

python scripts/summarize_v3_results.py `
  --raw results/v3/benchmark_v3_bsd68_raw.csv `
  --output-dir results/v3/summary_bsd68 `
  --literature references/literature_bm3d_dncnn.csv
```

## 9. Vẽ hình cho report

```powershell
python scripts/make_v3_report_figures.py `
  --summary results/v3/summary_set12/summary_by_dataset_method_sigma.csv `
  --output-dir results/v3/figures_set12

python scripts/make_v3_report_figures.py `
  --summary results/v3/summary_bsd68/summary_by_dataset_method_sigma.csv `
  --output-dir results/v3/figures_bsd68
```

## 10. Tiêu chí chốt baseline classical

Bạn có thể coi benchmark classical hoàn thành khi:

- `BM3D_TAU` gần literature ở sigma 15, 25, 50.
- Gaussian/Median nhanh hơn đáng kể so với NLM/BM3D.
- NLM fast và NLM IPOL-like được tách riêng trong report.
- Báo cáo có bảng PSNR, SSIM, runtime và bảng gap so với BM3D literature.
- Bạn giải thích rõ benchmark cũ bị lệch runtime do Gaussian/Median dùng Python loop còn NLM dùng implementation tối ưu.

## 11. Bước tiếp theo sang AI denoising

Sau khi chốt V3, dùng `references/literature_bm3d_dncnn.csv` làm mốc để sang DnCNN/FFDNet:

- BM3D là classical upper baseline.
- DnCNN-S là deep-learning target ở sigma 15/25/50.
- DnCNN-B hoặc FFDNet là blind/noise-level-map target cho sigma range 0–75.
