"""
scripts/plot_results.py — Vẽ biểu đồ benchmark

Cấu trúc project:
    image-denoising/
    ├── src/
    ├── scripts/       ← file này nằm ở đây
    └── results/       ← input CSV và output figures đều ở đây

Chạy từ root của project:
    python scripts/plot_results.py

Input:  results/benchmark_raw.csv
Output: results/figures/*.png
"""

import sys, os
from pathlib import Path

# scripts/ nằm trong root/scripts/ → root = parent
ROOT = Path(__file__).parent.parent
os.chdir(ROOT)   # đảm bảo relative paths đúng
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Style Config
# ─────────────────────────────────────────────

COLORS = {
    'Noisy'    : '#aaaaaa',
    'Gaussian' : '#7F77DD',   # purple
    'Median'   : '#EF9F27',   # amber
    'NLM'      : '#1D9E75',   # teal
    'BM3D'     : '#D85A30',   # coral
    'KSVD'     : '#378ADD',   # blue
}

METHOD_ORDER = ['Noisy', 'Gaussian', 'Median', 'NLM', 'BM3D', 'KSVD']

SIGMAS = [15, 25, 50]

plt.rcParams.update({
    'figure.dpi'        : 150,
    'font.family'       : 'DejaVu Sans',
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.grid'         : True,
    'grid.alpha'        : 0.25,
    'grid.linewidth'    : 0.5,
})


def load_data(csv_path='results/benchmark_raw.csv'):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['psnr', 'ssim'])
    return df


def methods_in(df):
    present = [m for m in METHOD_ORDER if m in df['method'].unique()]
    return present


# ─────────────────────────────────────────────
# Plot 1: PSNR Bar Chart (sigma × method)
# ─────────────────────────────────────────────

def plot_psnr_bars(df, save_path='results/figures/01_psnr_bars.png'):
    """
    3 sigma × N methods: grouped bar chart.
    Dễ đọc nhất cho nhà tuyển dụng.
    """
    methods = methods_in(df)
    summary = df.groupby(['method', 'sigma'])['psnr'].mean().reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle('Average PSNR (dB) — Higher is Better', fontsize=14, y=1.02)

    for ax, sigma in zip(axes, SIGMAS):
        sub = summary[summary['sigma'] == sigma].copy()
        sub = sub.set_index('method').reindex(methods)

        bars = ax.bar(
            methods,
            sub['psnr'],
            color=[COLORS.get(m, '#888') for m in methods],
            edgecolor='white',
            linewidth=0.5,
            zorder=3,
        )

        # Value labels on bars
        for bar, val in zip(bars, sub['psnr']):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.1,
                        f'{val:.1f}',
                        ha='center', va='bottom', fontsize=9)

        ax.set_title(f'σ = {sigma}', fontsize=12)
        ax.set_ylabel('PSNR (dB)' if ax == axes[0] else '')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, fontsize=9)

        # Highlight best (excluding Noisy)
        best_val = sub.loc[sub.index != 'Noisy', 'psnr'].max()
        best_method = sub.loc[sub['psnr'] == best_val, 'psnr'].idxmax()
        if best_method in methods:
            i = methods.index(best_method)
            bars[i].set_edgecolor('#222')
            bars[i].set_linewidth(2)

    plt.tight_layout()
    os.makedirs(Path(save_path).parent, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


# ─────────────────────────────────────────────
# Plot 2: PSNR vs Sigma Line Chart
# ─────────────────────────────────────────────

def plot_psnr_vs_sigma(df, save_path='results/figures/02_psnr_vs_sigma.png'):
    """
    Line chart: PSNR vs σ cho từng method.
    Thể hiện rõ trend khi noise tăng.
    """
    methods = [m for m in methods_in(df) if m != 'Noisy']
    summary = df.groupby(['method', 'sigma'])['psnr'].agg(['mean', 'std']).reset_index()
    sigmas = sorted(df['sigma'].unique())

    fig, ax = plt.subplots(figsize=(8, 5))

    for method in methods:
        sub = summary[summary['method'] == method].sort_values('sigma')
        ax.plot(sub['sigma'], sub['mean'],
                'o-', color=COLORS.get(method, '#888'),
                label=method, linewidth=2, markersize=6, zorder=3)
        ax.fill_between(sub['sigma'],
                        sub['mean'] - sub['std'],
                        sub['mean'] + sub['std'],
                        alpha=0.08,
                        color=COLORS.get(method, '#888'))

    ax.set_xlabel('Noise level σ', fontsize=11)
    ax.set_ylabel('Average PSNR (dB)', fontsize=11)
    ax.set_title('PSNR vs Noise Level', fontsize=13)
    ax.set_xticks(sigmas)
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


# ─────────────────────────────────────────────
# Plot 3: PSNR vs Runtime scatter (trade-off)
# ─────────────────────────────────────────────

def plot_tradeoff(df, sigma=25, save_path='results/figures/03_tradeoff.png'):
    """
    Scatter: PSNR (quality) vs runtime (speed) — tại sigma cố định.
    Thể hiện trade-off rõ nhất.
    """
    methods = [m for m in methods_in(df) if m != 'Noisy']
    sub = df[df['sigma'] == sigma].groupby('method').agg(
        psnr_mean    = ('psnr',       'mean'),
        runtime_mean = ('runtime_ms', 'mean'),
    ).reindex(methods).dropna()

    fig, ax = plt.subplots(figsize=(8, 5))

    for method, row in sub.iterrows():
        color = COLORS.get(method, '#888')
        ax.scatter(row['runtime_mean'], row['psnr_mean'],
                   color=color, s=120, zorder=5)
        ax.annotate(method,
                    xy=(row['runtime_mean'], row['psnr_mean']),
                    xytext=(8, 4), textcoords='offset points',
                    fontsize=10, color=color)

    ax.set_xlabel('Runtime (ms / image)  — Lower is Better', fontsize=11)
    ax.set_ylabel('Average PSNR (dB)  — Higher is Better', fontsize=11)
    ax.set_title(f'Quality vs Speed Trade-off (σ={sigma})', fontsize=13)

    # Annotation arrow
    ax.annotate('', xy=(0.15, 0.92), xytext=(0.05, 0.92),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='<-', color='#666'))
    ax.text(0.16, 0.92, 'Faster', transform=ax.transAxes,
            fontsize=9, color='#666', va='center')
    ax.annotate('', xy=(0.98, 0.15), xytext=(0.98, 0.05),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle='<-', color='#666'))
    ax.text(0.98, 0.16, 'Better', transform=ax.transAxes,
            fontsize=9, color='#666', ha='right', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


# ─────────────────────────────────────────────
# Plot 4: SSIM Heatmap
# ─────────────────────────────────────────────

def plot_ssim_heatmap(df, save_path='results/figures/04_ssim_heatmap.png'):
    """
    Heatmap SSIM: method × sigma.
    Cho thấy structural preservation.
    """
    methods = methods_in(df)
    sigmas  = sorted(df['sigma'].unique())
    summary = df.groupby(['method', 'sigma'])['ssim'].mean().reset_index()

    matrix = np.zeros((len(methods), len(sigmas)))
    for i, m in enumerate(methods):
        for j, s in enumerate(sigmas):
            v = summary[(summary['method']==m) & (summary['sigma']==s)]['ssim'].values
            matrix[i, j] = v[0] if len(v) > 0 else np.nan

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto',
                   vmin=matrix.min(), vmax=matrix.max())

    ax.set_xticks(range(len(sigmas)))
    ax.set_xticklabels([f'σ={s}' for s in sigmas])
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_title('Average SSIM (green = better)', fontsize=13)

    for i in range(len(methods)):
        for j in range(len(sigmas)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f'{v:.3f}', ha='center', va='center',
                        fontsize=9,
                        color='white' if v < 0.65 else 'black')

    plt.colorbar(im, ax=ax, label='SSIM')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


# ─────────────────────────────────────────────
# Plot 5: Per-image variability boxplot
# ─────────────────────────────────────────────

def plot_boxplot(df, sigma=25, save_path='results/figures/05_boxplot.png'):
    """
    Boxplot PSNR phân phối theo từng method tại sigma cố định.
    Thể hiện ổn định của mỗi method trên nhiều ảnh.
    """
    methods = [m for m in methods_in(df) if m != 'Noisy']
    sub = df[df['sigma'] == sigma]

    data = [sub[sub['method'] == m]['psnr'].dropna().values for m in methods]
    colors = [COLORS.get(m, '#888') for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))

    bp = ax.boxplot(data, patch_artist=True, notch=False, widths=0.5,
                    medianprops=dict(color='black', linewidth=2))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    for i, (d, method) in enumerate(zip(data, methods)):
        ax.scatter(np.random.normal(i+1, 0.05, len(d)), d,
                   color=COLORS.get(method, '#888'), alpha=0.5,
                   s=20, zorder=5)

    ax.set_xticks(range(1, len(methods)+1))
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel('PSNR (dB)', fontsize=11)
    ax.set_title(f'PSNR Distribution per Method (σ={sigma})\n'
                 f'Dots = individual images', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


# ─────────────────────────────────────────────
# Plot 6: Runtime bar chart
# ─────────────────────────────────────────────

def plot_runtime(df, sigma=25, save_path='results/figures/06_runtime.png'):
    """
    Bar chart runtime: thể hiện tốc độ từng method.
    Log scale để dễ so sánh khi runtime chênh lệch lớn.
    """
    methods = [m for m in methods_in(df) if m != 'Noisy']
    sub = df[(df['sigma'] == sigma)].groupby('method')['runtime_ms'].mean().reindex(methods)

    fig, ax = plt.subplots(figsize=(8, 4))

    bars = ax.bar(methods, sub.values,
                  color=[COLORS.get(m, '#888') for m in methods],
                  edgecolor='white', linewidth=0.5, zorder=3)

    for bar, val in zip(bars, sub.values):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(sub.values)*0.01,
                    f'{val:.0f}ms',
                    ha='center', va='bottom', fontsize=9)

    ax.set_yscale('log')
    ax.set_ylabel('Runtime per image (ms, log scale)', fontsize=11)
    ax.set_title(f'Runtime Comparison (σ={sigma})', fontsize=13)
    ax.set_xlabel('Method')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")


# ─────────────────────────────────────────────
# Plot 7: Summary Table figure
# ─────────────────────────────────────────────

def plot_summary_table(df, save_path='results/figures/07_summary_table.png'):
    """
    Render bảng tóm tắt thành ảnh PNG — dễ chèn vào report.
    """
    methods = [m for m in methods_in(df)]
    sigmas  = sorted(df['sigma'].unique())
    summary = df.groupby(['method', 'sigma']).agg(
        psnr = ('psnr', 'mean'),
        ssim = ('ssim', 'mean'),
        rt   = ('runtime_ms', 'mean'),
    ).round({'psnr': 2, 'ssim': 3, 'rt': 0}).reset_index()

    # Build cell data
    col_labels = ['Method'] + [f'σ={s}\nPSNR / SSIM' for s in sigmas] + ['Runtime\n(ms)']
    rows = []
    for m in methods:
        row = [m]
        for s in sigmas:
            v = summary[(summary['method']==m) & (summary['sigma']==s)]
            if len(v):
                row.append(f"{v['psnr'].values[0]:.2f} / {v['ssim'].values[0]:.3f}")
            else:
                row.append('—')
        # Runtime at sigma=25
        v25 = summary[(summary['method']==m) & (summary['sigma']==25)]
        row.append(f"{v25['rt'].values[0]:.0f}" if len(v25) else '—')
        rows.append(row)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    tbl = ax.table(
        cellText   = rows,
        colLabels  = col_labels,
        cellLoc    = 'center',
        loc        = 'center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)

    # Color method cells
    for i, method in enumerate(methods):
        cell = tbl[(i+1, 0)]
        cell.set_facecolor(COLORS.get(method, '#eee'))
        cell.set_text_props(color='white' if method != 'Noisy' else '#555')

    # Header
    for j in range(len(col_labels)):
        tbl[(0, j)].set_facecolor('#2c2c2a')
        tbl[(0, j)].set_text_props(color='white', weight='bold')

    plt.title('Benchmark Summary Table', fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"  ✓ Saved: {save_path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def plot_all(csv_path='results/benchmark_raw.csv',
             output_dir='results/figures'):
    """
    Vẽ tất cả 7 biểu đồ từ CSV.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nLoading data from {csv_path}...")
    df = load_data(csv_path)
    print(f"  {len(df)} records, methods: {df['method'].unique()}")

    print("\nGenerating plots...")
    plot_psnr_bars(df,    f'{output_dir}/01_psnr_bars.png')
    plot_psnr_vs_sigma(df, f'{output_dir}/02_psnr_vs_sigma.png')
    plot_tradeoff(df,     f'{output_dir}/03_tradeoff.png')
    plot_ssim_heatmap(df, f'{output_dir}/04_ssim_heatmap.png')
    plot_boxplot(df,      f'{output_dir}/05_boxplot.png')
    plot_runtime(df,      f'{output_dir}/06_runtime.png')
    plot_summary_table(df, f'{output_dir}/07_summary_table.png')

    print(f"\n✓ All plots saved to {output_dir}/")
    print("  Files: 01_psnr_bars.png · 02_psnr_vs_sigma.png · 03_tradeoff.png")
    print("         04_ssim_heatmap.png · 05_boxplot.png · 06_runtime.png")
    print("         07_summary_table.png")


if __name__ == '__main__':
    plot_all()
