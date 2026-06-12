"""Generate standard figures from V3 benchmark summary CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_curves(summary: pd.DataFrame, dataset: str, metric: str, y_label: str, output_path: Path) -> None:
    subset = summary[summary["dataset"].eq(dataset)].copy()
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    value_col = metric
    for method, group in subset.groupby("method"):
        group = group.sort_values("sigma")
        ax.plot(group["sigma"], group[value_col], marker="o", label=method)
    ax.set_title(f"{dataset}: {y_label} vs noise sigma")
    ax.set_xlabel("Noise sigma")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_runtime(summary: pd.DataFrame, dataset: str, output_path: Path) -> None:
    subset = summary[summary["dataset"].eq(dataset)].copy()
    if subset.empty:
        return
    # Runtime can vary a lot; use median over sigmas for a compact report chart.
    rt = subset.groupby("method", as_index=False)["runtime_mean_ms"].median().sort_values("runtime_mean_ms")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(rt["method"], rt["runtime_mean_ms"])
    ax.set_title(f"{dataset}: median runtime across sigmas")
    ax.set_xlabel("Method")
    ax.set_ylabel("Runtime (ms/image)")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_figures(summary_csv: Path, output_dir: Path) -> None:
    summary = pd.read_csv(summary_csv)
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in sorted(summary["dataset"].dropna().unique()):
        plot_curves(summary, dataset, "psnr_mean", "PSNR (dB)", output_dir / f"{dataset}_psnr_curve.png")
        plot_curves(summary, dataset, "ssim_mean", "SSIM", output_dir / f"{dataset}_ssim_curve.png")
        plot_runtime(summary, dataset, output_dir / f"{dataset}_runtime_median.png")
    print(f"Saved figures into: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make V3 benchmark report figures")
    parser.add_argument("--summary", required=True, help="summary_by_dataset_method_sigma.csv")
    parser.add_argument("--output-dir", default="results/v3/figures")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_figures(Path(args.summary), Path(args.output_dir))
