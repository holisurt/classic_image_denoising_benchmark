"""Summarize V3 benchmark CSV and compare BM3D with literature values."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

METHOD_TO_LITERATURE_NAME = {
    "BM3D_TAU": "BM3D",
    "NLM_IPOL_Like": "NLM",
    "NLM_Skimage_Fast": "NLM",
}


def summarize(raw_csv: Path, output_dir: Path, literature_csv: Path | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(raw_csv)

    ok = df[df["status"].eq("ok")].copy()
    summary = (
        ok.groupby(["dataset", "method", "sigma"], as_index=False)
        .agg(
            psnr_mean=("psnr", "mean"),
            psnr_std=("psnr", "std"),
            ssim_mean=("ssim", "mean"),
            ssim_std=("ssim", "std"),
            runtime_mean_ms=("runtime_ms", "mean"),
            runtime_median_ms=("runtime_ms", "median"),
            n_images=("image", "nunique"),
        )
        .round(
            {
                "psnr_mean": 4,
                "psnr_std": 4,
                "ssim_mean": 6,
                "ssim_std": 6,
                "runtime_mean_ms": 3,
                "runtime_median_ms": 3,
            }
        )
    )
    summary_path = output_dir / "summary_by_dataset_method_sigma.csv"
    summary.to_csv(summary_path, index=False)

    for metric, column in [
        ("psnr", "psnr_mean"),
        ("ssim", "ssim_mean"),
        ("runtime", "runtime_mean_ms"),
    ]:
        pivot = summary.pivot_table(index=["dataset", "method"], columns="sigma", values=column)
        pivot.to_csv(output_dir / f"pivot_{metric}.csv")

    failures = df[~df["status"].eq("ok")].copy()
    if not failures.empty:
        failures.to_csv(output_dir / "failed_rows.csv", index=False)

    if literature_csv is not None and literature_csv.exists():
        lit = pd.read_csv(literature_csv)
        ours = summary.copy()
        ours["literature_method"] = ours["method"].map(METHOD_TO_LITERATURE_NAME).fillna(ours["method"])
        comp = ours.merge(
            lit,
            left_on=["dataset", "sigma", "literature_method"],
            right_on=["dataset", "sigma", "method"],
            suffixes=("_ours", "_ref"),
            how="inner",
        )
        if not comp.empty:
            comp["psnr_gap_ours_minus_ref"] = (comp["psnr_mean"] - comp["psnr_ref"]).round(4)
            if "ssim_ref" in comp.columns:
                comp["ssim_gap_ours_minus_ref"] = (comp["ssim_mean"] - comp["ssim_ref"]).round(6)
            comp.to_csv(output_dir / "literature_comparison.csv", index=False)

    print(f"Saved summary: {summary_path}")
    print(f"Saved pivots into: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize V3 benchmark results")
    parser.add_argument("--raw", required=True, help="Path to benchmark_v3_*_raw.csv")
    parser.add_argument("--output-dir", default="results/v3/summary")
    parser.add_argument("--literature", default="references/literature_bm3d_dncnn.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    lit = Path(args.literature) if args.literature else None
    summarize(Path(args.raw), Path(args.output_dir), lit)
