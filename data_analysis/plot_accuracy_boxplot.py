#!/usr/bin/env python3
"""
plot_accuracy_boxplot.py
------------------------
Boxplot of semilandmark accuracy (pixel error) across approaches.

Reads every "*_results.csv" produced by the AI/ML driver
(DL_models_run_v01.py) and the rembg adapter (rembg_transform_v01.py).
Each file has one row per semilandmark point with columns:

    img_name, model_type, index, X_new, Y_new, X_old, Y_old, dist_Old_to_New

The script pools all files, groups the per-point error `dist_Old_to_New`
(in pixels) by method, and draws one box per method.

Usage
-----
    python plot_accuracy_boxplot.py                       # current folder
    python plot_accuracy_boxplot.py --input-dir ./csvs
    python plot_accuracy_boxplot.py --max-px 50           # clip y-axis
    python plot_accuracy_boxplot.py --log                 # log y-axis
    python plot_accuracy_boxplot.py -o accuracy.png --dpi 200

Only needs: pandas, matplotlib.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe on headless machines; comment out to use plt.show()
import matplotlib.pyplot as plt


# Pretty display labels + a sensible left-to-right order for known methods.
DISPLAY = {
    "rembg": "rembg\n(image analysis)",
    "knn": "KNN", "mlp": "MLP", "cnn": "CNN", "gnn": "GNN", "vit": "ViT",
}
PREFERRED_ORDER = ["rembg", "knn", "mlp", "cnn", "gnn", "vit"]


def parse_args():
    p = argparse.ArgumentParser(description="Boxplot of pixel error per approach.")
    p.add_argument("--input-dir", type=Path, default=Path("."),
                   help="Folder containing the *_results.csv files (default: current).")
    p.add_argument("--pattern", default="*_results.csv",
                   help="Glob for the result files (default: *_results.csv).")
    p.add_argument("--metric", default="dist_Old_to_New",
                   help="Column holding the per-point error in px.")
    p.add_argument("-o", "--output", type=Path, default=Path("accuracy_boxplot.png"),
                   help="Output image path (.png/.pdf/.svg).")
    p.add_argument("--max-px", type=float, default=None,
                   help="Clip the y-axis at this many pixels (outliers still counted).")
    p.add_argument("--log", action="store_true",
                   help="Use a logarithmic y-axis.")
    p.add_argument("--sort-by-median", action="store_true",
                   help="Order boxes by median error instead of the canonical order.")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def load_results(input_dir: Path, pattern: str, metric: str) -> pd.DataFrame:
    files = sorted(input_dir.glob(pattern))
    # Never treat the summary files as results.
    files = [f for f in files if not f.name.endswith("_summary.csv")]
    if not files:
        raise SystemExit(f"No files matching '{pattern}' in '{input_dir.resolve()}'.")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        if metric not in df.columns:
            print(f"  skipping {f.name}: no '{metric}' column")
            continue
        # Prefer the model_type column; fall back to the filename stem.
        if "model_type" in df.columns and df["model_type"].notna().any():
            df["method"] = df["model_type"].astype(str)
        else:
            df["method"] = f.stem.replace("_results", "")
        df = df[["method", metric]].copy()
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        df = df.dropna(subset=[metric])
        frames.append(df)
        print(f"  loaded {f.name:32s} -> {len(df):6d} points "
              f"({df['method'].iloc[0] if len(df) else '?'})")

    if not frames:
        raise SystemExit("No usable data found.")
    return pd.concat(frames, ignore_index=True)


def order_methods(methods, medians, sort_by_median: bool):
    methods = list(methods)
    if sort_by_median:
        return sorted(methods, key=lambda m: medians[m])
    known = [m for m in PREFERRED_ORDER if m in methods]
    extra = sorted(m for m in methods if m not in PREFERRED_ORDER)
    return known + extra


def main():
    args = parse_args()
    print(f"Reading from: {args.input_dir.resolve()}")
    data = load_results(args.input_dir, args.pattern, args.metric)

    medians = data.groupby("method")[args.metric].median().to_dict()
    methods = order_methods(data["method"].unique(), medians, args.sort_by_median)

    # Per-method summary to stdout.
    print("\nPer-method error (px):")
    print(f"  {'method':<10}{'n':>8}{'median':>10}{'mean':>10}{'p90':>10}{'max':>10}")
    series = {}
    for m in methods:
        v = data.loc[data["method"] == m, args.metric].to_numpy()
        series[m] = v
        print(f"  {m:<10}{len(v):>8}{np.median(v):>10.3f}{v.mean():>10.3f}"
              f"{np.percentile(v, 90):>10.3f}{v.max():>10.3f}")

    # ── Plot ──────────────────────────────────────────────────────────────
    labels = [DISPLAY.get(m, m) for m in methods]
    box_data = [series[m] for m in methods]

    fig, ax = plt.subplots(figsize=(1.6 * len(methods) + 2, 6))

    bp = ax.boxplot(
        box_data, patch_artist=True, showfliers=True,
        widths=0.6, whis=1.5,
        medianprops=dict(color="black", linewidth=2),
        flierprops=dict(marker="o", markersize=3, markerfacecolor="none",
                        markeredgecolor="gray", alpha=0.35),
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="white",
                       markeredgecolor="black", markersize=6),
    )

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)

    cmap = plt.get_cmap("tab10")
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cmap(i % 10))
        patch.set_alpha(0.55)

    # Annotate each box with its median and n.
    ymax_for_text = args.max_px if args.max_px else max(np.percentile(series[m], 95)
                                                        for m in methods)
    for i, m in enumerate(methods, start=1):
        med = np.median(series[m])
        ax.annotate(f"med={med:.2f}\nn={len(series[m])}",
                    xy=(i, med), xytext=(i, ymax_for_text * 1.02),
                    ha="center", va="bottom", fontsize=8, color="black")

    ax.set_ylabel("Semilandmark error  |new - old|  (pixels)")
    ax.set_xlabel("Approach")
    ax.set_title("Semilandmark placement accuracy by approach\n"
                 "(lower is better; ◇ = mean, — = median)")
    ax.yaxis.grid(True, linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)

    if args.log:
        ax.set_yscale("log")
    elif args.max_px:
        ax.set_ylim(0, args.max_px)

    fig.tight_layout()
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"\nSaved boxplot -> {args.output.resolve()}")


if __name__ == "__main__":
    main()
