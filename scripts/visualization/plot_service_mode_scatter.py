#!/usr/bin/env python3
"""
Scatter plot of provider vs user performance for baseline-comparison outputs.

Usage examples:

python3 scripts/visualization/plot_service_mode_scatter.py \
  --results-dir results/baseline_comparison/synthetic_grid5x5_K3_L1000_wr1.0_wv10.0_ep1e-3_20260322_222613

python3 scripts/visualization/plot_service_mode_scatter.py \
  --csv results/baseline_comparison/.../summary.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_COLORS = {
    "OD": "#d62728",
    "Multi-FR": "#1f77b4",
    "Multi-FR-Detour": "#17becf",
    "SP-Lit": "#2ca02c",
    "Carlsson": "#9467bd",
    "TP-Lit": "#ff7f0e",
    "Joint-Nom": "#8c564b",
    "Joint-DRO": "#e377c2",
    "VCC": "#7f7f7f",
}


def _load_summary(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"Method", "Provider cost", "User cost", "Total cost"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")
    return df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, help="Folder containing summary.csv")
    parser.add_argument("--csv", type=Path, help="Direct path to summary.csv")
    parser.add_argument(
        "--out",
        type=Path,
        help="Output image path. Default: <results-dir>/service_mode_scatter.png",
    )
    parser.add_argument(
        "--label-points",
        action="store_true",
        help="Annotate every point with the method name.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Provider vs User Performance by Service Mode",
    )
    parser.add_argument(
        "--size-by-total-cost",
        action="store_true",
        help="Scale marker size by total cost.",
    )
    return parser.parse_args()


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.csv is None and args.results_dir is None:
        raise ValueError("Provide either --csv or --results-dir")
    if args.csv is not None and args.results_dir is not None:
        raise ValueError("Provide only one of --csv or --results-dir")

    csv_path = args.csv if args.csv is not None else args.results_dir / "summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find summary CSV: {csv_path}")

    out_path = args.out
    if out_path is None:
        out_dir = csv_path.parent
        out_path = out_dir / "service_mode_scatter.png"
    return csv_path, out_path


def main() -> None:
    args = _parse_args()
    csv_path, out_path = _resolve_paths(args)
    df = _load_summary(csv_path)

    x = df["Provider cost"]
    y = df["User cost"]
    methods = df["Method"]

    if args.size_by_total_cost:
        total = df["Total cost"]
        sizes = 120 + 480 * (total - total.min()) / max(total.max() - total.min(), 1e-9)
    else:
        sizes = [180] * len(df)

    fig, ax = plt.subplots(figsize=(10, 7))

    for idx, row in df.iterrows():
        method = row["Method"]
        color = DEFAULT_COLORS.get(method, "#333333")
        ax.scatter(
            row["Provider cost"],
            row["User cost"],
            s=sizes[idx] if not isinstance(sizes, list) else sizes[idx],
            color=color,
            edgecolors="white",
            linewidth=1.2,
            alpha=0.9,
        )
        if args.label_points:
            ax.annotate(
                method,
                (row["Provider cost"], row["User cost"]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=9,
            )

    ax.set_xlabel("Provider Cost")
    ax.set_ylabel("User Cost")
    ax.set_title(args.title)
    ax.grid(True, alpha=0.25, linestyle="--")

    x_pad = max((x.max() - x.min()) * 0.08, 0.5)
    y_pad = max((y.max() - y.min()) * 0.08, 0.05)
    ax.set_xlim(max(0, x.min() - x_pad), x.max() + x_pad)
    ax.set_ylim(max(0, y.min() - y_pad), y.max() + y_pad)

    handles = []
    labels = []
    for method in methods:
        if method in labels:
            continue
        labels.append(method)
        handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=DEFAULT_COLORS.get(method, "#333333"),
                markeredgecolor="white",
                markersize=9,
                linewidth=0,
            )
        )
    ax.legend(handles, labels, title="Service Mode", loc="best", frameon=True)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved scatter plot to {out_path}")


if __name__ == "__main__":
    main()
