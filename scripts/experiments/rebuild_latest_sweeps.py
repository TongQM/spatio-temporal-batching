#!/usr/bin/env python3
"""Rebuild aggregate CSV and cloud plot from latest_sweeps per-setting JSONs.

Usage:
    python scripts/experiments/rebuild_latest_sweeps.py

Reads all setting_*.json files under results/baseline_clouds/latest_sweeps/,
rebuilds aggregate_summary.csv, and regenerates the cloud plot. Run this
after copying new per-baseline results into latest_sweeps/.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import colorsys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from scripts.experiments.baseline_cloud_sweep import (
    SERVICE_STYLES, _convex_hull, _filter_cloud_outliers,
)
from matplotlib.lines import Line2D


LATEST_DIR = Path("results/baseline_clouds/latest_sweeps")


def lighten(color, amount):
    c = mc.cnames.get(color, color)
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], c[1] + (1.0 - c[1]) * amount, c[2])


def load_all_rows(d: Path) -> pd.DataFrame:
    rows = []
    for sub in sorted(d.iterdir()):
        if not sub.is_dir():
            continue
        for f in sorted(sub.iterdir()):
            if f.name.startswith("setting_") and f.suffix == ".json":
                p = json.loads(f.read_text())
                if "row" in p and p["row"] is not None:
                    rows.append(p["row"])
    return pd.DataFrame(rows)


def plot_cloud(df: pd.DataFrame, meta: dict, out_path: Path) -> None:
    regime_lambdas = sorted(set(meta.get("regime_lambdas") or []))
    regime_wrs = sorted(set(meta.get("regime_wrs") or []))

    lightness_for_wr = {
        w: 0.55 * (1 - i / max(len(regime_wrs) - 1, 1))
        for i, w in enumerate(regime_wrs)
    }
    size_for_lambda = {
        l: 50 + (i / max(len(regime_lambdas) - 1, 1)) * 150
        for i, l in enumerate(regime_lambdas)
    }

    fig, ax = plt.subplots(figsize=(13.5, 9))

    for bn, g in df.groupby("baseline", sort=False):
        st = SERVICE_STYLES.get(bn, {"color": "#333", "marker": "o"})
        base_col = st["color"]

        g_filt = _filter_cloud_outliers(g)
        x = g_filt["Provider cost"].to_numpy(float)
        y = g_filt["User Total (h)"].to_numpy(float)
        v = x > 0
        x, y = x[v], y[v]
        if len(x) >= 3:
            hull = _convex_hull(np.column_stack([np.log10(x), y]))
            if len(hull) >= 3:
                ax.fill(
                    10 ** hull[:, 0], hull[:, 1],
                    facecolor=base_col, edgecolor=base_col,
                    alpha=0.05, lw=0.6, zorder=1,
                )

        for lam in regime_lambdas:
            sub = g[g["Lambda"] == lam].sort_values("wr")
            if len(sub) >= 2:
                ax.plot(
                    sub["Provider cost"], sub["User Total (h)"],
                    "-", color=base_col, alpha=0.30, linewidth=1.1, zorder=2,
                )

        for _, row in g.iterrows():
            face = lighten(base_col, lightness_for_wr.get(float(row["wr"]), 0))
            ax.scatter(
                row["Provider cost"], row["User Total (h)"],
                s=size_for_lambda.get(float(row["Lambda"]), 100),
                color=face, marker=st["marker"],
                edgecolors=base_col, linewidths=1.2, alpha=0.92, zorder=4,
            )

    # Legends
    baseline_handles = [
        Line2D([0], [0], marker=SERVICE_STYLES.get(bn, {"marker": "o"})["marker"],
               color="w",
               markerfacecolor=SERVICE_STYLES.get(bn, {"color": "#333"})["color"],
               markeredgecolor=SERVICE_STYLES.get(bn, {"color": "#333"})["color"],
               markersize=9, label=bn)
        for bn in df["baseline"].unique()
    ]
    leg1 = ax.legend(handles=baseline_handles, loc="upper right", title="Method",
                     framealpha=0.92, fontsize=8, title_fontsize=9)
    ax.add_artist(leg1)

    lambda_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="#666", markeredgecolor="#333",
               markersize=np.sqrt(size_for_lambda[float(l)]),
               label=f"\u039b={int(l)}")
        for l in regime_lambdas
    ]
    leg2 = ax.legend(handles=lambda_handles, loc="center right",
                     bbox_to_anchor=(1.0, 0.55), title="\u039b (size)",
                     framealpha=0.92, fontsize=8, title_fontsize=9)
    ax.add_artist(leg2)

    wr_handles = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=lighten("#666", lightness_for_wr[float(w)]),
               markeredgecolor="#333", markersize=9, markeredgewidth=1.2,
               label=f"wr={w:g}")
        for w in regime_wrs
    ]
    ax.legend(handles=wr_handles, loc="center right",
              bbox_to_anchor=(1.0, 0.30), title="wr (fill shade)",
              framealpha=0.92, fontsize=8, title_fontsize=9)

    ax.set_xscale("log")
    ax.set_xlabel("Provider cost rate", fontsize=11)
    ax.set_ylabel("User time (hours per rider)", fontsize=11)

    fcr = meta.get("fleet_cost_rate", 0)
    ax.set_title(
        f"Baseline Performance Clouds — latest sweep results\n"
        + "\u039b \u2208 {" + ",".join(str(int(l)) for l in regime_lambdas) + "}, "
        + "wr \u2208 {" + ",".join(str(w) for w in regime_wrs) + "}, "
        + f"wv={meta.get('wv', '?')}, $c_f$={fcr}; "
        + f"each baseline picks best K (FR fleet fallback K={meta.get('districts', '?')})",
        fontsize=11, fontweight="bold",
    )
    ax.grid(True, alpha=0.2, ls="--", which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    d = LATEST_DIR
    if not d.exists():
        print(f"Error: {d} does not exist.")
        return

    meta_path = d / "experiment.json"
    if not meta_path.exists():
        print(f"Error: {meta_path} does not exist.")
        return
    meta = json.loads(meta_path.read_text())

    df = load_all_rows(d)
    if df.empty:
        print("No rows found in latest_sweeps/.")
        return

    # Status
    print(f"Loaded {len(df)} rows from {d}")
    for bn, g in df.groupby("baseline"):
        print(f"  {bn}: {len(g)}/9")

    # Write aggregate
    csv_path = d / "aggregate_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    # Plot
    png_path = d / "scatter_user_provider_cloud.png"
    plot_cloud(df, meta, png_path)
    print(f"Saved {png_path}")


if __name__ == "__main__":
    main()
