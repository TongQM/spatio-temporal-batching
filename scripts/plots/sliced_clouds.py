"""Generate sliced (2-panel) cloud figures by baseline family.

Left panel: Pareto-core methods (Joint-Nom, Joint-DRO, SP-Lit, Multi-FR, TP-Lit).
Right panel: Operational extremes (VCC, OD).

Each panel auto-scales its x-axis (provider cost) so the Pareto-core's
narrow range is not visually compressed by VCC/OD's wide range.
"""
from __future__ import annotations
import os, sys
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


CORE = ["Joint-Nom", "Joint-DRO", "SP-Lit (Carlsson)",
        "Multi-FR (Jacquillat)", "TP-Lit (Liu)"]
EXTREMES = ["VCC", "OD"]


def lighten(c, a):
    cn = mc.cnames.get(c, c)
    cn = colorsys.rgb_to_hls(*mc.to_rgb(cn))
    return colorsys.hls_to_rgb(cn[0], cn[1] + (1.0 - cn[1]) * a, cn[2])


def _draw_panel(ax, df, baselines, regime_lambdas, regime_wrs, title):
    size_lambda = {l: 50 + (i / max(len(regime_lambdas)-1, 1)) * 130
                   for i, l in enumerate(regime_lambdas)}
    shade_wr = {w: 0.55 * (1 - i / max(len(regime_wrs)-1, 1))
                for i, w in enumerate(regime_wrs)}
    for bn in baselines:
        g = df[df["baseline"] == bn]
        if g.empty:
            continue
        st = SERVICE_STYLES.get(bn, {"color": "#333", "marker": "o"})
        bc = st["color"]
        gf = _filter_cloud_outliers(g)
        x = gf["Provider cost"].to_numpy(float)
        y = gf["User Total (h)"].to_numpy(float)
        v = (x > 0) & np.isfinite(x) & np.isfinite(y)
        x, y = x[v], y[v]
        if len(x) >= 3:
            h = _convex_hull(np.column_stack([np.log10(x), y]))
            if len(h) >= 3:
                ax.fill(10 ** h[:, 0], h[:, 1],
                        facecolor=bc, edgecolor=bc,
                        alpha=0.06, lw=0.6, zorder=1)
        for lam in regime_lambdas:
            sub = g[g["Lambda"] == lam].sort_values("wr")
            if len(sub) >= 2:
                ax.plot(sub["Provider cost"], sub["User Total (h)"],
                        "-", color=bc, alpha=0.3, linewidth=1.0, zorder=2)
        for _, row in g.iterrows():
            face = lighten(bc, shade_wr.get(float(row["wr"]), 0))
            ax.scatter(row["Provider cost"], row["User Total (h)"],
                       s=size_lambda.get(float(row["Lambda"]), 80),
                       color=face, marker=st["marker"],
                       edgecolors=bc, linewidths=1.0, alpha=0.9, zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("Provider cost rate", fontsize=10)
    ax.set_ylabel("User time (h per rider)", fontsize=10)
    ax.set_title(title, fontsize=10, fontweight="semibold")
    ax.grid(True, alpha=0.2, ls="--", which="both")


def render_split(df, out_path: Path, suptitle: str, regime_lambdas=None,
                 regime_wrs=None):
    if regime_lambdas is None:
        regime_lambdas = sorted(set(df["Lambda"]))
    if regime_wrs is None:
        regime_wrs = sorted(set(df["wr"]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    _draw_panel(ax1, df, CORE, regime_lambdas, regime_wrs,
                 "(a) Pareto-core methods")
    _draw_panel(ax2, df, EXTREMES, regime_lambdas, regime_wrs,
                 "(b) Operational extremes")

    def legend_handles(names):
        return [Line2D([0], [0],
                       marker=SERVICE_STYLES.get(b, {"marker": "o"})["marker"],
                       color="w",
                       markerfacecolor=SERVICE_STYLES.get(b, {"color": "#333"})["color"],
                       markeredgecolor=SERVICE_STYLES.get(b, {"color": "#333"})["color"],
                       markersize=8, label=b)
                for b in names if not df[df["baseline"] == b].empty]
    ax1.legend(handles=legend_handles(CORE), loc="upper right",
                fontsize=8, framealpha=0.9)
    ax2.legend(handles=legend_handles(EXTREMES), loc="upper right",
                fontsize=8, framealpha=0.9)
    fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def render_four_city_split(out_path: Path):
    cities = [
        ("pittsburgh", "Pittsburgh"),
        ("chicago", "Chicago"),
        ("manhattan", "Manhattan"),
        ("la", "Los Angeles"),
    ]
    fig, axes = plt.subplots(4, 2, figsize=(13, 18))
    ref_lams = ref_wrs = None
    for ci, (c, title) in enumerate(cities):
        df = pd.read_csv(f"results/baseline_clouds/cities_road/{c}/aggregate_summary.csv")
        lams = sorted(set(df["Lambda"]))
        wrs = sorted(set(df["wr"]))
        if ref_lams is None:
            ref_lams, ref_wrs = lams, wrs
        _draw_panel(axes[ci, 0], df, CORE, lams, wrs,
                     f"{title} (a) Pareto-core")
        _draw_panel(axes[ci, 1], df, EXTREMES, lams, wrs,
                     f"{title} (b) Operational extremes")
    fig.suptitle(
        "Cross-city baseline clouds, sliced by baseline family\n"
        f"$\\Lambda \\in $ {ref_lams}, $w_r \\in $ {ref_wrs}",
        fontsize=12, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    out_root = Path("results/baseline_clouds/cities_road")
    # Per-city scatters split
    for c in ["pittsburgh", "chicago", "manhattan", "la"]:
        df = pd.read_csv(out_root / c / "aggregate_summary.csv")
        render_split(df,
                     out_root / c / "scatter_split.png",
                     suptitle=f"{c.capitalize()} cloud, sliced by baseline family")
    # 4-city split
    render_four_city_split(out_root / "four_city_clouds_split.png")
    # Synthetic split (find latest synthetic regime sweep)
    synth_csvs = sorted(
        Path("results/baseline_clouds").glob(
            "regime_synthetic_grid5x5_K3_L300_wr1.0_wv1.0_cloud_*/aggregate_summary.csv"))
    if synth_csvs:
        latest = synth_csvs[-1]
        df = pd.read_csv(latest)
        render_split(df, latest.parent / "scatter_split.png",
                     suptitle="Synthetic $5{\\times}5$ cloud, sliced by baseline family")
