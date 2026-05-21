"""Generate regime-faceted cloud figures.

Two faceting schemes per dataset:

  * "by_lambda" : one panel per Lambda; within each panel, baselines move
    along the wr axis (small-shade -> large-shade fill).
  * "by_wr"     : one panel per wr;     within each panel, baselines move
    along the Lambda axis (small marker -> large marker).

For single-city datasets each scheme produces a 1xN row of panels.
For the 4-city dataset each scheme produces a 4xN grid (rows are
cities, columns are facet values).
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


ORDER = ["Joint-Nom", "Joint-DRO", "SP-Lit (Carlsson)",
         "Multi-FR (Jacquillat)", "TP-Lit (Liu)", "VCC", "OD"]


def lighten(c, a):
    cn = mc.cnames.get(c, c)
    cn = colorsys.rgb_to_hls(*mc.to_rgb(cn))
    return colorsys.hls_to_rgb(cn[0], cn[1] + (1.0 - cn[1]) * a, cn[2])


def _panel_fixed_lambda(ax, df, Lambda_value, regime_wrs, title):
    """Fixed Lambda, varying wr. Each baseline plotted across the wr sweep."""
    sub = df[df["Lambda"] == Lambda_value]
    shade_wr = {w: 0.55 * (1 - i / max(len(regime_wrs) - 1, 1))
                for i, w in enumerate(regime_wrs)}
    for bn in ORDER:
        g = sub[sub["baseline"] == bn].sort_values("wr")
        if g.empty:
            continue
        st = SERVICE_STYLES.get(bn, {"color": "#333", "marker": "o"})
        bc = st["color"]
        x = g["Provider cost"].to_numpy(float)
        y = g["User Total (h)"].to_numpy(float)
        v = (x > 0) & np.isfinite(x) & np.isfinite(y)
        x, y = x[v], y[v]
        wrs_v = g["wr"].to_numpy(float)[v] if v.any() else np.array([])
        if len(x) >= 2:
            ax.plot(x, y, "-", color=bc, alpha=0.35, linewidth=1.0, zorder=2)
        for px, py, w in zip(x, y, wrs_v):
            face = lighten(bc, shade_wr.get(float(w), 0))
            ax.scatter(px, py, s=72, color=face, marker=st["marker"],
                       edgecolors=bc, linewidths=1.0, alpha=0.9, zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("Provider cost rate", fontsize=9)
    ax.set_ylabel("User time (h)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="semibold")
    ax.grid(True, alpha=0.2, ls="--", which="both")


def _panel_fixed_wr(ax, df, wr_value, regime_lambdas, title):
    """Fixed wr, varying Lambda. Each baseline plotted across the Lambda sweep."""
    sub = df[df["wr"] == wr_value]
    size_lambda = {l: 40 + (i / max(len(regime_lambdas) - 1, 1)) * 130
                   for i, l in enumerate(regime_lambdas)}
    for bn in ORDER:
        g = sub[sub["baseline"] == bn].sort_values("Lambda")
        if g.empty:
            continue
        st = SERVICE_STYLES.get(bn, {"color": "#333", "marker": "o"})
        bc = st["color"]
        x = g["Provider cost"].to_numpy(float)
        y = g["User Total (h)"].to_numpy(float)
        v = (x > 0) & np.isfinite(x) & np.isfinite(y)
        x, y = x[v], y[v]
        lams_v = g["Lambda"].to_numpy(float)[v] if v.any() else np.array([])
        if len(x) >= 2:
            ax.plot(x, y, "-", color=bc, alpha=0.35, linewidth=1.0, zorder=2)
        for px, py, lam in zip(x, y, lams_v):
            ax.scatter(px, py, s=size_lambda.get(float(lam), 80),
                       color=bc, marker=st["marker"],
                       edgecolors="white", linewidths=0.6, alpha=0.85, zorder=4)
    ax.set_xscale("log")
    ax.set_xlabel("Provider cost rate", fontsize=9)
    ax.set_ylabel("User time (h)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="semibold")
    ax.grid(True, alpha=0.2, ls="--", which="both")


def _legend_handles(df):
    return [Line2D([0], [0],
                   marker=SERVICE_STYLES.get(b, {"marker": "o"})["marker"],
                   color="w",
                   markerfacecolor=SERVICE_STYLES.get(b, {"color": "#333"})["color"],
                   markeredgecolor=SERVICE_STYLES.get(b, {"color": "#333"})["color"],
                   markersize=7, label=b)
            for b in ORDER if not df[df["baseline"] == b].empty]


def render_by_lambda(df, out_path: Path, suptitle: str):
    lams = sorted(set(df["Lambda"]))
    wrs = sorted(set(df["wr"]))
    fig, axes = plt.subplots(1, len(lams), figsize=(4.6 * len(lams), 4.6),
                              sharey=True)
    if len(lams) == 1:
        axes = np.array([axes])
    for ax, lam in zip(axes, lams):
        _panel_fixed_lambda(ax, df, lam, wrs,
                             f"$\\Lambda = {int(lam)}$ (fill shade $=w_r$)")
    fig.legend(handles=_legend_handles(df), loc="upper center",
               ncol=min(len(ORDER), 7), fontsize=8, framealpha=0.92,
               bbox_to_anchor=(0.5, 0.99))
    fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=1.05)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def render_by_wr(df, out_path: Path, suptitle: str):
    lams = sorted(set(df["Lambda"]))
    wrs = sorted(set(df["wr"]))
    fig, axes = plt.subplots(1, len(wrs), figsize=(4.6 * len(wrs), 4.6),
                              sharey=True)
    if len(wrs) == 1:
        axes = np.array([axes])
    for ax, w in zip(axes, wrs):
        _panel_fixed_wr(ax, df, w, lams,
                         f"$w_r = {w:g}$ (marker size $\\propto \\Lambda$)")
    fig.legend(handles=_legend_handles(df), loc="upper center",
               ncol=min(len(ORDER), 7), fontsize=8, framealpha=0.92,
               bbox_to_anchor=(0.5, 0.99))
    fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=1.05)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def render_4city_grid(scheme: str, out_path: Path):
    """4 cities x 3 facets (Lambda or wr). scheme in {'lambda', 'wr'}."""
    cities = [
        ("pittsburgh", "Pittsburgh"),
        ("chicago", "Chicago"),
        ("manhattan", "Manhattan"),
        ("la", "Los Angeles"),
    ]
    dfs = {c: pd.read_csv(f"results/baseline_clouds/cities_road/{c}/aggregate_summary.csv")
           for c, _ in cities}
    ref = dfs["pittsburgh"]
    facet_values = (sorted(set(ref["Lambda"])) if scheme == "lambda"
                    else sorted(set(ref["wr"])))
    fig, axes = plt.subplots(len(cities), len(facet_values),
                              figsize=(4.6 * len(facet_values), 4.0 * len(cities)),
                              sharey="row")
    for ci, (c, title) in enumerate(cities):
        df = dfs[c]
        for j, val in enumerate(facet_values):
            ax = axes[ci, j]
            if scheme == "lambda":
                _panel_fixed_lambda(ax, df, val, sorted(set(df["wr"])),
                                     f"{title}: $\\Lambda = {int(val)}$")
            else:
                _panel_fixed_wr(ax, df, val, sorted(set(df["Lambda"])),
                                 f"{title}: $w_r = {val:g}$")
    fig.legend(handles=_legend_handles(ref), loc="upper center",
               ncol=min(len(ORDER), 7), fontsize=9, framealpha=0.92,
               bbox_to_anchor=(0.5, 0.995))
    suptitle = ("Cross-city clouds, fixed $\\Lambda$ per column, varying $w_r$ within panel"
                if scheme == "lambda" else
                "Cross-city clouds, fixed $w_r$ per column, varying $\\Lambda$ within panel")
    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    out_root = Path("results/baseline_clouds/cities_road")

    # Per-city by-Lambda and by-wr
    for c in ["pittsburgh", "chicago", "manhattan", "la"]:
        df = pd.read_csv(out_root / c / "aggregate_summary.csv")
        render_by_lambda(df, out_root / c / "scatter_by_lambda.png",
                         suptitle=f"{c.capitalize()} cloud, sliced by $\\Lambda$")
        render_by_wr(df, out_root / c / "scatter_by_wr.png",
                     suptitle=f"{c.capitalize()} cloud, sliced by $w_r$")

    # 4-city by-Lambda and by-wr
    render_4city_grid("lambda", out_root / "four_city_clouds_by_lambda.png")
    render_4city_grid("wr",     out_root / "four_city_clouds_by_wr.png")

    # Synthetic by-Lambda and by-wr (most recent regime sweep with all 7 baselines)
    synth_csvs = sorted(
        Path("results/baseline_clouds").glob(
            "regime_synthetic_grid5x5_K3_L300_wr1.0_wv1.0_cloud_*/aggregate_summary.csv"))
    synth_pick = None
    for csv in reversed(synth_csvs):
        df = pd.read_csv(csv)
        if df["baseline"].nunique() >= 7:
            synth_pick = csv
            break
    if synth_pick is not None:
        df = pd.read_csv(synth_pick)
        render_by_lambda(df, synth_pick.parent / "scatter_by_lambda.png",
                         suptitle="Synthetic $5{\\times}5$ cloud, sliced by $\\Lambda$")
        render_by_wr(df, synth_pick.parent / "scatter_by_wr.png",
                     suptitle="Synthetic $5{\\times}5$ cloud, sliced by $w_r$")
        print(f"Synthetic source: {synth_pick.parent.name}")
