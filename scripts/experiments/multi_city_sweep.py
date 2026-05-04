"""Multi-city regime sweep.

Runs Joint-Nom over the full regime grid (3x3 = 9 points) for each of
4 cities (Pittsburgh, Chicago, Manhattan, LA). Produces:
  - per-city aggregate CSV
  - 4-panel scatter cloud comparison
  - 4-panel design map at Λ=150, wr=1
"""
from __future__ import annotations

import os
import sys
import time
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc
import colorsys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from lib.city_geodata import CityGeoData, CITY_CONFIGS
from lib.baselines import JointNom
from scripts.experiments.baseline_comparison import zero_J, build_prob_dict


CITY_KEYS = ["pittsburgh", "chicago", "manhattan", "la"]
LAMBDAS = [50, 150, 300]
WRS = [1, 3, 10]
K_GRID = [2, 3, 4, 5, 6]


def lighten(color, amount):
    c = mc.cnames.get(color, color)
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], c[1] + (1.0 - c[1]) * amount, c[2])


def run_one(cg, rn, prob, omega, Lambda, wr, wv, k_grid):
    """Run Joint-Nom with K sweep at one regime point. Pick best by aggregate cost."""
    best_result = None
    best_sel = float("inf")
    best_K = None
    for K in k_grid:
        method = JointNom(max_iters=20, synthetic_refine_factor=1)
        try:
            design = method.design(cg, prob, omega, zero_J,
                                    num_districts=K,
                                    Lambda=Lambda, wr=wr, wv=wv)
            result = method.evaluate(design, cg, prob, omega, zero_J,
                                      Lambda=Lambda, wr=wr, wv=wv,
                                      fleet_cost_rate=4.0, road_network=rn)
            sel = result.provider_cost + wr * result.total_user_time * Lambda
            if sel < best_sel:
                best_sel = sel
                best_result = result
                best_K = K
        except Exception as e:
            print(f"      K={K} failed: {e}")
    return best_result, best_K


def sweep_city(city_key):
    print(f"\n[{city_key.upper()}]")
    cg = CityGeoData(city_key)
    rn = cg.road_network
    prob = build_prob_dict(cg)
    omega = {}
    rows = []
    t0 = time.monotonic()
    for Lam in LAMBDAS:
        for wr in WRS:
            t = time.monotonic()
            res, k = run_one(cg, rn, prob, omega, Lam, wr, 1.0, K_GRID)
            print(f"  Λ={Lam:>3} wr={wr:>2}: K={k}, prov={res.provider_cost:.1f}, "
                  f"user_t={res.total_user_time:.2f}h ({time.monotonic()-t:.1f}s)")
            rows.append({
                "city": city_key,
                "Lambda": float(Lam),
                "wr": float(wr),
                "K": k,
                "provider_cost": res.provider_cost,
                "user_total": res.total_user_time,
                "wait": res.avg_wait_time,
                "in_vehicle": res.avg_invehicle_time,
                "raw_fleet": int(res.raw_fleet_size),
                "T_avg": res.avg_dispatch_interval,
            })
    print(f"  total: {time.monotonic()-t0:.1f}s")
    return rows


def plot_clouds(df, out_path):
    """4-panel cloud, one per city."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    lambdas = sorted(df["Lambda"].unique())
    wrs = sorted(df["wr"].unique())
    size_for_lambda = {l: 60 + (i / max(len(lambdas) - 1, 1)) * 160
                       for i, l in enumerate(lambdas)}
    lightness_for_wr = {w: 0.55 * (1 - i / max(len(wrs) - 1, 1))
                        for i, w in enumerate(wrs)}

    base_color = "#9467bd"  # Joint-Nom purple

    for ax, k in zip(axes, CITY_KEYS):
        sub = df[df["city"] == k]
        cfg = CITY_CONFIGS[k]
        for lam in lambdas:
            line = sub[sub["Lambda"] == lam].sort_values("wr")
            if len(line) >= 2:
                ax.plot(line["provider_cost"], line["user_total"],
                        "-", color=base_color, alpha=0.4, linewidth=1.5,
                        zorder=2)
        for _, row in sub.iterrows():
            face = lighten(base_color, lightness_for_wr.get(row["wr"], 0))
            ax.scatter(row["provider_cost"], row["user_total"],
                       s=size_for_lambda.get(row["Lambda"], 100),
                       color=face, marker="o",
                       edgecolors=base_color, linewidths=1.5, alpha=0.9,
                       zorder=4)
        ax.set_xscale("log")
        ax.set_xlabel("Provider cost rate", fontsize=10)
        ax.set_ylabel("User time (hours per rider)", fontsize=10)
        ax.set_title(f"{k.title()} — {cfg.description}", fontsize=11,
                     fontweight="semibold")
        ax.grid(True, alpha=0.2, ls="--", which="both")

    fig.suptitle(
        "Joint-Nom regime cloud across road topologies\n"
        + f"Λ ∈ {{{','.join(str(int(l)) for l in lambdas)}}}, "
        + f"wr ∈ {{{','.join(str(int(w)) for w in wrs)}}}, "
        + "K swept ∈ {2,...,6}, 5×5 grid in each city's bbox",
        fontsize=12, fontweight="bold", y=1.0,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved cloud: {out_path}")


def plot_overlay(df, out_path):
    """All 4 cities on one cloud for direct ranking."""
    fig, ax = plt.subplots(figsize=(12, 8))

    city_palette = {
        "pittsburgh": ("#cc4444", "o"),
        "chicago":    ("#4477cc", "s"),
        "manhattan":  ("#449944", "D"),
        "la":         ("#cc8844", "^"),
    }
    lambdas = sorted(df["Lambda"].unique())
    size_for_lambda = {l: 60 + (i / max(len(lambdas) - 1, 1)) * 160
                       for i, l in enumerate(lambdas)}

    for k in CITY_KEYS:
        col, mk = city_palette[k]
        sub = df[df["city"] == k]
        for lam in lambdas:
            line = sub[sub["Lambda"] == lam].sort_values("wr")
            if len(line) >= 2:
                ax.plot(line["provider_cost"], line["user_total"],
                        "-", color=col, alpha=0.45, linewidth=1.4, zorder=2)
        sizes = [size_for_lambda.get(l, 100) for l in sub["Lambda"]]
        ax.scatter(sub["provider_cost"], sub["user_total"],
                   s=sizes, color=col, marker=mk, alpha=0.85,
                   edgecolors="white", linewidths=0.7, label=k.title(),
                   zorder=4)

    ax.set_xscale("log")
    ax.set_xlabel("Provider cost rate", fontsize=11)
    ax.set_ylabel("User time (hours per rider)", fontsize=11)
    ax.legend(loc="upper right", fontsize=10, title="City")
    ax.set_title(
        "Joint-Nom across road topologies — overlay\n"
        + "lines connect wr sweep at fixed Λ; size ∝ Λ",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, alpha=0.2, ls="--", which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved overlay: {out_path}")


def main():
    out_dir = "results/baseline_clouds/multi_city_pilot"
    os.makedirs(out_dir, exist_ok=True)

    all_rows = []
    for k in CITY_KEYS:
        all_rows.extend(sweep_city(k))

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(out_dir, "multi_city_jointnom.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")

    plot_clouds(df, os.path.join(out_dir, "multi_city_clouds.png"))
    plot_overlay(df, os.path.join(out_dir, "multi_city_overlay.png"))


if __name__ == "__main__":
    main()
