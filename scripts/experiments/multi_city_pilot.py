"""Multi-city pilot: Joint-Nom at one regime point on 4 cities.

Validates the CityGeoData + RoadNetwork pipeline across road topologies
and produces a 4-panel design map for visual comparison.
"""
from __future__ import annotations

import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from lib.city_geodata import CityGeoData, CITY_CONFIGS
from lib.baselines import JointNom
from scripts.experiments.baseline_comparison import zero_J, build_prob_dict


DISTRICT_COLORS = ["#cf9eea", "#e6c585", "#9bd0a8", "#f0a8a8", "#a4cee5", "#cccccc"]


def design_and_evaluate(city_key: str, Lambda: float, wr: float, wv: float):
    cg = CityGeoData(city_key)
    rn = cg.road_network
    prob = build_prob_dict(cg)
    omega = {}

    # synthetic_refine_factor=1 disables synthetic refinement, which would
    # otherwise replace our real-city positions with a unit-cell grid and
    # collapse all cities to identical results.
    method = JointNom(max_iters=20, synthetic_refine_factor=1)
    t0 = time.monotonic()
    design = method.design(cg, prob, omega, zero_J,
                           num_districts=3,
                           Lambda=Lambda, wr=wr, wv=wv)
    t1 = time.monotonic()

    # Use evaluate_design via the method's evaluate() with road_network
    result = method.evaluate(design, cg, prob, omega, zero_J,
                             Lambda=Lambda, wr=wr, wv=wv,
                             fleet_cost_rate=4.0, road_network=rn)
    t2 = time.monotonic()
    return cg, rn, design, result, (t1 - t0, t2 - t1)


def draw_basemap(ax, cg, rn):
    """Faint road network + block-group outlines."""
    bg = cg.gdf.to_crs("EPSG:4326")

    if rn is not None:
        try:
            G = rn.G
            for u, v, data in G.edges(data=True):
                if "geometry" in data:
                    xs, ys = data["geometry"].xy
                    ax.plot(xs, ys, color="#888888", linewidth=0.6,
                            alpha=0.8, zorder=1.5)
                else:
                    x = [G.nodes[u]["x"], G.nodes[v]["x"]]
                    y = [G.nodes[u]["y"], G.nodes[v]["y"]]
                    ax.plot(x, y, color="#888888", linewidth=0.6,
                            alpha=0.8, zorder=1.5)
        except Exception as e:
            print(f"  road draw skipped: {e}")

    bg.plot(ax=ax, facecolor="none", edgecolor="#777777",
            linewidth=0.6, zorder=2)


def draw_partition(ax, cg, design):
    block_ids = cg.short_geoid_list
    bg = cg.gdf.to_crs("EPSG:4326")

    if design.assignment is None:
        return
    N = len(block_ids)

    roots = [block_ids[i] for i in range(N) if round(design.assignment[i, i]) == 1]
    root_idx = {r: k for k, r in enumerate(roots)}

    for i, bid in enumerate(block_ids):
        col = "#dddddd"
        for j in range(N):
            if round(design.assignment[i, j]) == 1:
                root_id = block_ids[j]
                if root_id in root_idx:
                    col = DISTRICT_COLORS[root_idx[root_id] % len(DISTRICT_COLORS)]
                break
        try:
            geom = bg.iloc[i].geometry
            if geom.geom_type == "Polygon":
                ax.fill(*geom.exterior.xy, color=col, alpha=0.35, zorder=2.5)
            else:
                for poly in geom.geoms:
                    ax.fill(*poly.exterior.xy, color=col, alpha=0.35, zorder=2.5)
        except Exception:
            pass

    coords = [(g.centroid.x, g.centroid.y) for g in bg.geometry]
    if design.depot_id and design.depot_id in block_ids:
        d_lon, d_lat = coords[block_ids.index(design.depot_id)]
        ax.plot(d_lon, d_lat, "*", color="black", markersize=14, zorder=6,
                markeredgecolor="white", markeredgewidth=1)

        depot_xy = (d_lon, d_lat)
        for k, root in enumerate(roots):
            blocks_in_d = [
                block_ids[i] for i in range(N)
                if round(design.assignment[i, block_ids.index(root)]) == 1
            ]
            pts = [coords[block_ids.index(b)] for b in blocks_in_d]
            if not pts:
                continue
            cur = depot_xy
            order = []
            remaining = list(range(len(pts)))
            while remaining:
                nx_idx = min(remaining,
                             key=lambda i: (pts[i][0]-cur[0])**2 + (pts[i][1]-cur[1])**2)
                order.append(nx_idx)
                cur = pts[nx_idx]
                remaining.remove(nx_idx)
            tour = [depot_xy] + [pts[i] for i in order] + [depot_xy]
            xs, ys = zip(*tour)
            tour_color = DISTRICT_COLORS[k % len(DISTRICT_COLORS)]
            ax.plot(xs, ys, "-", color=tour_color, linewidth=1.2,
                    alpha=0.9, zorder=4.5)


def main():
    Lambda = 150.0
    wr = 1.0
    wv = 1.0
    city_keys = ["pittsburgh", "chicago", "manhattan", "la"]

    print("=" * 70)
    print(f"MULTI-CITY PILOT: Joint-Nom at Λ={Lambda}, wr={wr}, K=3")
    print("=" * 70)

    results = {}
    for k in city_keys:
        print(f"\n[{k.upper()}]")
        try:
            cg, rn, design, result, (t_design, t_eval) = design_and_evaluate(
                k, Lambda, wr, wv,
            )
            print(f"  Design: {t_design:.1f}s, Evaluate: {t_eval:.1f}s")
            print(f"  Depot: {design.depot_id}")
            print(f"  T*: {[round(t, 3) for t in design.dispatch_intervals.values()]}")
            print(f"  Provider: {result.provider_cost:.2f}")
            print(f"  User time: {result.total_user_time:.3f} h")
            print(f"  Raw fleet: {result.raw_fleet_size}")
            results[k] = (cg, rn, design, result)
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()

    # 4-panel map: 2x2 layout for better visibility
    n = len(results)
    rows = 2 if n >= 3 else 1
    cols = (n + rows - 1) // rows
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).flatten()

    for ax, (k, (cg, rn, design, result)) in zip(axes, results.items()):
        draw_basemap(ax, cg, rn)
        draw_partition(ax, cg, design)
        cfg = CITY_CONFIGS[k]
        ax.set_title(
            f"{k.title()} — {cfg.description}\n"
            + f"provider={result.provider_cost:.1f}, "
            + f"user_t={result.total_user_time:.2f}h, "
            + f"fleet={result.raw_fleet_size}",
            fontsize=10, fontweight="semibold",
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    for ax in axes[len(results):]:
        ax.axis("off")

    fig.suptitle(
        f"Joint-Nom across road topologies (Λ={Lambda:g}, wr={wr:g}, K=3, 5x5 grid)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    out = "results/baseline_clouds/multi_city_pilot/joint_nom_4cities.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
