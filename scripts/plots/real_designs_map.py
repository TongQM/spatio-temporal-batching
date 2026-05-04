"""Visualize service designs from the real-data regime sweep, overlayed on
the Pittsburgh road network.

For one chosen (Lambda, wr) regime point, runs each baseline's design and
plots a per-baseline panel with:
  - Road network as faint grey edges (background)
  - Block-group polygons (dark border, no fill)
  - Service-design overlay (mode-appropriate)

Usage:
    python scripts/plots/real_designs_map.py [--Lambda 150] [--wr 1]
    python scripts/plots/real_designs_map.py --baselines joint_nom sp_lit od
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPoly
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from real_case.inputs import load_geodata, load_probability_from_population
from lib.data import GeoData
from lib.road_network import RoadNetwork
from config import DATA_PATHS
from lib.baselines import (
    JointNom, JointDRO, FRDetour, CarlssonPartition,
    TemporalOnly, VCC, FullyOD,
)
from lib.baselines.base import _get_pos
from scripts.experiments.baseline_comparison import zero_J


SERVICE_PALETTE = {
    "Joint-Nom":              "#9467bd",
    "Joint-DRO":              "#8c564b",
    "Multi-FR (Jacquillat)":  "#1f77b4",
    "SP-Lit (Carlsson)":      "#ff7f0e",
    "TP-Lit (Liu)":           "#d62728",
    "VCC":                    "#e377c2",
    "OD":                     "#7f7f7f",
}

DISTRICT_COLORS = ["#cf9eea", "#e6c585", "#9bd0a8", "#f0a8a8", "#a4cee5", "#cccccc"]


def _build_subset(n_subset: int = 25):
    g_full = load_geodata()
    positions = np.array([g_full.pos[b] for b in g_full.short_geoid_list])
    center = positions.mean(axis=0)
    order = np.argsort(np.linalg.norm(positions - center, axis=1))
    subset = [g_full.short_geoid_list[i] for i in order[:n_subset]]
    g = GeoData(DATA_PATHS["shapefile"], subset, level="block_group")
    return g


def _build_road_network(g):
    bounds = g.gdf.to_crs("EPSG:4326").total_bounds
    west, south, east, north = bounds
    pad = 0.005
    return RoadNetwork.from_bbox(north + pad, south - pad, east + pad, west - pad)


def _draw_basemap(ax, g, rn):
    """Faint road network edges + block-group outlines."""
    bg_gdf = g.gdf.to_crs("EPSG:4326")

    # Road edges (faint grey)
    if rn is not None:
        try:
            G = rn.G  # already projected to EPSG:4326
            for u, v, data in G.edges(data=True):
                if "geometry" in data:
                    xs, ys = data["geometry"].xy
                    ax.plot(xs, ys, color="#cccccc", linewidth=0.4, alpha=0.6, zorder=1)
                else:
                    x = [G.nodes[u]["x"], G.nodes[v]["x"]]
                    y = [G.nodes[u]["y"], G.nodes[v]["y"]]
                    ax.plot(x, y, color="#cccccc", linewidth=0.4, alpha=0.6, zorder=1)
        except Exception as e:
            print(f"  (road draw skipped: {e})")

    # Block-group polygons
    bg_gdf.plot(ax=ax, facecolor="none", edgecolor="#777777", linewidth=0.6, zorder=2)


def _block_lonlat(g, block_id):
    """Return (lon, lat) for a block id."""
    bg = g.gdf.to_crs("EPSG:4326")
    sg = g.short_geoid_list.index(block_id)
    geom = bg.iloc[sg].geometry
    if geom is None:
        return None
    c = geom.centroid
    return (c.x, c.y)


def _all_block_lonlat(g):
    """List of (lon, lat) for each block in g.short_geoid_list order."""
    bg = g.gdf.to_crs("EPSG:4326")
    return [(geom.centroid.x, geom.centroid.y) for geom in bg.geometry]


def _draw_partition(ax, g, design, title, color):
    block_ids = g.short_geoid_list
    N = len(block_ids)
    coords = _all_block_lonlat(g)
    bg = g.gdf.to_crs("EPSG:4326")

    if design.assignment is None:
        ax.set_title(title)
        return

    # Color block-groups by district
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
                ax.fill(*geom.exterior.xy, color=col, alpha=0.55, zorder=2.5)
            else:
                for poly in geom.geoms:
                    ax.fill(*poly.exterior.xy, color=col, alpha=0.55, zorder=2.5)
        except Exception:
            pass

    # Depot
    if design.depot_id and design.depot_id in block_ids:
        d_lon, d_lat = coords[block_ids.index(design.depot_id)]
        ax.plot(d_lon, d_lat, "*", color="black", markersize=14, zorder=6,
                markeredgecolor="white", markeredgewidth=1)

    # Tour through each district centroid (depot -> blocks in district -> depot)
    if design.depot_id and design.depot_id in block_ids:
        depot_xy = coords[block_ids.index(design.depot_id)]
        for k, root in enumerate(roots):
            blocks_in_d = [
                block_ids[i] for i in range(N)
                if round(design.assignment[i, block_ids.index(root)]) == 1
            ]
            pts = [coords[block_ids.index(b)] for b in blocks_in_d]
            if not pts:
                continue
            # Simple NN tour from depot
            order = []
            cur = depot_xy
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
            ax.plot(xs, ys, "-", color=tour_color, linewidth=1.2, alpha=0.9,
                    zorder=4.5)

    ax.set_title(title, fontsize=10, fontweight="semibold")


def _draw_fr(ax, g, design, title):
    """Fixed-route checkpoint loop overlay."""
    coords = _all_block_lonlat(g)
    block_ids = g.short_geoid_list
    bg = g.gdf.to_crs("EPSG:4326")

    # Light gray fill for all blocks
    bg.plot(ax=ax, facecolor="#f0f0f0", edgecolor="#bbbbbb",
            linewidth=0.5, zorder=2.4, alpha=0.5)

    # Depot
    if design.depot_id and design.depot_id in block_ids:
        d_lon, d_lat = coords[block_ids.index(design.depot_id)]
        ax.plot(d_lon, d_lat, "*", color="black", markersize=14, zorder=6,
                markeredgecolor="white", markeredgewidth=1)

    # Plot routes from service_metadata.fr_detour.checkpoint_sequence_by_root
    md = design.service_metadata.get("fr_detour", {})
    routes = md.get("checkpoint_sequence_by_root", {})
    line_colors = ["#1f77b4", "#2ca02c", "#9467bd", "#e377c2", "#ff7f0e"]
    for k, (rid, ckpts) in enumerate(routes.items()):
        if not ckpts:
            continue
        # Each route is a list of block ids; loop back to depot at the end
        pts = []
        for c in ckpts:
            if c in block_ids:
                pts.append(coords[block_ids.index(c)])
        if len(pts) < 2:
            continue
        # Close the loop with depot
        if design.depot_id and design.depot_id in block_ids:
            depot_xy = coords[block_ids.index(design.depot_id)]
            pts = [depot_xy] + pts + [depot_xy]
        xs, ys = zip(*pts)
        col = line_colors[k % len(line_colors)]
        ax.plot(xs, ys, "-o", color=col, linewidth=1.8, markersize=4,
                zorder=4.5, alpha=0.85)
        # Annotate route start
        ax.text(pts[1][0], pts[1][1], f"R{k+1}", fontsize=7,
                color=col, fontweight="bold", zorder=6)

    ax.set_title(title, fontsize=10, fontweight="semibold")


def _draw_unscheduled(ax, g, design, title, color, label):
    """Generic depot + sample lines for OD/VCC."""
    coords = _all_block_lonlat(g)
    block_ids = g.short_geoid_list
    bg = g.gdf.to_crs("EPSG:4326")
    bg.plot(ax=ax, facecolor="#f0f0f0", edgecolor="#bbbbbb",
            linewidth=0.5, zorder=2.4, alpha=0.5)

    if design.depot_id and design.depot_id in block_ids:
        d_lon, d_lat = coords[block_ids.index(design.depot_id)]
        ax.plot(d_lon, d_lat, "*", color="black", markersize=14, zorder=6,
                markeredgecolor="white", markeredgewidth=1)

        # Draw a sample of "depot -> block" rays
        rng = np.random.default_rng(7)
        targets = rng.choice(len(coords), size=min(8, len(coords)),
                             replace=False)
        for t in targets:
            tx, ty = coords[t]
            ax.plot([d_lon, tx], [d_lat, ty], "-", color=color,
                    linewidth=0.8, alpha=0.45, zorder=4.5)
        ax.scatter([c[0] for c in coords], [c[1] for c in coords],
                   s=20, color=color, alpha=0.65, zorder=5)

    ax.set_title(title, fontsize=10, fontweight="semibold")


def _design_baseline(name, g, prob, omega, Lambda, wr, wv):
    """Return a design object for the named baseline at this regime."""
    if name == "joint_nom":
        m = JointNom(max_iters=20)
    elif name == "joint_dro":
        m = JointDRO(epsilon=0.5, max_iters=20)
    elif name == "sp_lit":
        m = CarlssonPartition(T_fixed=0.5)
    elif name == "tp_lit":
        m = TemporalOnly(T_fixed=0.25, max_fleet=50, n_candidates=25)
    elif name == "vcc":
        m = VCC(fleet_size=20, max_walk_km=0.3)
    elif name == "od":
        m = FullyOD(fleet_size=15)
    elif name == "multi_fr":
        m = FRDetour(delta=0.0, max_fleet=3, max_iters=3, n_saa=10,
                     max_cg_iters=10, T_values=[0.25, 0.5, 1.0],
                     name="Multi-FR")
    else:
        return None
    return (name, m, m.design(g, prob, omega, zero_J,
                               num_districts=3,
                               Lambda=Lambda, wr=wr, wv=wv))


PARTITION_BASELINES = {
    "joint_nom":  ("Joint-Nom",         _draw_partition),
    "joint_dro":  ("Joint-DRO",         _draw_partition),
    "sp_lit":     ("SP-Lit (Carlsson)", _draw_partition),
}
SCHED_BASELINES = {
    "multi_fr":   ("Multi-FR (Jacquillat)", _draw_fr),
}
UNSCHED_BASELINES = {
    "vcc":  ("VCC", _draw_unscheduled, "#e377c2"),
    "od":   ("OD",  _draw_unscheduled, "#7f7f7f"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--Lambda", type=float, default=150.0)
    p.add_argument("--wr", type=float, default=1.0)
    p.add_argument("--wv", type=float, default=1.0)
    p.add_argument("--n_subset", type=int, default=25)
    p.add_argument("--baselines", nargs="*",
                   default=["joint_nom", "joint_dro", "sp_lit",
                            "multi_fr", "vcc", "od"])
    p.add_argument("--out", type=str,
                   default="results/baseline_clouds/latest_sweeps_real/designs_map.png")
    args = p.parse_args()

    print(f"Building real geodata subset ({args.n_subset} BGs)...")
    g = _build_subset(args.n_subset)
    prob = load_probability_from_population(g)
    omega = {}

    print("Loading road network (cached)...")
    rn = _build_road_network(g)

    print(f"Designing {len(args.baselines)} baselines at Λ={args.Lambda}, wr={args.wr}...")
    designs = {}
    for bn in args.baselines:
        t0 = time.monotonic()
        try:
            res = _design_baseline(bn, g, prob, omega,
                                    args.Lambda, args.wr, args.wv)
            if res is not None:
                designs[bn] = res
                _, _, design = res
                print(f"  {bn}: {time.monotonic()-t0:.1f}s, "
                      f"depot={design.depot_id}, "
                      f"districts={len(design.district_roots)}")
        except Exception as e:
            print(f"  {bn}: FAILED — {e}")

    n_panels = len(designs)
    cols = min(3, n_panels)
    rows = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 5.5))
    if n_panels == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).flatten()

    for ax, (bn, (_, _, design)) in zip(axes, designs.items()):
        _draw_basemap(ax, g, rn)
        if bn in PARTITION_BASELINES:
            disp_name, drawer = PARTITION_BASELINES[bn]
            drawer(ax, g, design, disp_name, SERVICE_PALETTE.get(disp_name, "#333333"))
        elif bn in SCHED_BASELINES:
            disp_name, drawer = SCHED_BASELINES[bn]
            drawer(ax, g, design, disp_name)
        elif bn in UNSCHED_BASELINES:
            disp_name, drawer, color = UNSCHED_BASELINES[bn]
            drawer(ax, g, design, disp_name, color, disp_name)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused axes
    for ax in axes[len(designs):]:
        ax.axis("off")

    fig.suptitle(
        f"Service designs on Pittsburgh road network "
        f"(Λ={args.Lambda:g}, wr={args.wr:g}, K=3, "
        f"{args.n_subset} block groups)",
        fontsize=12, fontweight="bold", y=1.0,
    )
    fig.tight_layout()
    out = args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
