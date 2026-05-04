"""Visualize service designs for each of 4 cities at one regime point.

Produces a grid of panels: rows = cities, columns = baselines.
Each panel overlays the chosen design on the city's real road network.

Usage:
    python scripts/plots/multi_city_designs_map.py [--Lambda 150] [--wr 1]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from lib.data import GeoData
from lib.road_network import RoadNetwork
from lib.baselines import (
    JointNom, JointDRO, FRDetour, CarlssonPartition,
    TemporalOnly, VCC, FullyOD,
)
from scripts.experiments.baseline_comparison import zero_J


CITY_CONFIG = {
    "pittsburgh": dict(
        shapefile="data/2023_shape_files/census_shape_block_group/tl_2023_42_bg.shp",
        geoid_list_file="data/city_bg_lists/pittsburgh.txt",
        title="Pittsburgh — irregular hilly with rivers",
    ),
    "chicago": dict(
        shapefile="data/tiger_2023/tl_2023_17_bg/tl_2023_17_bg.shp",
        geoid_list_file="data/city_bg_lists/chicago.txt",
        title="Chicago Loop — strict cardinal grid",
    ),
    "manhattan": dict(
        shapefile="data/tiger_2023/tl_2023_36_bg/tl_2023_36_bg.shp",
        geoid_list_file="data/city_bg_lists/manhattan.txt",
        title="Manhattan Midtown — avenue+street grid",
    ),
    "la": dict(
        shapefile="data/tiger_2023/tl_2023_06_bg/tl_2023_06_bg.shp",
        geoid_list_file="data/city_bg_lists/la.txt",
        title="Los Angeles DTLA — sprawled grid + freeways",
    ),
}

DISTRICT_COLORS = ["#cf9eea", "#e6c585", "#9bd0a8", "#f0a8a8", "#a4cee5", "#cccccc"]


def load_city_geodata(city: str) -> GeoData:
    cfg = CITY_CONFIG[city]
    with open(cfg["geoid_list_file"]) as f:
        geoids = [l.strip() for l in f if l.strip()]
    return GeoData(cfg["shapefile"], geoids, level="block_group")


def build_road_network(g: GeoData) -> RoadNetwork:
    bounds = g.gdf.to_crs("EPSG:4326").total_bounds
    west, south, east, north = bounds
    pad = 0.005
    return RoadNetwork.from_bbox(north + pad, south - pad, east + pad, west - pad)


def all_block_lonlat(g: GeoData):
    bg = g.gdf.to_crs("EPSG:4326")
    return [(geom.centroid.x, geom.centroid.y) for geom in bg.geometry]


def draw_basemap(ax, g, rn):
    bg_4326 = g.gdf.to_crs("EPSG:4326")
    if rn is not None:
        try:
            G = rn.G
            for u, v, data in G.edges(data=True):
                if "geometry" in data:
                    xs, ys = data["geometry"].xy
                    ax.plot(xs, ys, color="#bbbbbb", linewidth=0.4,
                            alpha=0.7, zorder=1)
                else:
                    x = [G.nodes[u]["x"], G.nodes[v]["x"]]
                    y = [G.nodes[u]["y"], G.nodes[v]["y"]]
                    ax.plot(x, y, color="#bbbbbb", linewidth=0.4,
                            alpha=0.7, zorder=1)
        except Exception:
            pass
    bg_4326.plot(ax=ax, facecolor="none", edgecolor="#777777",
                  linewidth=0.5, zorder=2)


def draw_partition(ax, g, design):
    block_ids = g.short_geoid_list
    coords = all_block_lonlat(g)
    bg = g.gdf.to_crs("EPSG:4326")
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
                ax.fill(*geom.exterior.xy, color=col, alpha=0.45, zorder=2.5)
            else:
                for poly in geom.geoms:
                    ax.fill(*poly.exterior.xy, color=col, alpha=0.45, zorder=2.5)
        except Exception:
            pass

    if design.depot_id and design.depot_id in block_ids:
        d_lon, d_lat = coords[block_ids.index(design.depot_id)]
        ax.plot(d_lon, d_lat, "*", color="black", markersize=10, zorder=6,
                markeredgecolor="white", markeredgewidth=0.8)
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
            ax.plot(xs, ys, "-", color=tour_color, linewidth=0.9,
                    alpha=0.9, zorder=4.5)


def draw_fr(ax, g, design):
    block_ids = g.short_geoid_list
    coords = all_block_lonlat(g)
    bg = g.gdf.to_crs("EPSG:4326")
    bg.plot(ax=ax, facecolor="#f5f5f5", edgecolor="#aaaaaa",
            linewidth=0.4, zorder=2.4, alpha=0.5)
    if design.depot_id and design.depot_id in block_ids:
        d_lon, d_lat = coords[block_ids.index(design.depot_id)]
        ax.plot(d_lon, d_lat, "*", color="black", markersize=10, zorder=6,
                markeredgecolor="white", markeredgewidth=0.8)
    md = design.service_metadata.get("fr_detour", {})
    routes = md.get("checkpoint_sequence_by_root", {})
    line_colors = ["#1f77b4", "#2ca02c", "#9467bd", "#e377c2", "#ff7f0e"]
    for k, (rid, ckpts) in enumerate(routes.items()):
        if not ckpts:
            continue
        pts = []
        for c in ckpts:
            if c in block_ids:
                pts.append(coords[block_ids.index(c)])
        if len(pts) < 2:
            continue
        if design.depot_id and design.depot_id in block_ids:
            depot_xy = coords[block_ids.index(design.depot_id)]
            pts = [depot_xy] + pts + [depot_xy]
        xs, ys = zip(*pts)
        col = line_colors[k % len(line_colors)]
        ax.plot(xs, ys, "-o", color=col, linewidth=1.4, markersize=2.5,
                zorder=4.5, alpha=0.85)


def draw_unscheduled(ax, g, design, color):
    block_ids = g.short_geoid_list
    coords = all_block_lonlat(g)
    bg = g.gdf.to_crs("EPSG:4326")
    bg.plot(ax=ax, facecolor="#f5f5f5", edgecolor="#aaaaaa",
            linewidth=0.4, zorder=2.4, alpha=0.5)
    if design.depot_id and design.depot_id in block_ids:
        d_lon, d_lat = coords[block_ids.index(design.depot_id)]
        ax.plot(d_lon, d_lat, "*", color="black", markersize=10, zorder=6,
                markeredgecolor="white", markeredgewidth=0.8)
        rng = np.random.default_rng(7)
        targets = rng.choice(len(coords), size=min(8, len(coords)),
                             replace=False)
        for t in targets:
            tx, ty = coords[t]
            ax.plot([d_lon, tx], [d_lat, ty], "-", color=color,
                    linewidth=0.6, alpha=0.4, zorder=4.5)
        ax.scatter([c[0] for c in coords], [c[1] for c in coords],
                   s=12, color=color, alpha=0.6, zorder=5)


# Baseline label, factory, drawer
BASELINE_SPECS = [
    ("Joint-Nom",  lambda: JointNom(max_iters=20),                                  "partition"),
    ("Joint-DRO",  lambda: JointDRO(epsilon=0.5, max_iters=20),                     "partition"),
    ("SP-Lit",     lambda: CarlssonPartition(T_fixed=0.5),                          "partition"),
    ("Multi-FR",   lambda: FRDetour(delta=0.0, max_fleet=3, max_iters=3, n_saa=10,
                                     max_cg_iters=10,
                                     T_values=[0.25, 0.5, 1.0], name="Multi-FR"),    "fr"),
    ("VCC",        lambda: VCC(fleet_size=20, max_walk_km=0.3),                     "vcc"),
    ("OD",         lambda: FullyOD(fleet_size=15),                                   "od"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--Lambda", type=float, default=150.0)
    p.add_argument("--wr", type=float, default=1.0)
    p.add_argument("--cities", nargs="+",
                   default=["pittsburgh", "chicago", "manhattan", "la"])
    p.add_argument("--out", type=str,
                   default="results/baseline_clouds/cities/four_city_designs_map.png")
    args = p.parse_args()

    cities = args.cities
    nC = len(cities)
    nB = len(BASELINE_SPECS)

    # Pre-load each city
    city_data = {}
    for c in cities:
        print(f"\n[{c}] loading...")
        g = load_city_geodata(c)
        rn = build_road_network(g)
        prob = {b: 1.0/len(g.short_geoid_list) for b in g.short_geoid_list}
        city_data[c] = (g, rn, prob)

    # Build figure
    fig, axes = plt.subplots(nC, nB, figsize=(3.6 * nB, 3.6 * nC))
    if nC == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for ci, city in enumerate(cities):
        g, rn, prob = city_data[city]
        for bi, (bname, factory, drawer_kind) in enumerate(BASELINE_SPECS):
            ax = axes[ci, bi]
            t0 = time.monotonic()
            try:
                m = factory()
                d = m.design(g, prob, {}, zero_J,
                             num_districts=3,
                             Lambda=args.Lambda, wr=args.wr, wv=1.0)
                draw_basemap(ax, g, rn)
                if drawer_kind == "partition":
                    draw_partition(ax, g, d)
                elif drawer_kind == "fr":
                    draw_fr(ax, g, d)
                elif drawer_kind == "vcc":
                    draw_unscheduled(ax, g, d, "#e377c2")
                elif drawer_kind == "od":
                    draw_unscheduled(ax, g, d, "#7f7f7f")
                elapsed = time.monotonic() - t0
                print(f"  [{city}|{bname}] {elapsed:.1f}s")
            except Exception as e:
                print(f"  [{city}|{bname}] FAILED: {e}")
                ax.text(0.5, 0.5, f"{bname}\n(failed)",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=8, color="red")
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if ci == 0:
                ax.set_title(bname, fontsize=10, fontweight="semibold")
            if bi == 0:
                ax.set_ylabel(CITY_CONFIG[city]["title"].split(" — ")[0],
                              fontsize=10, fontweight="semibold")

    fig.suptitle(
        f"Service designs across cities (Λ={args.Lambda:g}, wr={args.wr:g}, K=3, "
        + "real BGs + road network)",
        fontsize=12, fontweight="bold", y=1.0,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
