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
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from lib.data import GeoData
from lib.road_network import RoadNetwork
from lib.baselines import (
    JointNom, JointDRO, FRDetour, CarlssonPartition,
    TemporalOnly, VCC, FullyOD,
)
from lib.baselines.base import _get_pos
from lib.baselines.temporal_only import _sample_discrete_demand, _solve_vrp
from scripts.experiments.baseline_comparison import zero_J

try:
    from python_tsp.exact import solve_tsp_dynamic_programming
    from python_tsp.heuristics import solve_tsp_local_search
    _PYTHON_TSP = True
except Exception:
    _PYTHON_TSP = False


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


def _road_polyline(rn, lonlat_seq):
    """Return a list of (lon, lat) waypoints along the road network connecting
    the input sequence of (lon, lat) points. Falls back to straight lines if
    the road network is unavailable or no path exists.
    """
    if rn is None or len(lonlat_seq) < 2:
        return list(lonlat_seq)
    try:
        import networkx as nx
    except Exception:
        return list(lonlat_seq)
    G = rn.G
    polyline = []
    for i in range(len(lonlat_seq) - 1):
        a = lonlat_seq[i]
        b = lonlat_seq[i + 1]
        try:
            n1 = rn._nearest_node(a[1], a[0])
            n2 = rn._nearest_node(b[1], b[0])
            nodes = nx.shortest_path(G, n1, n2, weight="length")
            seg = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in nodes]
        except Exception:
            seg = [a, b]
        if i == 0:
            polyline.extend(seg)
        else:
            polyline.extend(seg[1:])
    return polyline


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


def draw_partition(ax, g, design, rn=None):
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
            tour_pts = [depot_xy] + pts
            n = len(tour_pts)
            arr = np.array(tour_pts)
            dist = np.linalg.norm(arr[:, None, :] - arr[None, :, :], axis=2)
            order = list(range(n))
            if _PYTHON_TSP and n >= 3:
                try:
                    if n <= 12:
                        order, _ = solve_tsp_dynamic_programming(dist)
                    else:
                        order, _ = solve_tsp_local_search(dist)
                except Exception:
                    pass
            tour = [tour_pts[i] for i in order] + [tour_pts[order[0]]]
            poly = _road_polyline(rn, tour)
            xs, ys = zip(*poly)
            tour_color = DISTRICT_COLORS[k % len(DISTRICT_COLORS)]
            ax.plot(xs, ys, "-", color=tour_color, linewidth=0.9,
                    alpha=0.9, zorder=4.5)


def draw_fr(ax, g, design, rn=None):
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
        poly = _road_polyline(rn, pts)
        xs, ys = zip(*poly)
        col = line_colors[k % len(line_colors)]
        ax.plot(xs, ys, "-", color=col, linewidth=1.4, alpha=0.85, zorder=4.5)
        # Mark checkpoints separately so they're visible on top of the polyline
        cx, cy = zip(*pts)
        ax.plot(cx, cy, "o", color=col, markersize=2.5, zorder=5, alpha=0.95)


def draw_od(ax, g, design, color="#7f7f7f", rn=None):
    """OD: depot + sampled radial trips to a few destination blocks."""
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
            poly = _road_polyline(rn, [(d_lon, d_lat), (tx, ty)])
            xs, ys = zip(*poly)
            ax.plot(xs, ys, "-", color=color,
                    linewidth=0.6, alpha=0.4, zorder=4.5)
        ax.scatter([c[0] for c in coords], [c[1] for c in coords],
                   s=12, color=color, alpha=0.6, zorder=5)


def draw_vcc(ax, g, design, color="#e377c2", rn=None):
    """VCC: depot + per-block walk-tolerance balls + meeting-point candidates.

    The walk-radius circles show where each block's pickup region is
    (riders walk up to ``max_walk_km`` to a vehicle's coordinated meeting
    point), which is the structural difference from OD's pure
    door-to-door service.
    """
    block_ids = g.short_geoid_list
    coords = all_block_lonlat(g)
    bg = g.gdf.to_crs("EPSG:4326")
    bg.plot(ax=ax, facecolor="#f5f5f5", edgecolor="#aaaaaa",
            linewidth=0.4, zorder=2.4, alpha=0.5)
    meta = design.service_metadata or {}
    max_walk_km = float(meta.get("max_walk_km", 0.3))
    # Convert km radius to ~lon/lat degrees for the city's latitude
    if coords:
        avg_lat = np.mean([c[1] for c in coords])
        deg_per_km_lat = 1.0 / 111.0
        deg_per_km_lon = 1.0 / (111.0 * max(np.cos(np.deg2rad(avg_lat)), 0.1))
    else:
        deg_per_km_lat = deg_per_km_lon = 1.0 / 111.0
    for (lon, lat) in coords:
        ax.add_patch(mpatches.Ellipse(
            (lon, lat),
            width=2.0 * max_walk_km * deg_per_km_lon,
            height=2.0 * max_walk_km * deg_per_km_lat,
            facecolor=color, edgecolor=color,
            alpha=0.10, linewidth=0.4, zorder=3,
        ))
    ax.scatter([c[0] for c in coords], [c[1] for c in coords],
               s=10, color=color, alpha=0.7, zorder=5,
               edgecolors="white", linewidths=0.4)
    if design.depot_id and design.depot_id in block_ids:
        d_lon, d_lat = coords[block_ids.index(design.depot_id)]
        ax.plot(d_lon, d_lat, "*", color="black", markersize=10, zorder=6,
                markeredgecolor="white", markeredgewidth=0.8)


def draw_tp_lit(ax, g, design, prob, Lambda, wr, rn=None, n_waves=3):
    """TP-Lit: depot + realized VRP routes from multiple dispatch waves.

    TP-Lit's design persists only depot + fleet + dispatch interval; the
    actual routes are realized per scenario in evaluation. We sample
    ``n_waves`` independent dispatches (different RNG seeds) and plot
    each wave's vehicle routes color-coded by wave so the reader can
    see how route topology changes across realizations. Customer
    markers are colored by the wave that served them.
    """
    block_ids = g.short_geoid_list
    coords = all_block_lonlat(g)
    bg = g.gdf.to_crs("EPSG:4326")
    bg.plot(ax=ax, facecolor="#f5f5f5", edgecolor="#aaaaaa",
            linewidth=0.4, zorder=2.4, alpha=0.5)
    meta = design.service_metadata or {}
    fleet_size = int(meta.get("fleet_size", 3))
    capacity = int(meta.get("capacity", 15))
    T = float(meta.get("T_fixed", 0.5))
    pos_m = np.array([_get_pos(g, b) for b in block_ids])
    cand_km = pos_m / 1000.0
    probs = np.array([prob.get(b, 0.0) for b in block_ids], dtype=float)
    if probs.sum() > 1e-15:
        probs = probs / probs.sum()
    centroid = pos_m.mean(axis=0)
    depot_idx = int(np.argmin(np.linalg.norm(pos_m - centroid, axis=1)))
    depot_km = pos_m[depot_idx] / 1000.0
    depot_lon, depot_lat = coords[depot_idx]
    Lambda_T = Lambda * T

    wave_palettes = [
        ["#1f77b4", "#5aa0d8", "#9ec5ec"],   # blues  — wave 1
        ["#d62728", "#e87f7f", "#f4b9b9"],   # reds   — wave 2
        ["#2ca02c", "#7fc97f", "#bce0ad"],   # greens — wave 3
        ["#ff7f0e", "#fbb168", "#fcd9b1"],   # oranges (extra)
    ]

    drew_anything = False
    for w in range(n_waves):
        rng = np.random.default_rng(11 + 7 * w)
        active_pos, demands = _sample_discrete_demand(probs, cand_km, Lambda_T, rng)
        if len(active_pos) == 0:
            continue
        total_d = int(demands.sum())
        if total_d > fleet_size * capacity:
            order = np.argsort(-demands)
            cum = np.cumsum(demands[order])
            keep = int(np.searchsorted(cum, fleet_size * capacity, side="right")) + 1
            keep = min(keep, len(active_pos))
            sel = order[:keep]
            active_pos = active_pos[sel]
            demands = demands[sel]
        K_eff = min(fleet_size, len(active_pos))
        routes, _, _ = _solve_vrp(depot_km, active_pos, K_eff, capacity,
                                   demands=demands)
        active_lonlat = []
        for ap_km in active_pos:
            d2 = np.linalg.norm(cand_km - ap_km, axis=1)
            idx = int(np.argmin(d2))
            active_lonlat.append(coords[idx])
        palette = wave_palettes[w % len(wave_palettes)]
        wave_main = palette[0]
        for r_idx, route in enumerate(routes):
            if not route:
                continue
            col = palette[r_idx % len(palette)]
            seq = [(depot_lon, depot_lat)] + [active_lonlat[j] for j in route] \
                  + [(depot_lon, depot_lat)]
            poly = _road_polyline(rn, seq)
            xs, ys = zip(*poly)
            # Earlier waves drawn first, lower zorder; alpha decreases per wave
            ax.plot(xs, ys, "-", color=col, linewidth=1.1,
                    alpha=0.78 - 0.10 * w, zorder=4.5 - 0.1 * w)
        sizes = 5.0 + 3.0 * np.array(demands, dtype=float)
        ax.scatter([p[0] for p in active_lonlat], [p[1] for p in active_lonlat],
                   s=sizes, color=wave_main, alpha=0.55,
                   zorder=5 - 0.1 * w,
                   edgecolors="white", linewidths=0.3)
        drew_anything = True

    if not drew_anything:
        ax.text(0.5, 0.05, "(empty samples)", transform=ax.transAxes,
                fontsize=7, ha="center", color="#666")
    ax.plot(depot_lon, depot_lat, "*", color="black", markersize=10,
            zorder=6.5, markeredgecolor="white", markeredgewidth=0.8)


# Baseline label, factory, drawer
BASELINE_SPECS = [
    ("Joint-Nom",  lambda: JointNom(max_iters=20),                                  "partition"),
    ("Joint-DRO",  lambda: JointDRO(epsilon=0.5, max_iters=20),                     "partition"),
    ("SP-Lit",     lambda: CarlssonPartition(T_fixed=0.5),                          "partition"),
    ("Multi-FR",   lambda: FRDetour(delta=0.0, max_fleet=5, max_iters=3, n_saa=10,
                                     max_cg_iters=10,
                                     T_values=[0.25, 0.5, 1.0], name="Multi-FR"),    "fr"),
    ("TP-Lit",     lambda: TemporalOnly(n_candidates=0, T_fixed=0.5,
                                         n_eval_scenarios=10),                       "tp_lit"),
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

    # Build figure (panels are squarish; bigger when only one city is plotted)
    panel = 5.0 if nC == 1 else 3.6
    fig, axes = plt.subplots(nC, nB, figsize=(panel * nB, panel * nC))
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
                    draw_partition(ax, g, d, rn=rn)
                elif drawer_kind == "fr":
                    draw_fr(ax, g, d, rn=rn)
                elif drawer_kind == "vcc":
                    draw_vcc(ax, g, d, "#e377c2", rn=rn)
                elif drawer_kind == "tp_lit":
                    draw_tp_lit(ax, g, d, prob, args.Lambda, args.wr, rn=rn)
                elif drawer_kind == "od":
                    draw_od(ax, g, d, "#7f7f7f", rn=rn)
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
        f"Service designs across cities (Λ={args.Lambda:g}, wr={args.wr:g}, "
        f"K=3 fixed for visualization; cloud sweeps pick K per regime; "
        f"real BGs + road network)",
        fontsize=11, fontweight="bold", y=1.0,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
