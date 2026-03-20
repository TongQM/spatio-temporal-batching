#!/usr/bin/env python3
"""
Visualize the original (nearest‑route) partition and denote the depot as the
stop where three routes intersect.

Outputs:
  figures/original_partition_with_depot.pdf

Notes:
  - The "original partition" assigns each BG to its nearest route.
  - The depot is inferred by clustering stops from all routes and picking a
    cluster that is shared by ≥ 3 distinct routes (within ~30 meters).
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from collections import defaultdict
from typing import Tuple, Dict, List

import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from real_case.inputs import load_geodata
from lib.data import RouteData
import polyline


def find_three_route_intersection(rd: RouteData, tol_m: float = 30.0) -> Tuple[float, float]:
    """Return projected (x,y) of a stop location shared by ≥3 routes.

    We cluster stops from all routes by rounding projected coordinates to a
    tolerance grid and count distinct routes per cluster; choose the cluster
    with the maximum distinct route count (ties broken by stop count).
    """
    # Collect all stops with projected coordinates
    buckets: Dict[Tuple[int, int], List[Tuple[str, float, float]]] = defaultdict(list)
    for route in rd.routes_info:
        rname = route.get('Description', f"Route {route.get('RouteID')}")
        for s in route.get('Stops', []):
            lon = s.get('Longitude'); lat = s.get('Latitude')
            if lon is None or lat is None:
                continue
            try:
                p = rd.project_geometry(Point(lon, lat))  # shapely Point in EPSG:2163 meters
                x, y = float(p.x), float(p.y)
                key = (int(round(x / tol_m)), int(round(y / tol_m)))
                buckets[key].append((rname, x, y))
            except Exception:
                continue

    best_key = None
    best_routes = set()
    for key, items in buckets.items():
        routes = {r for (r, _, _) in items}
        if len(routes) >= 3:
            if best_key is None:
                best_key, best_routes = key, routes
            else:
                # Prefer more routes, then more items
                if len(routes) > len(best_routes) or (
                    len(routes) == len(best_routes) and len(items) > len(buckets[best_key])
                ):
                    best_key, best_routes = key, routes

    if best_key is None:
        # Fallback: choose the densest cluster even if <3 routes
        best_key = max(buckets.keys(), key=lambda k: len(buckets[k]))

    items = buckets[best_key]
    xs = [x for (_, x, _) in items]
    ys = [y for (_, _, y) in items]
    return float(np.mean(xs)), float(np.mean(ys))


def assign_nearest_route(rd: RouteData, geodata) -> Dict[str, str]:
    """Return mapping short_GEOID -> route name by nearest stop."""
    nearest = rd.find_nearest_stops()
    return {idx: (nearest[idx]['route'] if idx in nearest else 'Unassigned')
            for idx in geodata.short_geoid_list}


def plot_original_partition():
    geodata = load_geodata()
    rd = RouteData('data/hct_routes.json', geodata)

    # 1) Assign BGs to nearest route
    route_by_bg = assign_nearest_route(rd, geodata)
    gdf = geodata.gdf.copy()
    gdf['district'] = gdf.index.map(route_by_bg.get)

    # 2) Find depot as 3‑route intersection
    depot_x, depot_y = find_three_route_intersection(rd, tol_m=30.0)

    # 3) Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8), constrained_layout=True)

    routes = sorted(gdf['district'].dropna().unique())
    cmap = plt.get_cmap('tab20', max(1, len(routes)))
    color_map = {r: cmap(i) for i, r in enumerate(routes)}

    # Plot polygons by route
    for r in routes:
        try:
            gdf[gdf['district'] == r].plot(ax=ax, color=color_map[r], edgecolor='white', linewidth=0.2)
        except Exception:
            continue

    # Overlay routes
    for route in rd.routes_info:
        color = route.get('MapLineColor', '#008080')
        poly = route.get('EncodedPolyline')
        if not poly:
            continue
        try:
            coords = polyline.decode(poly)  # (lat, lon)
            line = LineString([(lng, lat) for (lat, lng) in coords])
            proj = rd.project_geometry(line)
            ax.plot(*proj.xy, color=color, linewidth=1.2, alpha=0.9)
        except Exception:
            continue

    # Mark depot with a red square
    ax.plot(depot_x, depot_y, marker='s', markersize=8, color='red',
            markeredgecolor='white', markeredgewidth=0.8, zorder=10)

    # ax.set_title('Original Partition by Nearest Route with Depot (3‑Route Intersection)')
    # Build legend: depot + route/district colors
    depot_handle = Line2D([0], [0], marker='s', color='red', label='Depot (3‑route intersection)',
                          markerfacecolor='red', markeredgecolor='white', markeredgewidth=0.8,
                          linestyle='None', markersize=8)
    route_handles = [
        Patch(facecolor=color_map[r], edgecolor='white', label=r)
        for r in routes
    ]
    handles = [depot_handle] + route_handles
    ax.legend(
        handles=handles,
        loc='right',
        frameon=True,
        fontsize=12,
        ncol=1,
        # title='Nearest‑route Districts',
        # title_fontsize=12,
    )
    ax.set_axis_off()

    os.makedirs('results/figures', exist_ok=True)
    out_fig = os.path.join('results/figures', 'original_partition_with_depot.pdf')
    fig.savefig(out_fig, dpi=200)
    plt.close(fig)
    print(f"Saved: {out_fig}")


if __name__ == '__main__':
    plot_original_partition()
