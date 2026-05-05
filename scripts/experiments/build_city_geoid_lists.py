"""Build per-city block-group geoid lists from TIGER 2023 shapefiles.

For each city, filters the state-level BG shapefile to BGs whose centroid
falls inside a focal bbox, then keeps the N centermost BGs (by distance
from the bbox centroid). Saves the resulting GEOID list to a text file
for use with `--geoid_list` in the regime sweep.

Output: data/city_bg_lists/{pittsburgh,chicago,manhattan,la}.txt
"""
from __future__ import annotations

import argparse
import os

import geopandas as gpd
import numpy as np


# bbox = (north, south, east, west) — same convention as the rest of the repo
CITY_AREAS = {
    "pittsburgh": dict(
        # Use the existing config.BG_GEOID_LIST as the truth for Pittsburgh.
        existing_config=True,
        shapefile="data/2023_shape_files/census_shape_block_group/tl_2023_42_bg.shp",
    ),
    "chicago": dict(
        bbox=(41.91, 41.86, -87.61, -87.66),  # Loop / River North
        shapefile="data/tiger_2023/tl_2023_17_bg/tl_2023_17_bg.shp",
    ),
    "manhattan": dict(
        # Wider Midtown + bordering blocks; original 1.2x1.6 km Midtown-only
        # bbox was too small for FR-Detour candidate-line library to produce
        # feasible trips. ~3x larger bounding box now covers ~3-4 km^2 with 50 BGs.
        bbox=(40.795, 40.735, -73.94, -74.01),
        shapefile="data/tiger_2023/tl_2023_36_bg/tl_2023_36_bg.shp",
    ),
    "la": dict(
        bbox=(34.075, 34.025, -118.22, -118.28),  # DTLA
        shapefile="data/tiger_2023/tl_2023_06_bg/tl_2023_06_bg.shp",
    ),
}


def _format_geoid(g):
    """TIGER GEOID is 12-char string for BG; the Pittsburgh config uses
    the prefix '1500000US' so we keep that style for consistency."""
    return f"1500000US{g}"


def build_city_list(city: str, n_subset: int = 25) -> list:
    cfg = CITY_AREAS[city]
    shp = cfg["shapefile"]
    if not os.path.isfile(shp):
        raise FileNotFoundError(f"Shapefile not found: {shp}")

    if cfg.get("existing_config"):
        from config import BG_GEOID_LIST
        # Keep only the n_subset centermost (matches Pittsburgh real-data sweep).
        gdf = gpd.read_file(shp)
        gdf["geoid_full"] = "1500000US" + gdf["GEOID"].astype(str)
        keep = gdf[gdf["geoid_full"].isin(BG_GEOID_LIST)].copy()
        if keep.empty:
            raise RuntimeError("No matching GEOIDs from BG_GEOID_LIST")
        # Pick centermost n_subset
        keep = keep.to_crs("EPSG:2163")
        keep["x"] = keep.geometry.centroid.x
        keep["y"] = keep.geometry.centroid.y
        cx, cy = keep["x"].mean(), keep["y"].mean()
        keep["d"] = np.hypot(keep["x"] - cx, keep["y"] - cy)
        keep = keep.sort_values("d").head(n_subset)
        return keep["geoid_full"].tolist()

    # bbox-based selection
    north, south, east, west = cfg["bbox"]
    print(f"  {city}: loading shapefile {shp}")
    gdf = gpd.read_file(shp)
    print(f"  {city}: {len(gdf)} BGs in state, filtering to bbox...")
    bg_4326 = gdf.to_crs("EPSG:4326")
    cents = bg_4326.geometry.centroid
    inside = (
        (cents.y >= south) & (cents.y <= north) &
        (cents.x >= west) & (cents.x <= east)
    )
    sel = gdf[inside].copy()
    print(f"  {city}: {len(sel)} BGs in bbox; selecting {n_subset} centermost")
    sel_4326 = sel.to_crs("EPSG:4326")
    sel_lat = sel_4326.geometry.centroid.y
    sel_lon = sel_4326.geometry.centroid.x
    cx = 0.5 * (north + south)
    cy = 0.5 * (east + west)
    d = np.hypot(sel_lat - cx, sel_lon - cy)
    sel = sel.assign(_d=d.values).sort_values("_d").head(n_subset)
    return [_format_geoid(g) for g in sel["GEOID"].astype(str).tolist()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_subset", type=int, default=25)
    p.add_argument("--out_dir", type=str, default="data/city_bg_lists")
    p.add_argument("--cities", nargs="+", default=None,
                   help="Restrict to a subset of cities (default: all)")
    p.add_argument("--n_subset_per_city", nargs="*", default=None,
                   help="Per-city overrides as 'city:n', e.g. manhattan:50")
    args = p.parse_args()

    overrides = {}
    if args.n_subset_per_city:
        for kv in args.n_subset_per_city:
            k, v = kv.split(":")
            overrides[k.strip()] = int(v)

    os.makedirs(args.out_dir, exist_ok=True)
    cities = args.cities or list(CITY_AREAS)
    for city in cities:
        n = overrides.get(city, args.n_subset)
        print(f"\n[{city}] n_subset={n}")
        try:
            geoids = build_city_list(city, n_subset=n)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        out = os.path.join(args.out_dir, f"{city}.txt")
        with open(out, "w") as f:
            f.write("\n".join(geoids) + "\n")
        print(f"  Wrote {len(geoids)} geoids → {out}")
        print(f"  First 3: {geoids[:3]}")


if __name__ == "__main__":
    main()
