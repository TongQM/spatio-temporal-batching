"""
Compute real ODD features (Ω) per block group for the Pittsburgh case study.

Omega definition (Immediate real-ODD):
  - omega1 (Exposure): normalized ACS block-group population (B01003).
  - omega2 (Service intensity): normalized mix of GTFS weekday trips/hour and
    WPRDC boardings+alightings aggregated to block groups.

Inputs
  - BG shapefile: lbbd_config.DATA_PATHS['shapefile'] (2023 block groups)
  - ACS BG population CSV: lbbd_config.DATA_PATHS['population_data']
  - WPRDC stop usage CSV: data/prt_bus_stop_usage_latest.csv (from hct_odd.py)
  - PRT GTFS ZIP: latest data/prt_gtfs_*.zip (from hct_odd.py)

Outputs
  - data/bg_odd_features_real.csv with columns:
      GEOID, short_GEOID, omega1, omega2,
      pop, trips_per_hour_bg, wprdc_activity_bg
  - figures/odd_features_real.png visualization for selected BGs

Notes on data sources are annotated on the figure.
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import re
import io
import glob
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import zipfile

from config import DATA_PATHS, BG_GEOID_LIST, SMALL_BG_GEOID_LIST


def _latest_gtfs_zip(data_dir: str = "data") -> str:
    cands = sorted(glob.glob(os.path.join(data_dir, "prt_gtfs_*.zip")))
    if not cands:
        raise FileNotFoundError("GTFS ZIP not found in data/. Run hct_odd.py first.")
    return cands[-1]


def load_block_groups() -> gpd.GeoDataFrame:
    shp = DATA_PATHS['shapefile']
    gdf = gpd.read_file(shp)
    gdf['short_GEOID'] = gdf['GEOID'].str[-7:]
    gdf = gdf.set_index('short_GEOID', drop=False)
    short_list = [g[-7:] for g in BG_GEOID_LIST]
    gdf = gdf.loc[short_list]
    if gdf.crs and gdf.crs.is_geographic:
        gdf = gdf.to_crs(epsg=4326)  # for joining with lon/lat points
    return gdf


def load_acs_population() -> pd.DataFrame:
    """Load BG-level population. Tries configured B01003 first; falls back to
    2023_target_blockgroup_population.csv (B02001_001E ≈ total pop)."""
    path = DATA_PATHS['population_data']
    if os.path.exists(path):
        df = pd.read_csv(path, dtype=str)
        geocol = next((c for c in df.columns if c.lower().startswith('geography')), 'Geography')
        popcol = None
        for c in df.columns:
            cl = c.lower().strip()
            if cl.startswith('estimate') and 'total' in cl:
                popcol = c
                break
        if popcol is None:
            raise ValueError("Could not find ACS population 'Estimate Total' column in configured CSV")
        out = pd.DataFrame({'GEOID': df[geocol], 'pop': pd.to_numeric(df[popcol], errors='coerce')})
        out['short_GEOID'] = out['GEOID'].str[-7:]
        return out.dropna(subset=['pop'])

    # Fallback to project-provided aggregation
    alt = os.path.join('data', '2023_target_blockgroup_population', '2023_target_blockgroup_population.csv')
    if os.path.exists(alt):
        df = pd.read_csv(alt, dtype={'GEO_ID':str})
        if 'B02001_001E' not in df.columns:
            raise ValueError("Fallback population CSV missing B02001_001E")
        out = pd.DataFrame({'GEOID': df['GEO_ID'], 'pop': pd.to_numeric(df['B02001_001E'], errors='coerce')})
        out['short_GEOID'] = out['GEOID'].str[-7:]
        return out.dropna(subset=['pop'])

    raise FileNotFoundError(f"Population CSV not found in either {path} or fallback {alt}")


def load_wprdc_stop_usage(path: str = os.path.join('data', 'prt_bus_stop_usage_latest.csv')) -> gpd.GeoDataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"WPRDC stop usage file not found: {path}")
    df = pd.read_csv(path)
    # Identify boarding/alighting columns
    cand_ons = [c for c in df.columns if 'on' in c.lower()]
    cand_offs = [c for c in df.columns if 'off' in c.lower()]
    # Prefer avg_ons/avg_offs if present
    ons_col = next((c for c in df.columns if c.lower() == 'avg_ons'), cand_ons[0] if cand_ons else None)
    offs_col = next((c for c in df.columns if c.lower() == 'avg_offs'), cand_offs[0] if cand_offs else None)
    if ons_col is None or offs_col is None:
        # Fallback: if no explicit on/off columns, create zeros
        df['activity'] = 0.0
    else:
        df['activity'] = pd.to_numeric(df[ons_col], errors='coerce').fillna(0) + \
                         pd.to_numeric(df[offs_col], errors='coerce').fillna(0)
    # Geometry
    lat_col = next((c for c in df.columns if 'lat' in c.lower()), None)
    lon_col = next((c for c in df.columns if 'lon' in c.lower()), None)
    if not lat_col or not lon_col:
        raise ValueError("Latitude/Longitude columns not found in WPRDC CSV")
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[lon_col], df[lat_col], crs='EPSG:4326'))
    # drop rows without geometry
    gdf = gdf.dropna(subset=['geometry'])
    return gdf[['activity', 'geometry']]


def gtfs_trips_per_hour_by_stop(gtfs_zip: str) -> pd.DataFrame:
    """Compute weekday trips/hour per stop from GTFS inside a ZIP."""
    with zipfile.ZipFile(gtfs_zip, 'r') as zf:
        # Resolve paths possibly inside a subfolder
        def find(name):
            for n in zf.namelist():
                if n.lower().endswith('/' + name) or n.lower() == name:
                    return n
            raise FileNotFoundError(name)
        cal_p = find('calendar.txt')
        trips_p = find('trips.txt')
        stops_p = find('stops.txt')

        def read_csv(path, usecols=None, dtype=None):
            with zf.open(path) as f:
                return pd.read_csv(f, usecols=usecols, dtype=dtype)

        # stop_times may not be present in the published feed; handle gracefully
        try:
            st_p = find('stop_times.txt')
        except FileNotFoundError:
            # Return stops with zero trips_per_hour; caller can fallback to stop counts
            stops = read_csv(stops_p, usecols=['stop_id','stop_lat','stop_lon'], dtype={'stop_id':str})
            stops['stop_id'] = stops['stop_id'].astype(str)
            stops['trips_per_hour'] = 0.0
            return stops

        cal = read_csv(cal_p, usecols=['service_id','monday','tuesday','wednesday','thursday','friday'])
        # Use any weekday-active service
        wd = cal[(cal[['monday','tuesday','wednesday','thursday','friday']].sum(axis=1) > 0)]
        svc = set(wd['service_id'].astype(str))

        trips = read_csv(trips_p, usecols=['route_id','service_id','trip_id'], dtype=str)
        trips = trips[trips['service_id'].astype(str).isin(svc)]
        trip_ids = set(trips['trip_id'].astype(str))

        stops = read_csv(stops_p, usecols=['stop_id','stop_lat','stop_lon'], dtype={'stop_id':str})
        stops['stop_id'] = stops['stop_id'].astype(str)

        # stop_times may be large; iterate in chunks using BytesIO
        with zf.open(st_p) as f:
            st_df = pd.read_csv(f, usecols=['trip_id','stop_id','arrival_time'], dtype={'trip_id':str,'stop_id':str})
        st_df = st_df[st_df['trip_id'].isin(trip_ids)]
        # Count trips per stop by hour (arrival_time HH:MM:SS, may exceed 24: wrap to 0-23)
        def hh_to_int(s: str) -> int:
            try:
                h = int(str(s).split(':',1)[0])
                return h % 24
            except Exception:
                return 0
        st_df['hour'] = st_df['arrival_time'].map(hh_to_int)
        cnt = st_df.groupby(['stop_id','hour']).size().reset_index(name='cnt')
        tph = cnt.groupby('stop_id')['cnt'].mean().rename('trips_per_hour').reset_index()
        tph['stop_id'] = tph['stop_id'].astype(str)

        # Merge TPH onto stops
        stops_tph = stops.merge(tph, on='stop_id', how='left').fillna({'trips_per_hour':0.0})
    return stops_tph


def gtfs_stops_df(gtfs_zip: str) -> pd.DataFrame:
    """Return stops with lon/lat from GTFS ZIP."""
    with zipfile.ZipFile(gtfs_zip, 'r') as zf:
        def find(name):
            for n in zf.namelist():
                if n.lower().endswith('/' + name) or n.lower() == name:
                    return n
            raise FileNotFoundError(name)
        stops_p = find('stops.txt')
        with zf.open(stops_p) as f:
            stops = pd.read_csv(f, usecols=['stop_id','stop_lat','stop_lon'], dtype={'stop_id':str})
        return stops


def minmax01(x: pd.Series) -> pd.Series:
    mn, mx = x.min(), x.max()
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - mn) / (mx - mn)


def compute_real_odd() -> Tuple[pd.DataFrame, gpd.GeoDataFrame]:
    # Load BGs
    bg = load_block_groups()

    # ACS population
    pop = load_acs_population()[['short_GEOID','pop']]
    pop = pop.groupby('short_GEOID', as_index=False)['pop'].sum()

    # WPRDC stop usage → BG aggregate
    wprdc = load_wprdc_stop_usage()
    # Spatial join (EPSG:4326 across both)
    joined = gpd.sjoin(wprdc, bg[['geometry']], how='inner', predicate='within')
    w_activity = joined.groupby('short_GEOID')['activity'].sum().rename('wprdc_activity_bg')
    w_activity = w_activity.to_frame().reset_index()

    # GTFS trips/hour per stop → BG aggregate
    gtfs_zip = _latest_gtfs_zip()
    stops_tph = gtfs_trips_per_hour_by_stop(gtfs_zip)
    stops_gdf = gpd.GeoDataFrame(
        stops_tph,
        geometry=gpd.points_from_xy(stops_tph['stop_lon'], stops_tph['stop_lat'], crs='EPSG:4326')
    )
    sj = gpd.sjoin(stops_gdf, bg[['geometry']], how='inner', predicate='within')
    tph_bg = sj.groupby('short_GEOID')['trips_per_hour'].sum().rename('trips_per_hour_bg').reset_index()
    # Also compute stop counts for fallback when stop_times.txt absent
    stops_only = gtfs_stops_df(gtfs_zip)
    stops_only_g = gpd.GeoDataFrame(
        stops_only,
        geometry=gpd.points_from_xy(stops_only['stop_lon'], stops_only['stop_lat'], crs='EPSG:4326')
    )
    sj2 = gpd.sjoin(stops_only_g, bg[['geometry']], how='inner', predicate='within')
    stop_counts = sj2.groupby('short_GEOID').size().rename('stops_per_bg').reset_index()

    # Assemble features
    df = pd.DataFrame({'short_GEOID': bg.index})
    df = df.merge(pop, on='short_GEOID', how='left')
    df = df.merge(tph_bg, on='short_GEOID', how='left')
    df = df.merge(stop_counts, on='short_GEOID', how='left')
    df = df.merge(w_activity, on='short_GEOID', how='left')
    df[['pop','trips_per_hour_bg','wprdc_activity_bg','stops_per_bg']] = df[['pop','trips_per_hour_bg','wprdc_activity_bg','stops_per_bg']].fillna(0.0)

    # Normalize
    df['omega1'] = minmax01(df['pop']) * 5.0
    # Combine service components. If trips/hour is all zeros (no stop_times), use stops_per_bg instead.
    if (df['trips_per_hour_bg'] > 0).any():
        service_idx = minmax01(df['trips_per_hour_bg']) * 0.5 + minmax01(df['wprdc_activity_bg']) * 0.5
    else:
        service_idx = minmax01(df['stops_per_bg']) * 0.5 + minmax01(df['wprdc_activity_bg']) * 0.5
    df['omega2'] = service_idx * 5.0

    # Attach full GEOID for output
    df = df.merge(bg[['short_GEOID','GEOID']].reset_index(drop=True), on='short_GEOID', how='left')
    return df, bg


def save_outputs(df: pd.DataFrame) -> str:
    out_dir = 'data'
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, 'bg_odd_features_real.csv')
    cols = ['GEOID','short_GEOID','omega1','omega2','pop','trips_per_hour_bg','stops_per_bg','wprdc_activity_bg']
    df[cols].to_csv(out, index=False)
    return out


def visualize(df: pd.DataFrame, bg: gpd.GeoDataFrame) -> str:
    fig_dir = 'results/figures'
    os.makedirs(fig_dir, exist_ok=True)
    out_fig = os.path.join(fig_dir, 'odd_features_real.pdf')

    gplot = bg.copy().reset_index(drop=True)
    gplot = gplot.merge(df[['short_GEOID','omega1','omega2']], on='short_GEOID', how='left')

    # Plot two panels
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    # Use more muted, low-saturation palettes
    gplot.plot(column='omega1', cmap='Greys', legend=True, ax=axes[0], edgecolor='#888888', linewidth=0.2, vmin=0, vmax=5)
    axes[0].set_title('Omega1: Exposure (ACS population)')
    axes[0].axis('off')

    gplot.plot(column='omega2', cmap='GnBu', legend=True, ax=axes[1], edgecolor='#888888', linewidth=0.2, vmin=0, vmax=5)
    axes[1].set_title('Omega2: Service intensity (GTFS + WPRDC)')
    axes[1].axis('off')

    src_note = (
        'Data sources: ACS 5-year B01003 (BG pop),\n'
        'PRT GTFS (weekday stop_times), WPRDC stop usage.\n'
        'BGs: lbbd_config.BG_GEOID_LIST'
    )
    fig.text(0.01, 0.01, src_note, fontsize=8)
    fig.suptitle('Real ODD Features per Block Group (Ω)', fontsize=12)
    fig.savefig(out_fig, dpi=200)
    plt.close(fig)
    return out_fig


def main():
    df, bg = compute_real_odd()
    out_csv = save_outputs(df)
    out_fig = visualize(df, bg)
    # Print subset preview
    small_short = {g[-7:] for g in SMALL_BG_GEOID_LIST}
    print(f"Saved: {out_csv}\nFigure: {out_fig}\n\nSubset (SMALL_BG_GEOID_LIST):")
    for sid in sorted(small_short):
        r = df[df['short_GEOID'] == sid]
        if not r.empty:
            o1 = float(r['omega1'].iloc[0])
            o2 = float(r['omega2'].iloc[0])
            print(f"  {sid}: omega=[{o1:.3f}, {o2:.3f}]")


if __name__ == '__main__':
    main()
