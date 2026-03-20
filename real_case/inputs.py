"""
Real-case input loaders for the optimization experiment.

Provides:
- load_geodata(): GeoData for BG_GEOID_LIST
- load_probability_from_population(): probability dict from ACS population
- load_real_odd(): Omega_dict from compute_real_odd output CSV
- load_baseline_assignment(): route-based baseline assignment (nearest-stop)
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List

from lib.data import GeoData, RouteData
from config import DATA_PATHS, BG_GEOID_LIST


def load_geodata() -> GeoData:
    shp = DATA_PATHS['shapefile']
    return GeoData(shp, BG_GEOID_LIST, level='block_group')


def compute_vehicle_speed_kmh(geodata: GeoData, routes_json: str = 'data/hct_routes.json') -> float:
    """Estimate vehicle speed in km/h from HCT routes.

    For each route, decode the polyline, project to EPSG:2163 using RouteData's
    projector, compute total path length (kilometers) and divide by total
    scheduled time (hours) from SecondsToNextStop. Return the median km/h across routes.
    """
    try:
        rd = RouteData(routes_json, geodata)
        import polyline
        from shapely.geometry import LineString

        kmh_values = []
        for route in rd.routes_info:
            poly = route.get('EncodedPolyline')
            stops = route.get('Stops', [])
            total_secs = sum(s.get('SecondsToNextStop', 0) for s in stops)
            if not poly or total_secs <= 0:
                continue
            try:
                coords = polyline.decode(poly)  # list of (lat, lon)
                line = LineString([(lng, lat) for (lat, lng) in coords])
                proj_line = rd.project_geometry(line)  # EPSG:2163 meters
                km = float(proj_line.length) / 1000.0
                hours = float(total_secs) / 3600.0
                kmh = km / hours if hours > 0 else None
                if kmh and kmh > 0:
                    kmh_values.append(kmh)
            except Exception:
                continue
        if kmh_values:
            import statistics
            return float(statistics.median(kmh_values))
    except Exception:
        pass
    # Fallback if routes not available
    return 20.0  # km/h fallback


def set_geodata_speeds(
    geodata: GeoData,
    routes_json: str = 'data/hct_routes.json',
    walk_kmh: float = 0.0,
    vehicle_kmh: float = 18.710765208297367,
    compute_from_routes: bool = False,
) -> tuple[float, float]:
    """Set speeds on geodata in km/h and return (wv_kmh, wr_kmh).

    Defaults align with the notebook and common practice:
    - walk_kmh = 5.04 km/h (exactly 1.40 m/s)
    - vehicle_kmh = 18.710765208297367 km/h (median service speed used in analyses)

    If compute_from_routes=True, vehicle speed is recomputed from routes instead
    of using the constant default.
    """
    if compute_from_routes:
        vehicle_kmh = compute_vehicle_speed_kmh(geodata, routes_json)
    geodata.wv_kmh = vehicle_kmh
    geodata.wr_kmh = walk_kmh
    # Also store mph for any legacy code paths
    geodata.wv = vehicle_kmh / 1.60934
    geodata.wr = 0.0 if walk_kmh == 0 else walk_kmh / 1.60934
    return vehicle_kmh, walk_kmh


def _load_population_df() -> pd.DataFrame:
    # Prefer configured B01003 CSV; fallback to provided aggregate
    path = DATA_PATHS.get('population_data')
    if path and os.path.exists(path):
        df = pd.read_csv(path, dtype=str)
        geocol = next((c for c in df.columns if c.lower().startswith('geography')), 'Geography')
        popcol = next((c for c in df.columns if c.lower().startswith('estimate total')), None)
        if popcol is None:
            for c in df.columns:
                if c.lower().startswith('estimate') and 'total' in c.lower():
                    popcol = c
                    break
        out = pd.DataFrame({'GEOID': df[geocol], 'pop': pd.to_numeric(df[popcol], errors='coerce')})
    else:
        alt = os.path.join('data', '2023_target_blockgroup_population', '2023_target_blockgroup_population.csv')
        df = pd.read_csv(alt, dtype={'GEO_ID': str})
        out = pd.DataFrame({'GEOID': df['GEO_ID'], 'pop': pd.to_numeric(df['B02001_001E'], errors='coerce')})
    out['short_GEOID'] = out['GEOID'].str[-7:]
    return out.dropna(subset=['pop'])


def load_probability_from_population(geodata: GeoData) -> Dict[str, float]:
    df = _load_population_df()[['short_GEOID', 'pop']]
    df = df[df['short_GEOID'].isin(geodata.short_geoid_list)]
    if df['pop'].sum() <= 0:
        # fallback uniform
        n = len(geodata.short_geoid_list)
        return {sid: 1.0 / n for sid in geodata.short_geoid_list}
    df['p'] = df['pop'] / df['pop'].sum()
    return dict(zip(df['short_GEOID'], df['p']))


def load_probability_from_commuting(geodata: GeoData,
                                    tract_csv: str = 'data/2023_target_tract_commuting/2023_target_tract_commuting.csv',
                                    bg_csv: str = 'data/2023_target_blockgroup_population/2023_target_blockgroup_population.csv') -> Dict[str, float]:
    """Replicate the notebook's nominal mass based on ACS commuting counts.

    Steps (matching data_process.ipynb logic):
    - For each tract, compute tract_commuting = carpooled + public transportation + taxicab/motorcycle/other
      using S0801 estimate columns.
    - For each block group, compute its population share within the tract.
    - Demand at block group = tract_commuting * population_share.
    - Normalize over the selected BG set to produce a probability dict.
    """
    # Load tract-level commuting counts
    if not os.path.exists(tract_csv) or not os.path.exists(bg_csv):
        # Fallback to population-based if inputs are missing
        return load_probability_from_population(geodata)

    import pandas as pd
    tract = pd.read_csv(tract_csv, dtype=str)
    # Ensure numeric for needed estimates
    for c in ['S0801_C01_004E', 'S0801_C01_009E', 'S0801_C01_012E']:
        if c not in tract.columns:
            # Columns not present; fallback
            return load_probability_from_population(geodata)
        tract[c] = pd.to_numeric(tract[c], errors='coerce').fillna(0.0)
    # Extract 6-digit tract GEOID from GEO_ID
    # Example GEO_ID like '1400000US42003504100' -> short tract '504100'
    tract['short_tract_GEOID'] = tract['GEO_ID'].str[-6:]
    tract['tract_commuting'] = tract['S0801_C01_004E'] + tract['S0801_C01_009E'] + tract['S0801_C01_012E']
    tract_comm = tract[['short_tract_GEOID', 'tract_commuting']]

    # Load block-group population and compute share within tract
    bg = pd.read_csv(bg_csv, dtype={'GEO_ID': str, 'B02001_001E': float})
    bg['short_GEOID'] = bg['GEO_ID'].str[-7:]
    bg['short_tract_GEOID'] = bg['GEO_ID'].str[-7:-1]  # drop the last digit (block-group id)
    # tract population from BG sum
    tract_pop = (
        bg.groupby('short_tract_GEOID', as_index=False)['B02001_001E']
          .sum().rename(columns={'B02001_001E': 'tract_population'})
    )
    bg = bg.merge(tract_pop, on='short_tract_GEOID', how='left')
    bg['population_share'] = bg['B02001_001E'] / bg['tract_population']

    # Merge commuting counts onto BG, compute BG commuting demand
    bg = bg.merge(tract_comm, on='short_tract_GEOID', how='left')
    bg['bg_commuting'] = bg['population_share'] * bg['tract_commuting']

    # Filter to model's BGs
    bg = bg[bg['short_GEOID'].isin(geodata.short_geoid_list)]
    total = float(bg['bg_commuting'].sum())
    if total <= 0:
        return load_probability_from_population(geodata)
    bg['p'] = bg['bg_commuting'] / total
    return dict(zip(bg['short_GEOID'], bg['p']))


def load_real_odd(csv_path: str = 'data/odd_features.csv') -> Dict[str, np.ndarray]:
    """
    Load engineered ODD features and convert them into scalar penalties using Table 2/3 coefficients.
    Elevation is shifted down by 200 m before applying the coefficient.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"ODD features CSV not found: {csv_path}")

    # Force GEOID to string to avoid scientific-notation float parsing
    df = pd.read_csv(csv_path, dtype=str)
    # Normalize column names by stripping unit suffixes " [..]"
    rename_map = {}
    for col in df.columns:
        if '[' in col:
            rename_map[col] = col.split(' [', 1)[0]
    if rename_map:
        df = df.rename(columns=rename_map)

    geocol = next((c for c in df.columns if c.lower().startswith('geoid')), 'GEOID')
    df['GEOID'] = df[geocol].astype(str)
    df['short_GEOID'] = df['GEOID'].str.replace(r'\\.0$', '', regex=True).str[-7:]

    coeffs = {
        "curvature": 5.0,
        "grade": 3.0,
        "lanes": 2.0,
        "speed": 2.5,
        "is_bridge": 4.5,
        "rail_present": 2.0,
        "is_tunnel": 6.0,
        "mean_elevation": 1.0,
    }
    road_type_coeffs = {
        "busway": 1.0,
        "motorway": 1.5,
        "trunk": 3.0,
        "primary": 5.5,
        "secondary": 9.0,
        "tertiary": 13.5,
        "residential": 19.0,
        "unclassified": 25.0,
    }

    odd_dict: Dict[str, float] = {}
    for _, row in df.iterrows():
        cost = 0.0
        for feature, coeff in coeffs.items():
            val = row.get(feature)
            if pd.isna(val):
                continue
            if feature == "mean_elevation":
                val = float(val) - 200.0
            cost += coeff * float(val)

        road_type = str(row.get("road_type", "unclassified")).strip().lower()
        cost += road_type_coeffs.get(road_type, road_type_coeffs["unclassified"])

        odd_dict[str(row['short_GEOID'])] = float(cost)

    return {k: np.array([v], dtype=float) for k, v in odd_dict.items()}


def load_baseline_assignment(geodata: GeoData, routes_json: str = 'data/hct_routes.json', visualize: bool = False) -> Tuple[np.ndarray, List[str]]:
    rd = RouteData(routes_json, geodata)
    assignment, centers = rd.build_assignment_matrix(visualize=visualize)
    # Convert centers (which are short_GEOID values) to strings for consistency
    centers = [str(c) for c in centers]
    return assignment, centers
