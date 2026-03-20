import os
import re
import gzip
import io
import json
import csv
from datetime import datetime
from typing import Optional, Dict, Any, List

import requests
import pandas as pd
from tqdm import tqdm

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ---------- Helpers ----------
def download_file(url: str, out_path: str, chunk=1024 * 256) -> None:
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(out_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=os.path.basename(out_path)) as pbar:
        for b in r.iter_content(chunk_size=chunk):
            if b:
                f.write(b)
                pbar.update(len(b))

def try_get_json(url: str) -> Dict[str, Any]:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()

def head_print(df: pd.DataFrame, name: str, n: int = 5):
    print(f"\n=== {name}: shape={df.shape} columns={list(df.columns)[:10]}... ===")
    print(df.head(n))

# ---------- 1) WPRDC: PRT Bus Stop Usage (boardings/alightings) ----------
def fetch_wprdc_stop_usage() -> str:
    """
    Uses CKAN API to find the latest CSV resource under dataset 'prt-transit-stop-usage'
    and downloads it.
    """
    pkg_url = "https://data.wprdc.org/api/3/action/package_show?id=prt-transit-stop-usage"
    pkg = try_get_json(pkg_url)
    assert pkg.get("success"), "WPRDC CKAN returned success=false"
    resources = pkg["result"]["resources"]

    # Prefer CSV resources; if there are multiple months, pick the most recently modified one
    csv_resources = [r for r in resources if r.get("format", "").upper() == "CSV" and r.get("url")]
    if not csv_resources:
        raise RuntimeError("No CSV resources found for PRT stop usage on WPRDC.")

    def parse_dt(s: Optional[str]) -> datetime:
        if not s:
            return datetime.min
        try:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        except Exception:
            return datetime.min

    latest = max(csv_resources, key=lambda r: parse_dt(r.get("last_modified") or r.get("created")))
    url = latest["url"]
    out = os.path.join(DATA_DIR, "prt_bus_stop_usage_latest.csv")
    download_file(url, out)
    return out

# ---------- 2) PRT GTFS static feed ----------
def fetch_prt_gtfs() -> str:
    """
    Downloads the latest public GTFS ZIP provided on PRT's Open GIS Hub.
    We try current (2506) then fall back to (2502). If ArcGIS item changes in the future,
    update the ITEM_IDS list below.
    """
    # Known ArcGIS item ids for 2025 schedule periods:
    ITEM_IDS = [
        "a05bc2bd3afc4549b4f384f8737e045f",  # 2506 (effective Jun 2025)
        "592b63c9258143539d55102d1c518513",  # 2502 (effective Feb 2025)
    ]
    for item in ITEM_IDS:
        # ArcGIS "data" endpoint returns the actual zip bytes when the item stores a file
        url = f"https://www.arcgis.com/sharing/rest/content/items/{item}/data"
        out = os.path.join(DATA_DIR, f"prt_gtfs_{item}.zip")
        try:
            r = requests.get(url, stream=True, timeout=60)
            r.raise_for_status()
            # crude check: ensure it looks like a ZIP by the first two bytes "PK"
            start = r.raw.read(2)
            if start != b"PK":
                # not directly streaming? try with download fallback
                r2 = requests.get(url + "?download", stream=True, timeout=60)
                r2.raise_for_status()
                with open(out, "wb") as f:
                    for b in r2.iter_content(1024 * 256):
                        f.write(b)
            else:
                with open(out, "wb") as f:
                    f.write(start)
                    for b in r.iter_content(1024 * 256):
                        f.write(b)
            print(f"Downloaded PRT GTFS from item {item} → {out}")
            return out
        except Exception as e:
            print(f"GTFS download attempt for item {item} failed: {e}")
    raise RuntimeError("Unable to download PRT GTFS. Update ITEM_IDS if the feed has rolled over.")

# ---------- 3) LEHD LODES OD flows (PA) ----------
def lodes_pa_od_urls(year=2021, seg="JT00") -> List[str]:
    """
    Return a list of candidate LODES OD URLs (newest to oldest hosting) for PA.
    As of recent years, LODES8 hosts 2020+, while LODES7 hosts earlier years.
    We'll probe these in order and use the first that exists (HTTP 200).
    """
    base_patterns = [
        # Newer host
        "https://lehd.ces.census.gov/data/lodes/LODES8/pa/od/pa_od_main_{seg}_{year}.csv.gz",
        # Historical host
        "https://lehd.ces.census.gov/data/lodes/LODES7/pa/od/pa_od_main_{seg}_{year}.csv.gz",
    ]
    return [p.format(seg=seg, year=year) for p in base_patterns]

def pick_first_ok_url(urls: List[str]) -> Optional[str]:
    for u in urls:
        try:
            r = requests.head(u, timeout=30)
            if r.status_code == 200:
                return u
        except Exception:
            pass
    return None

def fetch_lodes_pa(year=2021, seg="JT00") -> str:
    candidates = lodes_pa_od_urls(year, seg)
    url = pick_first_ok_url(candidates)
    if not url:
        # try simple fallbacks across nearby years (newest first)
        for y in [year, year - 1, year - 2, year - 3]:
            url = pick_first_ok_url(lodes_pa_od_urls(y, seg))
            if url:
                year = y
                break
    if not url:
        raise RuntimeError("Unable to locate a valid LODES OD file for PA across known hosts/years.")
    out = os.path.join(DATA_DIR, f"pa_od_main_{seg}_{year}.csv.gz")
    download_file(url, out)
    return out

def read_lodes_sample(path_gz: str, n_rows=10) -> pd.DataFrame:
    with gzip.open(path_gz, "rb") as f:
        return pd.read_csv(f, nrows=n_rows)

# ---------- 4) Census API: ACS tract population (Allegheny County, PA) ----------
def fetch_acs_population_tracts_pa_allegheny(year=2023) -> pd.DataFrame:
    """
    B01003_001E = total population. geography: tract in county 003 (Allegheny), state 42 (PA)
    API doc: https://api.census.gov/data.html
    """
    base = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        "get": "NAME,B01003_001E",
        "for": "tract:*",
        "in": "state:42 county:003"
    }
    r = requests.get(base, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    headers = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=headers)
    # cast population to numeric
    df["B01003_001E"] = pd.to_numeric(df["B01003_001E"], errors="coerce")
    return df

# ---------- Run everything + quick tests ----------
def main():
    print("Fetching WPRDC stop usage…")
    stop_csv = fetch_wprdc_stop_usage()
    stop_df = pd.read_csv(stop_csv)
    head_print(stop_df, "WPRDC Stop Usage")
    # minimal checks
    expected_cols = {"stop_id", "boardings", "alightings"}
    if expected_cols.intersection(set(map(str.lower, stop_df.columns.str.lower()))) == set():
        print("Note: Column names differ by month in WPRDC; inspect the CSV to map fields.")

    print("\nFetching PRT GTFS…")
    gtfs_zip = fetch_prt_gtfs()
    # show contained files quickly using Python's zipfile without full extraction
    import zipfile
    with zipfile.ZipFile(gtfs_zip, "r") as zf:
        names = zf.namelist()
        print(f"GTFS entries ({len(names)} files):", [n for n in names if n.endswith(".txt")][:10])
        # simple schema check for core GTFS tables (allow for subdirectories like GTFS_2506/)
        core = ["stops.txt", "routes.txt", "trips.txt", "stop_times.txt"]
        missing = []
        for c in core:
            if not any(n.endswith("/" + c) or n == c for n in names):
                missing.append(c)
        if missing:
            print("Warning: GTFS appears to be missing:", missing)

    print("\nFetching LODES OD flows for PA (JT00 all jobs, 2021)…")
    lodes_gz = fetch_lodes_pa(year=2021, seg="JT00")
    lodes_head = read_lodes_sample(lodes_gz, n_rows=5)
    head_print(lodes_head, "LODES OD (sample)")
    # sanity: typical columns include 'h_geocode','w_geocode','S000', etc.
    lodes_expected = {"h_geocode", "w_geocode", "S000"}
    if not lodes_expected.issubset(set(lodes_head.columns)):
        print("Note: LODES columns not fully matched; check version docs for field names.")

    print("\nFetching ACS tract population for Allegheny County, PA (ACS 5-year)…")
    acs_df = fetch_acs_population_tracts_pa_allegheny(year=2023)
    head_print(acs_df, "ACS tract population (Allegheny)")
    if acs_df["B01003_001E"].isna().all():
        print("Warning: ACS population parsed as NaN; verify variables/year combination.")

    # Quick joins example (optional): summarize WPRDC by stop id if present
    print("\nQuick checks complete. Files saved in:", os.path.abspath(DATA_DIR))

if __name__ == "__main__":
    main()
