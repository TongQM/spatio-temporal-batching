"""Fetch ACS-2023 population per block group for each city via Census API.

For each city's BG list, parses STATE+COUNTY+TRACT+BG codes from each GEOID
and queries the public Census Data API for B01003_001E (total population).
Saves a JSON {GEOID -> population} per city under data/city_bg_populations/.

Usage:
    python scripts/experiments/fetch_city_populations.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import urllib.request
import urllib.parse


CITIES = ["pittsburgh", "chicago", "manhattan", "la"]
LIST_DIR = "data/city_bg_lists"
OUT_DIR = "data/city_bg_populations"
ACS_VINTAGE = "2023"


def parse_geoid(g: str) -> Tuple[str, str, str, str]:
    """1500000US360610137002 → (state=36, county=061, tract=010137, bg=2)"""
    g = g.replace("1500000US", "")
    if len(g) != 12:
        raise ValueError(f"Bad GEOID length: {g!r} (got {len(g)})")
    return g[0:2], g[2:5], g[5:11], g[11:12]


def fetch_county_bg_pops(state: str, county: str, retries: int = 2) -> Dict[str, int]:
    """Fetch B01003_001E for every BG in a county. Returns {GEOID -> pop}."""
    url = (
        f"https://api.census.gov/data/{ACS_VINTAGE}/acs/acs5"
        f"?get=B01003_001E&for=block%20group:*"
        f"&in=state:{state}%20county:{county}%20tract:*"
    )
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                data = json.loads(r.read())
            break
        except Exception as e:
            if attempt == retries:
                raise
            time.sleep(2)
    # data[0] is header: ['B01003_001E', 'state', 'county', 'tract', 'block group']
    out = {}
    for row in data[1:]:
        pop, st, co, tr, bg = row[0], row[1], row[2], row[3], row[4]
        try:
            pop_i = int(pop) if pop is not None else 0
        except ValueError:
            pop_i = 0
        geoid = f"{st}{co}{tr}{bg}"
        out[geoid] = pop_i
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for city in CITIES:
        list_file = os.path.join(LIST_DIR, f"{city}.txt")
        if not os.path.isfile(list_file):
            print(f"[{city}] missing {list_file}")
            continue
        with open(list_file) as f:
            geoids = [l.strip() for l in f if l.strip()]
        # Group by (state, county); fetch each county once
        by_county: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for g in geoids:
            st, co, tr, bg = parse_geoid(g)
            by_county[(st, co)].append(g.replace("1500000US", ""))
        print(f"\n[{city}] {len(geoids)} BGs in {len(by_county)} counties")
        all_pops: Dict[str, int] = {}
        for (st, co), city_bgs in by_county.items():
            print(f"  Fetching state={st} county={co} ...")
            pops = fetch_county_bg_pops(st, co)
            print(f"    Got {len(pops)} BGs from API")
            for bg in city_bgs:
                all_pops[bg] = int(pops.get(bg, 0))
        print(f"  Coverage: {sum(1 for v in all_pops.values() if v > 0)}/{len(geoids)} non-zero")
        # Save with full 1500000US prefix to match our config
        out = {f"1500000US{bg}": pop for bg, pop in all_pops.items()}
        out_path = os.path.join(OUT_DIR, f"{city}.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        total = sum(out.values())
        print(f"  Total population: {total:,}")
        print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
