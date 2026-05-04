#!/usr/bin/env bash
# Run regime sweep for fast baselines on each city.
# Slow baselines (TP-Lit, Multi-FR) excluded; can be added later.
# (Avoids associative arrays for compatibility with macOS bash 3.x)

set -euo pipefail

source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh
conda activate optimization

FAST_SERVICES="joint_nom joint_dro sp_lit od vcc"

run_city() {
  local CITY="$1"
  local SHAPEFILE="$2"
  echo "========================================"
  echo "  CITY: $CITY"
  echo "========================================"
  local OUT="results/baseline_clouds/cities/${CITY}"
  python scripts/experiments/baseline_cloud_sweep.py \
    --view regime \
    --mode real \
    --road_network \
    --shapefile "$SHAPEFILE" \
    --geoid_list "data/city_bg_lists/${CITY}.txt" \
    --real_subset_size 25 \
    --regime_lambdas 50 150 300 \
    --regime_wrs 1 3 10 \
    --epsilon 0.5 \
    --fleet_cost_rate 4.0 \
    --services $FAST_SERVICES \
    --jobs 1 \
    --resume \
    --out_dir "$OUT" 2>&1 | tail -15
  echo ""
}

run_city pittsburgh "data/2023_shape_files/census_shape_block_group/tl_2023_42_bg.shp"
run_city chicago    "data/tiger_2023/tl_2023_17_bg/tl_2023_17_bg.shp"
run_city manhattan  "data/tiger_2023/tl_2023_36_bg/tl_2023_36_bg.shp"
run_city la         "data/tiger_2023/tl_2023_06_bg/tl_2023_06_bg.shp"

echo "Done. Results in results/baseline_clouds/cities/"
