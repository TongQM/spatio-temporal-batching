#!/usr/bin/env bash
# Full per-city regime sweep with population-weighted demand.
# Runs all 7 baselines (fast 5 + Multi-FR + reduced-grid TP-Lit).
# Resumable: per-setting JSON cache means restarts don't lose progress.
#
# Compatible with macOS bash 3.x (no associative arrays).

set -euo pipefail

source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh
conda activate optimization

ALL_SERVICES="joint_nom joint_dro sp_lit od vcc multi_fr tp_lit"

run_city() {
  local CITY="$1"
  local SHAPEFILE="$2"
  local SUBSET_SIZE="${3:-25}"
  echo "========================================"
  echo "  CITY: $CITY  (subset_size=$SUBSET_SIZE, population-weighted demand)"
  echo "========================================"
  local OUT="results/baseline_clouds/cities_road/${CITY}"
  python scripts/experiments/baseline_cloud_sweep.py \
    --view regime \
    --mode real \
    --road_network \
    --shapefile "$SHAPEFILE" \
    --geoid_list "data/city_bg_lists/${CITY}.txt" \
    --real_subset_size "$SUBSET_SIZE" \
    --real_use_population_prob \
    --population_json "data/city_bg_populations/${CITY}.json" \
    --regime_lambdas 50 150 300 \
    --regime_wrs 1 3 10 \
    --epsilon 0.5 \
    --fleet_cost_rate 4.0 \
    --services $ALL_SERVICES \
    --jobs 1 \
    --resume \
    --out_dir "$OUT" 2>&1 | tail -10
  echo ""
}

# Manhattan uses 50 BGs (~2.4 km^2): the original 25-BG Midtown footprint
# (~1 km^2) was too small for FR-Detour candidate-line library to produce
# feasible trips. The other cities stay at 25 BGs.
run_city pittsburgh "data/2023_shape_files/census_shape_block_group/tl_2023_42_bg.shp"  25
run_city chicago    "data/tiger_2023/tl_2023_17_bg/tl_2023_17_bg.shp"                    25
run_city manhattan  "data/tiger_2023/tl_2023_36_bg/tl_2023_36_bg.shp"                    50
run_city la         "data/tiger_2023/tl_2023_06_bg/tl_2023_06_bg.shp"                    25

echo "Done. Results in results/baseline_clouds/cities_pop/"
