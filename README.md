# Spatio-Temporal Batching: Transit Service District Partitioning

This project implements optimization algorithms for partitioning a geographic area into contiguous transit service districts, with simultaneous depot location optimization under demand uncertainty. The primary application is the Pittsburgh High Capacity Transit (HCT) network.

## Problem

Given a geographic service area, partition it into *K* contiguous districts and select a depot location to minimize total operational cost:

- **Linehaul cost**: depot to each district
- **ODD (On-Demand Delivery) cost**: district-level service intensity
- **Travel cost**: vehicle routing within districts
- **User cost**: passenger wait time and travel time

Demand is modeled as uncertain, handled via **Distributionally Robust Optimization (DRO)** with Wasserstein ambiguity sets.

## Repository Structure

```
├── lib/                        # Core algorithm library
│   ├── algorithm.py            # LBBD, random search, local search
│   ├── data.py                 # Geographic data (GeoData, RouteData)
│   ├── evaluate.py             # Cost evaluation
│   └── constants.py            # Shared constants
│
├── scripts/
│   ├── experiments/            # Main experiment runners
│   │   ├── run_lbbd.py         # CLI for LBBD experiments
│   │   ├── lbbd_config.py      # LBBD run profiles and settings
│   │   ├── design_perf.py      # DRO vs nominal design performance
│   │   ├── simulate_service_designs.py   # DRO vs non-DRO simulation
│   │   └── pareto_frontier_analysis.py  # Provider/user cost trade-offs
│   ├── analysis/               # Theoretical validation
│   │   ├── validate_sample_cutoff.py    # Sample size cutoff validation
│   │   ├── test_radius_effect.py        # Wasserstein radius sensitivity
│   │   ├── verify_fleet_sizing.py       # Fleet sizing propositions
│   │   ├── verify_proposition4.py       # Monotonicity analysis
│   │   └── compare_appearance_effect.py # Appearance probability comparison
│   ├── visualization/          # Figure generation
│   │   ├── visualize_rs_solution.py
│   │   ├── visualize_original_partition.py
│   │   └── visualize_worst_case_distribution.py
│   └── data_processing/        # Data pipeline
│       ├── hct_odd.py          # Download PRT GTFS and WPRDC stop usage
│       ├── compute_real_odd.py # Compute ODD features per block group
│       └── check_available_blocks.py
│
├── tests/
│   ├── test_lbbd.py            # LBBD algorithm test (4×4 grid)
│   └── test_rs_heuristic.py    # Random search heuristic test
│
├── real_case/                  # Pittsburgh HCT case study
│   ├── experiment.py           # Full optimization vs baseline comparison
│   └── inputs.py               # Data loaders for real-world inputs
│
├── notebooks/                  # Jupyter notebooks
│   ├── SDP2025.ipynb           # Main research notebook
│   ├── data_process.ipynb      # Data processing pipeline
│   ├── BHH_estimation.ipynb    # Wait time valuation (β = 0.7120)
│   └── ODD_features_retrieval.ipynb
│
├── data/                       # Raw input data (not tracked by git)
│   ├── 2023_shape_files/       # Census block group shapefiles (PA)
│   ├── 2023_target_blockgroup_population/
│   ├── 2023_target_tract_commuting/
│   ├── hct_routes.json         # Pittsburgh HCT route definitions
│   └── odd_features.csv        # Precomputed ODD features
│
├── results/                    # Outputs (not tracked by git)
│   ├── figures/
│   └── lbbd/
│
├── docs/                       # Extended documentation
├── config.py                   # Shared data paths and block group GEOIDs
└── cache/                      # OSM/distance cache (auto-generated)
```

## Setup

### Environment

```bash
conda activate optimization
```

Required packages: `numpy`, `scipy`, `matplotlib`, `gurobipy`, `networkx`, `geopandas`, `pandas`, `shapely`, `pyproj`, `libpysal`, `simpy`, `polyline`

> Gurobi requires a valid license (free academic license available at [gurobi.com](https://www.gurobi.com/academia/academic-program-and-licenses/)).

### Data

Raw data is not tracked by git. To run the real-case experiments, obtain:
- 2023 Census block group shapefiles for Pennsylvania
- ACS population (B01003) and commuting (B08303) data
- PRT GTFS feed and WPRDC stop usage (via `scripts/data_processing/hct_odd.py`)

Place all data under `data/` as structured above.

## Usage

All scripts should be run from the **project root**.

### Run LBBD on synthetic grid

```bash
python tests/test_lbbd.py
```

### Run LBBD on real Pittsburgh data

```bash
python -m real_case.experiment --method LBBD --districts 3
```

### Run heuristic experiments

```bash
python scripts/experiments/simulate_service_designs.py
python scripts/experiments/design_perf.py
python scripts/experiments/pareto_frontier_analysis.py
```

### Custom LBBD configuration

Edit `scripts/experiments/lbbd_config.py` to define run profiles, then:

```bash
python scripts/experiments/run_lbbd.py --config standard
```

## Algorithm Overview

### Logic-Based Benders Decomposition (LBBD)

- **Master problem**: district assignment (Hess model) + depot selection
- **Subproblems**: per-district dispatch optimization (CQCP) with Wasserstein DRO
- **Multi-cut acceleration**: multiple Benders cuts per iteration

### Heuristics

- **Random search**: samples depot locations and district configurations, applies local search
- **Local search**: block reassignment with depot reoptimization until convergence

### Uncertainty Model

Demand is treated as distributionally uncertain within a Wasserstein ball of radius ε = ε₀/√n around the empirical distribution from *n* samples.

## Key Results

- DRO designs consistently outperform nominal designs on worst-case cost
- ODD cost is the dominant cost component (~30–40% of total)
- Optimized partition reduces total cost vs nearest-route baseline on Pittsburgh HCT
