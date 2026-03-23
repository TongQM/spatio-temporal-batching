#!/usr/bin/env python3
"""
Baseline comparison experiment.

Runs all service-design baselines (FR, OD, TP-Lit, SP-Lit, Joint-Nom, Joint-DRO)
and saves a JSON result file under a uniquely-named output folder so that
different experiment configurations never overwrite each other.

Output folder pattern:
  results/baseline_comparison/{mode}_{descriptor}_K{K}_L{Lambda}_wr{wr}_wv{wv}_ep{eps}_{timestamp}/
    results.json   ← full structured results + experiment metadata
    summary.csv    ← human-readable table

Usage
-----
# Synthetic 5×5 grid, 3 districts
python scripts/experiments/baseline_comparison.py \
    --mode synthetic --grid_size 5 --districts 3

# Real city with road-network TSP (requires osmnx)
python scripts/experiments/baseline_comparison.py \
    --mode real \
    --shapefile data/2023_shape_files/census_shape_block_group/tl_2023_42_bg.shp \
    --bbox 40.50 40.35 -79.85 -80.10 \
    --districts 3 --road_network
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))


# ---------------------------------------------------------------------------
# Synthetic ToyGeoData
# ---------------------------------------------------------------------------

def make_toy_geodata(grid_size: int):
    import networkx as nx
    from lib.data import GeoData

    class ToyGeoData(GeoData):
        def __init__(self, n, gs):
            self.n_blocks = n
            self.grid_size = gs
            self.level = 'block_group'
            self.gdf = None
            self.short_geoid_list = [f"BLK{i:03d}" for i in range(n)]
            self.block_to_coord = {i: (i // gs, i % gs) for i in range(n)}
            self.pos = {
                self.short_geoid_list[i]: (float(col) * 1000.0, float(row) * 1000.0)
                for i, (row, col) in self.block_to_coord.items()
            }
            # Contiguity graph (expected as geodata.G by algorithm.py)
            self.G = nx.Graph()
            for bid in self.short_geoid_list:
                self.G.add_node(bid)
            for i in range(n):
                for j in range(i + 1, n):
                    r1, c1 = self.block_to_coord[i]
                    r2, c2 = self.block_to_coord[j]
                    if abs(r1 - r2) + abs(c1 - c2) == 1:
                        self.G.add_edge(self.short_geoid_list[i], self.short_geoid_list[j])
            self.block_graph = self.G
            self.arc_list = []
            for e in self.G.edges():
                self.arc_list.append(e)
                self.arc_list.append((e[1], e[0]))
            # Precompute Manhattan distances
            self._dist = {}
            for b1 in self.short_geoid_list:
                i1 = self.short_geoid_list.index(b1)
                r1, c1 = self.block_to_coord[i1]
                for b2 in self.short_geoid_list:
                    i2 = self.short_geoid_list.index(b2)
                    r2, c2 = self.block_to_coord[i2]
                    self._dist[(b1, b2)] = float(abs(r1 - r2) + abs(c1 - c2))

        def get_dist(self, b1, b2):
            return self._dist.get((b1, b2), float('inf'))

        def get_area(self, _):
            return 1.0

        def get_K(self, _):
            return 2.0

        def get_F(self, _):
            return 0.0

        def get_arc_list(self):
            return self.arc_list

    return ToyGeoData(grid_size * grid_size, grid_size)


# ---------------------------------------------------------------------------
# Default ODD helpers
# ---------------------------------------------------------------------------

def identity_J(omega):
    return float(np.sum(omega))


# ---------------------------------------------------------------------------
# Output folder
# ---------------------------------------------------------------------------

def make_output_dir(args) -> str:
    """Return a uniquely-named output folder for this experiment run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.mode == 'synthetic':
        descriptor = f"grid{args.grid_size}x{args.grid_size}"
    else:
        descriptor = "real"
    eps_str = f"{args.epsilon:.0e}".replace('-0', '-').replace('+0', '')
    folder_name = (
        f"{args.mode}_{descriptor}"
        f"_K{args.districts}"
        f"_L{int(args.Lambda)}"
        f"_wr{args.wr}_wv{args.wv}"
        f"_ep{eps_str}"
        f"_{timestamp}"
    )
    out_dir = os.path.join("results", "baseline_comparison", folder_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


# ---------------------------------------------------------------------------
# Data builders
# ---------------------------------------------------------------------------

def build_geodata(args):
    if args.mode == 'synthetic':
        print(f"Building synthetic {args.grid_size}×{args.grid_size} grid …")
        return make_toy_geodata(args.grid_size)
    from lib.data import GeoData
    if args.shapefile is None:
        raise ValueError("--shapefile required for real mode")
    if args.geoid_list:
        with open(args.geoid_list) as f:
            geoid_list = [l.strip() for l in f if l.strip()]
    else:
        from config import BG_GEOID_LIST
        geoid_list = BG_GEOID_LIST
    print(f"Loading GeoData ({len(geoid_list)} block groups) …")
    return GeoData(args.shapefile, geoid_list, level='block_group')


def build_prob_dict(geodata):
    block_ids = geodata.short_geoid_list
    p = 1.0 / len(block_ids)
    return {b: p for b in block_ids}


def build_omega_dict(geodata):
    return {b: np.array([1.0]) for b in geodata.short_geoid_list}


def build_road_network(args):
    if not args.road_network:
        return None
    if args.bbox is None:
        print("Warning: --road_network needs --bbox; skipping.")
        return None
    from lib.road_network import RoadNetwork
    north, south, east, west = args.bbox
    print(f"Fetching road network (bbox {north},{south},{east},{west}) …")
    return RoadNetwork.from_bbox(north, south, east, west)


# ---------------------------------------------------------------------------
# Run one baseline
# ---------------------------------------------------------------------------

def run_baseline(method, name, geodata, prob_dict, Omega_dict, J_function,
                 num_districts, Lambda, wr, wv, road_network):
    from lib.baselines.base import simulate_design, EvaluationResult
    try:
        print(f"  [{name}] designing …", end=" ", flush=True)
        design = method.design(
            geodata, prob_dict, Omega_dict, J_function, num_districts,
            Lambda=Lambda, wr=wr, wv=wv,
        )
        print("simulating …", end=" ", flush=True)

        # Methods with custom simulation (e.g. TP-Lit ADP+VRP)
        if hasattr(method, 'custom_simulate'):
            result = method.custom_simulate(
                design, geodata, prob_dict, Omega_dict, J_function,
                Lambda, wr, wv,
            )
        else:
            # Methods with custom route costs (FR, VCC) provide overrides
            walk_time_h = 0.0
            tour_km_overrides = None
            if hasattr(method, 'compute_route_costs'):
                tour_km_overrides, walk_time_h = method.compute_route_costs(
                    design, geodata, prob_dict, Lambda, wr,
                )

            result = simulate_design(
                design, geodata, prob_dict, Omega_dict, J_function,
                Lambda, wr, wv,
                walk_time_override=walk_time_h,
                tour_km_overrides=tour_km_overrides,
            )
        # Override name to match the baseline label
        result = EvaluationResult(
            name=name,
            **{k: v for k, v in result.__dict__.items() if k != 'name'},
        )
        print(f"done  (total_cost={result.total_cost:.4f})")
        return result
    except NotImplementedError as e:
        print(f"SKIPPED — {e}")
        return None
    except Exception as e:
        import traceback
        print(f"ERROR — {e}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Baseline comparison experiment")
    p.add_argument('--mode', choices=['synthetic', 'real'], default='synthetic')
    p.add_argument('--grid_size', type=int, default=5)
    p.add_argument('--districts', type=int, default=3)
    p.add_argument('--shapefile', type=str, default=None)
    p.add_argument('--geoid_list', type=str, default=None)
    p.add_argument('--bbox', type=float, nargs=4,
                   metavar=('NORTH', 'SOUTH', 'EAST', 'WEST'))
    p.add_argument('--road_network', action='store_true')
    p.add_argument('--Lambda', type=float, default=1000.0,
                   help='Overall arrival rate (passengers/hour)')
    p.add_argument('--wr', type=float, default=1.0,
                   help='Rider time cost weight')
    p.add_argument('--wv', type=float, default=10.0,
                   help='Vehicle operating cost weight (per km)')
    p.add_argument('--epsilon', type=float, default=1e-3,
                   help='Wasserstein radius for Joint-DRO')
    p.add_argument('--rs_iters', type=int, default=100,
                   help='Random-search iterations for SP-Lit / Joint baselines')
    p.add_argument('--T_fixed', type=float, default=0.5,
                   help='Fixed headway for FR and SP-Lit (hours)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out_dir = make_output_dir(args)

    geodata = build_geodata(args)
    prob_dict = build_prob_dict(geodata)
    Omega_dict = build_omega_dict(geodata)
    J_function = identity_J
    road_network = build_road_network(args) if args.mode == 'real' else None

    from lib.baselines import (
        FullyOD, TemporalOnly,
        JointNom, JointDRO, VCC, FRDetour, CarlssonPartition,
    )

    # Table order: FR, spatial, temporal, joint, VCC, OD
    baselines = [
        ("Multi-FR (Jacquillat)",        FRDetour(delta=0.0, max_iters=args.rs_iters, name="Multi-FR (Jacquillat)")),
        ("Multi-FR-Detour (Jacquillat)", FRDetour(delta=0.3, max_iters=args.rs_iters)),
        ("SP-Lit (Carlsson)",            CarlssonPartition(T_fixed=args.T_fixed)),
        ("TP-Lit (Liu)",                 TemporalOnly()),
        ("Joint-Nom",                    JointNom(max_iters=args.rs_iters)),
        ("Joint-DRO",                    JointDRO(epsilon=args.epsilon, max_iters=args.rs_iters)),
        ("VCC",                          VCC()),
        ("OD",                           FullyOD()),
    ]

    print(f"\nRunning {len(baselines)} baselines")
    print(f"  mode={args.mode}, K={args.districts}, Λ={args.Lambda}, "
          f"wr={args.wr}, wv={args.wv}, ε={args.epsilon}\n")

    results = []
    for name, method in baselines:
        result = run_baseline(
            method, name, geodata, prob_dict, Omega_dict, J_function,
            num_districts=args.districts,
            Lambda=args.Lambda,
            wr=args.wr,
            wv=args.wv,
            road_network=road_network,
        )
        if result is not None:
            results.append(result)

    if not results:
        print("No results — check errors above.")
        return

    # ------------------------------------------------------------------ #
    # JSON output
    # ------------------------------------------------------------------ #
    experiment_meta = {
        "timestamp":    datetime.now().isoformat(),
        "mode":         args.mode,
        "grid_size":    args.grid_size if args.mode == 'synthetic' else None,
        "districts":    args.districts,
        "Lambda":       args.Lambda,
        "wr":           args.wr,
        "wv":           args.wv,
        "epsilon":      args.epsilon,
        "T_fixed":      args.T_fixed,
        "rs_iters":     args.rs_iters,
        "road_network": args.road_network,
        "shapefile":    args.shapefile,
        "bbox":         args.bbox,
    }

    json_payload = {
        "experiment": experiment_meta,
        "results":    [r.to_dict() for r in results],
    }

    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, 'w') as f:
        json.dump(json_payload, f, indent=2)

    # ------------------------------------------------------------------ #
    # CSV summary — columns match paper table
    # ------------------------------------------------------------------ #
    rows = []
    for r in results:
        rows.append({
            "Method":              r.name,
            # User performance
            "Wait (h)":            round(r.avg_wait_time, 4),
            "In-vehicle (h)":      round(r.avg_invehicle_time, 4),
            "Walk (h)":            round(r.avg_walk_time, 4),
            "User Total (h)":      round(r.total_user_time, 4),
            # Provider performance
            "Linehaul":            round(r.linehaul_cost, 4),
            "In-district":         round(r.in_district_cost, 4),
            "Total travel":        round(r.total_travel_cost, 4),
            "Fleet":               round(r.fleet_size, 2),
            "Dispatch interval (h)": round(r.avg_dispatch_interval, 4),
            # Aggregates
            "ODD cost":            round(r.odd_cost, 4),
            "Provider cost":       round(r.provider_cost, 4),
            "User cost":           round(r.user_cost, 4),
            "Total cost":          round(r.total_cost, 4),
        })

    df = pd.DataFrame(rows).set_index("Method")
    csv_path = os.path.join(out_dir, "summary.csv")
    df.to_csv(csv_path)

    # ------------------------------------------------------------------ #
    # Print two-panel table matching the figure layout
    # ------------------------------------------------------------------ #
    user_cols     = ["Wait (h)", "In-vehicle (h)", "Walk (h)", "User Total (h)"]
    provider_cols = ["Linehaul", "In-district", "Total travel", "Fleet",
                     "Dispatch interval (h)"]

    print("\n" + "=" * 95)
    print("BASELINE COMPARISON — USER PERFORMANCE")
    print("=" * 95)
    print(df[user_cols].to_string())
    print("\n" + "=" * 95)
    print("BASELINE COMPARISON — PROVIDER PERFORMANCE")
    print("=" * 95)
    print(df[provider_cols].to_string())
    print("\n" + "=" * 95)
    print(f"Output saved to: {out_dir}/")
    print(f"  results.json  — full structured data")
    print(f"  summary.csv   — all columns")
    print("=" * 95)


if __name__ == '__main__':
    main()
