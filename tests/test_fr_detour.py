#!/usr/bin/env python3
"""Deterministic acceptance tests for FRDetour MiND-style semantics."""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lib.baselines.base import VEHICLE_SPEED_KMH
from lib.baselines.fr_detour import (
    FRDetour,
    ReferenceLine,
    SubpathArc,
    _build_initial_arcs,
    _line_access_km,
    _route_execution_km,
    _sample_global_service_assignments,
    _solve_restricted_routing_lp,
)


HAS_GUROBI = importlib.util.find_spec("gurobipy") is not None


class TinyGeoData:
    """Minimal GeoData-like object for FRDetour tests."""

    def __init__(self, positions_km, checkpoint_ids=None):
        self.short_geoid_list = list(positions_km.keys())
        self.n_blocks = len(self.short_geoid_list)
        self.level = "block_group"
        self.gdf = None
        self.pos = {
            b: (xy[0] * 1000.0, xy[1] * 1000.0) for b, xy in positions_km.items()
        }
        self.checkpoint_ids = list(checkpoint_ids or self.short_geoid_list)
        self.G = {}
        self.block_graph = {}
        self.arc_list = []
        for i, u in enumerate(self.short_geoid_list):
            for v in self.short_geoid_list[i + 1:]:
                self.arc_list.append((u, v))
                self.arc_list.append((v, u))

    def get_dist(self, b1, b2):
        p1 = np.array(self.pos[b1], dtype=float)
        p2 = np.array(self.pos[b2], dtype=float)
        return float(np.linalg.norm(p1 - p2) / 1000.0)

    def get_area(self, _):
        return 0.25

    def get_K(self, _):
        return 2.0

    def get_F(self, _):
        return 0.0

    def get_arc_list(self):
        return self.arc_list


def identity_J(omega):
    return float(np.sum(omega))


class FRDetourTests(unittest.TestCase):
    def setUp(self):
        self.block_pos = {
            "A": np.array([0.0, 0.0], dtype=float),
            "B": np.array([1.0, 0.5], dtype=float),
            "C": np.array([2.0, 0.0], dtype=float),
            "D": np.array([1.0, 2.0], dtype=float),
        }

    def test_route_execution_uses_depot_legs_once(self):
        depot_pos = np.array([0.0, 0.0], dtype=float)
        block_pos = {
            "B0": np.array([0.0, 0.0], dtype=float),
            "B1": np.array([10.0, 0.0], dtype=float),
            "B2": np.array([0.0, 1.0], dtype=float),
        }
        line = ReferenceLine(
            idx=0,
            checkpoints=["B0", "B1", "B2"],
            reachable_locations={"B0", "B1", "B2"},
            est_tour_km=0.0,
        )

        access_km = _line_access_km(line, depot_pos, block_pos)
        total_km = _route_execution_km(line, depot_pos, block_pos, K_skip=1)

        self.assertAlmostEqual(access_km, 1.0, places=6)
        self.assertAlmostEqual(total_km, 2.0, places=6)

    def test_delta_zero_only_builds_checkpoint_arcs(self):
        line = ReferenceLine(
            idx=0,
            checkpoints=["A", "C"],
            reachable_locations={"A", "B", "C"},
            est_tour_km=0.0,
        )
        arcs = _build_initial_arcs(
            line, ["A", "B", "C"], self.block_pos, delta=0.0, K_skip=0,
            speed_kmh=VEHICLE_SPEED_KMH, alpha_time=0.5,
        )
        self.assertTrue(arcs)
        self.assertTrue(all("B" not in arc.served_locations for arc in arcs))

    def test_delta_positive_builds_corridor_detour_arc(self):
        line = ReferenceLine(
            idx=0,
            checkpoints=["A", "C"],
            reachable_locations={"A", "B", "C"},
            est_tour_km=0.0,
        )
        arcs = _build_initial_arcs(
            line, ["A", "B", "C"], self.block_pos, delta=0.8, K_skip=0,
            speed_kmh=VEHICLE_SPEED_KMH, alpha_time=0.5,
        )
        self.assertTrue(any("B" in arc.served_locations for arc in arcs))

    def test_schedule_feasibility_rejects_nearby_but_too_slow_detour(self):
        line = ReferenceLine(
            idx=0,
            checkpoints=["A", "C"],
            reachable_locations={"A", "B", "C"},
            est_tour_km=0.0,
        )
        arcs = _build_initial_arcs(
            line, ["A", "B", "C"], self.block_pos, delta=0.8, K_skip=0,
            speed_kmh=VEHICLE_SPEED_KMH, alpha_time=0.0,
        )
        self.assertTrue(arcs)
        self.assertTrue(all("B" not in arc.served_locations for arc in arcs))

    def test_skip_limit_changes_available_subpaths(self):
        checkpoints = ["A", "B", "C"]
        line = ReferenceLine(
            idx=0,
            checkpoints=checkpoints,
            reachable_locations=set(checkpoints),
            est_tour_km=0.0,
        )
        arcs_k0 = _build_initial_arcs(
            line, checkpoints, self.block_pos, delta=0.0, K_skip=0,
            speed_kmh=VEHICLE_SPEED_KMH, alpha_time=0.5,
        )
        arcs_k1 = _build_initial_arcs(
            line, checkpoints, self.block_pos, delta=0.0, K_skip=1,
            speed_kmh=VEHICLE_SPEED_KMH, alpha_time=0.5,
        )
        self.assertFalse(any(arc.cp_from == 0 and arc.cp_to == 2 for arc in arcs_k0))
        self.assertTrue(any(arc.cp_from == 0 and arc.cp_to == 2 for arc in arcs_k1))

    def test_global_assignment_prefers_served_stop_over_checkpoint(self):
        block_pos = {
            "CP1": np.array([0.0, 0.0], dtype=float),
            "CP2": np.array([10.0, 0.0], dtype=float),
            "STOP": np.array([1.0, 0.0], dtype=float),
            "DEM": np.array([1.1, 0.0], dtype=float),
        }
        areas = {b: 0.0 for b in block_pos}
        service_info = {
            "CP1": {"kind": "checkpoint", "arrival_h": 0.2, "headway_h": 1.0, "root": "r1"},
            "CP2": {"kind": "checkpoint", "arrival_h": 0.3, "headway_h": 1.0, "root": "r2"},
            "STOP": {"kind": "detour_stop", "arrival_h": 0.25, "headway_h": 1.0, "root": "r2"},
        }
        walk_km, wait_h, inveh_h, service_counts, total_riders, outcomes = (
            _sample_global_service_assignments(
                demand_s={"DEM": 1},
                demand_blocks=["DEM"],
                block_pos_km=block_pos,
                areas_km2=areas,
                service_point_info=service_info,
                rng=np.random.default_rng(7),
            )
        )
        self.assertEqual(total_riders, 1)
        self.assertEqual(service_counts.get("STOP"), 1)
        self.assertEqual(outcomes["DEM"]["detour_stop_served"], 1)
        self.assertAlmostEqual(walk_km, 0.1, places=6)
        self.assertGreater(wait_h, 0.0)
        self.assertAlmostEqual(inveh_h, 0.25, places=6)

    def test_unselected_checkpoint_demand_falls_back_to_selected_checkpoint(self):
        block_pos = {
            "CP_SEL": np.array([0.0, 0.0], dtype=float),
            "CP_UNSEL": np.array([2.0, 0.0], dtype=float),
        }
        areas = {b: 0.0 for b in block_pos}
        service_info = {
            "CP_SEL": {"kind": "checkpoint", "arrival_h": 0.4, "headway_h": 1.0, "root": "r1"},
        }
        walk_km, _, inveh_h, service_counts, total_riders, outcomes = (
            _sample_global_service_assignments(
                demand_s={"CP_UNSEL": 2},
                demand_blocks=["CP_UNSEL"],
                block_pos_km=block_pos,
                areas_km2=areas,
                service_point_info=service_info,
                rng=np.random.default_rng(9),
            )
        )
        self.assertEqual(total_riders, 2)
        self.assertEqual(service_counts.get("CP_SEL"), 2)
        self.assertEqual(outcomes["CP_UNSEL"]["checkpoint_fallback"], 2)
        self.assertAlmostEqual(walk_km, 4.0, places=6)
        self.assertAlmostEqual(inveh_h, 0.8, places=6)

    @unittest.skipUnless(HAS_GUROBI, "requires gurobipy")
    def test_fixed_origin_lp_serves_corridor_stop_with_detour(self):
        line = ReferenceLine(
            idx=0,
            checkpoints=["A", "C"],
            reachable_locations={"A", "B", "C"},
            est_tour_km=0.0,
        )
        arcs = _build_initial_arcs(
            line, ["A", "B", "C"], self.block_pos, delta=0.8, K_skip=0,
            speed_kmh=VEHICLE_SPEED_KMH, alpha_time=0.5,
        )
        obj, _, _, u_vals = _solve_restricted_routing_lp(
            arcs=arcs,
            checkpoints=["A", "C"],
            m=2,
            capacity=4,
            z_star={"B": 1.0},
            demand_s={"A": 0, "B": 1, "C": 0},
            all_blocks={"B"},
            cp_set={"A", "C"},
            cost_mult=1.0,
            walk_penalties={"B": 10.0},
            return_solution=True,
        )
        self.assertIsNotNone(obj)
        self.assertAlmostEqual(u_vals["B"], 0.0, places=6)

    @unittest.skipUnless(HAS_GUROBI, "requires gurobipy")
    def test_fixed_origin_lp_rejects_outside_corridor_stop(self):
        line = ReferenceLine(
            idx=0,
            checkpoints=["A", "C"],
            reachable_locations={"A", "C", "D"},
            est_tour_km=0.0,
        )
        arcs = _build_initial_arcs(
            line, ["A", "C", "D"], self.block_pos, delta=0.8, K_skip=0,
            speed_kmh=VEHICLE_SPEED_KMH, alpha_time=0.5,
        )
        obj, _, _, u_vals = _solve_restricted_routing_lp(
            arcs=arcs,
            checkpoints=["A", "C"],
            m=2,
            capacity=4,
            z_star={"D": 1.0},
            demand_s={"A": 0, "C": 0, "D": 1},
            all_blocks={"D"},
            cp_set={"A", "C"},
            cost_mult=1.0,
            walk_penalties={"D": 10.0},
            return_solution=True,
        )
        self.assertIsNotNone(obj)
        self.assertAlmostEqual(u_vals["D"], 1.0, places=6)

    @unittest.skipUnless(HAS_GUROBI, "requires gurobipy")
    def test_design_metadata_exposes_schedule_and_loads(self):
        positions = {
            "A": (0.0, 0.0),
            "B": (1.0, 0.4),
            "C": (2.0, 0.0),
        }
        geodata = TinyGeoData(positions, checkpoint_ids=["A", "C"])
        prob_dict = {"A": 0.05, "B": 0.90, "C": 0.05}
        Omega_dict = {b: np.array([1.0]) for b in positions}

        method = FRDetour(
            delta=0.8,
            max_iters=2,
            num_scenarios=2,
            n_saa=2,
            n_angular=1,
            K_skip=0,
            T_values=[1.0],
            capacity=4,
            max_fleet=1.0,
            load_factor_kappa=0.5,
        )
        candidate = [
            ReferenceLine(
                idx=0,
                checkpoints=["A", "C"],
                reachable_locations=set(),
                est_tour_km=4.0,
            )
        ]
        crn = np.full((2, 3), 0.60, dtype=float)
        shared = method._prepare_shared_inputs(
            geodata,
            1,
            candidate_lines=candidate,
            crn_uniforms=crn,
        )
        design = method.design(
            geodata,
            prob_dict,
            Omega_dict,
            identity_J,
            num_districts=1,
            Lambda=4.0,
            wr=1.0,
            wv=10.0,
            **shared,
        )

        meta = design.service_metadata["fr_detour"]
        self.assertEqual(meta["solve_mode"], "dd_relaxed")
        self.assertIn("checkpoint_schedule_by_root", meta)
        self.assertIn("scenario_assigned_loads_by_root", meta)
        self.assertIn("selected_checkpoints_by_root", meta)
        self.assertIn("assigned_stop_points_by_root", meta)
        self.assertIn("reachable_stop_points_by_root", meta)
        self.assertIn("demand_support_locations", meta)
        self.assertTrue(meta["checkpoint_sequence_by_root"])
        for root in design.district_roots:
            checkpoints = meta["checkpoint_sequence_by_root"][root]
            schedule = meta["checkpoint_schedule_by_root"][root]
            loads = meta["scenario_assigned_loads_by_root"][root]
            self.assertEqual(len(checkpoints), len(schedule))
            self.assertTrue(all(schedule[i] <= schedule[i + 1] for i in range(len(schedule) - 1)))
            self.assertTrue(all(load <= (1.0 + method.load_factor_kappa) * method.capacity + 1e-6 for load in loads))


if __name__ == "__main__":
    unittest.main()
