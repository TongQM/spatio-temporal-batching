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

from lib.baselines.base import ServiceDesign, VEHICLE_SPEED_KMH
from lib.baselines.fr_detour import (
    FRDetour,
    ReferenceLine,
    SubpathArc,
    _build_initial_arcs,
    _solve_projected_service_scenario,
    _line_access_km,
    _project_riders_to_candidate_locations,
    _finalize_projected_rider_assignments,
    _route_execution_km,
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

    def test_projection_prefers_nearest_assigned_stop_point(self):
        riders = [{"origin_block": "DEM", "origin_pos": np.array([1.1, 0.0], dtype=float)}]
        candidate_info = {
            "CP1": {"kind": "checkpoint", "root": "r1"},
            "STOP": {"kind": "stop_point", "root": "r2"},
        }
        block_pos = {
            "CP1": np.array([0.0, 0.0], dtype=float),
            "STOP": np.array([1.0, 0.0], dtype=float),
        }
        projected_counts, projected_stop_by_root, projected_cp = _project_riders_to_candidate_locations(
            riders, candidate_info, block_pos
        )
        self.assertEqual(projected_counts["STOP"], 1)
        self.assertEqual(projected_stop_by_root["r2"]["STOP"], 1)
        self.assertEqual(projected_cp, {})
        self.assertEqual(riders[0]["projected_loc"], "STOP")

    def test_final_assignment_reassigns_unserved_projected_stop_to_checkpoint(self):
        riders = [
            {
                "origin_block": "DEM",
                "origin_pos": np.array([1.1, 0.0], dtype=float),
                "projected_loc": "STOP",
                "projected_kind": "stop_point",
                "projected_root": "r2",
            }
        ]
        block_pos = {
            "CP1": np.array([0.0, 0.0], dtype=float),
            "STOP": np.array([1.0, 0.0], dtype=float),
        }
        service_info = {
            "CP1": {"kind": "checkpoint", "arrival_h": 0.4, "headway_h": 1.0, "root": "r1"},
        }
        walk_km, _, inveh_h, service_counts, total_riders, outcomes, records = (
            _finalize_projected_rider_assignments(
                riders,
                block_pos,
                service_info,
                np.random.default_rng(9),
                return_assignments=True,
            )
        )
        self.assertEqual(total_riders, 1)
        self.assertEqual(service_counts.get("CP1"), 1)
        self.assertEqual(outcomes["DEM"]["checkpoint_served"], 1)
        self.assertEqual(outcomes["DEM"]["reassigned_service"], 1)
        self.assertAlmostEqual(walk_km, 1.1, places=6)
        self.assertAlmostEqual(inveh_h, 0.4, places=6)
        self.assertEqual(records[0]["service_loc"], "CP1")
        self.assertTrue(records[0]["reassigned"])

    def test_final_assignment_keeps_projected_checkpoint(self):
        riders = [
            {
                "origin_block": "CP1",
                "origin_pos": np.array([0.0, 0.0], dtype=float),
                "projected_loc": "CP1",
                "projected_kind": "checkpoint",
                "projected_root": "r1",
            }
        ]
        block_pos = {"CP1": np.array([0.0, 0.0], dtype=float)}
        service_info = {
            "CP1": {"kind": "checkpoint", "arrival_h": 0.2, "headway_h": 1.0, "root": "r1"},
        }
        walk_km, _, _, service_counts, total_riders, outcomes = _finalize_projected_rider_assignments(
            riders,
            block_pos,
            service_info,
            np.random.default_rng(3),
        )
        self.assertEqual(total_riders, 1)
        self.assertEqual(service_counts.get("CP1"), 1)
        self.assertEqual(outcomes["CP1"]["checkpoint_served"], 1)
        self.assertEqual(outcomes["CP1"]["reassigned_service"], 0)
        self.assertAlmostEqual(walk_km, 0.0, places=6)

    def test_zero_projected_stop_demand_is_not_served(self):
        positions = {
            "A": (0.0, 0.0),
            "B": (1.0, 0.4),
            "C": (2.0, 0.0),
        }
        geodata = TinyGeoData(positions, checkpoint_ids=["A", "C"])
        design = ServiceDesign(
            name="Multi-FR-Detour",
            assignment=np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
            depot_id="A",
            district_roots=["A"],
            dispatch_intervals={"A": 1.0},
            district_routes={"A": [0, 2]},
            service_metadata={
                "fr_detour": {
                    "checkpoint_sequence_by_root": {"A": ["A", "C"]},
                    "selected_checkpoints_by_root": {"A": ["A", "C"]},
                    "assigned_locations_by_root": {"A": ["A", "B", "C"]},
                    "assigned_stop_points_by_root": {"A": ["B"]},
                    "demand_support_locations": ["A", "C"],
                }
            },
        )
        method = FRDetour(delta=0.8, max_iters=1, num_scenarios=1, n_saa=1, n_angular=1, K_skip=0, T_values=[1.0])
        block_ids = geodata.short_geoid_list
        block_pos_map = {b: np.array(geodata.pos[b], dtype=float) / 1000.0 for b in block_ids}
        areas_km2 = {b: float(geodata.get_area(b)) for b in block_ids}
        walk_saved_km = {"A": 0.0, "B": 0.0, "C": 0.0}
        scenario = _solve_projected_service_scenario(
            design,
            method,
            geodata,
            {"A": 1.0, "B": 0.0, "C": 0.0},
            block_ids,
            block_pos_map,
            {"A": 1, "B": 0, "C": 0},
            block_pos_map["A"],
            areas_km2,
            walk_saved_km,
            1.0,
            1.0,
            np.random.default_rng(5),
        )
        self.assertEqual(scenario["projected_stop_counts_by_root"].get("A", {}), {})
        self.assertEqual(scenario["route_services"]["A"]["served_stops"], set())

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
        checkpoint_candidates = set(meta["checkpoint_candidate_locations"])
        self.assertTrue(meta["checkpoint_sequence_by_root"])
        for root in design.district_roots:
            checkpoints = meta["checkpoint_sequence_by_root"][root]
            schedule = meta["checkpoint_schedule_by_root"][root]
            loads = meta["scenario_assigned_loads_by_root"][root]
            self.assertEqual(len(checkpoints), len(schedule))
            self.assertTrue(all(schedule[i] <= schedule[i + 1] for i in range(len(schedule) - 1)))
            self.assertTrue(all(load <= (1.0 + method.load_factor_kappa) * method.capacity + 1e-6 for load in loads))
            self.assertTrue(
                checkpoint_candidates.isdisjoint(meta["assigned_stop_points_by_root"][root])
            )
            self.assertTrue(
                checkpoint_candidates.isdisjoint(meta["reachable_stop_points_by_root"][root])
            )


if __name__ == "__main__":
    unittest.main()
