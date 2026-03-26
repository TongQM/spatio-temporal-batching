#!/usr/bin/env python3
"""Focused tests for FRDetour shared-input and route semantics."""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import unittest

import networkx as nx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lib.baselines.base import ServiceDesign, VEHICLE_SPEED_KMH
from lib.baselines.fr_detour import (
    FRDetour,
    ReferenceLine,
    SubpathArc,
    _line_access_km,
    _route_execution_km,
    _solve_restricted_routing_lp,
)


HAS_GUROBI = importlib.util.find_spec("gurobipy") is not None


class TinyGeoData:
    """Minimal GeoData-like object for FRDetour tests."""

    def __init__(self, positions_km):
        self.short_geoid_list = list(positions_km.keys())
        self.n_blocks = len(self.short_geoid_list)
        self.level = "block_group"
        self.gdf = None
        self.pos = {
            b: (xy[0] * 1000.0, xy[1] * 1000.0) for b, xy in positions_km.items()
        }
        self.G = nx.complete_graph(self.short_geoid_list)
        self.block_graph = self.G
        self.arc_list = []
        for u, v in self.G.edges():
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

    @unittest.skipUnless(HAS_GUROBI, "requires gurobipy")
    def test_terminal_checkpoint_counts_as_served(self):
        arcs = [
            SubpathArc(
                arc_id=0,
                cp_from=0,
                cp_to=1,
                cost_km=1.0,
                travel_time_h=1.0 / 30.0,
                served_locations=frozenset(),
            )
        ]
        obj, _, _, u_vals = _solve_restricted_routing_lp(
            arcs=arcs,
            checkpoints=["A", "B"],
            m=2,
            capacity=5,
            z_star={"A": 1.0, "B": 1.0},
            demand_s={"A": 0, "B": 1},
            all_blocks={"A", "B"},
            cp_set={"A", "B"},
            cost_mult=1.0,
            walk_penalties={"A": 0.0, "B": 10.0},
            return_solution=True,
        )
        self.assertIsNotNone(obj)
        self.assertAlmostEqual(u_vals["B"], 0.0, places=6)

    @unittest.skipUnless(HAS_GUROBI, "requires gurobipy")
    def test_shared_inputs_metadata_and_zero_demand_eval(self):
        positions = {
            "B0": (0.0, 0.0),
            "B1": (10.0, 0.0),
            "B2": (0.0, 1.0),
        }
        geodata = TinyGeoData(positions)
        prob_dict = {b: 1.0 / len(positions) for b in positions}
        Omega_dict = {b: np.array([1.0]) for b in positions}

        method = FRDetour(
            delta=0.8,
            max_iters=1,
            num_scenarios=1,
            n_saa=2,
            n_angular=2,
            K_skip=1,
            T_values=[1.0],
        )
        shared = method._prepare_shared_inputs(geodata, 1)
        self.assertEqual(shared["crn_uniforms"].shape, (2, 3))

        design_method = FRDetour(
            delta=0.8,
            max_iters=1,
            num_scenarios=1,
            n_saa=2,
            n_angular=2,
            K_skip=1,
            T_values=[1.0],
        )
        design = design_method.design(
            geodata,
            prob_dict,
            Omega_dict,
            identity_J,
            num_districts=1,
            Lambda=1.0,
            wr=1.0,
            wv=10.0,
            **shared,
        )

        meta = design.service_metadata["fr_detour"]
        self.assertTrue(meta["shared_candidate_library"])
        self.assertTrue(meta["shared_crn"])
        self.assertEqual(meta["depot_id"], shared["depot_id"])
        self.assertIn("assigned_locations_by_root", meta)
        self.assertIn("reachable_locations_by_root", meta)
        for root, locations in meta["assigned_locations_by_root"].items():
            self.assertTrue(set(locations).issubset(set(meta["reachable_locations_by_root"][root])))

        manual_design = ServiceDesign(
            name="Manual-FR",
            assignment=np.array(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            ),
            depot_id="B0",
            district_roots=["B0"],
            dispatch_intervals={"B0": 1.0},
            district_routes={"B0": [0, 1, 2]},
            service_metadata={
                "fr_detour": {
                    "assigned_locations_by_root": {"B0": ["B0", "B1", "B2"]},
                    "checkpoint_sequence_by_root": {"B0": ["B0", "B1", "B2"]},
                }
            },
        )
        eval_method = FRDetour(
            delta=0.0,
            num_scenarios=1,
            K_skip=1,
        )
        result = eval_method.evaluate(
            manual_design,
            geodata,
            prob_dict,
            Omega_dict,
            identity_J,
            Lambda=1.0,
            wr=1.0,
            wv=10.0,
            crn_uniforms=np.zeros((1, 3), dtype=float),
        )

        expected_route_km = 2.0
        self.assertAlmostEqual(
            result.avg_invehicle_time,
            expected_route_km / (2.0 * VEHICLE_SPEED_KMH),
            places=6,
        )
        self.assertAlmostEqual(result.avg_wait_time, 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
