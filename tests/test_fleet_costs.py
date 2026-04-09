#!/usr/bin/env python3
"""Tests for raw fleet costing vs amortized fleet reporting."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lib.baselines.base import ServiceDesign, evaluate_design
from lib.baselines.fr_detour import FRDetour
from lib.baselines.temporal_only import TemporalOnly


class TinyGeo:
    def __init__(self):
        self.short_geoid_list = ["B0", "B1"]
        self.pos = {
            "B0": (0.0, 0.0),
            "B1": (10000.0, 0.0),
        }

    def get_area(self, _):
        return 0.25

    def get_dist(self, b1, b2):
        p1 = np.array(self.pos[b1], dtype=float)
        p2 = np.array(self.pos[b2], dtype=float)
        return float(np.linalg.norm(p1 - p2) / 1000.0)


class FleetCostTests(unittest.TestCase):
    def test_evaluation_result_serializes_raw_fleet_and_costs_with_it(self):
        design = ServiceDesign(
            name="SharedEval",
            assignment=np.array([[1.0, 0.0], [1.0, 0.0]], dtype=float),
            depot_id="B0",
            district_roots=["B0"],
            dispatch_intervals={"B0": 1.0},
        )
        geo = TinyGeo()

        result = evaluate_design(
            design=design,
            geodata=geo,
            prob_dict={"B0": 0.5, "B1": 0.5},
            Omega_dict={},
            J_function=lambda omega: 0.0,
            Lambda=1.0,
            wr=1.0,
            wv=1.0,
            fleet_cost_rate=7.0,
            tour_km_overrides={"B0": 3.0},
        )

        self.assertAlmostEqual(result.fleet_size, 0.1, places=6)
        self.assertEqual(result.raw_fleet_size, 1)
        self.assertAlmostEqual(
            result.provider_cost - (result.total_travel_cost + result.odd_cost),
            7.0,
            places=6,
        )
        payload = result.to_dict()
        self.assertIn("raw_fleet_size", payload["provider"])
        self.assertEqual(payload["provider"]["raw_fleet_size"], 1)

    def test_temporal_only_uses_metadata_fleet_for_fixed_cost(self):
        design = ServiceDesign(
            name="TP-Lit",
            depot_id="B0",
            service_mode="temporal",
            service_metadata={
                "fleet_size": 4,
                "dispatch_K": 2,
                "T_fixed": 1.0,
                "capacity": 3,
                "n_eval_scenarios": 2,
            },
        )
        method = TemporalOnly(n_eval_scenarios=2)
        geo = TinyGeo()

        with patch(
            "lib.baselines.temporal_only._sample_discrete_demand",
            return_value=(np.array([[10.0, 0.0]], dtype=float), np.array([1], dtype=int)),
        ), patch(
            "lib.baselines.temporal_only._solve_vrp",
            return_value=([np.array([0], dtype=int)], 6.0, [6.0]),
        ), patch(
            "lib.baselines.temporal_only._delivery_times_from_routes",
            return_value=(0.5, 1),
        ):
            result = method.custom_simulate(
                design=design,
                geodata=geo,
                prob_dict={"B0": 0.5, "B1": 0.5},
                Omega_dict={},
                J_function=lambda omega: 0.0,
                Lambda=1.0,
                wr=1.0,
                wv=1.0,
                fleet_cost_rate=3.0,
            )

        self.assertEqual(result.raw_fleet_size, 4)
        self.assertAlmostEqual(result.fleet_size, 6.0 / 30.0, places=6)
        self.assertAlmostEqual(
            result.provider_cost - result.total_travel_cost,
            12.0,
            places=6,
        )

    def test_fr_detour_reports_raw_fleet_separately(self):
        geo = TinyGeo()
        design = ServiceDesign(
            name="Multi-FR",
            assignment=np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float),
            depot_id="B0",
            district_roots=["B1"],
            dispatch_intervals={"B1": 1.0},
            district_routes={"B1": [1]},
            service_metadata={
                "fr_detour": {
                    "demand_support_locations": ["B1"],
                    "selected_checkpoints_by_root": {"B1": ["B1"]},
                    "checkpoint_sequence_by_root": {"B1": ["B1"]},
                }
            },
        )
        method = FRDetour(delta=0.0, num_scenarios=1)

        result = method.evaluate(
            design=design,
            geodata=geo,
            prob_dict={"B0": 0.0, "B1": 1.0},
            Omega_dict={},
            J_function=lambda omega: 0.0,
            Lambda=1.0,
            wr=1.0,
            wv=1.0,
            fleet_cost_rate=2.0,
            crn_uniforms=np.zeros((1, 2), dtype=float),
        )

        self.assertEqual(result.raw_fleet_size, 1)
        self.assertGreater(result.fleet_size, 0.0)
        self.assertLess(result.fleet_size, 1.0)
        self.assertAlmostEqual(
            result.provider_cost - result.total_travel_cost,
            2.0,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
