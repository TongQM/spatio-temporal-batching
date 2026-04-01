#!/usr/bin/env python3
"""Focused tests for depot-origin VCC overflow semantics."""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lib.baselines.base import ServiceDesign
from lib.baselines.vcc import VCC, _Request, _overflow_request_metrics, _scenario_metrics


class VCCTests(unittest.TestCase):
    def test_overflow_request_waits_until_horizon(self):
        req = _Request(
            req_id=0,
            dest_block="B1",
            dest_pos=np.array([1.0, 0.0], dtype=float),
            release_h=0.25,
            deadline_h=1.0,
            direct_km=1.0,
            direct_time_h=1.0 / 30.0,
        )

        metrics = _overflow_request_metrics(
            req=req,
            start_pos=np.array([0.0, 0.0], dtype=float),
            planning_horizon_h=1.0,
        )

        self.assertAlmostEqual(metrics["travel_km"], 2.0, places=6)
        self.assertAlmostEqual(metrics["wait_h"], 0.75, places=6)
        self.assertAlmostEqual(metrics["drop_walk_h"], 0.0, places=6)
        self.assertAlmostEqual(metrics["invehicle_h"], 1.0 / 30.0, places=6)

    def test_scenario_metrics_include_overflow_wait(self):
        req = _Request(
            req_id=0,
            dest_block="B1",
            dest_pos=np.array([1.0, 0.0], dtype=float),
            release_h=0.4,
            deadline_h=1.0,
            direct_km=1.0,
            direct_time_h=1.0 / 30.0,
        )

        stats = _scenario_metrics(
            route_evals=[],
            rejected=[req],
            requests=[req],
            start_pos=np.array([0.0, 0.0], dtype=float),
            planning_horizon_h=1.0,
        )

        self.assertAlmostEqual(stats["wait_h"], 0.6, places=6)
        self.assertAlmostEqual(stats["walk_h"], 0.0, places=6)
        self.assertAlmostEqual(stats["invehicle_h"], 1.0 / 30.0, places=6)
        self.assertAlmostEqual(stats["travel_km"], 2.0, places=6)

    def test_custom_simulate_amortizes_provider_cost_by_horizon(self):
        design = ServiceDesign(
            name="VCC",
            depot_id="B0",
            service_mode="vcc",
            service_metadata={
                "fleet_size": 3,
                "vehicle_capacity": 3,
                "max_walk_km": 0.3,
                "deadline_slack_h": 0.5,
                "walk_speed_kmh": 5.0,
                "vehicle_speed_kmh": 30.0,
                "planning_horizon_h": 2.0,
                "n_scenarios": 2,
                "max_exact_requests": 4,
                "random_seed": 42,
            },
        )
        method = VCC(n_scenarios=2, random_seed=42)
        scenario_stats = [
            {"wait_h": 0.5, "walk_h": 0.1, "invehicle_h": 0.2, "travel_km": 12.0},
            {"wait_h": 0.7, "walk_h": 0.2, "invehicle_h": 0.4, "travel_km": 8.0},
        ]

        with patch.object(VCC, "_simulate_one_scenario", side_effect=scenario_stats):
            result = method.custom_simulate(
                design=design,
                geodata=object(),
                prob_dict={},
                Omega_dict={},
                J_function=lambda omega: 0.0,
                Lambda=1.0,
                wr=2.0,
                wv=10.0,
            )

        self.assertAlmostEqual(result.avg_wait_time, 0.6, places=6)
        self.assertAlmostEqual(result.avg_walk_time, 0.15, places=6)
        self.assertAlmostEqual(result.avg_invehicle_time, 0.3, places=6)
        self.assertAlmostEqual(result.in_district_cost, 5.0, places=6)
        self.assertAlmostEqual(result.total_travel_cost, 5.0, places=6)
        self.assertAlmostEqual(result.provider_cost, 5.0, places=6)
        self.assertAlmostEqual(result.fleet_size, 10.0 / 60.0, places=6)
        self.assertAlmostEqual(result.avg_dispatch_interval, 2.0, places=6)

    def test_design_uses_explicit_fleet_size_override(self):
        class TinyGeo:
            short_geoid_list = ["B0", "B1", "B2"]
            pos = {
                "B0": (0.0, 0.0),
                "B1": (1000.0, 0.0),
                "B2": (0.0, 1000.0),
            }

        method = VCC(fleet_size=30, vehicle_capacity=3)
        design = method.design(
            geodata=TinyGeo(),
            prob_dict={"B0": 1 / 3, "B1": 1 / 3, "B2": 1 / 3},
            Omega_dict={},
            J_function=lambda omega: 0.0,
            num_districts=3,
            Lambda=300.0,
            wr=1.0,
            wv=10.0,
        )

        self.assertEqual(design.service_metadata["fleet_size"], 30)
        self.assertAlmostEqual(design.service_metadata["planning_horizon_h"], 0.3, places=6)


if __name__ == "__main__":
    unittest.main()
