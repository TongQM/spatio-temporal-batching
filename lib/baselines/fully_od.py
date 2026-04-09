"""
Fully on-demand (OD) baseline: one district, dispatch every T → 0.
Represents maximum route flexibility, no spatial or temporal batching.
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from lib.baselines.base import (
    BETA, BaselineMethod, EvaluationResult, ServiceDesign,
    VEHICLE_SPEED_KMH, _effective_fleet_size, _get_pos,
)
from lib.constants import DEFAULT_FLEET_COST_RATE


class FullyOD(BaselineMethod):
    """Single district, T ≈ 0 (T_min = 1 min) dispatch interval.

    Represents the on-demand extreme: every request is served individually
    (or nearly so), minimising wait time at the expense of high travel cost.
    """

    def __init__(self, T_min: float = 1.0 / 60.0):
        """
        Parameters
        ----------
        T_min : float
            Dispatch interval in hours (default: 1 minute = 1/60 h).
        """
        self.T_min = T_min

    def design(
        self,
        geodata,
        prob_dict: Dict[str, float],
        Omega_dict,
        J_function: Callable,
        num_districts: int,
        **kwargs,
    ) -> ServiceDesign:
        block_ids = geodata.short_geoid_list
        N = len(block_ids)

        # One district: assign every block to the first block as root
        # Root = block whose index is 0; all blocks assigned to column 0
        assignment = np.zeros((N, N))
        assignment[:, 0] = 1.0
        assignment[0, 0] = 1.0  # root self-assigns
        root = block_ids[0]

        # Depot = block closest to area centroid (approximated by geographic mean)
        positions = np.array([_get_pos(geodata, b) for b in block_ids])
        centroid = positions.mean(axis=0)
        dist_to_centroid = np.linalg.norm(positions - centroid, axis=1)
        depot_idx = int(np.argmin(dist_to_centroid))
        depot_id = block_ids[depot_idx]

        return ServiceDesign(
            name="OD",
            assignment=assignment,
            depot_id=depot_id,
            district_roots=[root],
            dispatch_intervals={root: self.T_min},
        )

    def custom_simulate(
        self,
        design: ServiceDesign,
        geodata,
        prob_dict: Dict[str, float],
        Omega_dict,
        J_function: Callable,
        Lambda: float,
        wr: float,
        wv: float,
        fleet_cost_rate: float = DEFAULT_FLEET_COST_RATE,
        n_scenarios: int = 500,
    ) -> EvaluationResult:
        """Simulate depot-origin fully on-demand service.

        Each realized demand is served by its own vehicle:
          depot -> destination -> depot

        A group sharing the same destination is treated as one demand atom by
        the Poisson draw at the location level.
        """
        block_ids = geodata.short_geoid_list
        block_pos_m = np.array([_get_pos(geodata, b) for b in block_ids], dtype=float)
        block_pos_km = block_pos_m / 1000.0
        depot_idx = block_ids.index(design.depot_id)
        depot_pos_km = block_pos_km[depot_idx]
        T = float(next(iter(design.dispatch_intervals.values()), self.T_min))

        areas_km2 = np.array([max(float(geodata.get_area(b)), 1e-12) for b in block_ids])
        probs = np.array([prob_dict.get(b, 0.0) for b in block_ids], dtype=float)

        span_km = block_pos_km.max(axis=0) - block_pos_km.min(axis=0)
        pad_km = float(span_km.max()) * 0.05 + 0.05
        lo_km = block_pos_km.min(axis=0) - pad_km
        hi_km = block_pos_km.max(axis=0) + pad_km
        resolution = 100
        xs = np.linspace(lo_km[0], hi_km[0], resolution)
        ys = np.linspace(lo_km[1], hi_km[1], resolution)
        dA = float((xs[1] - xs[0]) * (ys[1] - ys[0])) if resolution > 1 else 1.0
        cell_dx = float(xs[1] - xs[0]) if len(xs) > 1 else 0.0
        cell_dy = float(ys[1] - ys[0]) if len(ys) > 1 else 0.0

        xx, yy = np.meshgrid(xs, ys)
        grid_pos_km = np.column_stack([xx.ravel(), yy.ravel()])
        dists_gb = np.linalg.norm(
            grid_pos_km[:, None, :] - block_pos_km[None, :, :], axis=2,
        )
        nearest_block = np.argmin(dists_gb, axis=1)
        f_dA = np.maximum(probs[nearest_block] / areas_km2[nearest_block], 0.0) * dA
        total_mass = float(f_dA.sum())
        if total_mass > 1e-15:
            cell_prob = f_dA / total_mass
        else:
            cell_prob = np.ones(len(grid_pos_km), dtype=float) / max(len(grid_pos_km), 1)

        rng = np.random.default_rng(42)
        total_wait_h = 0.0
        total_invehicle_h = 0.0
        total_route_km = 0.0
        total_demands = 0

        for _ in range(n_scenarios):
            demand_counts = rng.poisson(Lambda * T * probs)
            n_demands = int(demand_counts.sum())
            if n_demands <= 0:
                continue

            arrivals_h = rng.uniform(0.0, T, size=n_demands)
            total_wait_h += float((T - arrivals_h).sum())

            sampled = rng.choice(len(grid_pos_km), size=n_demands, p=cell_prob)
            dest_points = grid_pos_km[sampled].copy()
            dest_points[:, 0] += rng.uniform(-0.5 * cell_dx, 0.5 * cell_dx, size=n_demands)
            dest_points[:, 1] += rng.uniform(-0.5 * cell_dy, 0.5 * cell_dy, size=n_demands)

            one_way_km = np.linalg.norm(dest_points - depot_pos_km[None, :], axis=1)
            total_invehicle_h += float((one_way_km / VEHICLE_SPEED_KMH).sum())
            total_route_km += float((2.0 * one_way_km).sum())
            total_demands += n_demands

        avg_wait_h = total_wait_h / max(total_demands, 1)
        avg_invehicle_h = total_invehicle_h / max(total_demands, 1)
        total_user_time = avg_wait_h + avg_invehicle_h

        avg_route_km_per_cycle = total_route_km / max(n_scenarios, 1)
        in_district_cost = avg_route_km_per_cycle / max(T, 1e-9)
        fleet_size = avg_route_km_per_cycle / (VEHICLE_SPEED_KMH * max(T, 1e-9))
        raw_fleet_size = _effective_fleet_size(avg_route_km_per_cycle, T)
        provider_cost = in_district_cost + fleet_cost_rate * raw_fleet_size
        user_cost = wr * total_user_time
        total_cost = provider_cost + user_cost

        return EvaluationResult(
            name="OD",
            avg_wait_time=avg_wait_h,
            avg_invehicle_time=avg_invehicle_h,
            avg_walk_time=0.0,
            total_user_time=total_user_time,
            linehaul_cost=0.0,
            in_district_cost=in_district_cost,
            total_travel_cost=in_district_cost,
            fleet_size=fleet_size,
            raw_fleet_size=raw_fleet_size,
            avg_dispatch_interval=T,
            odd_cost=0.0,
            provider_cost=provider_cost,
            user_cost=user_cost,
            total_cost=total_cost,
        )

    def evaluate(
        self,
        design: ServiceDesign,
        geodata,
        prob_dict,
        Omega_dict,
        J_function,
        Lambda: float,
        wr: float,
        wv: float,
        fleet_cost_rate: float = DEFAULT_FLEET_COST_RATE,
        beta: float = BETA,
        road_network=None,
    ) -> EvaluationResult:
        return self.custom_simulate(
            design,
            geodata,
            prob_dict,
            Omega_dict,
            J_function,
            Lambda,
            wr,
            wv,
            fleet_cost_rate,
        )
