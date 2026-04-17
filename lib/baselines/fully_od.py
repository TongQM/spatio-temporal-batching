"""
Fully on-demand (OD) baseline: one district, dispatch every T → 0.
Represents maximum route flexibility, no spatial or temporal batching.

Supports finite fleet with queueing:
  - ``fleet_size`` caps the number of vehicles operating in parallel.
  - Each request is served depot → destination → depot; a request waits
    in queue when all vehicles are busy.
  - When ``fleet_size_grid`` is supplied, the method sweeps over the grid
    per call and keeps the fleet size that minimises total cost at the
    given (Lambda, wr, wv, fleet_cost_rate), analogous to SP-Lit's T*
    optimisation.
"""
from __future__ import annotations

import heapq
from typing import Callable, Dict, Iterable, Optional

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

    def __init__(
        self,
        T_min: float = 1.0 / 60.0,
        fleet_size: Optional[int] = None,
        fleet_size_grid: Optional[Iterable[int]] = None,
        horizon_h: float = 1.0,
    ):
        """
        Parameters
        ----------
        T_min : float
            Dispatch interval in hours (default: 1 minute = 1/60 h).
        fleet_size : int or None
            Fixed fleet cap. None = infinite fleet (no queueing).
        fleet_size_grid : iterable[int] or None
            When provided, ``custom_simulate`` tries each candidate fleet
            size and keeps the one minimising total_cost at the given
            (Lambda, wr, wv, fleet_cost_rate). Overrides ``fleet_size``.
        horizon_h : float
            Simulation horizon per scenario (hours). Longer horizons
            give tighter estimates of steady-state queue dynamics.
        """
        self.T_min = T_min
        self.fleet_size = int(fleet_size) if fleet_size is not None else None
        self.fleet_size_grid = (
            tuple(int(f) for f in fleet_size_grid)
            if fleet_size_grid is not None else None
        )
        self.horizon_h = float(horizon_h)

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
        n_scenarios: int = 50,
    ) -> EvaluationResult:
        """Simulate depot-origin fully on-demand service with queueing.

        Each realized demand is served by its own vehicle:
          depot -> destination -> depot

        When ``fleet_size`` (or a chosen candidate from ``fleet_size_grid``)
        is finite, an event-driven queue assigns each arrival to the next
        free vehicle; arrivals wait in queue when all vehicles are busy.
        """
        # ---- Precompute geometry / demand density (independent of fleet_size) ----
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

        # ---- Pre-sample arrivals and destinations once, reuse across fleet sizes ----
        horizon_h = self.horizon_h
        speed = VEHICLE_SPEED_KMH
        rng = np.random.default_rng(42)

        scenario_arrivals = []        # list of (arrivals_h, one_way_km)
        for _ in range(n_scenarios):
            n_demands = int(rng.poisson(Lambda * horizon_h))
            if n_demands == 0:
                scenario_arrivals.append((np.array([]), np.array([])))
                continue
            arrivals = np.sort(rng.uniform(0.0, horizon_h, size=n_demands))
            sampled = rng.choice(len(grid_pos_km), size=n_demands, p=cell_prob)
            dest_points = grid_pos_km[sampled].copy()
            dest_points[:, 0] += rng.uniform(-0.5 * cell_dx, 0.5 * cell_dx, size=n_demands)
            dest_points[:, 1] += rng.uniform(-0.5 * cell_dy, 0.5 * cell_dy, size=n_demands)
            one_way = np.linalg.norm(dest_points - depot_pos_km[None, :], axis=1)
            scenario_arrivals.append((arrivals, one_way))

        # ---- Try each candidate fleet size and keep the best ----
        if self.fleet_size_grid is not None:
            candidates = list(self.fleet_size_grid)
        elif self.fleet_size is not None:
            candidates = [self.fleet_size]
        else:
            candidates = [None]  # infinite fleet (legacy)

        best_result = None
        best_total = float("inf")

        for f_cap in candidates:
            result = self._simulate_one_fleet(
                scenario_arrivals=scenario_arrivals,
                fleet_cap=f_cap,
                horizon_h=horizon_h,
                speed=speed,
                T=T,
                wr=wr,
                fleet_cost_rate=fleet_cost_rate,
            )
            if result.total_cost < best_total:
                best_total = result.total_cost
                best_result = result
                best_result.name = f"OD"
                # Stash chosen fleet size for diagnostics
                best_fleet_cap = f_cap

        if best_result is None:
            raise RuntimeError("No OD fleet size candidate produced a result.")
        return best_result

    def _simulate_one_fleet(
        self,
        scenario_arrivals,
        fleet_cap: Optional[int],
        horizon_h: float,
        speed: float,
        T: float,
        wr: float,
        fleet_cost_rate: float,
    ) -> EvaluationResult:
        total_wait_h = 0.0
        total_invehicle_h = 0.0
        total_route_km = 0.0
        total_demands = 0
        # For provider cost rate: average km/hour over horizon across scenarios
        total_horizon_km = 0.0

        for arrivals, one_way_km in scenario_arrivals:
            n = len(arrivals)
            if n == 0:
                continue

            # Event-driven queueing with fleet_cap vehicles.
            if fleet_cap is None or fleet_cap <= 0:
                # Infinite fleet: every arrival served immediately (wait=T/2 dispatch only).
                queue_wait = np.zeros(n)
            else:
                free_times = [0.0] * fleet_cap
                heapq.heapify(free_times)
                queue_wait = np.empty(n)
                for i in range(n):
                    earliest = heapq.heappop(free_times)
                    start = max(arrivals[i], earliest)
                    queue_wait[i] = start - arrivals[i]
                    trip_dur = 2.0 * one_way_km[i] / speed  # round-trip
                    heapq.heappush(free_times, start + trip_dur)

            # User times per rider: dispatch wait (T/2) + queue wait + in-vehicle (one-way)
            total_wait_h += float((T / 2.0 + queue_wait).sum())
            total_invehicle_h += float((one_way_km / speed).sum())
            total_route_km += float((2.0 * one_way_km).sum())
            total_horizon_km += float((2.0 * one_way_km).sum())  # alias
            total_demands += n

        n_scen = max(len(scenario_arrivals), 1)
        avg_wait_h = total_wait_h / max(total_demands, 1)
        avg_invehicle_h = total_invehicle_h / max(total_demands, 1)
        total_user_time = avg_wait_h + avg_invehicle_h

        # Provider cost rate: km per hour = total_km / (n_scenarios * horizon)
        in_district_rate_km_per_h = total_horizon_km / (n_scen * horizon_h)
        fleet_size_metric = in_district_rate_km_per_h / speed  # continuous utilisation
        if fleet_cap is not None:
            raw_fleet_size = int(fleet_cap)
        else:
            raw_fleet_size = _effective_fleet_size(
                total_horizon_km / n_scen, horizon_h,
            )

        provider_cost = in_district_rate_km_per_h + fleet_cost_rate * raw_fleet_size
        user_cost = wr * total_user_time
        total_cost = provider_cost + user_cost

        return EvaluationResult(
            name="OD",
            avg_wait_time=avg_wait_h,
            avg_invehicle_time=avg_invehicle_h,
            avg_walk_time=0.0,
            total_user_time=total_user_time,
            linehaul_cost=0.0,
            in_district_cost=in_district_rate_km_per_h,
            total_travel_cost=in_district_rate_km_per_h,
            fleet_size=fleet_size_metric,
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
