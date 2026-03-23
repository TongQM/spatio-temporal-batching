"""
Base classes and shared evaluation logic for baseline service designs.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from lib.constants import TSP_TRAVEL_DISCOUNT

BETA = 0.7120          # BHH constant
VEHICLE_SPEED_KMH = 30.0   # assumed shuttle speed for fleet/transit-time estimates


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ServiceDesign:
    """Fully describes a service design produced by a baseline method."""
    name: str
    # N×N binary matrix: assignment[j, i] = 1 iff block j is in district i
    assignment: np.ndarray
    depot_id: str
    district_roots: List[str]
    # root → T* (hours)
    dispatch_intervals: Dict[str, float] = field(default_factory=dict)
    # root → fixed-route ordering (list of local block indices in visit order)
    district_routes: Dict[str, List[int]] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    """
    Cost/performance breakdown matching the comparison table:

    User performance
    ----------------
    avg_wait_time      : average wait time per rider (hours); T/2
    avg_invehicle_time : average in-vehicle transit time per rider (hours)
    avg_walk_time      : average walk time per rider (hours); non-zero only for FR
    total_user_time    : avg_wait + avg_invehicle + avg_walk (hours)

    Provider performance
    --------------------
    linehaul_cost      : Σ K_i / T_i  (amortised, per-district weighted)
    in_district_cost   : Σ (in-district tour) / discount / T_i  (weighted)
    total_travel_cost  : linehaul_cost + in_district_cost
    fleet_size         : total vehicles needed  Σ_i ceil(tour_i / (speed * T_i))
    avg_dispatch_interval : demand-weighted mean T* across districts (hours)

    Aggregates
    ----------
    odd_cost           : Σ F_i / T_i  (ODD operating cost)
    provider_cost      : linehaul + in_district + odd
    user_cost          : wr * total_user_time (weighted by demand)
    total_cost         : provider_cost + user_cost
    """
    name: str

    # --- User performance ---
    avg_wait_time: float          # hours
    avg_invehicle_time: float     # hours
    avg_walk_time: float          # hours (0 for non-FR modes)
    total_user_time: float        # hours

    # --- Provider performance ---
    linehaul_cost: float
    in_district_cost: float
    total_travel_cost: float
    fleet_size: float
    avg_dispatch_interval: float  # hours

    # --- Aggregates ---
    odd_cost: float
    provider_cost: float
    user_cost: float
    total_cost: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "user": {
                "avg_wait_time_h":       round(self.avg_wait_time, 6),
                "avg_invehicle_time_h":  round(self.avg_invehicle_time, 6),
                "avg_walk_time_h":       round(self.avg_walk_time, 6),
                "total_user_time_h":     round(self.total_user_time, 6),
                "user_cost":             round(self.user_cost, 6),
            },
            "provider": {
                "linehaul_cost":         round(self.linehaul_cost, 6),
                "in_district_cost":      round(self.in_district_cost, 6),
                "total_travel_cost":     round(self.total_travel_cost, 6),
                "odd_cost":              round(self.odd_cost, 6),
                "fleet_size":            round(self.fleet_size, 4),
                "avg_dispatch_interval_h": round(self.avg_dispatch_interval, 6),
                "provider_cost":         round(self.provider_cost, 6),
            },
            "total_cost": round(self.total_cost, 6),
        }


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaselineMethod(ABC):
    @abstractmethod
    def design(
        self,
        geodata,
        prob_dict: Dict[str, float],
        Omega_dict: Dict[str, np.ndarray],
        J_function: Callable[[np.ndarray], float],
        num_districts: int,
        **kwargs,
    ) -> ServiceDesign: ...

    @abstractmethod
    def evaluate(
        self,
        design: ServiceDesign,
        geodata,
        prob_dict: Dict[str, float],
        Omega_dict: Dict[str, np.ndarray],
        J_function: Callable[[np.ndarray], float],
        Lambda: float,
        wr: float,
        wv: float,
        beta: float = BETA,
        road_network=None,
    ) -> EvaluationResult:
        """
        road_network=None  → BHH/Euclidean approximation (synthetic).
        road_network=RoadNetwork → TSP on actual road distances (real cities).
        """


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _newton_T_star(
    K_i: float,
    F_i: float,
    alpha_i: float,
    wr: float,
    wv: float,
    discount: float = TSP_TRAVEL_DISCOUNT,
    T_min: float = 1e-3,
    T_max: float = 24.0,
) -> float:
    """Newton's method for optimal T* (dispatch interval) for one district."""
    alpha_provider = alpha_i / discount if discount else alpha_i
    c_min, c_max = math.sqrt(T_min), math.sqrt(T_max)

    def g_prime(c):
        return (-2.0 * (K_i + F_i) * c ** -3
                - alpha_provider * c ** -2
                + (wr / wv) * alpha_i
                + 2.0 * wr * c)

    def g_double_prime(c):
        return (6.0 * (K_i + F_i) * c ** -4
                + 2.0 * alpha_provider * c ** -3
                + 2.0 * wr)

    c = math.sqrt(max(T_min, min(T_max, wv / wr))) if wr > 0 else math.sqrt(T_min + 1.0)
    for _ in range(30):
        gp_ = g_prime(c)
        gpp = g_double_prime(c)
        if abs(gp_) < 1e-9 or gpp == 0:
            break
        c_new = max(c_min, min(c - gp_ / gpp, c_max))
        if abs(c_new - c) < 1e-10:
            c = c_new
            break
        c = c_new
    return c ** 2


def _get_pos(geodata, block_id: str) -> Tuple[float, float]:
    """(x, y) in metres; works for both GeoData and synthetic ToyGeoData."""
    if hasattr(geodata, 'pos') and geodata.pos is not None:
        return geodata.pos[block_id]
    if hasattr(geodata, 'block_to_coord') and hasattr(geodata, 'short_geoid_list'):
        idx = geodata.short_geoid_list.index(block_id)
        row, col = geodata.block_to_coord[idx]
        return float(col) * 1000.0, float(row) * 1000.0
    raise AttributeError(f"geodata has no 'pos' for block '{block_id}'")


def _bhh_alpha(geodata, block_ids: List[str], prob_dict: Dict[str, float]) -> float:
    """BHH spatial term for a district: Σ_j sqrt(p_j * A_j)."""
    return sum(
        math.sqrt(prob_dict.get(b, 0.0)) * math.sqrt(float(geodata.get_area(b)))
        for b in block_ids
    )


def _centroid_block(geodata, block_ids: List[str]) -> str:
    """Block closest to the geographic centroid of a set of blocks."""
    positions = np.array([_get_pos(geodata, b) for b in block_ids])
    centroid = positions.mean(axis=0)
    dist = np.linalg.norm(positions - centroid, axis=1)
    return block_ids[int(np.argmin(dist))]


# ---------------------------------------------------------------------------
# Core shared evaluation
# ---------------------------------------------------------------------------

def evaluate_design(
    design: ServiceDesign,
    geodata,
    prob_dict: Dict[str, float],
    Omega_dict: Dict[str, np.ndarray],
    J_function: Callable[[np.ndarray], float],
    Lambda: float,
    wr: float,
    wv: float,
    beta: float = BETA,
    road_network=None,
    discount: float = TSP_TRAVEL_DISCOUNT,
    walk_time_override: float = 0.0,
    tour_km_overrides: Optional[Dict[str, float]] = None,
) -> EvaluationResult:
    """
    Standard per-district cost evaluation.

    Per district i (root r_i, blocks B_i, dispatch interval T_i):
      K_i  = dist(depot → closest block in B_i)           [linehaul, km]
      F_i  = J(max ODD vector over B_i)                   [ODD cost]
      tour_i = BHH β√(Λ T_i) α_i  OR  road TSP on B_i    [in-district, km]
      fleet_i = tour_i / (speed × T_i)                    [vehicles]

    User costs (per rider, averaged over demand):
      wait      = T_i / 2
      in-vehicle ≈ tour_i / (2 × speed)                   [half-tour approximation]
      walk      = walk_time_override (0 for all non-FR modes)

    Parameters
    ----------
    walk_time_override : float
        Pre-computed average walk time (hours) for FR mode; 0 elsewhere.
    """
    from lib.tsp_solver import tsp_tour_length

    block_ids = geodata.short_geoid_list
    N = len(block_ids)
    assignment = design.assignment
    depot_id = design.depot_id

    # Build CRS transformer once for road-network mode
    transformer = None
    if road_network is not None:
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:2163", "EPSG:4326", always_xy=False)

    # Accumulators (all demand-weighted)
    acc_linehaul = 0.0
    acc_in_district = 0.0
    acc_odd = 0.0
    acc_wait = 0.0          # hours × prob
    acc_invehicle = 0.0     # hours × prob
    acc_fleet = 0.0         # vehicles (summed, not averaged)
    acc_T_weighted = 0.0    # T_i × prob  (for demand-weighted mean T)
    acc_prob = 0.0

    for i, root in enumerate(block_ids):
        if round(assignment[i, i]) != 1:
            continue

        assigned = [j for j in range(N) if round(assignment[j, i]) == 1]
        if not assigned:
            continue
        assigned_blocks = [block_ids[j] for j in assigned]

        # K_i: linehaul (km to closest block in district)
        K_i = min(geodata.get_dist(depot_id, b) for b in assigned_blocks)

        # F_i: ODD cost
        if Omega_dict and J_function:
            ref = next(iter(Omega_dict.values()))
            district_omega = np.zeros_like(ref)
            for b in assigned_blocks:
                district_omega = np.maximum(district_omega, Omega_dict.get(b, np.zeros_like(ref)))
            F_i = J_function(district_omega)
        else:
            F_i = 0.0

        T_i = design.dispatch_intervals.get(root, 1.0)
        district_prob = sum(prob_dict.get(block_ids[j], 0.0) for j in assigned)
        alpha_i = _bhh_alpha(geodata, assigned_blocks, prob_dict)

        # In-district tour length (km)
        if tour_km_overrides and root in tour_km_overrides:
            tour_km = tour_km_overrides[root]
        elif road_network is None:
            tour_km = beta * math.sqrt(Lambda * T_i) * alpha_i
        else:
            latlon_list = []
            for b in assigned_blocks:
                x, y = _get_pos(geodata, b)
                lat, lon = transformer.transform(x, y)
                latlon_list.append((lat, lon))
            if len(latlon_list) >= 2:
                D = road_network.get_dist_matrix_km(latlon_list)
                tour_km = tsp_tour_length(D)
            else:
                tour_km = 0.0

        provider_in_district = tour_km / discount  # vehicle-km accounting for autonomy discount

        # Amortised period costs
        period_linehaul    = K_i / T_i
        period_in_district = provider_in_district / T_i
        period_odd         = F_i / T_i

        # Fleet for this district: vehicles needed so one vehicle finishes before next dispatch
        fleet_i = tour_km / (VEHICLE_SPEED_KMH * T_i)   # continuous; round up in practice

        # User times per rider (hours)
        wait_i      = T_i / 2.0
        invehicle_i = tour_km / (2.0 * VEHICLE_SPEED_KMH)  # half-tour approximation

        acc_linehaul    += period_linehaul    * district_prob
        acc_in_district += period_in_district * district_prob
        acc_odd         += period_odd         * district_prob
        acc_wait        += wait_i             * district_prob
        acc_invehicle   += invehicle_i        * district_prob
        acc_fleet       += fleet_i                             # fleet is summed, not averaged
        acc_T_weighted  += T_i               * district_prob
        acc_prob        += district_prob

    acc_prob = max(acc_prob, 1e-12)

    # Per-rider averages
    avg_wait      = acc_wait      / acc_prob
    avg_invehicle = acc_invehicle / acc_prob
    avg_walk      = walk_time_override
    total_user_time = avg_wait + avg_invehicle + avg_walk

    linehaul_cost      = acc_linehaul
    in_district_cost   = acc_in_district
    total_travel_cost  = linehaul_cost + in_district_cost
    odd_cost           = acc_odd
    fleet_size         = acc_fleet
    avg_dispatch_interval = acc_T_weighted / acc_prob

    provider_cost = linehaul_cost + in_district_cost + odd_cost
    user_cost     = wr * total_user_time * acc_prob   # aggregate over demand
    total_cost    = provider_cost + user_cost

    return EvaluationResult(
        name=design.name,
        avg_wait_time=avg_wait,
        avg_invehicle_time=avg_invehicle,
        avg_walk_time=avg_walk,
        total_user_time=total_user_time,
        linehaul_cost=linehaul_cost,
        in_district_cost=in_district_cost,
        total_travel_cost=total_travel_cost,
        fleet_size=fleet_size,
        avg_dispatch_interval=avg_dispatch_interval,
        odd_cost=odd_cost,
        provider_cost=provider_cost,
        user_cost=user_cost,
        total_cost=total_cost,
    )
