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
    # root → fixed-route ordering (list of global block indices in visit order)
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

def _nn_tsp_length(depot: np.ndarray, points: np.ndarray) -> float:
    """Nearest-neighbour TSP tour: depot → points → depot.  Returns km."""
    n = len(points)
    if n == 0:
        return 0.0
    if n == 1:
        return 2.0 * float(np.linalg.norm(points[0] - depot))

    used = np.zeros(n, dtype=bool)
    current = depot.copy()
    total = 0.0
    for _ in range(n):
        dists = np.linalg.norm(points - current, axis=1)
        dists[used] = np.inf
        nearest = int(np.argmin(dists))
        total += float(dists[nearest])
        current = points[nearest]
        used[nearest] = True
    total += float(np.linalg.norm(current - depot))
    return total


def simulate_design(
    design: ServiceDesign,
    geodata,
    prob_dict: Dict[str, float],
    Omega_dict,
    J_function: Callable,
    Lambda: float,
    wr: float,
    wv: float,
    beta: float = BETA,
    discount: float = TSP_TRAVEL_DISCOUNT,
    walk_time_override: float = 0.0,
    tour_km_overrides: Optional[Dict[str, float]] = None,
    n_scenarios: int = 500,
    resolution: int = 100,
) -> EvaluationResult:
    """
    Continuous-space Monte Carlo evaluation.

    For each scenario:
      1. Sample n_i ~ Poisson(Lambda * T_i * p_i) demand per district.
      2. Sample demand-point locations from the continuous density f(x)
         within each district (grid-cell sampling + uniform jitter).
      3. Compute NN-TSP tour through the realised points.
      4. Derive per-district costs from the actual tour length.

    All spatial computations use km (positions converted from metres).
    """
    block_ids = geodata.short_geoid_list
    N = len(block_ids)
    assignment = design.assignment

    # --- Positions in km ---
    block_pos_m = np.array([_get_pos(geodata, b) for b in block_ids])
    block_pos = block_pos_m / 1000.0
    depot_idx = block_ids.index(design.depot_id)
    depot_pos = block_pos[depot_idx]

    # --- Identify districts from assignment matrix ---
    roots = []  # root block indices
    for i in range(N):
        if round(assignment[i, i]) == 1:
            roots.append(i)
    n_districts = len(roots)

    block_district = np.full(N, -1, dtype=int)
    for d, root_idx in enumerate(roots):
        for j in range(N):
            if round(assignment[j, root_idx]) == 1:
                block_district[j] = d

    # --- Fine grid in km for continuous density ---
    areas_km2 = np.array([max(float(geodata.get_area(b)), 1e-12)
                           for b in block_ids])
    probs = np.array([prob_dict.get(b, 0.0) for b in block_ids])

    span = block_pos.max(axis=0) - block_pos.min(axis=0)
    pad = float(span.max()) * 0.05 + 0.05
    lo = block_pos.min(axis=0) - pad
    hi = block_pos.max(axis=0) + pad

    xs = np.linspace(lo[0], hi[0], resolution)
    ys = np.linspace(lo[1], hi[1], resolution)
    dA = float((xs[1] - xs[0]) * (ys[1] - ys[0]))  # km²
    cell_dx = float(xs[1] - xs[0]) if len(xs) > 1 else 0.0
    cell_dy = float(ys[1] - ys[0]) if len(ys) > 1 else 0.0

    xx, yy = np.meshgrid(xs, ys)
    grid_pos = np.column_stack([xx.ravel(), yy.ravel()])  # (M, 2) km

    # Nearest-block interpolation for density
    dists_gb = np.linalg.norm(
        grid_pos[:, None, :] - block_pos[None, :, :], axis=2,
    )  # (M, N)
    nearest_block = np.argmin(dists_gb, axis=1)  # (M,)

    f_vals = probs[nearest_block] / areas_km2[nearest_block]  # per km²
    f_dA = np.maximum(f_vals, 0.0) * dA               # probability mass

    # Grid cell → district (via nearest block's assignment)
    grid_district = block_district[nearest_block]

    # --- Per-district measures and sampling distributions ---
    p_dist = np.zeros(n_districts)
    K_dist = np.zeros(n_districts)
    F_dist = np.zeros(n_districts)
    T_dist = np.zeros(n_districts)

    # Per-district: cell indices, positions, sampling probabilities
    district_cells = []       # list of (cell_indices, cell_positions, cell_probs)

    for d in range(n_districts):
        root_id = block_ids[roots[d]]
        T_dist[d] = design.dispatch_intervals.get(root_id, 1.0)

        cell_mask = grid_district == d
        cell_idx = np.where(cell_mask)[0]
        mass = f_dA[cell_idx]
        total_mass = float(mass.sum())
        p_dist[d] = total_mass

        if len(cell_idx) > 0 and total_mass > 1e-15:
            cell_prob = mass / total_mass
            K_dist[d] = float(np.linalg.norm(
                grid_pos[cell_idx] - depot_pos, axis=1,
            ).min())
        else:
            cell_prob = np.ones(max(len(cell_idx), 1)) / max(len(cell_idx), 1)
        district_cells.append((cell_idx, grid_pos[cell_idx], cell_prob))

        # F_i: ODD cost (inherently discrete — max over blocks)
        assigned_blocks = [block_ids[j] for j in range(N)
                           if block_district[j] == d]
        if Omega_dict and J_function and assigned_blocks:
            ref = next(iter(Omega_dict.values()))
            omega_max = np.zeros_like(ref)
            for b in assigned_blocks:
                omega_max = np.maximum(
                    omega_max, Omega_dict.get(b, np.zeros_like(ref)))
            F_dist[d] = J_function(omega_max)

    # --- Monte Carlo: sample demand points, compute NN-TSP tours ---
    rng = np.random.default_rng(42)
    mu = Lambda * T_dist * p_dist  # expected demand per district (D,)

    n_demand = rng.poisson(
        lam=np.broadcast_to(mu, (n_scenarios, n_districts)).copy(),
    )  # (S, D)

    tour_km_scenarios = np.zeros((n_scenarios, n_districts))

    for d in range(n_districts):
        root_id = block_ids[roots[d]]
        # Skip Monte Carlo if tour is overridden (e.g. fixed-route designs)
        if tour_km_overrides and root_id in tour_km_overrides:
            tour_km_scenarios[:, d] = tour_km_overrides[root_id]
            continue

        cell_idx, cell_pos, cell_prob = district_cells[d]
        if len(cell_idx) == 0:
            continue

        for s in range(n_scenarios):
            n_d = int(n_demand[s, d])
            if n_d == 0:
                continue
            # Sample demand-point locations from the district density
            sampled = rng.choice(len(cell_idx), size=n_d, p=cell_prob)
            points = cell_pos[sampled].copy()
            # Uniform jitter within cell for continuous sampling
            points[:, 0] += rng.uniform(-0.5 * cell_dx, 0.5 * cell_dx, size=n_d)
            points[:, 1] += rng.uniform(-0.5 * cell_dy, 0.5 * cell_dy, size=n_d)
            # NN-TSP tour through realised demand points
            tour_km_scenarios[s, d] = _nn_tsp_length(depot_pos, points)

    avg_tour_km = tour_km_scenarios.mean(axis=0)  # (D,)

    # --- Aggregate costs (demand-weighted) ---
    total_prob = max(float(p_dist.sum()), 1e-12)

    acc_linehaul    = float((K_dist / T_dist * p_dist).sum())
    acc_in_district = float((avg_tour_km / (discount * T_dist) * p_dist).sum())
    acc_odd         = float((F_dist / T_dist * p_dist).sum())
    acc_fleet       = float((avg_tour_km / (VEHICLE_SPEED_KMH * T_dist)).sum())
    acc_wait        = float((T_dist / 2.0 * p_dist).sum())
    acc_invehicle   = float((avg_tour_km / (2.0 * VEHICLE_SPEED_KMH) * p_dist).sum())
    acc_T_w         = float((T_dist * p_dist).sum())

    avg_wait      = acc_wait / total_prob
    avg_invehicle = acc_invehicle / total_prob
    avg_walk      = walk_time_override
    total_user_time = avg_wait + avg_invehicle + avg_walk

    linehaul_cost     = acc_linehaul
    in_district_cost  = acc_in_district
    total_travel_cost = linehaul_cost + in_district_cost
    odd_cost          = acc_odd
    fleet_size        = acc_fleet
    avg_dispatch_interval = acc_T_w / total_prob

    provider_cost = linehaul_cost + in_district_cost + odd_cost
    user_cost     = wr * total_user_time * total_prob
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
