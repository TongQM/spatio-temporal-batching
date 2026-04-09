"""
Multi Fixed-Route with Detour — MiND-structured adaptation (Martin-Iradi et al. 2025)

Adaptation of the double decomposition from:
  "A Double Decomposition Algorithm for Network Planning and Operations in
  Deviated Fixed-route Microtransit"
  (Martin-Iradi, Schmid, Cummings, Jacquillat 2025, arXiv:2402.01265)

to a steady-state district-based service design setting.

Repo interpretation:
  - fixed common origin / depot
  - stochastic destination demand
  - last-mile delivery semantics

Some earlier discussion in this repo used the reverse first-mile wording
("stochastic origins to a common destination"). For the simplified Euclidean
service model used here, those views are largely symmetric mathematically, but
the intended semantic interpretation in code, plots, and docs is:
  riders start from the common origin and are dropped off at realized
  destination/service locations.

Algorithm structure (from the paper):
  Double decomposition:
    - Outer: SAA multi-cut Benders with scenario-wise optimality cuts
    - Inner: column generation with label-setting DP pricing

  First stage — line selection + demand assignment:
    - Master MIP selects reference trips endogenously under a fleet budget,
      chooses dispatch intervals, and assigns stopping locations to trips
    - Variables: x[ℓ,t] (trip selection), z[n,ℓ,t] (binary stop-to-trip assignment),
      θ[s,ℓ,t] (scenario recourse)
    - Hard coverage, expected-demand load bounds, fleet budget

  Second stage — subpath routing on load-expanded checkpoint network:
    - Nodes: (checkpoint, passenger_load) pairs
    - Subpath arcs between checkpoint pairs (skip ≤ K_skip), time-feasible
    - Routing LP with flow balance, linking (Σy ≤ z*), coverage (Σy+u ≥ z*)
    - Column generation: restricted LP → label-setting DP pricing → iterate
    - Benders cut coefficients: π[b] + μ[b] (linking + coverage duals)

Adaptations from the paper (honest deviations):

  Structural:
    - Stop-location destination-demand aggregation (not individual passengers)
    - Steady-state timing surrogate: time_budget = (1+α)*direct_dist/speed,
      not the paper's scheduled checkpoint arrival times
    - Line library via angular partition + NN-TSP (not pre-existing network)
    - Expected-demand master load bounds; scenario capacity in 2nd stage

  Cost model (repo-specific, differs from the paper):
    - Master objective uses FR-consistent cost structure:
        * No linehaul term (depot travel is part of the route tour)
        * No ODD term (vehicles run on a fixed schedule)
        * Wait cost is demand-weighted: wr * T/2 * Σ prob[b] * z[b,ℓ,t]
    - This aligns the master with this repo's evaluation semantics,
      not the paper's original planning + service-quality objective.

  Reachability:
    - Geometric δ-distance to route segments, plus time-budget feasibility
      for a single-block detour.  Still weaker than the paper's full subpath
      feasibility model (which considers multi-block detour paths).

  Evaluation:
    - Final scoring uses the same subpath-routing semantics as the Benders
      subproblem, with common-random-number demand samples available for
      reproducible FR vs FR-Detour comparisons.
    - Provider/user cost reporting still follows this repo's baseline table
      semantics rather than the paper's original service-quality objective.

When delta=0: "Multi-FR" (no deviations).
When delta>0: "Multi-FR-Detour" (deviation within δ-corridor).

Reference: Martin-Iradi, Schmid, Cummings, Jacquillat (2025, arXiv:2402.01265).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np
try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    tqdm = None

from lib.baselines.base import (
    BETA, VEHICLE_SPEED_KMH, BaselineMethod, EvaluationResult, ServiceDesign,
    _effective_fleet_size, _get_pos,
)
from lib.constants import DEFAULT_FLEET_COST_RATE, TSP_TRAVEL_DISCOUNT


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ReferenceLine:
    idx: int
    checkpoints: List[str]
    reachable_locations: Set[str]
    est_tour_km: float

    def __init__(
        self,
        idx: int,
        checkpoints: List[str],
        reachable_locations: Optional[Set[str]] = None,
        est_tour_km: float = 0.0,
        reachable_blocks: Optional[Set[str]] = None,
    ):
        if reachable_locations is None:
            reachable_locations = reachable_blocks
        self.idx = idx
        self.checkpoints = checkpoints
        self.reachable_locations = set(reachable_locations or set())
        self.est_tour_km = est_tour_km

    @property
    def reachable_blocks(self) -> Set[str]:
        """Backward-compatible alias for older repo code."""
        return self.reachable_locations


@dataclass
class SubpathArc:
    arc_id: int
    cp_from: int                    # checkpoint index (source end)
    cp_to: int                      # checkpoint index (destination end)
    cost_km: float                  # vehicle travel distance (km)
    travel_time_h: float            # time for vehicle to traverse this subpath
    served_locations: FrozenSet[str]
    # Note: load_delta is scenario-dependent (= Σ demand_s[b] for b in served_locations),
    # computed at LP-build time, NOT stored on the arc.

    def __init__(
        self,
        arc_id: int,
        cp_from: int,
        cp_to: int,
        cost_km: float,
        travel_time_h: float,
        served_locations: Optional[FrozenSet[str]] = None,
        served_blocks: Optional[FrozenSet[str]] = None,
    ):
        if served_locations is None:
            served_locations = served_blocks
        self.arc_id = arc_id
        self.cp_from = cp_from
        self.cp_to = cp_to
        self.cost_km = cost_km
        self.travel_time_h = travel_time_h
        self.served_locations = frozenset(served_locations or frozenset())

    @property
    def served_blocks(self) -> FrozenSet[str]:
        """Backward-compatible alias for older repo code."""
        return self.served_locations


@dataclass
class ReferenceTrip:
    line_idx: int
    time_idx: int
    headway_h: float
    checkpoints: Tuple[str, ...]
    checkpoint_schedule_h: Tuple[float, ...]
    reachable_locations: FrozenSet[str]
    est_tour_km: float


def _checkpoint_time_budget(cp_i_pos, cp_j_pos, speed_kmh, alpha_time=0.5):
    """Max allowed time for a subpath between checkpoints i and j.

    Steady-state timing surrogate: (1 + alpha_time) * direct_dist / speed.
    """
    direct_km = float(np.linalg.norm(cp_j_pos - cp_i_pos))
    if direct_km < 1e-12:
        return 1e-6  # degenerate: same position
    return (1.0 + alpha_time) * direct_km / speed_kmh


def _checkpoint_schedule_offsets(
    checkpoints: List[str],
    block_pos_km: Dict[str, np.ndarray],
    speed_kmh: float,
    alpha_time: float = 0.5,
) -> List[float]:
    """Scheduled arrival offsets at checkpoints for one reference trip.

    This is still built from geometry, but unlike the old segment budget helper
    it creates an explicit checkpoint schedule that downstream logic can use.
    """
    if not checkpoints:
        return []
    offsets = [0.0]
    for idx in range(1, len(checkpoints)):
        prev = block_pos_km[checkpoints[idx - 1]]
        cur = block_pos_km[checkpoints[idx]]
        offsets.append(offsets[-1] + _checkpoint_time_budget(prev, cur, speed_kmh, alpha_time))
    return offsets


def _trip_time_budget(
    checkpoint_schedule_h: List[float] | Tuple[float, ...],
    cp_from: int,
    cp_to: int,
) -> float:
    """Available scheduled time between two checkpoints on a reference trip."""
    if cp_to <= cp_from:
        return 0.0
    return max(float(checkpoint_schedule_h[cp_to]) - float(checkpoint_schedule_h[cp_from]), 1e-6)


def _build_reference_trip(
    line: ReferenceLine,
    time_idx: int,
    headway_h: float,
    block_pos_km: Dict[str, np.ndarray],
    speed_kmh: float,
    alpha_time: float,
) -> ReferenceTrip:
    schedule_h = tuple(
        _checkpoint_schedule_offsets(
            line.checkpoints,
            block_pos_km,
            speed_kmh=speed_kmh,
            alpha_time=alpha_time,
        )
    )
    return ReferenceTrip(
        line_idx=line.idx,
        time_idx=time_idx,
        headway_h=float(headway_h),
        checkpoints=tuple(line.checkpoints),
        checkpoint_schedule_h=schedule_h,
        reachable_locations=frozenset(line.reachable_locations),
        est_tour_km=float(line.est_tour_km),
    )


def _default_depot_id(geodata, block_ids: List[str]) -> str:
    """Use the geographic centroid block as the depot."""
    positions_m = np.array([_get_pos(geodata, b) for b in block_ids])
    centroid = positions_m.mean(axis=0)
    depot_idx = int(np.argmin(np.linalg.norm(positions_m - centroid, axis=1)))
    return block_ids[depot_idx]


def _shortest_checkpoint_path_km(
    checkpoints: List[str], block_pos_km: Dict[str, np.ndarray], K_skip: int,
) -> float:
    """Shortest checkpoint-to-checkpoint path allowed by skip limit."""
    m = len(checkpoints)
    if m <= 1:
        return 0.0
    dp = [float("inf")] * m
    dp[0] = 0.0
    for j in range(1, m):
        for i in range(max(0, j - K_skip - 1), j):
            d = float(np.linalg.norm(block_pos_km[checkpoints[j]]
                                     - block_pos_km[checkpoints[i]]))
            dp[j] = min(dp[j], dp[i] + d)
    return dp[-1]


def _line_access_km(
    line: ReferenceLine, depot_pos_km: np.ndarray, block_pos_km: Dict[str, np.ndarray],
) -> float:
    """Depot-to-first and last-to-depot legs for one route execution."""
    if not line.checkpoints:
        return 0.0
    start = float(np.linalg.norm(block_pos_km[line.checkpoints[0]] - depot_pos_km))
    end = float(np.linalg.norm(block_pos_km[line.checkpoints[-1]] - depot_pos_km))
    return start + end


def _route_execution_km(
    line: ReferenceLine, depot_pos_km: np.ndarray, block_pos_km: Dict[str, np.ndarray],
    K_skip: int, checkpoint_travel_km: float | None = None,
) -> float:
    """Full executed route length: depot legs plus checkpoint travel."""
    checkpoint_path = (
        _shortest_checkpoint_path_km(line.checkpoints, block_pos_km, K_skip)
        if checkpoint_travel_km is None else checkpoint_travel_km
    )
    return _line_access_km(line, depot_pos_km, block_pos_km) + checkpoint_path


def _coerce_crn_uniforms(U_base: Optional[np.ndarray], num_scenarios: int, num_blocks: int):
    """Validate or create common-random-number uniforms."""
    if U_base is None:
        rng = np.random.default_rng(42)
        return rng.uniform(0.0, 1.0, size=(num_scenarios, num_blocks))
    arr = np.asarray(U_base, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != num_blocks:
        raise ValueError(
            f"crn_uniforms must have {num_blocks} columns, got {arr.shape}")
    return np.clip(arr, 0.0, 1.0 - 1e-10)


# ---------------------------------------------------------------------------
# Preserved geometry helpers
# ---------------------------------------------------------------------------

def _greedy_nn_route(points_km: np.ndarray, depot_km: np.ndarray) -> List[int]:
    """Nearest-neighbour route from depot. Returns permutation of range(n)."""
    n = len(points_km)
    if n <= 1:
        return list(range(n))
    unvisited = list(range(n))
    route: List[int] = []
    cur = depot_km.copy()
    while unvisited:
        dists = [float(np.linalg.norm(points_km[j] - cur)) for j in unvisited]
        idx = int(np.argmin(dists))
        nearest = unvisited[idx]
        route.append(nearest)
        cur = points_km[nearest]
        unvisited.pop(idx)
    return route


def _two_opt_improve(
    route: List[int], points_km: np.ndarray,
    depot_km: np.ndarray, max_iters: int = 100,
) -> List[int]:
    """2-opt improvement on a route ordering."""
    r = list(route)
    n = len(r)
    if n <= 2:
        return r

    def _tour_len(perm):
        pts = np.vstack([depot_km[None]]
                        + [points_km[i][None] for i in perm]
                        + [depot_km[None]])
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    best = _tour_len(r)
    for _ in range(max_iters):
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                r2 = r[:i] + r[i:j + 1][::-1] + r[j + 1:]
                d = _tour_len(r2)
                if d < best - 1e-12:
                    r = r2
                    best = d
                    improved = True
        if not improved:
            break
    return r


def _fixed_route_tour_km(
    route_order: List[int], points_km: np.ndarray,
    depot_km: np.ndarray, active_mask: np.ndarray,
) -> float:
    """Tour distance following a fixed route, skipping inactive blocks."""
    active_stops = [i for i in route_order if active_mask[i]]
    if not active_stops:
        return 0.0
    pts = np.vstack([depot_km[None]]
                    + [points_km[i][None] for i in active_stops]
                    + [depot_km[None]])
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _block_geometry(areas_km2: np.ndarray, delta: float):
    """Per-block corridor and walk distances."""
    r = np.sqrt(areas_km2 / math.pi)
    r_safe = np.maximum(r, 1e-12)
    corridor_r = np.minimum(delta, r)
    walk_no_det = 2.0 * r / 3.0
    walk_det = np.where(
        delta < r_safe,
        walk_no_det - delta + delta**3 / (3.0 * r_safe**2),
        0.0,
    )
    walk_det = np.maximum(walk_det, 0.0)
    return corridor_r, walk_det, walk_no_det


def _sample_walk_distances(
    demand_s: Dict[str, int],
    assigned_blocks: List[str],
    block_pos_km: Dict[str, np.ndarray],
    cp_set: set,
    u_vals: Dict[str, float],
    areas_km2: Dict[str, float],
    rng: np.random.Generator,
) -> float:
    """Simulation-based walk distance computation for FR-Detour evaluation.

    For each demand in a block, sample a random position uniformly within the
    block's circular area, then assign to nearest checkpoint or stopping location.

    Walk rules:
      - Demand assigned to a checkpoint: walk = Euclidean distance to checkpoint.
      - Demand assigned to a stopping location AND detour happens (u ≈ 0):
        walk = distance to that stopping location.
      - Demand assigned to a stopping location AND no detour (u ≈ 1):
        walk = distance to the nearest checkpoint instead.

    Returns total walk_km * n_riders (unnormalized, to be accumulated).
    """
    # Build arrays of checkpoint and stopping-location positions
    cp_list = [b for b in assigned_blocks if b in cp_set]
    stop_list = [b for b in assigned_blocks if b not in cp_set]
    if not cp_list:
        # No checkpoints — fallback: all blocks are service points
        cp_list = assigned_blocks
        stop_list = []

    cp_pos = np.array([block_pos_km[b] for b in cp_list])   # (n_cp, 2)
    if stop_list:
        stop_pos = np.array([block_pos_km[b] for b in stop_list])  # (n_stop, 2)
    else:
        stop_pos = np.empty((0, 2))

    # Precompute which stopping locations are served (detour happens)
    stop_served = np.array([
        u_vals.get(b, 1.0) < 0.5 for b in stop_list
    ], dtype=bool)

    total_walk_riders = 0.0

    for b in assigned_blocks:
        d_b = demand_s.get(b, 0)
        if d_b == 0:
            continue

        # Sample d_b positions within block b's circular area
        center = block_pos_km[b]
        area_b = areas_km2.get(b, 0.25)
        r_b = math.sqrt(area_b / math.pi)

        if r_b < 1e-9:
            # Degenerate block — all demands at center
            positions = np.tile(center, (d_b, 1))
        else:
            # Uniform sampling in a disk: r * sqrt(U), theta ~ Uniform(0, 2pi)
            radii = r_b * np.sqrt(rng.uniform(0, 1, size=d_b))
            angles = rng.uniform(0, 2 * math.pi, size=d_b)
            offsets = np.column_stack([radii * np.cos(angles),
                                       radii * np.sin(angles)])
            positions = center + offsets  # (d_b, 2)

        # Distance from each demand to each checkpoint
        dist_to_cp = np.linalg.norm(
            positions[:, None, :] - cp_pos[None, :, :], axis=2)  # (d_b, n_cp)
        min_cp_dist = dist_to_cp.min(axis=1)  # (d_b,)

        if len(stop_list) == 0:
            # No stopping locations — all walk to nearest checkpoint
            total_walk_riders += float(min_cp_dist.sum())
            continue

        # Distance from each demand to each stopping location
        dist_to_stop = np.linalg.norm(
            positions[:, None, :] - stop_pos[None, :, :], axis=2)  # (d_b, n_stop)

        # Assign each demand to nearest service point (checkpoint or stop)
        min_stop_dist = dist_to_stop.min(axis=1)  # (d_b,)
        nearest_stop_idx = dist_to_stop.argmin(axis=1)  # (d_b,)

        # Is nearest point a checkpoint or stopping location?
        assign_to_stop = min_stop_dist < min_cp_dist  # True → nearest is a stop

        for p in range(d_b):
            if not assign_to_stop[p]:
                # Assigned to checkpoint — always walk there
                total_walk_riders += min_cp_dist[p]
            else:
                # Assigned to stopping location
                si = nearest_stop_idx[p]
                if stop_served[si]:
                    # Detour happens — walk to stopping location
                    total_walk_riders += min_stop_dist[p]
                else:
                    # No detour — must walk to nearest checkpoint
                    total_walk_riders += min_cp_dist[p]

    return total_walk_riders


def _sample_passenger_assignments(
    demand_s: Dict[str, int],
    assigned_blocks: List[str],
    block_pos_km: Dict[str, np.ndarray],
    cp_set: set,
    u_vals: Dict[str, float],
    areas_km2: Dict[str, float],
    rng: np.random.Generator,
    service_blocks: Optional[List[str]] = None,
):
    """Sample passenger locations and assign each passenger to a service point.

    Parameters
    ----------
    assigned_blocks : list
        Blocks that generate demand for this trip (partition-based accounting).
    service_blocks : list, optional
        All blocks that can act as service points for this trip (may be wider
        than assigned_blocks when stops are shared across routes).  If None,
        defaults to assigned_blocks for backward compatibility.

    Returns
    -------
    total_walk_riders_km : float
        Sum of passenger walk distances in km.
    service_counts : Dict[str, int]
        Number of passengers delivered to each service point.
    total_riders : int
        Total passengers in the scenario/district.
    """
    if service_blocks is None:
        service_blocks = assigned_blocks
    cp_list = [b for b in service_blocks if b in cp_set]
    stop_list = [b for b in service_blocks if b not in cp_set]
    if not cp_list:
        cp_list = assigned_blocks
        stop_list = []

    cp_pos = np.array([block_pos_km[b] for b in cp_list]) if cp_list else np.empty((0, 2))
    stop_pos = np.array([block_pos_km[b] for b in stop_list]) if stop_list else np.empty((0, 2))
    stop_served = np.array([u_vals.get(b, 1.0) < 0.5 for b in stop_list], dtype=bool)

    total_walk_riders = 0.0
    service_counts: Dict[str, int] = {}
    total_riders = 0

    for b in assigned_blocks:
        d_b = int(demand_s.get(b, 0))
        if d_b <= 0:
            continue
        total_riders += d_b

        center = block_pos_km[b]
        area_b = areas_km2.get(b, 0.25)
        r_b = math.sqrt(area_b / math.pi)

        if r_b < 1e-9:
            positions = np.tile(center, (d_b, 1))
        else:
            radii = r_b * np.sqrt(rng.uniform(0, 1, size=d_b))
            angles = rng.uniform(0, 2 * math.pi, size=d_b)
            offsets = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            positions = center + offsets

        dist_to_cp = np.linalg.norm(positions[:, None, :] - cp_pos[None, :, :], axis=2)
        min_cp_dist = dist_to_cp.min(axis=1)
        nearest_cp_idx = dist_to_cp.argmin(axis=1)

        if len(stop_list) == 0:
            total_walk_riders += float(min_cp_dist.sum())
            for idx in nearest_cp_idx:
                sp = cp_list[int(idx)]
                service_counts[sp] = service_counts.get(sp, 0) + 1
            continue

        dist_to_stop = np.linalg.norm(positions[:, None, :] - stop_pos[None, :, :], axis=2)
        min_stop_dist = dist_to_stop.min(axis=1)
        nearest_stop_idx = dist_to_stop.argmin(axis=1)
        assign_to_stop = min_stop_dist < min_cp_dist

        for p in range(d_b):
            cp_name = cp_list[int(nearest_cp_idx[p])]
            if not assign_to_stop[p]:
                total_walk_riders += float(min_cp_dist[p])
                service_counts[cp_name] = service_counts.get(cp_name, 0) + 1
                continue

            si = int(nearest_stop_idx[p])
            if stop_served[si]:
                stop_name = stop_list[si]
                total_walk_riders += float(min_stop_dist[p])
                service_counts[stop_name] = service_counts.get(stop_name, 0) + 1
            else:
                total_walk_riders += float(min_cp_dist[p])
                service_counts[cp_name] = service_counts.get(cp_name, 0) + 1

    return total_walk_riders, service_counts, total_riders


def _checkpoint_candidate_ids(geodata, block_ids: List[str]) -> Set[str]:
    """Candidate route-geometry anchors.

    For synthetic FR instances this is the `C..` pool; on older inputs we
    fall back to treating every location as a candidate checkpoint.
    """
    return set(getattr(geodata, "checkpoint_ids", block_ids))


def _stop_point_ids(geodata, block_ids: List[str]) -> Set[str]:
    """Return the globally disjoint non-checkpoint service-location set."""
    return set(block_ids) - _checkpoint_candidate_ids(geodata, block_ids)


def _sample_rider_destinations(
    demand_s: Dict[str, int],
    demand_blocks: List[str],
    block_pos_km: Dict[str, np.ndarray],
    areas_km2: Dict[str, float],
    rng: np.random.Generator,
) -> List[Dict[str, object]]:
    """Sample continuous rider destinations from block-level realized demand."""
    riders: List[Dict[str, object]] = []
    for b in demand_blocks:
        d_b = int(demand_s.get(b, 0))
        if d_b <= 0:
            continue
        center = block_pos_km[b]
        area_b = areas_km2.get(b, 0.25)
        r_b = math.sqrt(area_b / math.pi)
        if r_b < 1e-9:
            positions = np.tile(center, (d_b, 1))
        else:
            radii = r_b * np.sqrt(rng.uniform(0, 1, size=d_b))
            angles = rng.uniform(0, 2 * math.pi, size=d_b)
            offsets = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            positions = center + offsets
        for p in range(d_b):
            riders.append(
                {
                    "origin_block": b,
                    "origin_pos": np.array(positions[p], dtype=float),
                }
            )
    return riders


def _project_riders_to_candidate_locations(
    riders: List[Dict[str, object]],
    candidate_service_info: Dict[str, Dict[str, object]],
    block_pos_km: Dict[str, np.ndarray],
):
    """Greedily project each rider to the nearest candidate service location."""
    if not riders or not candidate_service_info:
        return {}, {}, {}

    candidate_locs = list(candidate_service_info.keys())
    candidate_pos = np.array([block_pos_km[loc] for loc in candidate_locs], dtype=float)
    positions = np.array([r["origin_pos"] for r in riders], dtype=float)
    dist = np.linalg.norm(positions[:, None, :] - candidate_pos[None, :, :], axis=2)
    nearest_idx = dist.argmin(axis=1)

    projected_counts: Dict[str, int] = {}
    projected_stop_counts_by_root: Dict[str, Dict[str, int]] = {}
    projected_checkpoint_counts: Dict[str, int] = {}

    for i, rider in enumerate(riders):
        loc = candidate_locs[int(nearest_idx[i])]
        info = candidate_service_info[loc]
        rider["projected_loc"] = loc
        rider["projected_kind"] = str(info.get("kind", ""))
        rider["projected_root"] = info.get("root")
        projected_counts[loc] = projected_counts.get(loc, 0) + 1
        if info.get("kind") == "stop_point":
            root = str(info.get("root", ""))
            projected_stop_counts_by_root.setdefault(root, {})
            projected_stop_counts_by_root[root][loc] = (
                projected_stop_counts_by_root[root].get(loc, 0) + 1
            )
        elif info.get("kind") == "checkpoint":
            projected_checkpoint_counts[loc] = projected_checkpoint_counts.get(loc, 0) + 1

    return projected_counts, projected_stop_counts_by_root, projected_checkpoint_counts


def _finalize_projected_rider_assignments(
    riders: List[Dict[str, object]],
    block_pos_km: Dict[str, np.ndarray],
    service_point_info: Dict[str, Dict[str, object]],
    rng: np.random.Generator,
    return_assignments: bool = False,
):
    """Finalize rider service after second-stage detour decisions."""
    if not riders or not service_point_info:
        if return_assignments:
            return 0.0, 0.0, 0.0, {}, 0, {}, []
        return 0.0, 0.0, 0.0, {}, 0, {}

    service_locs = list(service_point_info.keys())
    service_pos = np.array([block_pos_km[b] for b in service_locs], dtype=float)

    total_walk_riders = 0.0
    total_wait_h = 0.0
    total_invehicle_h = 0.0
    total_riders = len(riders)
    service_counts: Dict[str, int] = {}
    demand_outcomes: Dict[str, Dict[str, int]] = {}
    assignment_records = [] if return_assignments else None

    for rider in riders:
        origin_block = str(rider["origin_block"])
        origin_pos = np.array(rider["origin_pos"], dtype=float)
        projected_loc = str(rider["projected_loc"])

        demand_outcomes.setdefault(
            origin_block,
            {
                "checkpoint_served": 0,
                "detour_stop_served": 0,
                "reassigned_service": 0,
            },
        )

        if projected_loc in service_point_info:
            final_loc = projected_loc
            reassigned = False
            final_dist = float(np.linalg.norm(origin_pos - block_pos_km[final_loc]))
        else:
            dist_to_service = np.linalg.norm(origin_pos[None, :] - service_pos, axis=1)
            nearest_service_idx = int(dist_to_service.argmin())
            final_loc = service_locs[nearest_service_idx]
            reassigned = True
            final_dist = float(dist_to_service[nearest_service_idx])

        info = service_point_info[final_loc]
        total_walk_riders += final_dist
        total_invehicle_h += float(info.get("arrival_h", 0.0))
        headway_h = float(info.get("headway_h", 0.0))
        if headway_h > 0:
            total_wait_h += headway_h - float(rng.uniform(0.0, headway_h))

        service_counts[final_loc] = service_counts.get(final_loc, 0) + 1
        if info.get("kind") == "detour_stop":
            demand_outcomes[origin_block]["detour_stop_served"] += 1
        else:
            demand_outcomes[origin_block]["checkpoint_served"] += 1
        if reassigned:
            demand_outcomes[origin_block]["reassigned_service"] += 1

        if return_assignments:
            assignment_records.append(
                {
                    "origin_block": origin_block,
                    "origin_pos": origin_pos,
                    "projected_loc": projected_loc,
                    "service_loc": final_loc,
                    "service_pos": np.array(block_pos_km[final_loc], dtype=float),
                    "service_kind": str(info.get("kind", "")),
                    "walk_km": final_dist,
                    "reassigned": reassigned,
                }
            )

    result = (
        total_walk_riders,
        total_wait_h,
        total_invehicle_h,
        service_counts,
        total_riders,
        demand_outcomes,
    )
    if return_assignments:
        return result + (assignment_records,)
    return result


def _solve_projected_service_scenario(
    design: ServiceDesign,
    method,
    geodata,
    prob_dict: Dict[str, float],
    block_ids: List[str],
    block_pos_km: Dict[str, np.ndarray],
    demand_s: Dict[str, int],
    depot_pos_km: np.ndarray,
    areas_km2: Dict[str, float],
    walk_saved_km: Dict[str, float],
    rider_time_weight: float,
    vehicle_cost_weight: float,
    walk_rng: np.random.Generator,
    return_assignments: bool = False,
):
    """Scenario evaluation via demand projection -> detour solve -> reassignment."""
    fr_meta = design.service_metadata.get("fr_detour", {})
    assignment = design.assignment
    checkpoint_sequence_meta = fr_meta.get("checkpoint_sequence_by_root", {})
    selected_checkpoints_meta = fr_meta.get(
        "selected_checkpoints_by_root",
        checkpoint_sequence_meta,
    )
    assigned_locations_meta = fr_meta.get(
        "assigned_locations_by_root",
        fr_meta.get("assigned_blocks_by_root", {}),
    )
    assigned_stop_points_meta = fr_meta.get("assigned_stop_points_by_root", {})
    demand_support_locations = list(fr_meta.get(
        "demand_support_locations",
        [b for b in block_ids if prob_dict.get(b, 0.0) > 0.0],
    ))
    stop_point_set = _stop_point_ids(geodata, block_ids)

    roots_iter = design.district_roots or [
        block_ids[i] for i in range(len(block_ids))
        if assignment is not None and round(assignment[i, i]) == 1
    ]

    route_data = []
    candidate_service_info: Dict[str, Dict[str, object]] = {}
    for root in roots_iter:
        T_i = float(design.dispatch_intervals.get(root, 1.0))
        route_global = design.district_routes.get(root, [])
        checkpoints = [block_ids[gi] for gi in route_global]
        if not checkpoints and root in checkpoint_sequence_meta:
            checkpoints = list(checkpoint_sequence_meta[root])
        if not checkpoints and root in selected_checkpoints_meta:
            checkpoints = list(selected_checkpoints_meta[root])
        if not checkpoints:
            continue

        assigned_locations = list(assigned_locations_meta.get(root, []))
        assigned_stop_points = [
            b for b in assigned_stop_points_meta.get(root, [])
            if b in stop_point_set
        ]
        if not assigned_stop_points:
            assigned_stop_points = [
                b for b in assigned_locations if b in stop_point_set
            ]

        line_obj = ReferenceLine(
            idx=0,
            checkpoints=checkpoints,
            reachable_locations=set(checkpoints) | set(assigned_stop_points),
            est_tour_km=_tour_distance(depot_pos_km, checkpoints, block_pos_km),
        )
        arcs = _build_initial_arcs(
            line_obj, block_ids, block_pos_km, method.delta,
            method.K_skip, VEHICLE_SPEED_KMH, method.alpha_time
        )
        route_data.append({
            "root": root,
            "T_i": T_i,
            "route_global": route_global,
            "checkpoints": checkpoints,
            "checkpoint_set": set(checkpoints),
            "assigned_locations": assigned_locations,
            "assigned_stop_points": assigned_stop_points,
            "line_obj": line_obj,
            "arcs": arcs,
            "access_km": _line_access_km(line_obj, depot_pos_km, block_pos_km),
            "base_route_km": _route_execution_km(
                line_obj, depot_pos_km, block_pos_km, method.K_skip
            ),
        })

        for cp in checkpoints:
            candidate_service_info[cp] = {
                "kind": "checkpoint",
                "root": root,
            }
        for stop in assigned_stop_points:
            candidate_service_info[stop] = {
                "kind": "stop_point",
                "root": root,
            }

    riders = _sample_rider_destinations(
        demand_s,
        demand_support_locations,
        block_pos_km,
        areas_km2,
        walk_rng,
    )
    (
        projected_counts,
        projected_stop_counts_by_root,
        projected_checkpoint_counts,
    ) = _project_riders_to_candidate_locations(
        riders, candidate_service_info, block_pos_km
    )

    service_point_info: Dict[str, Dict[str, object]] = {}
    route_services = {}
    discount = TSP_TRAVEL_DISCOUNT
    speed = VEHICLE_SPEED_KMH

    for rd in route_data:
        root = rd["root"]
        checkpoints = rd["checkpoints"]
        cp_set = rd["checkpoint_set"]
        assigned_stop_points = rd["assigned_stop_points"]
        T_i = rd["T_i"]
        projected_stop_counts = {
            stop: int(projected_stop_counts_by_root.get(root, {}).get(stop, 0))
            for stop in assigned_stop_points
            if projected_stop_counts_by_root.get(root, {}).get(stop, 0) > 0
        }
        active_stop_points = sorted(projected_stop_counts.keys())
        all_blocks_set = set(checkpoints) | set(active_stop_points)

        projected_demand_s = {cp: int(projected_checkpoint_counts.get(cp, 0)) for cp in checkpoints}
        for stop, count in projected_stop_counts.items():
            projected_demand_s[stop] = int(count)

        eval_cost_mult = (
            vehicle_cost_weight / (discount * T_i)
            + rider_time_weight / (2.0 * speed)
        )
        walk_penalties = {
            b: (
                method.coverage_penalty_factor
                * method.walk_time_weight
                * rider_time_weight
                * walk_saved_km.get(b, 0.0)
                * projected_demand_s.get(b, 0)
                / method.walk_speed_kmh
            )
            for b in active_stop_points
        }

        route_km_s = rd["base_route_km"]
        active_arcs = []
        served_stops: Set[str] = set()
        if len(checkpoints) > 1 and rd["arcs"] and active_stop_points:
            z_star = {b: 1.0 for b in all_blocks_set}
            result = _solve_routing_lp_with_cg(
                rd["arcs"], rd["line_obj"], block_ids, block_pos_km,
                len(checkpoints), method.capacity, z_star, projected_demand_s,
                all_blocks_set, cp_set, eval_cost_mult, walk_penalties,
                method.delta, speed, method.alpha_time, method.K_skip,
                max_cg_iters=method.max_cg_iters, return_solution=True,
                return_active_arcs=True,
            )
            if result[0] is not None:
                _, _, detour_km, route_u_vals, active_arcs = result
                if detour_km is not None:
                    route_km_s = rd["access_km"] + detour_km
                served_stops = {
                    stop for stop in active_stop_points
                    if route_u_vals.get(stop, 1.0) < 0.5
                }

        arrival_h = _build_service_arrival_times(
            depot_pos_km, checkpoints, block_pos_km, active_arcs, speed
        )
        for cp in checkpoints:
            cp_info = {
                "kind": "checkpoint",
                "arrival_h": float(arrival_h.get(cp, 0.0)),
                "headway_h": float(T_i),
                "root": root,
            }
            prev = service_point_info.get(cp)
            if prev is None or cp_info["arrival_h"] < float(prev.get("arrival_h", 1e18)):
                service_point_info[cp] = cp_info
        for stop in served_stops:
            stop_info = {
                "kind": "detour_stop",
                "arrival_h": float(arrival_h.get(stop, 0.0)),
                "headway_h": float(T_i),
                "root": root,
            }
            prev = service_point_info.get(stop)
            if prev is None or stop_info["arrival_h"] < float(prev.get("arrival_h", 1e18)):
                service_point_info[stop] = stop_info

        skip_arcs = [(arc, flow) for arc, flow in active_arcs if arc.cp_to > arc.cp_from + 1]
        route_services[root] = {
            "root": root,
            "checkpoints": checkpoints,
            "assigned_locations": rd["assigned_locations"],
            "assigned_stop_points": assigned_stop_points,
            "active_stop_points": active_stop_points,
            "projected_stop_counts": projected_stop_counts,
            "served_stops": served_stops,
            "skip_arcs": skip_arcs,
            "route_global": rd["route_global"],
            "route_km_s": route_km_s,
        }

    final_result = _finalize_projected_rider_assignments(
        riders,
        block_pos_km,
        service_point_info,
        walk_rng,
        return_assignments=return_assignments,
    )
    (
        total_walk_km,
        total_wait_h,
        total_invehicle_h,
        service_counts,
        total_riders,
        demand_outcomes,
        *rest,
    ) = final_result
    assignment_records = rest[0] if rest else None

    route_km_by_root = {root: float(info["route_km_s"]) for root, info in route_services.items()}
    return {
        "route_services": route_services,
        "service_point_info": service_point_info,
        "service_counts": service_counts,
        "demand_outcomes": demand_outcomes,
        "projected_counts": projected_counts,
        "projected_checkpoint_counts": projected_checkpoint_counts,
        "projected_stop_counts_by_root": projected_stop_counts_by_root,
        "route_km_by_root": route_km_by_root,
        "riders": riders,
        "assignment_records": assignment_records,
        "total_walk_km": float(total_walk_km),
        "total_wait_h": float(total_wait_h),
        "total_invehicle_h": float(total_invehicle_h),
        "total_riders": int(total_riders),
    }


def _sample_global_service_assignments(
    demand_s: Dict[str, int],
    demand_blocks: List[str],
    block_pos_km: Dict[str, np.ndarray],
    areas_km2: Dict[str, float],
    service_point_info: Dict[str, Dict[str, object]],
    rng: np.random.Generator,
    return_assignments: bool = False,
):
    """Assign sampled passengers to the nearest realized service point globally.

    Parameters
    ----------
    demand_blocks : list
        Demand-support locations with positive probability.
    service_point_info : dict
        service_location_id -> {
            "kind": "checkpoint" | "detour_stop",
            "arrival_h": float,
            "headway_h": float,
            "root": str,
        }

    Returns
    -------
    total_walk_riders_km : float
    total_wait_h : float
    total_invehicle_h : float
    service_counts : Dict[str, int]
    total_riders : int
    demand_outcomes : Dict[str, Dict[str, int]]
        Per-demand-block counts for:
        - checkpoint_served
        - detour_stop_served
        - checkpoint_fallback
        - served_here_checkpoint
        - served_here_detour_stop
        - walk_to_service
        - walk_to_checkpoint
        - walk_to_detour_stop
    return_assignments : bool, optional
        When true, also return a list of rider-level assignment records for
        one-scenario visualization.
    """
    if not service_point_info:
        if return_assignments:
            return 0.0, 0.0, 0.0, {}, 0, {}, []
        return 0.0, 0.0, 0.0, {}, 0, {}

    service_locs = list(service_point_info.keys())
    service_pos = np.array([block_pos_km[b] for b in service_locs], dtype=float)

    total_walk_riders = 0.0
    total_wait_h = 0.0
    total_invehicle_h = 0.0
    total_riders = 0
    service_counts: Dict[str, int] = {}
    demand_outcomes: Dict[str, Dict[str, int]] = {}
    assignment_records = [] if return_assignments else None

    selected_checkpoint_set = {
        loc for loc, info in service_point_info.items()
        if info.get("kind") == "checkpoint"
    }

    for b in demand_blocks:
        d_b = int(demand_s.get(b, 0))
        if d_b <= 0:
            continue
        total_riders += d_b
        demand_outcomes[b] = {
            "checkpoint_served": 0,
            "detour_stop_served": 0,
            "checkpoint_fallback": 0,
            "served_here_checkpoint": 0,
            "served_here_detour_stop": 0,
            "walk_to_service": 0,
            "walk_to_checkpoint": 0,
            "walk_to_detour_stop": 0,
        }

        center = block_pos_km[b]
        area_b = areas_km2.get(b, 0.25)
        r_b = math.sqrt(area_b / math.pi)
        if r_b < 1e-9:
            positions = np.tile(center, (d_b, 1))
        else:
            radii = r_b * np.sqrt(rng.uniform(0, 1, size=d_b))
            angles = rng.uniform(0, 2 * math.pi, size=d_b)
            offsets = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            positions = center + offsets

        dist_to_service = np.linalg.norm(
            positions[:, None, :] - service_pos[None, :, :], axis=2
        )
        nearest_service_idx = dist_to_service.argmin(axis=1)
        min_service_dist = dist_to_service.min(axis=1)

        for p in range(d_b):
            service_loc = service_locs[int(nearest_service_idx[p])]
            info = service_point_info[service_loc]
            total_walk_riders += float(min_service_dist[p])
            total_invehicle_h += float(info.get("arrival_h", 0.0))
            headway_h = float(info.get("headway_h", 0.0))
            if headway_h > 0:
                total_wait_h += headway_h - float(rng.uniform(0.0, headway_h))
            service_counts[service_loc] = service_counts.get(service_loc, 0) + 1

            served_here = (service_loc == b)
            if not served_here:
                demand_outcomes[b]["walk_to_service"] += 1

            if info.get("kind") == "detour_stop":
                demand_outcomes[b]["detour_stop_served"] += 1
                if served_here:
                    demand_outcomes[b]["served_here_detour_stop"] += 1
                else:
                    demand_outcomes[b]["walk_to_detour_stop"] += 1
            elif b in selected_checkpoint_set:
                demand_outcomes[b]["checkpoint_served"] += 1
                if served_here:
                    demand_outcomes[b]["served_here_checkpoint"] += 1
                else:
                    demand_outcomes[b]["walk_to_checkpoint"] += 1
            else:
                demand_outcomes[b]["checkpoint_fallback"] += 1
                demand_outcomes[b]["walk_to_checkpoint"] += 1

            if return_assignments:
                assignment_records.append(
                    {
                        "origin_block": b,
                        "origin_pos": np.array(positions[p], dtype=float),
                        "service_loc": service_loc,
                        "service_pos": np.array(block_pos_km[service_loc], dtype=float),
                        "service_kind": str(info.get("kind", "")),
                        "walk_km": float(min_service_dist[p]),
                        "walked_elsewhere": not served_here,
                    }
                )

    result = (
        total_walk_riders,
        total_wait_h,
        total_invehicle_h,
        service_counts,
        total_riders,
        demand_outcomes,
    )
    if return_assignments:
        return result + (assignment_records,)
    return result


def _ordered_subpath_locations(
    start_loc: str,
    end_loc: str,
    served_locations: List[str],
    block_pos_km: Dict[str, np.ndarray],
) -> List[str]:
    """Greedy service ordering between two checkpoints.

    Mirrors the NN heuristic used in `_subpath_cost`.
    """
    if not served_locations:
        return []

    remaining = list(served_locations)
    ordered: List[str] = []
    cur = np.array(block_pos_km[start_loc], dtype=float)
    while remaining:
        dists = [float(np.linalg.norm(block_pos_km[loc] - cur)) for loc in remaining]
        idx = int(np.argmin(dists))
        nxt = remaining.pop(idx)
        ordered.append(nxt)
        cur = np.array(block_pos_km[nxt], dtype=float)
    return ordered


def _build_service_arrival_times(
    depot_pos_km: np.ndarray,
    checkpoints: List[str],
    block_pos_km: Dict[str, np.ndarray],
    active_arc_flows,
    speed_kmh: float,
) -> Dict[str, float]:
    """Arrival time at each service point along the realized route."""
    if not checkpoints:
        return {}

    arrival_h: Dict[str, float] = {}
    cur_pos = depot_pos_km.copy()
    cur_time = 0.0

    first_cp = checkpoints[0]
    cur_time += float(np.linalg.norm(block_pos_km[first_cp] - cur_pos)) / speed_kmh
    arrival_h[first_cp] = cur_time
    cur_pos = np.array(block_pos_km[first_cp], dtype=float)

    if not active_arc_flows:
        for cp in checkpoints[1:]:
            nxt = np.array(block_pos_km[cp], dtype=float)
            cur_time += float(np.linalg.norm(nxt - cur_pos)) / speed_kmh
            arrival_h[cp] = cur_time
            cur_pos = nxt
        return arrival_h

    active_by_from = {}
    for arc, flow in active_arc_flows:
        if flow <= 1e-9:
            continue
        prev = active_by_from.get(arc.cp_from)
        if prev is None or flow > prev[1]:
            active_by_from[arc.cp_from] = (arc, flow)

    cp_idx = 0
    while cp_idx < len(checkpoints) - 1:
        selected = active_by_from.get(cp_idx)
        if selected is None:
            next_idx = cp_idx + 1
            next_cp = checkpoints[next_idx]
            nxt = np.array(block_pos_km[next_cp], dtype=float)
            cur_time += float(np.linalg.norm(nxt - cur_pos)) / speed_kmh
            arrival_h[next_cp] = cur_time
            cur_pos = nxt
            cp_idx = next_idx
            continue

        arc = selected[0]
        end_idx = min(max(arc.cp_to, cp_idx + 1), len(checkpoints) - 1)
        end_cp = checkpoints[end_idx]
        served = [loc for loc in arc.served_locations if loc != checkpoints[cp_idx] and loc != end_cp]
        ordered = _ordered_subpath_locations(checkpoints[cp_idx], end_cp, served, block_pos_km)
        for loc in ordered:
            nxt = np.array(block_pos_km[loc], dtype=float)
            cur_time += float(np.linalg.norm(nxt - cur_pos)) / speed_kmh
            arrival_h[loc] = cur_time
            cur_pos = nxt
        nxt = np.array(block_pos_km[end_cp], dtype=float)
        cur_time += float(np.linalg.norm(nxt - cur_pos)) / speed_kmh
        arrival_h[end_cp] = cur_time
        cur_pos = nxt
        cp_idx = end_idx

    return arrival_h


def _fr_detour_tour_stochastic(
    points_km, depot_km, areas_km2, delta, prob_weights,
    Lambda, T, route_order, wr=1.0,
    discount=TSP_TRAVEL_DISCOUNT, num_scenarios=200, rng=None,
):
    """Stochastic tour under Poisson demand with δ-corridor deviation.

    Returns (expected_tour_km, demand_weighted_avg_walk_km).
    """
    n = len(points_km)
    if n == 0:
        return 0.0, 0.0
    corridor_r, walk_det, walk_no_det = _block_geometry(areas_km2, delta)
    walk_saved = walk_no_det - walk_det
    walk_speed = 5.0
    rates = Lambda * T * prob_weights
    if rng is None:
        rng = np.random.default_rng(42)
    ns = max(num_scenarios, 200)
    acc_tour = 0.0
    acc_walk_riders = 0.0
    acc_riders = 0.0
    for _ in range(ns):
        demand = rng.poisson(rates)
        has_demand = demand > 0
        if not np.any(has_demand):
            continue
        n_total = float(demand.sum())
        tsp_km = _fixed_route_tour_km(route_order, points_km, depot_km, has_demand)
        if delta > 0:
            active = np.where(has_demand)[0]
            n_j = demand[active].astype(float)
            cr = corridor_r[active]
            mini_km = np.where(
                n_j > 1,
                BETA * np.sqrt(n_j * math.pi) * cr,
                2.0 * walk_saved[active],
            )
            # Marginal total-cost impact of detouring to block j.
            # Walk is a per-rider average in evaluate(), so the marginal
            # walk saving from detouring to j is walk_saved_j * n_j / n_total
            # (contribution to the demand-weighted average).
            benefit = wr * walk_saved[active] * n_j / (n_total * walk_speed)
            cost = (mini_km / (discount * T)
                    + wr * mini_km / (2.0 * VEHICLE_SPEED_KMH))
            deviate = benefit > cost
            acc_tour += tsp_km + float(mini_km[deviate].sum())
            wk = np.where(deviate, walk_det[active], walk_no_det[active])
            acc_walk_riders += float((wk * n_j).sum())
        else:
            acc_tour += tsp_km
            active = np.where(has_demand)[0]
            acc_walk_riders += float((walk_no_det[active]
                                     * demand[active].astype(float)).sum())
        acc_riders += n_total
    tour_km = acc_tour / ns
    avg_walk_km = acc_walk_riders / max(acc_riders, 1.0)
    return tour_km, avg_walk_km


# ---------------------------------------------------------------------------
# Line library generation (Step 1)
# ---------------------------------------------------------------------------

def _point_to_segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance from point p to line segment [a, b]."""
    ab = b - a
    ab_sq = float(np.dot(ab, ab))
    if ab_sq < 1e-20:
        return float(np.linalg.norm(p - a))
    t = max(0.0, min(1.0, float(np.dot(p - a, ab)) / ab_sq))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def _compute_reachable(checkpoints, block_ids, block_pos_km, delta,
                       speed_kmh=VEHICLE_SPEED_KMH, alpha_time=0.5):
    """Blocks within delta of a route segment AND time-feasible to detour to.

    A block is reachable if:
      1. it is within delta of some checkpoint-to-checkpoint segment, AND
      2. a single-block detour from one segment endpoint through the block
         to the other endpoint fits within that segment's time budget.

    This tightens the purely geometric check to match what _build_initial_arcs
    can actually produce as a feasible subpath arc.
    """
    reachable = set(checkpoints)
    if delta <= 0:
        return reachable
    checkpoint_schedule_h = _checkpoint_schedule_offsets(
        list(checkpoints), block_pos_km, speed_kmh=speed_kmh, alpha_time=alpha_time
    )
    for b in block_ids:
        if b in reachable:
            continue
        b_pos = block_pos_km[b]
        # Check each consecutive segment
        for k in range(len(checkpoints) - 1):
            ci_pos = block_pos_km[checkpoints[k]]
            cj_pos = block_pos_km[checkpoints[k + 1]]
            # Geometric check: within delta of segment
            d = _point_to_segment_dist(b_pos, ci_pos, cj_pos)
            if d > delta + 1e-9:
                continue
            # Time-feasibility check: detour ci → b → cj within time budget
            time_budget = _trip_time_budget(checkpoint_schedule_h, k, k + 1)
            detour_km = (float(np.linalg.norm(b_pos - ci_pos))
                         + float(np.linalg.norm(cj_pos - b_pos)))
            detour_time = detour_km / speed_kmh if speed_kmh > 0 else 0.0
            if detour_time <= time_budget + 1e-9:
                reachable.add(b)
                break
        if b in reachable:
            continue
        # Also check endpoint-only reachability (blocks near a single CP)
        for cp in checkpoints:
            if float(np.linalg.norm(b_pos - block_pos_km[cp])) <= delta + 1e-9:
                reachable.add(b)
                break
    return reachable


def _tour_distance(depot_km, checkpoints, block_pos_km):
    """Depot → checkpoints → depot distance (km)."""
    if not checkpoints:
        return 0.0
    pts = np.vstack([depot_km[None]]
                    + [block_pos_km[c][None] for c in checkpoints]
                    + [depot_km[None]])
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _generate_line_library(geodata, depot_id, block_ids, n_angular,
                           num_districts=None):
    """Generate candidate reference lines from depot through checkpoint subsets.

    Routes are built through checkpoints only.  Stopping locations (non-
    checkpoint blocks) are served via corridor detours in the second stage.

    Includes sector lines, merged-sector lines, partition lines (sized for
    K-way coverage), and singleton fallbacks.
    """
    N = len(block_ids)
    depot_pos = np.array(_get_pos(geodata, depot_id), dtype=float) / 1000.0
    block_pos_km = {}
    for b in block_ids:
        block_pos_km[b] = np.array(_get_pos(geodata, b), dtype=float) / 1000.0

    # Use only checkpoint IDs for building route geometry.
    # Stopping locations are reachable via corridor detours.
    cp_ids = getattr(geodata, 'checkpoint_ids', block_ids)

    # Angular position of each checkpoint relative to depot
    positions = np.array([block_pos_km[b] for b in cp_ids])
    dx = positions[:, 0] - depot_pos[0]
    dy = positions[:, 1] - depot_pos[1]
    angles = np.arctan2(dy, dx)

    # Sort checkpoints by angle
    N_cp = len(cp_ids)
    order = np.argsort(angles)
    sorted_blocks = [cp_ids[i] for i in order]

    def _make_line(block_list, idx_counter):
        """Helper to create a ReferenceLine from a block list."""
        pts = np.array([block_pos_km[b] for b in block_list])
        nn = _greedy_nn_route(pts, depot_pos)
        nn = _two_opt_improve(nn, pts, depot_pos)
        cps = [block_list[i] for i in nn]
        km = _tour_distance(depot_pos, cps, block_pos_km)
        return ReferenceLine(idx=idx_counter, checkpoints=cps,
                             reachable_locations=set(), est_tour_km=km)

    lines: List[ReferenceLine] = []
    idx = 0

    # Build sector lists (from checkpoints only)
    n_ang = max(n_angular, 1)
    sector_lists: List[List[str]] = [[] for _ in range(n_ang)]
    for rank, b in enumerate(sorted_blocks):
        sector_lists[rank * n_ang // N_cp].append(b)

    # 1. Sector lines (single sectors)
    for sector in sector_lists:
        if not sector:
            continue
        lines.append(_make_line(sector, idx))
        idx += 1

    # 2. Merged adjacent sectors (pairs)
    for s in range(n_ang):
        merged = sector_lists[s] + sector_lists[(s + 1) % n_ang]
        if len(merged) <= 1:
            continue
        lines.append(_make_line(merged, idx))
        idx += 1

    # 3. Triple-merged sectors (for larger coverage per line)
    for s in range(n_ang):
        merged = (sector_lists[s]
                  + sector_lists[(s + 1) % n_ang]
                  + sector_lists[(s + 2) % n_ang])
        if len(merged) <= 1:
            continue
        lines.append(_make_line(merged, idx))
        idx += 1

    # 4. Partition lines: divide checkpoints into K equal groups by angle.
    if num_districts is not None and num_districts > 0:
        K = num_districts
        part_size = max(1, (N_cp + K - 1) // K)
        for g in range(K):
            group = sorted_blocks[g * part_size: min((g + 1) * part_size, N_cp)]
            if not group:
                continue
            lines.append(_make_line(group, idx))
            idx += 1

    # 5. Singleton fallback lines (every checkpoint reachable by at least one line)
    for b in cp_ids:
        est_km = 2.0 * float(np.linalg.norm(block_pos_km[b] - depot_pos))
        lines.append(ReferenceLine(idx=idx, checkpoints=[b],
                                   reachable_locations=set(), est_tour_km=est_km))
        idx += 1

    return lines, block_pos_km, depot_pos


def _apply_line_reachability(
    candidate_lines: List[ReferenceLine], block_ids: List[str],
    block_pos_km: Dict[str, np.ndarray], delta: float,
    speed_kmh=VEHICLE_SPEED_KMH, alpha_time=0.5,
) -> List[ReferenceLine]:
    """Clone candidate lines with delta-specific assignment compatibility."""
    realized: List[ReferenceLine] = []
    for line in candidate_lines:
        reachable = _compute_reachable(
            line.checkpoints, block_ids, block_pos_km, delta, speed_kmh, alpha_time)
        realized.append(ReferenceLine(
            idx=line.idx,
            checkpoints=list(line.checkpoints),
            reachable_locations=reachable,
            est_tour_km=line.est_tour_km,
        ))
    return realized


# ---------------------------------------------------------------------------
# Subpath arc enumeration (Step 2a)
# ---------------------------------------------------------------------------

def _subpath_cost(pos_from, pos_to, served_pos_list):
    """NN route distance from pos_from through served points to pos_to."""
    if not served_pos_list:
        return float(np.linalg.norm(pos_to - pos_from))
    cur = pos_from.copy()
    total = 0.0
    remaining = list(range(len(served_pos_list)))
    for _ in range(len(served_pos_list)):
        dists = [float(np.linalg.norm(served_pos_list[j] - cur)) for j in remaining]
        best_idx = int(np.argmin(dists))
        total += dists[best_idx]
        cur = served_pos_list[remaining[best_idx]]
        remaining.pop(best_idx)
    total += float(np.linalg.norm(pos_to - cur))
    return total


def _build_initial_arcs(line, block_ids, block_pos_km, delta, K_skip,
                        speed_kmh, alpha_time):
    """Build seed subpath arcs for column generation.

    Returns list of SubpathArc objects. These are the initial columns for the
    restricted routing LP; column generation adds more as needed.

    Endpoint convention: cp_i and cp_j are NOT in served_locations (they are
    visited by flow-balance). Skipped checkpoints between them ARE in
    served_locations.
    """
    checkpoints = line.checkpoints
    cp_set = set(checkpoints)
    m = len(checkpoints)
    if m <= 1:
        return []  # singleton: no inter-checkpoint arcs
    checkpoint_schedule_h = _checkpoint_schedule_offsets(
        checkpoints, block_pos_km, speed_kmh=speed_kmh, alpha_time=alpha_time
    )

    arcs: List[SubpathArc] = []
    arc_id = 0

    for i in range(m):
        for j in range(i + 1, min(i + K_skip + 2, m)):
            ci_pos = block_pos_km[checkpoints[i]]
            cj_pos = block_pos_km[checkpoints[j]]

            # Time budget for this segment
            time_budget = _trip_time_budget(checkpoint_schedule_h, i, j)

            # Skipped checkpoints (always in served_locations)
            skipped_cps = frozenset(
                checkpoints[k] for k in range(i + 1, j))

            # Corridor blocks: off-checkpoint blocks within δ of segment,
            # restricted to blocks reachable by this line (avoids leaking
            # demand from blocks assigned to other routes into arc loads).
            corridor: List[str] = []
            if delta > 0:
                reachable = line.reachable_locations
                for b in block_ids:
                    if b in cp_set or b not in reachable:
                        continue
                    d = _point_to_segment_dist(block_pos_km[b], ci_pos, cj_pos)
                    if d <= delta + 1e-9:
                        corridor.append(b)

            # --- Seed arc 1: direct (serves skipped CPs only) ---
            direct_km = float(np.linalg.norm(cj_pos - ci_pos))
            direct_time = direct_km / speed_kmh if speed_kmh > 0 else 0.0
            if direct_time <= time_budget + 1e-9:
                arcs.append(SubpathArc(
                    arc_id=arc_id, cp_from=i, cp_to=j,
                    cost_km=direct_km, travel_time_h=direct_time,
                    served_locations=skipped_cps,
                ))
                arc_id += 1

            # --- Seed arcs: single-block deviations ---
            for b in corridor:
                served = skipped_cps | {b}
                served_pos = [block_pos_km[b]]
                cost = _subpath_cost(ci_pos, cj_pos, served_pos)
                travel_time = cost / speed_kmh if speed_kmh > 0 else 0.0
                if travel_time <= time_budget + 1e-9:
                    arcs.append(SubpathArc(
                        arc_id=arc_id, cp_from=i, cp_to=j,
                        cost_km=cost, travel_time_h=travel_time,
                        served_locations=frozenset(served),
                    ))
                    arc_id += 1

            # --- Seed arcs: small multi-block combinations (≤ 3) ---
            # Cap at nearest 6 corridor blocks to avoid combinatorial explosion
            # when the reachable set is large.  CG pricing discovers the rest.
            if len(corridor) > 6:
                seg_mid = (ci_pos + cj_pos) / 2.0
                dists = [float(np.linalg.norm(block_pos_km[b] - seg_mid))
                         for b in corridor]
                nearest = sorted(range(len(corridor)), key=lambda i: dists[i])[:6]
                corridor_enum = [corridor[i] for i in nearest]
            else:
                corridor_enum = corridor
            max_enum = min(len(corridor_enum), 3)
            for size in range(2, max_enum + 1):
                for subset in combinations(range(len(corridor_enum)), size):
                    blocks_in_subset = [corridor_enum[s] for s in subset]
                    served = skipped_cps | frozenset(blocks_in_subset)
                    served_pos = [block_pos_km[b] for b in blocks_in_subset]
                    cost = _subpath_cost(ci_pos, cj_pos, served_pos)
                    travel_time = cost / speed_kmh if speed_kmh > 0 else 0.0
                    if travel_time <= time_budget + 1e-9:
                        arcs.append(SubpathArc(
                            arc_id=arc_id, cp_from=i, cp_to=j,
                            cost_km=cost, travel_time_h=travel_time,
                            served_locations=frozenset(served),
                        ))
                        arc_id += 1

    return arcs


# ---------------------------------------------------------------------------
# Restricted routing LP on load-expanded checkpoint network
# ---------------------------------------------------------------------------

def _solve_restricted_routing_lp(arcs, checkpoints, m, capacity, z_star,
                                  demand_s, all_blocks, cp_set, cost_mult,
                                  walk_penalties, return_solution=False,
                                  return_active_arcs=False):
    """Solve the restricted subpath routing LP for one scenario on one line.

    Load is measured in **passengers** (scenario-dependent), not blocks.
    Checkpoint demand enters load transitions: each arc's load delta includes
    demand at its destination checkpoint, and the source includes demand at
    the first checkpoint.

    Walking cost model: u[b] penalty = walk_saved_cost[b] (the cost of NOT
    detouring to serve stopping location b — passengers walk to a checkpoint instead).
    This replaces the former big-M penalty and lets the optimizer see the
    detour-vs-walking trade-off.

    Args:
        arcs: list of SubpathArc (current column pool)
        checkpoints: ordered list of checkpoint block IDs
        m: number of checkpoints
        capacity: vehicle passenger capacity
        z_star: dict location_id → assignment value from master
        demand_s: dict location_id → passenger count for this scenario
        all_blocks: set of candidate stopping locations that participate in linking/coverage
        cp_set: set of all checkpoint IDs
        cost_mult: km → cost conversion factor
        walk_penalties: dict location_id → penalty for u[b] (walking cost saved
            by detouring = wr * walk_saved_km * demand_s / walk_speed)

    Returns:
        (obj, duals_dict) where duals_dict has keys 'flow', 'link', 'coverage'
        or (None, None) if infeasible.
    """
    import gurobipy as gp
    from gurobipy import GRB

    model = gp.Model("routing_lp")
    model.setParam('OutputFlag', 0)

    assigned_blocks = {
        b for b in all_blocks if z_star.get(b, 0.0) > 1e-9 and demand_s.get(b, 0) > 0
    }
    # Fixed-origin MiND-VRP semantics: load starts full and decreases as riders
    # are dropped off along the realized subpath route.
    cp_demand = [
        demand_s.get(checkpoints[c], 0)
        if checkpoints[c] in assigned_blocks else 0
        for c in range(m)
    ]
    total_assigned = int(sum(demand_s.get(b, 0) for b in assigned_blocks))
    start_load = total_assigned - (cp_demand[0] if m > 0 else 0)
    if start_load < 0 or start_load > capacity:
        return None, None

    # Compute scenario-dependent delivered load per arc.
    arc_load: Dict[int, int] = {}
    for a in arcs:
        arc_load[a.arc_id] = (
            sum(demand_s.get(b, 0) for b in a.served_locations if b in assigned_blocks)
            + cp_demand[a.cp_to]
        )

    # Build arc-id-to-arc lookup
    arc_by_id: Dict[int, SubpathArc] = {a.arc_id: a for a in arcs}

    # Pre-compute incidence: which arcs serve each block.
    # For checkpoint endpoints, service = node visitation in the path.
    # We proxy that with outgoing arcs for non-terminal checkpoints and
    # incoming arcs for the final checkpoint.
    cp_index = {checkpoints[c]: c for c in range(m)}
    arcs_serving: Dict[str, List[SubpathArc]] = {}
    for b in all_blocks:
        # Standard: arcs that have b in served_locations
        serving = [a for a in arcs if b in a.served_locations]
        if serving:
            arcs_serving[b] = serving
        elif b in cp_index:
            c = cp_index[b]
            if c == m - 1:
                arcs_serving[b] = [a for a in arcs if a.cp_to == c]
            else:
                arcs_serving[b] = [a for a in arcs if a.cp_from == c]
        else:
            arcs_serving[b] = []

    # --- Variables ---
    # y[arc_id, load_from] for each arc and valid starting load.
    # Here load_from denotes remaining onboard passengers before traversing arc.
    y: Dict[Tuple[int, int], object] = {}
    for a in arcs:
        ld = arc_load[a.arc_id]
        for l in range(ld, capacity + 1):
            y[(a.arc_id, l)] = model.addVar(lb=0.0, name=f"y_{a.arc_id}_{l}")

    # Terminal arcs: (last_checkpoint, load) → super-sink
    term: Dict[int, object] = {}
    for l in range(capacity + 1):
        term[l] = model.addVar(lb=0.0, name=f"term_{l}")

    # Unserved slack per block
    u: Dict[str, object] = {}
    for b in all_blocks:
        u[b] = model.addVar(lb=0.0, name=f"u_{b}")

    model.update()

    # --- (i) Flow balance at each (checkpoint, load) node ---
    flow_constrs: Dict[Tuple[int, int], object] = {}
    for c in range(m):
        for l in range(capacity + 1):
            outflow = gp.LinExpr()
            inflow = gp.LinExpr()

            # Outflow from (c, l)
            for a in arcs:
                ld = arc_load[a.arc_id]
                if a.cp_from == c and l >= ld:
                    outflow += y[(a.arc_id, l)]
            if c == m - 1:
                outflow += term[l]

            # Inflow to (c, l)
            for a in arcs:
                ld = arc_load[a.arc_id]
                if a.cp_to == c:
                    l_from = l + ld
                    if ld <= l_from <= capacity:
                        inflow += y[(a.arc_id, l_from)]

            rhs = 1.0 if (c == 0 and l == start_load) else 0.0
            flow_constrs[(c, l)] = model.addConstr(
                outflow - inflow == rhs, name=f"flow_{c}_{l}")

    # --- (ii) Super-sink ---
    sink_constr = model.addConstr(
        gp.quicksum(term[l] for l in range(capacity + 1)) == 1.0,
        name="sink")

    # Pre-index y-variables by arc_id for fast lookup
    y_by_arc: Dict[int, List[object]] = {}
    for (aid, l), var in y.items():
        y_by_arc.setdefault(aid, []).append(var)

    # Helper to build service expression for a block
    def _service_expr(b):
        """Sum of y-variables that indicate block b is served."""
        serving = arcs_serving.get(b, [])
        if not serving:
            return gp.LinExpr()
        expr = gp.LinExpr()
        for a in serving:
            for var in y_by_arc.get(a.arc_id, []):
                expr += var
        return expr

    # --- (iii) Linking: Σ service(b) ≤ z*[b] for all blocks ---
    link_constrs: Dict[str, object] = {}
    for b in all_blocks:
        zval = z_star.get(b, 0.0)
        expr = _service_expr(b)
        if expr.size() > 0:
            link_constrs[b] = model.addConstr(expr <= zval, name=f"link_{b}")

    # --- (iv) Coverage with slack: service(b) + u[b] ≥ z*[b] ---
    # Only for blocks with demand in this scenario.
    cov_constrs: Dict[str, object] = {}
    for b in all_blocks:
        if demand_s.get(b, 0) <= 0:
            continue
        zval = z_star.get(b, 0.0)
        if zval < 1e-9:
            continue
        expr = _service_expr(b)
        cov_constrs[b] = model.addConstr(
            expr + u[b] >= zval, name=f"cov_{b}")

    # --- Objective ---
    # Vehicle routing cost + walking penalty for unserved blocks.
    # walk_penalties[b] = wr * walk_saved_km[b] * demand_s[b] / walk_speed
    # This makes the optimizer balance detour cost vs walking cost.
    obj = gp.LinExpr()
    for a in arcs:
        ld = arc_load[a.arc_id]
        for l in range(ld, capacity + 1):
            obj += cost_mult * a.cost_km * y[(a.arc_id, l)]
    for b in all_blocks:
        wp = walk_penalties.get(b, 0.0)
        obj += wp * u[b]
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        return None, None

    # Extract structured duals
    duals = {
        'flow': {k: c.Pi for k, c in flow_constrs.items()},
        'link': {b: c.Pi for b, c in link_constrs.items()},
        'coverage': {b: c.Pi for b, c in cov_constrs.items()},
    }

    if return_solution:
        # Extract primal solution: route_km and unserved slack per block
        route_km = 0.0
        active_arc_flows = []
        for a in arcs:
            ld = arc_load[a.arc_id]
            arc_flow = 0.0
            for l in range(ld, capacity + 1):
                if (a.arc_id, l) in y:
                    flow_val = y[(a.arc_id, l)].X
                    route_km += a.cost_km * flow_val
                    arc_flow += flow_val
            if return_active_arcs and arc_flow > 1e-9:
                active_arc_flows.append((a, arc_flow))
        u_vals = {b: u[b].X for b in all_blocks}
        if return_active_arcs:
            return model.ObjVal, duals, route_km, u_vals, active_arc_flows
        return model.ObjVal, duals, route_km, u_vals

    return model.ObjVal, duals


# ---------------------------------------------------------------------------
# Column generation pricing — label-setting DP
# ---------------------------------------------------------------------------

def _price_subpaths(line, block_ids, block_pos_km, m, capacity,
                    demand_s, duals, cost_mult, delta, speed_kmh,
                    alpha_time, K_skip):
    """Find negative-reduced-cost subpath arcs via label-setting DP.

    For each checkpoint segment (cp_i, cp_j), searches for a subpath with
    rc < -eps. Returns list of new SubpathArc objects to add to the pool.

    Reduced cost of arc a from cp_i to cp_j at starting remaining load l:
      rc(a, l) = cost_mult * a.cost_km
                 - (ν[i, l] - ν[j, l - δ_a])
                 - Σ_{b ∈ S_a} (π[b] + μ[b])

    DP label: (node, travel_km, travel_time_h, passenger_load, served_set)
    Resources: time ≤ time_budget, load ≤ capacity.
    """
    import heapq

    checkpoints = line.checkpoints
    cp_set = set(checkpoints)
    flow_d = duals.get('flow', {})
    link_d = duals.get('link', {})
    cov_d = duals.get('coverage', {})

    new_arcs: List[SubpathArc] = []
    eps = 1e-6
    checkpoint_schedule_h = _checkpoint_schedule_offsets(
        checkpoints, block_pos_km, speed_kmh=speed_kmh, alpha_time=alpha_time
    )

    for i in range(m):
        for j in range(i + 1, min(i + K_skip + 2, m)):
            ci_pos = block_pos_km[checkpoints[i]]
            cj_pos = block_pos_km[checkpoints[j]]
            time_budget = _trip_time_budget(checkpoint_schedule_h, i, j)

            # Skipped checkpoints (always in served_locations of any arc on this segment)
            skipped_cps = [checkpoints[k] for k in range(i + 1, j)]
            skipped_set = frozenset(skipped_cps)

            # Corridor blocks (off-checkpoint, within delta of segment,
            # restricted to blocks reachable by this line)
            corridor: List[str] = []
            if delta > 0:
                reachable = line.reachable_locations
                for b in block_ids:
                    if b in cp_set or b not in reachable:
                        continue
                    d = _point_to_segment_dist(block_pos_km[b], ci_pos, cj_pos)
                    if d <= delta + 1e-9:
                        corridor.append(b)

            # Candidate blocks the DP can visit (corridor only; skipped CPs are always served)
            if not corridor:
                continue  # direct arc already in pool; nothing to price

            # Dual reward per block: π[b] + μ[b]
            # π[b] ≤ 0 (linking), μ[b] ≥ 0 (coverage), both from Gurobi .Pi
            block_reward = {}
            for b in corridor:
                block_reward[b] = link_d.get(b, 0.0) + cov_d.get(b, 0.0)
            # Skipped CPs also get dual reward
            skipped_reward = sum(
                link_d.get(b, 0.0) + cov_d.get(b, 0.0) for b in skipped_cps)

            # Label: (neg_dual_reward_so_far, id, node, travel_km, travel_time,
            #         passenger_load, served_corridor_set)
            # We track corridor-only served in DP; skipped CPs added at completion.
            # Priority = travel_cost_so_far - corridor_dual_reward (want minimum).

            initial_cost = 0.0
            initial_reward = 0.0
            heap = [(0.0, 0, checkpoints[i], 0.0, 0.0, 0, frozenset())]
            # key for dominance: (node) -> list of (rc_partial, time, load, served)
            best_labels: Dict[str, List] = {}

            best_rc_found = -eps  # only want strictly negative
            best_arc = None

            label_id = 1
            max_labels = 5000  # bound computation

            while heap and label_id < max_labels:
                rc_partial, _, node, km, time_h, load, served = heapq.heappop(heap)

                # Try completing: go to cp_j
                cj_km = float(np.linalg.norm(cj_pos - block_pos_km[node]))
                cj_time = cj_km / speed_kmh if speed_kmh > 0 else 0.0
                total_km = km + cj_km
                total_time = time_h + cj_time

                if total_time <= time_budget + 1e-9:
                    # Full served set
                    full_served = skipped_set | served
                    # Scenario-dependent load delta:
                    # corridor load + skipped CP load + destination CP load
                    load_delta = (load
                                  + sum(demand_s.get(b, 0) for b in skipped_cps)
                                  + demand_s.get(checkpoints[j], 0))

                    # Evaluate reduced cost for each feasible starting load l
                    arc_travel_cost = cost_mult * total_km
                    total_dual_reward = (
                        rc_partial  # = -(corridor reward accumulated)
                        + arc_travel_cost
                    )
                    # Actually, let's compute rc properly:
                    corridor_reward = -rc_partial + km * 0  # rc_partial = cost*km_so_far - reward_so_far
                    # Simpler: recompute from scratch
                    total_reward = (
                        sum(block_reward.get(b, 0.0) for b in served)
                        + skipped_reward)

                    for l_start in range(load_delta, capacity + 1):
                        nu_i = flow_d.get((i, l_start), 0.0)
                        nu_j = flow_d.get((j, l_start - load_delta), 0.0)
                        rc = (cost_mult * total_km
                              - (nu_i - nu_j)
                              - total_reward)
                        if rc < best_rc_found:
                            best_rc_found = rc
                            best_arc = SubpathArc(
                                arc_id=-1, cp_from=i, cp_to=j,
                                cost_km=total_km,
                                travel_time_h=total_time,
                                served_locations=frozenset(full_served),
                            )

                # Extend: visit an unserved corridor block
                for b in corridor:
                    if b in served:
                        continue
                    b_pos = block_pos_km[b]
                    step_km = float(np.linalg.norm(b_pos - block_pos_km[node]))
                    step_time = step_km / speed_kmh if speed_kmh > 0 else 0.0
                    new_time = time_h + step_time

                    # Check time feasibility: can still reach cp_j?
                    remain_km = float(np.linalg.norm(cj_pos - b_pos))
                    remain_time = remain_km / speed_kmh if speed_kmh > 0 else 0.0
                    if new_time + remain_time > time_budget + 1e-9:
                        continue

                    new_load = load + demand_s.get(b, 0)
                    # Check capacity (including skipped CPs + dest CP load)
                    total_load_if_complete = (new_load
                        + sum(demand_s.get(cp, 0) for cp in skipped_cps)
                        + demand_s.get(checkpoints[j], 0))
                    if total_load_if_complete > capacity:
                        continue

                    new_km = km + step_km
                    new_served = served | {b}

                    # Partial reduced cost (travel cost - corridor reward so far)
                    new_rc_partial = (cost_mult * new_km
                                     - sum(block_reward.get(bb, 0.0)
                                           for bb in new_served))

                    # Dominance check
                    key = b
                    dominated = False
                    if key in best_labels:
                        for (old_rc, old_time, old_load, old_served) in best_labels[key]:
                            if (old_rc <= new_rc_partial + 1e-9
                                    and old_time <= new_time + 1e-9
                                    and old_load <= new_load
                                    and old_served >= new_served):
                                dominated = True
                                break
                    if dominated:
                        continue

                    # Remove dominated old labels
                    if key in best_labels:
                        best_labels[key] = [
                            x for x in best_labels[key]
                            if not (new_rc_partial <= x[0] + 1e-9
                                    and new_time <= x[1] + 1e-9
                                    and new_load <= x[2]
                                    and new_served >= x[3])]
                    else:
                        best_labels[key] = []
                    best_labels[key].append(
                        (new_rc_partial, new_time, new_load, new_served))

                    heapq.heappush(heap, (
                        new_rc_partial, label_id, b, new_km, new_time,
                        new_load, new_served))
                    label_id += 1

            if best_arc is not None:
                new_arcs.append(best_arc)

    return new_arcs


# ---------------------------------------------------------------------------
# Column generation wrapper
# ---------------------------------------------------------------------------

def _solve_routing_lp_with_cg(initial_arcs, line, block_ids, block_pos_km,
                               m, capacity, z_star, demand_s, all_blocks,
                               cp_set, cost_mult, walk_penalties,
                               delta, speed_kmh,
                               alpha_time, K_skip, max_cg_iters=30,
                               return_solution=False, return_active_arcs=False):
    """Solve routing LP with column generation for one (line, scenario).

    Iterates: solve restricted LP → price new subpath arcs → add to pool.
    Returns (lp_obj, benders_coefficients) by default.
    If return_solution=True, returns (lp_obj, benders_coefficients, route_km, u_vals).
    """
    checkpoints = line.checkpoints
    arc_pool = list(initial_arcs)

    for cg_iter in range(max_cg_iters):
        obj, duals = _solve_restricted_routing_lp(
            arc_pool, checkpoints, m, capacity, z_star, demand_s,
            all_blocks, cp_set, cost_mult, walk_penalties)

        if obj is None:
            if return_solution:
                return (None, None, None, None, None) if return_active_arcs else (None, None, None, None)
            return (None, None)

        # Price new arcs
        new_arcs = _price_subpaths(
            line, block_ids, block_pos_km, m, capacity,
            demand_s, duals, cost_mult, delta, speed_kmh,
            alpha_time, K_skip)

        if not new_arcs:
            break  # optimal — no negative RC arcs

        # Re-index and add to pool
        next_id = max(a.arc_id for a in arc_pool) + 1
        for a in new_arcs:
            a.arc_id = next_id
            next_id += 1
            arc_pool.append(a)

    # Final solve to get definitive duals (and optionally solution values)
    result = _solve_restricted_routing_lp(
        arc_pool, checkpoints, m, capacity, z_star, demand_s,
        all_blocks, cp_set, cost_mult, walk_penalties,
        return_solution=return_solution, return_active_arcs=return_active_arcs)

    if return_solution:
        if return_active_arcs:
            obj, duals, route_km, u_vals, active_arcs = result
        else:
            obj, duals, route_km, u_vals = result
        if obj is None:
            return (None, None, None, None, None) if return_active_arcs else (None, None, None, None)
    else:
        obj, duals = result
        if obj is None:
            return None, None

    # Benders cut coefficient for block b = π[b] + μ[b] (from LP duals)
    # Caller adds walk_det_cost[b] to get full coefficient.
    link_d = duals.get('link', {})
    cov_d = duals.get('coverage', {})
    benders_coeffs = {}
    for b in all_blocks:
        coeff = link_d.get(b, 0.0) + cov_d.get(b, 0.0)
        if abs(coeff) > 1e-12:
            benders_coeffs[b] = coeff

    if return_solution:
        if return_active_arcs:
            return obj, benders_coeffs, route_km, u_vals, active_arcs
        return obj, benders_coeffs, route_km, u_vals
    return obj, benders_coeffs


# ---------------------------------------------------------------------------
# CRN demand generation (Step 4 setup)
# ---------------------------------------------------------------------------

def _crn_demand(U_base, Lambda, T, prob_dict, block_ids):
    """Generate Poisson demand from common random numbers.

    U_base: (S, N) uniform draws in [0, 1).
    Returns: (S, N) integer demand array.
    """
    from scipy.stats import poisson as poisson_dist

    rates = np.array([prob_dict.get(b, 0.0) for b in block_ids]) * Lambda * T
    S, N = U_base.shape
    demand = np.zeros((S, N), dtype=int)
    U_safe = np.clip(U_base, 0.0, 1.0 - 1e-10)
    for j in range(N):
        if rates[j] > 1e-12:
            demand[:, j] = poisson_dist.ppf(U_safe[:, j], mu=rates[j]).astype(int)
    return demand


# ---------------------------------------------------------------------------
# FRDetour class
# ---------------------------------------------------------------------------

class FRDetour(BaselineMethod):
    """
    Multi Fixed-Route with Detour — MiND-structured adaptation.

    Double decomposition (Martin-Iradi et al. 2025, arXiv:2402.01265):
      Outer: SAA multi-cut Benders (master = line+interval selection MIP)
      Inner: column generation with label-setting DP pricing inside each
             Benders subproblem (subpath routing LP on load-expanded
             checkpoint network)

    Key adaptations (see module docstring for full details):
      - FR-consistent master objective (no linehaul/ODD; demand-weighted wait)
      - Block-level demand aggregation (not individual passengers)
      - Steady-state timing surrogate (not scheduled checkpoint times)
      - Line library via angular partition (not pre-existing network)
      - Shared candidate-library / CRN inputs for reproducible comparisons

    Parameters
    ----------
    delta          : max vehicle deviation from route (km); 0 = Multi-FR
    max_iters      : max Benders iterations
    num_scenarios  : Monte Carlo scenarios for stochastic evaluation
    max_fleet      : optional fleet budget (max simultaneous vehicles)
    capacity       : vehicle passenger capacity (passengers, not blocks)
    n_angular      : number of angular sectors for line library
    K_skip         : max checkpoints that can be skipped between subpath ends
    n_saa          : SAA scenarios for multi-cut Benders
    T_values       : list of candidate dispatch intervals (hours)
    alpha_time     : time slack factor for checkpoint segments
    max_cg_iters   : max column generation iterations per subproblem
    load_factor_kappa : first-stage load-factor tolerance
    walk_time_weight : walk-time weight relative to in-vehicle time
    solve_mode     : dd_relaxed | dd_ils | ub_dd
    name           : display name override
    """

    def __init__(
        self,
        delta: float = 0.3,
        max_iters: int = 10,
        walk_speed_kmh: float = 5.0,
        num_scenarios: int = 200,
        max_fleet: float | None = None,
        capacity: int = 15,
        n_angular: int = 8,
        K_skip: int = 2,
        n_saa: int = 30,
        T_values: List[float] | None = None,
        alpha_time: float = 0.5,
        max_cg_iters: int = 30,
        load_factor_kappa: float = 0.25,
        walk_time_weight: float = 5.0,
        solve_mode: str = "dd_relaxed",
        coverage_penalty_factor: float = 1.0,
        show_progress: bool = False,
        name: str | None = None,
    ):
        self.delta = delta
        self.max_iters = max_iters
        self.walk_speed_kmh = walk_speed_kmh
        self.num_scenarios = num_scenarios
        self.max_fleet = max_fleet
        self.capacity = capacity
        self.n_angular = n_angular
        self.K_skip = K_skip
        self.n_saa = n_saa
        self.T_values = T_values or [0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.75, 1.0]
        self.alpha_time = alpha_time
        self.max_cg_iters = max_cg_iters
        self.load_factor_kappa = float(load_factor_kappa)
        self.walk_time_weight = float(walk_time_weight)
        self.solve_mode = str(solve_mode)
        self.coverage_penalty_factor = float(coverage_penalty_factor)
        self.show_progress = bool(show_progress)
        self._name = name or ("Multi-FR-Detour" if delta > 0 else "Multi-FR")

    def _progress(self, total: int, desc: str):
        if self.show_progress and tqdm is not None:
            return tqdm(total=total, desc=desc, leave=False)
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _district_data(geodata, block_ids, assigned_blocks, depot_id, prob_dict):
        """Extract positions, areas, probs, K_i for one district."""
        depot_pos_km = np.array(_get_pos(geodata, depot_id), dtype=float) / 1000.0
        pts_km = np.array(
            [np.array(_get_pos(geodata, b), dtype=float) / 1000.0
             for b in assigned_blocks])
        areas_km2 = np.array(
            [float(geodata.get_area(b)) for b in assigned_blocks])
        prob_weights = np.array(
            [float(prob_dict.get(b, 0.0)) for b in assigned_blocks])
        K_i = min(geodata.get_dist(depot_id, b) for b in assigned_blocks)
        return depot_pos_km, pts_km, areas_km2, prob_weights, K_i

    def _prepare_shared_inputs(
        self,
        geodata,
        num_districts: int,
        candidate_lines=None,
        block_pos_km=None,
        depot_id: str | None = None,
        crn_uniforms: Optional[np.ndarray] = None,
    ) -> dict:
        """Prepare reusable candidate lines and CRN draws."""
        # GeoData still exposes spatial units through `short_geoid_list`, but
        # in FR-Detour we interpret those identifiers as candidate stop locations.
        block_ids = geodata.short_geoid_list
        if depot_id is None:
            depot_id = _default_depot_id(geodata, block_ids)
        depot_pos = np.array(_get_pos(geodata, depot_id), dtype=float) / 1000.0

        if candidate_lines is None or block_pos_km is None:
            generated_lines, generated_block_pos, _ = _generate_line_library(
                geodata, depot_id, block_ids, self.n_angular, num_districts=num_districts)
            if candidate_lines is None:
                candidate_lines = generated_lines
            if block_pos_km is None:
                block_pos_km = generated_block_pos

        shared = {
            "candidate_lines": list(candidate_lines),
            "block_pos_km": {
                b: np.array(pos, dtype=float) for b, pos in block_pos_km.items()
            },
            "depot_id": depot_id,
            "depot_pos_km": depot_pos,
            "crn_uniforms": _coerce_crn_uniforms(
                crn_uniforms, self.n_saa, len(block_ids)),
        }
        return shared

    # ------------------------------------------------------------------
    # Design: Double decomposition (Benders + subpath routing LP)
    # ------------------------------------------------------------------

    def design(
        self,
        geodata,
        prob_dict: Dict[str, float],
        Omega_dict,
        J_function: Callable,
        num_districts: int,
        Lambda: float = 1.0,
        wr: float = 1.0,
        wv: float = 1.0,
        **kwargs,
    ) -> ServiceDesign:
        """
        Two-stage stochastic program solved by SAA multi-cut Benders.

        First stage: select reference trips from a fixed candidate library
        under a fleet budget, choose dispatch intervals, and assign stopping
        locations to trips.

        Weight note:
        - `wr` is the rider-time weight (wait and in-vehicle time)
        - `walk_time_weight * wr` prices walk time
        - `wv` is retained only as a deprecated vehicle-cost-weight alias

        Second stage: per selected trip, solve subpath routing LP on
        load-expanded checkpoint network with scenario-wise demand.

        Benders cuts: from LP duals of linking constraints (z* → y),
        added per-scenario for multi-cut structure.
        """
        import gurobipy as gp
        from gurobipy import GRB

        block_ids = geodata.short_geoid_list
        N = len(block_ids)
        K = num_districts  # library granularity + default fleet-budget proxy
        tol = 1e-2
        discount = TSP_TRAVEL_DISCOUNT
        rider_time_weight = float(wr)
        vehicle_cost_weight = float(wv)
        speed = VEHICLE_SPEED_KMH
        fleet_budget = (float(self.max_fleet)
                        if self.max_fleet is not None else float(num_districts))

        shared_inputs = self._prepare_shared_inputs(
            geodata,
            K,
            candidate_lines=kwargs.get("candidate_lines"),
            block_pos_km=kwargs.get("block_pos_km"),
            depot_id=kwargs.get("depot_id"),
            crn_uniforms=kwargs.get("crn_uniforms"),
        )
        depot_id = shared_inputs["depot_id"]
        depot_pos = shared_inputs["depot_pos_km"]
        block_pos_km = shared_inputs["block_pos_km"]
        candidate_lines = shared_inputs["candidate_lines"]
        shared_candidate_library = "candidate_lines" in kwargs
        shared_crn = "crn_uniforms" in kwargs

        # ---- Step 1: Reuse fixed candidate line library; apply delta-specific reachability ----
        lines = _apply_line_reachability(
            candidate_lines, block_ids, block_pos_km, self.delta,
            speed_kmh=speed, alpha_time=self.alpha_time)
        T_vals = self.T_values
        n_T = len(T_vals)
        block_index = {b: j for j, b in enumerate(block_ids)}
        reference_trips = {
            (line.idx, ti): _build_reference_trip(
                line,
                ti,
                T_t,
                block_pos_km,
                speed_kmh=speed,
                alpha_time=self.alpha_time,
            )
            for line in lines
            for ti, T_t in enumerate(T_vals)
        }

        # Pre-build initial seed arcs for each line (for CG)
        line_arcs = {}
        for line in lines:
            line_arcs[line.idx] = _build_initial_arcs(
                line, block_ids, block_pos_km, self.delta,
                self.K_skip, speed, self.alpha_time)
            for i, a in enumerate(line_arcs[line.idx]):
                a.arc_id = i

        # Pre-compute walking geometry for recourse objective.
        # walk_saved_km[b]: km saved per passenger by detouring (constant, ≥ 0)
        # walk_det_km[b]:   km walked per passenger when served by detour arc
        walk_saved_km: Dict[str, float] = {}
        walk_det_km: Dict[str, float] = {}
        for b in block_ids:
            area_b = float(geodata.get_area(b))
            r_b = math.sqrt(area_b / math.pi)
            r_safe = max(r_b, 1e-12)
            walk_no_det_b = 2.0 * r_b / 3.0
            if self.delta < r_safe:
                walk_det_b = walk_no_det_b - self.delta + self.delta**3 / (3.0 * r_safe**2)
            else:
                walk_det_b = 0.0
            walk_det_b = max(walk_det_b, 0.0)
            walk_saved_km[b] = walk_no_det_b - walk_det_b
            walk_det_km[b] = walk_det_b
        walk_speed = self.walk_speed_kmh

        # No preprocessing guard: the master's load bound is soft (lower only),
        # and vehicle capacity is enforced per-scenario in the second-stage LP.
        infeasible_pairs: Set[Tuple[int, int]] = set()

        # ---- SAA: Common random numbers ----
        S = self.n_saa
        U_base = shared_inputs["crn_uniforms"]
        demand_matrices = {
            ti: _crn_demand(U_base, Lambda, T_t, prob_dict, block_ids)
            for ti, T_t in enumerate(T_vals)
        }

        # ---- Step 3: Build master MIP ----
        def _build_master():
            """Build a paper-aligned MiND-style master over scenario assignments."""
            master = gp.Model("MiND_Benders_Master")
            master.setParam('OutputFlag', 0)

            x_v = {}      # x[ℓ, t] binary: trip operates
            z_v = {}      # z[s, b, ℓ, t] binary: scenario-stop assigned to trip
            theta_v = {}  # θ[s, ℓ, t] continuous: scenario recourse cost
            slack_low_v = {}
            slack_high_v = {}

            for line in lines:
                for ti, T_t in enumerate(T_vals):
                    if (line.idx, ti) in infeasible_pairs:
                        continue
                    x_v[(line.idx, ti)] = master.addVar(
                        vtype=GRB.BINARY, name=f"x_{line.idx}_{ti}")
                    for s in range(S):
                        slack_low_v[(s, line.idx, ti)] = master.addVar(
                            lb=0.0, name=f"slack_low_{s}_{line.idx}_{ti}")
                        slack_high_v[(s, line.idx, ti)] = master.addVar(
                            lb=0.0, name=f"slack_high_{s}_{line.idx}_{ti}")
                        theta_v[(s, line.idx, ti)] = master.addVar(
                            lb=0.0, name=f"theta_{s}_{line.idx}_{ti}")
                        for b in line.reachable_blocks:
                            z_v[(s, b, line.idx, ti)] = master.addVar(
                                vtype=GRB.BINARY, name=f"z_{s}_{b}_{line.idx}_{ti}")

            master.update()

            penalty_low = 100.0
            penalty_high = 100.0

            # Assignment requires operation
            for line in lines:
                for ti in range(n_T):
                    if (line.idx, ti) not in x_v:
                        continue
                    for s in range(S):
                        for b in line.reachable_blocks:
                            master.addConstr(
                                z_v[(s, b, line.idx, ti)] <= x_v[(line.idx, ti)],
                                name=f"assign_{s}_{b}_{line.idx}_{ti}")

            # Each realized stop-demand block is assigned to at most one active trip.
            for s in range(S):
                for b in block_ids:
                    assign_expr = gp.LinExpr()
                    for line in lines:
                        if b not in line.reachable_blocks:
                            continue
                        for ti in range(n_T):
                            if (s, b, line.idx, ti) in z_v:
                                assign_expr += z_v[(s, b, line.idx, ti)]
                    master.addConstr(assign_expr <= 1.0, name=f"pack_{s}_{b}")

            # One interval per line
            for line in lines:
                active_tis = [ti for ti in range(n_T)
                              if (line.idx, ti) in x_v]
                if active_tis:
                    master.addConstr(
                        gp.quicksum(x_v[(line.idx, ti)] for ti in active_tis) <= 1,
                        name=f"one_T_{line.idx}")

            # At least one active line, with endogenous route count under a fleet budget.
            master.addConstr(
                gp.quicksum(x_v[k] for k in x_v) >= 1,
                name="min_one_line")

            _lbi = {ln.idx: ln for ln in lines}
            fleet_expr = gp.LinExpr()
            for k in x_v:
                li, ti = k
                fleet_expr += (_lbi[li].est_tour_km
                               / (speed * T_vals[ti])) * x_v[k]
            master.addConstr(fleet_expr <= fleet_budget,
                             name="fleet_budget")

            # Route count: at most K lines can operate.
            # This prevents the optimizer from selecting many small
            # routes just to collect the coverage reward.
            master.addConstr(
                gp.quicksum(x_v[k] for k in x_v)
                <= int(fleet_budget) + 1,
                name="max_routes")

            # Scenario-specific load-factor bounds, with soft slacks for robustness.
            kappa = self.load_factor_kappa

            for line in lines:
                for ti, T_t in enumerate(T_vals):
                    if (line.idx, ti) not in x_v:
                        continue
                    demand_matrix_t = demand_matrices[ti]
                    for s in range(S):
                        assigned_pax = gp.quicksum(
                            float(demand_matrix_t[s, block_index[b]]) * z_v[(s, b, line.idx, ti)]
                            for b in line.reachable_blocks
                        )
                        master.addConstr(
                            assigned_pax + slack_low_v[(s, line.idx, ti)]
                            >= (1.0 - kappa) * self.capacity * x_v[(line.idx, ti)],
                            name=f"load_lo_{s}_{line.idx}_{ti}")
                        master.addConstr(
                            assigned_pax - slack_high_v[(s, line.idx, ti)]
                            <= (1.0 + kappa) * self.capacity * x_v[(line.idx, ti)],
                            name=f"load_hi_{s}_{line.idx}_{ti}")

            # θ active only when trip operates
            M_big = max(l.est_tour_km for l in lines) * 100.0 + 1000.0
            for (li, ti) in x_v:
                for s in range(S):
                    master.addConstr(
                        theta_v[(s, li, ti)] <= M_big * x_v[(li, ti)],
                        name=f"theta_link_{s}_{li}_{ti}")

            # Warm-start cuts: θ ≥ shortest-path route cost through checkpoint network
            for line in lines:
                sp_km = _route_execution_km(
                    line, depot_pos, block_pos_km, self.K_skip)
                for ti, T_t in enumerate(T_vals):
                    if (line.idx, ti) not in x_v:
                        continue
                    cm = (
                        vehicle_cost_weight / (discount * T_t)
                        + rider_time_weight / (2.0 * speed)
                    )
                    for s in range(S):
                        master.addConstr(
                            theta_v[(s, line.idx, ti)] >= sp_km * cm * x_v[(line.idx, ti)],
                            name=f"theta_ws_{s}_{line.idx}_{ti}")

            # Objective: recourse cost + soft load penalties + expected wait
            # + paper-style coverage reward proxy on assigned scenario demand.
            coverage_reward = rider_time_weight * 10.0

            obj_expr = gp.LinExpr()
            for li, ti in x_v:
                obj_expr += (1.0 / S) * gp.quicksum(
                    theta_v[(s, li, ti)] for s in range(S))
                for s in range(S):
                    obj_expr += penalty_low * slack_low_v[(s, li, ti)]
                    obj_expr += penalty_high * slack_high_v[(s, li, ti)]
                    demand_matrix_t = demand_matrices[ti]
                    T_t = T_vals[ti]
                    for b in block_ids:
                        key = (s, b, li, ti)
                        if key not in z_v:
                            continue
                        demand_coeff = float(demand_matrix_t[s, block_index[b]]) / S
                        if demand_coeff <= 1e-12:
                            continue
                        obj_expr += rider_time_weight * T_t / 2.0 * demand_coeff * z_v[key]
                        obj_expr -= coverage_reward * demand_coeff * z_v[key]

            master.setObjective(obj_expr, GRB.MINIMIZE)

            return master, x_v, z_v, theta_v

        master, x, z, theta = _build_master()
        master.optimize()
        if master.status != GRB.OPTIMAL:
            print(f"  [MiND] Master infeasible at initial solve")

        # ---- Step 4: Benders loop ----
        pending_cuts: List[dict] = []
        cuts_added = 0
        best_ub = float('inf')
        best_solution = None

        # Build line-obj lookup
        line_by_idx = {ln.idx: ln for ln in lines}

        pbar = self._progress(self.max_iters, f"{self._name} design")
        for benders_iter in range(self.max_iters):
            # Add pending cuts
            for cut in pending_cuts[cuts_added:]:
                s = cut['scenario']
                li, ti = cut['trip']
                if (s, li, ti) not in theta:
                    continue
                master.addConstr(
                    theta[(s, li, ti)] >= cut['const'] + gp.quicksum(
                        cut['coeffs'].get(b, 0.0) * z[(b, li, ti)]
                        for b in cut['blocks'] if (b, li, ti) in z
                    ),
                    name=f"cut_{s}_{cut['id']}")
            cuts_added = len(pending_cuts)

            # Solve master
            master.optimize()
            if master.status != GRB.OPTIMAL:
                print(f"  [MiND Benders] Master infeasible at iter {benders_iter}")
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix_str("master infeasible")
                break
            lb = master.ObjVal

            # Extract solution
            x_star = {}
            z_star_all = {}
            for k in x:
                x_star[k] = round(x[k].X)
            for k in z:
                z_star_all[k] = z[k].X

            # Active trips
            active_trips = [(li, ti) for (li, ti), v in x_star.items()
                            if v > 0.5]

            # Solve subproblems: subpath LP with CG for both Benders cuts and UB.
            # Same cost model in optimization and evaluation.
            ub_iter = 0.0
            iter_solution = {}

            for li, ti in active_trips:
                T_t = T_vals[ti]
                line_obj = line_by_idx.get(li)
                if line_obj is None:
                    continue

                mc = len(line_obj.checkpoints)
                initial_arcs = line_arcs.get(li, [])
                # Provider vehicle-km cost plus a lightweight in-vehicle delay proxy.
                cost_mult = (
                    vehicle_cost_weight / (discount * T_t)
                    + rider_time_weight / (2.0 * speed)
                )
                access_cost = cost_mult * _line_access_km(
                    line_obj, depot_pos, block_pos_km)
                base_route_cost = cost_mult * _route_execution_km(
                    line_obj, depot_pos, block_pos_km, self.K_skip)

                demand_matrix = demand_matrices[ti]
                scenario_costs = []
                op_wait = 0.0

                for s in range(S):
                    z_dict = {
                        b: z_star_all.get((s, b, li, ti), 0.0)
                        for b in line_obj.reachable_blocks
                    }
                    cp_set_local = set(line_obj.checkpoints)
                    all_blocks = {
                        b for b in line_obj.reachable_blocks
                        if z_dict.get(b, 0.0) > 0.5
                    }
                    demand_s = {block_ids[j]: int(demand_matrix[s, j])
                                for j in range(N)}
                    op_wait += (rider_time_weight * T_t / 2.0 / S) * sum(
                        demand_s.get(b, 0) * z_dict.get(b, 0.0) for b in line_obj.reachable_blocks
                    )

                    if not all_blocks:
                        scenario_costs.append(base_route_cost)
                        continue

                    # Per-scenario walking penalties and costs.
                    s_walk_pen: Dict[str, float] = {}
                    s_walk_det_cost: Dict[str, float] = {}
                    for b in all_blocks:
                        d_b = demand_s.get(b, 0)
                        s_walk_pen[b] = (
                            self.coverage_penalty_factor
                            * self.walk_time_weight
                            * rider_time_weight
                            * walk_saved_km.get(b, 0.0)
                            * d_b
                            / walk_speed
                        )
                        s_walk_det_cost[b] = (
                            self.walk_time_weight
                            * rider_time_weight
                            * walk_det_km.get(b, 0.0)
                            * d_b
                            / walk_speed
                        )

                    base_walk = sum(
                        s_walk_det_cost.get(b, 0.0) * z_dict.get(b, 0.0)
                        for b in all_blocks)

                    if mc <= 1 or len(initial_arcs) == 0:
                        terminal_walk = 0.0
                        full_coeffs = {}
                        for b in all_blocks:
                            walk_km = walk_det_km.get(b, 0.0)
                            if b not in cp_set_local:
                                walk_km += walk_saved_km.get(b, 0.0)
                            walk_coeff = (
                                self.walk_time_weight
                                * rider_time_weight
                                * walk_km
                                * demand_s.get(b, 0)
                                / walk_speed
                            )
                            terminal_walk += walk_coeff * z_dict.get(b, 0.0)
                            if abs(walk_coeff) > 1e-12:
                                full_coeffs[b] = walk_coeff
                        total_obj_s = access_cost + terminal_walk
                    else:
                        # Solve with column generation
                        lp_obj_s, lp_coeffs_s = _solve_routing_lp_with_cg(
                            initial_arcs, line_obj, block_ids, block_pos_km,
                            mc, self.capacity, z_dict, demand_s, all_blocks,
                            cp_set_local, cost_mult, s_walk_pen,
                            self.delta, speed, self.alpha_time, self.K_skip,
                            max_cg_iters=self.max_cg_iters)

                        if lp_obj_s is None:
                            scenario_costs.append(base_route_cost)
                            continue

                        total_obj_s = access_cost + lp_obj_s + base_walk

                        # Benders cut coefficients
                        full_coeffs = {}
                        for b in all_blocks:
                            lp_coeff = (lp_coeffs_s.get(b, 0.0)
                                        if lp_coeffs_s else 0.0)
                            wdc = s_walk_det_cost.get(b, 0.0)
                            coeff = lp_coeff + wdc
                            if abs(coeff) > 1e-12:
                                full_coeffs[b] = coeff

                    scenario_costs.append(total_obj_s)

                    if full_coeffs:
                        const = total_obj_s - sum(
                            full_coeffs.get(b, 0.0) * z_dict.get(b, 0.0)
                            for b in all_blocks)
                        pending_cuts.append({
                            'scenario': s,
                            'trip': (li, ti),
                            'const': const,
                            'coeffs': full_coeffs,
                            'blocks': set(full_coeffs.keys()),
                            'id': len(pending_cuts),
                        })

                avg_cost = float(np.mean(scenario_costs))
                ub_iter += op_wait + avg_cost
                iter_solution[(li, ti)] = (z_dict, T_t, avg_cost)

            # Update best
            if ub_iter < best_ub:
                best_ub = ub_iter
                best_solution = (dict(x_star), dict(z_star_all),
                                 dict(iter_solution), active_trips)

            # Convergence
            gap = (best_ub - lb) / max(abs(best_ub), 1.0)
            n_new_cuts = len(pending_cuts) - cuts_added
            if pbar is not None:
                if math.isfinite(best_ub):
                    pbar.set_postfix(
                        lb=f"{lb:.2f}",
                        ub=f"{best_ub:.2f}",
                        gap=f"{100.0 * gap:.1f}%",
                        trips=len(active_trips),
                        cuts=len(pending_cuts),
                    )
                else:
                    pbar.set_postfix(
                        lb=f"{lb:.2f}",
                        ub="inf",
                        trips=len(active_trips),
                        cuts=len(pending_cuts),
                    )
                pbar.update(1)
            if gap < tol:
                break
            if n_new_cuts == 0 and benders_iter > 0:
                break

        if pbar is not None:
            pbar.close()

        # ---- Step 5: Map to ServiceDesign ----
        if best_solution is None:
            # Fallback: trivial design
            assignment = np.eye(N)
            return ServiceDesign(
                name=self._name, assignment=assignment,
                depot_id=depot_id,
                district_roots=[block_ids[0]],
                dispatch_intervals={block_ids[0]: 1.0},
                district_routes={},
                service_metadata={
                    "fr_detour": {
                        "selected_line_ids_by_root": {},
                        "checkpoint_sequence_by_root": {},
                        "selected_checkpoints_by_root": {},
                        "checkpoint_schedule_by_root": {},
                        "compatible_locations_by_root": {},
                        "assigned_locations_by_root": {},
                        "assigned_stop_points_by_root": {},
                        "location_assignment_scores_by_root": {},
                        "reachable_locations_by_root": {},
                        "reachable_stop_points_by_root": {},
                        "assigned_blocks_by_root": {},
                        "reachable_blocks_by_root": {},
                        "selected_trip_indices_by_root": {},
                        "scenario_assigned_loads_by_root": {},
                        "checkpoint_candidate_locations": sorted(
                            _checkpoint_candidate_ids(geodata, block_ids)
                        ),
                        "demand_support_locations": sorted(
                            b for b in block_ids if prob_dict.get(b, 0.0) > 0.0
                        ),
                        "planned_fleet_size": 1.0,
                        "depot_id": depot_id,
                        "solve_mode": self.solve_mode,
                        "shared_candidate_library": shared_candidate_library,
                        "shared_crn": shared_crn,
                    }
                })

        _, z_star_final, _, active_trips = best_solution

        assignment = np.zeros((N, N))
        roots_final: List[str] = []
        dispatch_intervals: Dict[str, float] = {}
        district_routes: Dict[str, List[int]] = {}
        selected_line_ids_by_root: Dict[str, int] = {}
        checkpoint_sequence_by_root: Dict[str, List[str]] = {}
        selected_checkpoints_by_root: Dict[str, List[str]] = {}
        checkpoint_schedule_by_root: Dict[str, List[float]] = {}
        compatible_locations_by_root: Dict[str, List[str]] = {}
        assigned_locations_by_root: Dict[str, List[str]] = {}
        assigned_stop_points_by_root: Dict[str, List[str]] = {}
        location_assignment_scores_by_root: Dict[str, Dict[str, float]] = {}
        reachable_locations_by_root: Dict[str, List[str]] = {}
        reachable_stop_points_by_root: Dict[str, List[str]] = {}
        selected_trip_indices_by_root: Dict[str, Dict[str, int]] = {}
        scenario_assigned_loads_by_root: Dict[str, List[float]] = {}
        checkpoint_candidate_set = _checkpoint_candidate_ids(geodata, block_ids)
        stop_point_set = _stop_point_ids(geodata, block_ids)
        demand_support_locations = sorted(
            b for b in block_ids if prob_dict.get(b, 0.0) > 0.0
        )

        used_roots: Set[str] = set()
        route_info: List[Tuple[str, int, int, ReferenceLine, Dict[str, float]]] = []

        for li, ti in active_trips:
            T_t = T_vals[ti]
            line_obj = line_by_idx.get(li)
            if line_obj is None:
                continue

            z_scores = {
                b: float(np.mean([
                    z_star_final.get((s, b, li, ti), 0.0) for s in range(S)
                ]))
                for b in line_obj.reachable_blocks
            }

            # Route key: pick a unique checkpoint when possible.
            root_id = None
            for cp in line_obj.checkpoints:
                if cp not in used_roots:
                    root_id = cp
                    break
            if root_id is None:
                ranked_blocks = sorted(
                    line_obj.reachable_blocks,
                    key=lambda b: (-z_scores.get(b, 0.0), b),
                )
                for b in ranked_blocks:
                    if b not in used_roots:
                        root_id = b
                        break
            if root_id is None:
                continue  # all blocks already roots (degenerate)

            used_roots.add(root_id)

            roots_final.append(root_id)
            dispatch_intervals[root_id] = T_t
            selected_line_ids_by_root[root_id] = line_obj.idx
            compatible_locations_by_root[root_id] = sorted(line_obj.reachable_locations)
            reachable_locations_by_root[root_id] = sorted(line_obj.reachable_locations)
            reachable_stop_points_by_root[root_id] = sorted(
                b for b in line_obj.reachable_locations if b in stop_point_set
            )
            location_assignment_scores_by_root[root_id] = {
                b: float(score) for b, score in sorted(z_scores.items())
                if score > 1e-9
            }
            selected_trip_indices_by_root[root_id] = {
                "line_idx": int(li),
                "time_idx": int(ti),
            }
            checkpoint_schedule_by_root[root_id] = list(
                reference_trips[(li, ti)].checkpoint_schedule_h
            )
            scenario_assigned_loads_by_root[root_id] = [
                float(sum(
                    demand_matrices[ti][s, block_index[b]]
                    * z_star_final.get((s, b, li, ti), 0.0)
                    for b in line_obj.reachable_blocks
                ))
                for s in range(S)
            ]

            # District route: checkpoint ordering as global block indices
            route_order = []
            for cp in line_obj.checkpoints:
                route_order.append(block_ids.index(cp))
            district_routes[root_id] = route_order
            checkpoint_sequence_by_root[root_id] = [
                block_ids[gi] for gi in route_order
            ]
            selected_checkpoints_by_root[root_id] = list(checkpoint_sequence_by_root[root_id])
            route_info.append((root_id, li, ti, line_obj, z_scores))

        planned_fleet_size = 0.0
        for root_id, _li, _ti, line_obj, _z_scores in route_info:
            T_root = dispatch_intervals.get(root_id, 1.0)
            planned_fleet_size += (
                _route_execution_km(line_obj, depot_pos, block_pos_km, self.K_skip)
                / (speed * max(T_root, 1e-9))
            )

        # Recover one route-level compatibility set per selected trip from the
        # optimized z values, without forcing a full nearest-root partition.
        for root_id, _li, _ti, line_obj, _z_scores in route_info:
            assigned_locations_by_root[root_id] = []

        for b in block_ids:
            best_root = None
            best_score = 0.5
            for root_id, _li, _ti, line_obj, z_scores in route_info:
                if b not in line_obj.reachable_locations:
                    continue
                score = z_scores.get(b, 0.0)
                if score > best_score + 1e-12:
                    best_score = score
                    best_root = root_id
            if best_root is not None:
                assigned_locations_by_root[best_root].append(b)

        for root_id, _li, _ti, line_obj, _z_scores in route_info:
            for cp in line_obj.checkpoints:
                if cp not in assigned_locations_by_root[root_id]:
                    assigned_locations_by_root[root_id].append(cp)
            assigned_locations_by_root[root_id].sort()
            assigned_stop_points_by_root[root_id] = sorted(
                b for b in assigned_locations_by_root[root_id]
                if b in stop_point_set
            )
            root_idx = block_ids.index(root_id)
            for b in assigned_locations_by_root[root_id]:
                j = block_ids.index(b)
                assignment[j, root_idx] = 1.0

        design = ServiceDesign(
            name=self._name,
            assignment=assignment,
            depot_id=depot_id,
            district_roots=roots_final,
            dispatch_intervals=dispatch_intervals,
            district_routes=district_routes,
            service_metadata={
                    "fr_detour": {
                        "selected_line_ids_by_root": selected_line_ids_by_root,
                        "checkpoint_sequence_by_root": checkpoint_sequence_by_root,
                        "selected_checkpoints_by_root": selected_checkpoints_by_root,
                        "checkpoint_schedule_by_root": checkpoint_schedule_by_root,
                        "compatible_locations_by_root": compatible_locations_by_root,
                        "assigned_locations_by_root": assigned_locations_by_root,
                        "assigned_stop_points_by_root": assigned_stop_points_by_root,
                        "location_assignment_scores_by_root": location_assignment_scores_by_root,
                        "reachable_locations_by_root": reachable_locations_by_root,
                        "reachable_stop_points_by_root": reachable_stop_points_by_root,
                        "assigned_blocks_by_root": assigned_locations_by_root,
                        "reachable_blocks_by_root": reachable_locations_by_root,
                        "selected_trip_indices_by_root": selected_trip_indices_by_root,
                        "scenario_assigned_loads_by_root": scenario_assigned_loads_by_root,
                        "checkpoint_candidate_locations": sorted(checkpoint_candidate_set),
                        "demand_support_locations": demand_support_locations,
                        "planned_fleet_size": planned_fleet_size,
                        "depot_id": depot_id,
                        "solve_mode": self.solve_mode,
                    "shared_candidate_library": shared_candidate_library,
                    "shared_crn": shared_crn,
                }
            },
        )

        return design

    # ------------------------------------------------------------------
    # Custom simulate (called by run_baseline in experiment scripts)
    # ------------------------------------------------------------------

    def custom_simulate(
        self, design: ServiceDesign, geodata,
        prob_dict: Dict[str, float], Omega_dict,
        J_function: Callable, Lambda: float,
        wr: float, wv: float,
        fleet_cost_rate: float = DEFAULT_FLEET_COST_RATE,
    ) -> EvaluationResult:
        """Dispatch to evaluate() so run_baseline uses FR-specific costs."""
        return self.evaluate(
            design, geodata, prob_dict, Omega_dict, J_function,
            Lambda, wr, wv, fleet_cost_rate=fleet_cost_rate)

    # ------------------------------------------------------------------
    # Evaluate — subpath routing (same model as optimization)
    # ------------------------------------------------------------------

    def evaluate(
        self, design: ServiceDesign, geodata,
        prob_dict: Dict[str, float], Omega_dict,
        J_function: Callable, Lambda: float,
        wr: float, wv: float,
        fleet_cost_rate: float = DEFAULT_FLEET_COST_RATE,
        beta: float = BETA, road_network=None,
        crn_uniforms: Optional[np.ndarray] = None,
    ) -> EvaluationResult:
        """FR-specific evaluation using the same subpath-routing semantics.

        `wr` prices rider wait and in-vehicle time. `walk_time_weight * wr`
        prices walk time. `wv` is retained only as a deprecated provider-side
        vehicle-cost-weight alias.
        """
        block_ids = geodata.short_geoid_list
        N = len(block_ids)
        assignment = design.assignment
        fr_meta = design.service_metadata.get("fr_detour", {})
        depot_id = design.depot_id
        discount = TSP_TRAVEL_DISCOUNT
        speed = VEHICLE_SPEED_KMH
        walk_speed = self.walk_speed_kmh
        rider_time_weight = float(wr)
        vehicle_cost_weight = float(wv)

        # Build block positions in km
        block_pos_km = {}
        for b in block_ids:
            block_pos_km[b] = np.array(_get_pos(geodata, b),
                                       dtype=float) / 1000.0
        depot_pos_km = np.array(_get_pos(geodata, depot_id),
                                dtype=float) / 1000.0

        # Pre-compute walking geometry per block (still needed for optimization
        # walk penalties in the detour trade-off LP)
        walk_saved_km: Dict[str, float] = {}
        walk_det_km: Dict[str, float] = {}
        walk_no_det_km: Dict[str, float] = {}
        areas_km2: Dict[str, float] = {}
        for b in block_ids:
            area_b = float(geodata.get_area(b))
            areas_km2[b] = area_b
            r_b = math.sqrt(area_b / math.pi)
            r_safe = max(r_b, 1e-12)
            wnd = 2.0 * r_b / 3.0
            if self.delta < r_safe:
                wd = wnd - self.delta + self.delta**3 / (3.0 * r_safe**2)
            else:
                wd = 0.0
            wd = max(wd, 0.0)
            walk_no_det_km[b] = wnd
            walk_det_km[b] = wd
            walk_saved_km[b] = wnd - wd

        # RNG for simulation-based walk distance sampling
        walk_rng = np.random.default_rng(42)

        # SAA: generate demand scenarios
        if crn_uniforms is not None:
            U_base = _coerce_crn_uniforms(crn_uniforms, self.num_scenarios, N)
            n_eval = int(U_base.shape[0])
        else:
            n_eval = self.num_scenarios
            U_base = _coerce_crn_uniforms(None, n_eval, N)

        acc_in_district = 0.0
        acc_wait_h = 0.0
        acc_invehicle_h = 0.0
        acc_walk_riders = 0.0  # walk_km * n_riders accumulated
        acc_riders = 0.0       # total riders accumulated
        acc_fleet = 0.0
        acc_raw_fleet = 0
        acc_T_weighted = 0.0
        acc_prob = 0.0
        route_km_sum_by_root = {root: 0.0 for root in design.dispatch_intervals}

        T_global = (
            float(np.mean(list(design.dispatch_intervals.values())))
            if design.dispatch_intervals else 1.0
        )
        demand_matrix_global = _crn_demand(U_base, Lambda, T_global, prob_dict, block_ids)

        pbar = self._progress(n_eval, f"{self._name} eval")
        for s in range(n_eval):
            demand_s = {block_ids[j]: int(demand_matrix_global[s, j]) for j in range(N)}
            scenario_result = _solve_projected_service_scenario(
                design,
                self,
                geodata,
                prob_dict,
                block_ids,
                block_pos_km,
                demand_s,
                depot_pos_km,
                areas_km2,
                walk_saved_km,
                rider_time_weight,
                vehicle_cost_weight,
                walk_rng,
            )
            acc_walk_riders += scenario_result["total_walk_km"]
            acc_wait_h += scenario_result["total_wait_h"]
            acc_invehicle_h += scenario_result["total_invehicle_h"]
            acc_riders += scenario_result["total_riders"]
            for root, route_km_s in scenario_result["route_km_by_root"].items():
                route_km_sum_by_root[root] = route_km_sum_by_root.get(root, 0.0) + float(route_km_s)
            if pbar is not None:
                pbar.set_postfix(
                    riders=f"{acc_riders:.0f}",
                    routes=len(scenario_result["route_km_by_root"]),
                )
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        # Route-average fleet is computed from mean realized route-km per route.
        acc_prob = 1.0
        acc_T_weighted = float(np.mean(list(design.dispatch_intervals.values()))) if design.dispatch_intervals else 1.0

        acc_in_district = 0.0
        acc_fleet = 0.0
        acc_raw_fleet = 0
        for root, T_i in design.dispatch_intervals.items():
            avg_tour_km = route_km_sum_by_root.get(root, 0.0) / max(n_eval, 1)
            acc_in_district += avg_tour_km / max(discount, 1e-12) / max(float(T_i), 1e-12)
            acc_fleet += avg_tour_km / (speed * max(float(T_i), 1e-12))
            acc_raw_fleet += _effective_fleet_size(avg_tour_km, float(T_i), speed)

        avg_wait = acc_wait_h / max(acc_riders, 1.0)
        avg_invehicle = acc_invehicle_h / max(acc_riders, 1.0)
        avg_walk_km = acc_walk_riders / max(acc_riders, 1.0)
        avg_walk = avg_walk_km / walk_speed
        total_user_time = avg_wait + avg_invehicle + avg_walk

        in_district_cost = acc_in_district
        fleet_size = float(acc_fleet)
        raw_fleet_size = int(acc_raw_fleet)
        avg_dispatch_interval = acc_T_weighted / acc_prob

        provider_cost = in_district_cost + fleet_cost_rate * raw_fleet_size
        user_cost = (
            rider_time_weight * avg_wait
            + rider_time_weight * avg_invehicle
            + self.walk_time_weight * rider_time_weight * avg_walk
        ) * acc_prob
        total_cost = provider_cost + user_cost

        return EvaluationResult(
            name=self._name,
            avg_wait_time=avg_wait,
            avg_invehicle_time=avg_invehicle,
            avg_walk_time=avg_walk,
            total_user_time=total_user_time,
            linehaul_cost=0.0,
            in_district_cost=in_district_cost,
            total_travel_cost=in_district_cost,
            fleet_size=fleet_size,
            raw_fleet_size=raw_fleet_size,
            avg_dispatch_interval=avg_dispatch_interval,
            odd_cost=0.0,
            provider_cost=provider_cost,
            user_cost=user_cost,
            total_cost=total_cost,
        )
