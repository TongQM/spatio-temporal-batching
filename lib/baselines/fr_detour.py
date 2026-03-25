"""
Multi Fixed-Route with Detour — MiND-structured adaptation (Martin-Iradi et al. 2025)

Adaptation of the double decomposition from:
  "A Double Decomposition Algorithm for Network Planning and Operations in
  Deviated Fixed-route Microtransit"
  (Martin-Iradi, Schmid, Cummings, Jacquillat 2025, arXiv:2402.01265)

to a steady-state district-based service design setting.

Algorithm structure (from the paper):
  Double decomposition:
    - Outer: SAA multi-cut Benders with scenario-wise optimality cuts
    - Inner: column generation with label-setting DP pricing

  First stage — line selection + demand assignment:
    - Master MIP selects K lines, chooses dispatch intervals, assigns blocks
    - Variables: x[ℓ,t] (trip selection), z[b,ℓ,t] (assignment), θ[ℓ,t] (recourse)
    - Hard coverage, expected-demand load bounds, fleet budget

  Second stage — subpath routing on load-expanded checkpoint network:
    - Nodes: (checkpoint, passenger_load) pairs
    - Subpath arcs between checkpoint pairs (skip ≤ K_skip), time-feasible
    - Routing LP with flow balance, linking (Σy ≤ z*), coverage (Σy+u ≥ z*)
    - Column generation: restricted LP → label-setting DP pricing → iterate
    - Benders cut coefficients: π[b] + μ[b] (linking + coverage duals)

Adaptations from the paper (honest deviations):

  Structural:
    - Block-level demand aggregation (not individual passengers)
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
    - Final scoring uses a BHH-style stochastic tour simulator
      (_fr_detour_tour_stochastic), not the subpath-routing LP.
    - This means the design is optimized against one routing model
      (subpath LP via Benders) but scored by another (BHH Monte Carlo).
    - The gap is accepted as an optimization-vs-evaluation separation
      consistent with this repo's design philosophy.

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

from lib.baselines.base import (
    BETA, VEHICLE_SPEED_KMH, BaselineMethod, EvaluationResult, ServiceDesign,
    _get_pos,
)
from lib.constants import TSP_TRAVEL_DISCOUNT


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ReferenceLine:
    idx: int
    checkpoints: List[str]      # block IDs in visit order (excl. depot)
    reachable_blocks: Set[str]  # blocks within δ of any checkpoint
    est_tour_km: float          # estimated depot→checkpoints→depot distance


@dataclass
class SubpathArc:
    arc_id: int
    cp_from: int                    # checkpoint index (source end)
    cp_to: int                      # checkpoint index (destination end)
    cost_km: float                  # vehicle travel distance (km)
    travel_time_h: float            # time for vehicle to traverse this subpath
    served_blocks: FrozenSet[str]   # blocks served (skipped CPs + corridor)
    # Note: load_delta is scenario-dependent (= Σ demand_s[b] for b in served_blocks),
    # computed at LP-build time, NOT stored on the arc.


def _checkpoint_time_budget(cp_i_pos, cp_j_pos, speed_kmh, alpha_time=0.5):
    """Max allowed time for a subpath between checkpoints i and j.

    Steady-state timing surrogate: (1 + alpha_time) * direct_dist / speed.
    """
    direct_km = float(np.linalg.norm(cp_j_pos - cp_i_pos))
    if direct_km < 1e-12:
        return 1e-6  # degenerate: same position
    return (1.0 + alpha_time) * direct_km / speed_kmh


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
            time_budget = _checkpoint_time_budget(
                ci_pos, cj_pos, speed_kmh, alpha_time)
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


def _generate_line_library(geodata, depot_id, block_ids, n_angular, delta,
                           num_districts=None, speed_kmh=VEHICLE_SPEED_KMH,
                           alpha_time=0.5):
    """Generate candidate reference lines from depot through block subsets.

    Includes sector lines, merged-sector lines, partition lines (sized for
    K-way coverage), and singleton fallbacks.
    """
    N = len(block_ids)
    depot_pos = np.array(_get_pos(geodata, depot_id), dtype=float) / 1000.0
    block_pos_km = {}
    for b in block_ids:
        block_pos_km[b] = np.array(_get_pos(geodata, b), dtype=float) / 1000.0

    # Angular position of each block relative to depot
    positions = np.array([block_pos_km[b] for b in block_ids])
    dx = positions[:, 0] - depot_pos[0]
    dy = positions[:, 1] - depot_pos[1]
    angles = np.arctan2(dy, dx)

    # Sort blocks by angle
    order = np.argsort(angles)
    sorted_blocks = [block_ids[i] for i in order]

    def _make_line(block_list, idx_counter):
        """Helper to create a ReferenceLine from a block list."""
        pts = np.array([block_pos_km[b] for b in block_list])
        nn = _greedy_nn_route(pts, depot_pos)
        nn = _two_opt_improve(nn, pts, depot_pos)
        cps = [block_list[i] for i in nn]
        reach = _compute_reachable(cps, block_ids, block_pos_km, delta,
                                       speed_kmh, alpha_time)
        km = _tour_distance(depot_pos, cps, block_pos_km)
        return ReferenceLine(idx=idx_counter, checkpoints=cps,
                             reachable_blocks=reach, est_tour_km=km)

    lines: List[ReferenceLine] = []
    idx = 0

    # Build sector lists
    n_ang = max(n_angular, 1)
    sector_lists: List[List[str]] = [[] for _ in range(n_ang)]
    for rank, b in enumerate(sorted_blocks):
        sector_lists[rank * n_ang // N].append(b)

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

    # 4. Partition lines: divide blocks into K equal groups by angle.
    #    Guarantees that selecting these K lines covers all blocks.
    if num_districts is not None and num_districts > 0:
        K = num_districts
        part_size = max(1, (N + K - 1) // K)
        for g in range(K):
            group = sorted_blocks[g * part_size: min((g + 1) * part_size, N)]
            if not group:
                continue
            lines.append(_make_line(group, idx))
            idx += 1

    # 5. Singleton fallback lines (every block reachable by at least one line)
    for b in block_ids:
        reachable = _compute_reachable([b], block_ids, block_pos_km, delta,
                                           speed_kmh, alpha_time)
        est_km = 2.0 * float(np.linalg.norm(block_pos_km[b] - depot_pos))
        lines.append(ReferenceLine(idx=idx, checkpoints=[b],
                                   reachable_blocks=reachable, est_tour_km=est_km))
        idx += 1

    return lines, block_pos_km, depot_pos


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

    Endpoint convention: cp_i and cp_j are NOT in served_blocks (they are
    visited by flow-balance). Skipped checkpoints between them ARE in
    served_blocks.
    """
    checkpoints = line.checkpoints
    cp_set = set(checkpoints)
    m = len(checkpoints)
    if m <= 1:
        return []  # singleton: no inter-checkpoint arcs

    arcs: List[SubpathArc] = []
    arc_id = 0

    for i in range(m):
        for j in range(i + 1, min(i + K_skip + 2, m)):
            ci_pos = block_pos_km[checkpoints[i]]
            cj_pos = block_pos_km[checkpoints[j]]

            # Time budget for this segment
            time_budget = _checkpoint_time_budget(
                ci_pos, cj_pos, speed_kmh, alpha_time)

            # Skipped checkpoints (always in served_blocks)
            skipped_cps = frozenset(
                checkpoints[k] for k in range(i + 1, j))

            # Corridor blocks: off-checkpoint blocks within δ of segment
            corridor: List[str] = []
            if delta > 0:
                for b in block_ids:
                    if b in cp_set:
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
                    served_blocks=skipped_cps,
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
                        served_blocks=frozenset(served),
                    ))
                    arc_id += 1

            # --- Seed arcs: small multi-block combinations (≤ 3) ---
            max_enum = min(len(corridor), 3)
            for size in range(2, max_enum + 1):
                for subset in combinations(range(len(corridor)), size):
                    blocks_in_subset = [corridor[s] for s in subset]
                    served = skipped_cps | frozenset(blocks_in_subset)
                    served_pos = [block_pos_km[b] for b in blocks_in_subset]
                    cost = _subpath_cost(ci_pos, cj_pos, served_pos)
                    travel_time = cost / speed_kmh if speed_kmh > 0 else 0.0
                    if travel_time <= time_budget + 1e-9:
                        arcs.append(SubpathArc(
                            arc_id=arc_id, cp_from=i, cp_to=j,
                            cost_km=cost, travel_time_h=travel_time,
                            served_blocks=frozenset(served),
                        ))
                        arc_id += 1

    return arcs


# ---------------------------------------------------------------------------
# Restricted routing LP on load-expanded checkpoint network
# ---------------------------------------------------------------------------

def _solve_restricted_routing_lp(arcs, checkpoints, m, capacity, z_star,
                                  demand_s, all_blocks, cp_set, cost_mult,
                                  walk_penalties, return_solution=False):
    """Solve the restricted subpath routing LP for one scenario on one line.

    Load is measured in **passengers** (scenario-dependent), not blocks.
    Checkpoint demand enters load transitions: each arc's load delta includes
    demand at its destination checkpoint, and the source includes demand at
    the first checkpoint.

    Walking cost model: u[b] penalty = walk_saved_cost[b] (the cost of NOT
    detouring to serve block b — passengers walk to checkpoint instead).
    This replaces the former big-M penalty and lets the optimizer see the
    detour-vs-walking trade-off.

    Args:
        arcs: list of SubpathArc (current column pool)
        checkpoints: ordered list of checkpoint block IDs
        m: number of checkpoints
        capacity: vehicle passenger capacity
        z_star: dict block_id → assignment value from master
        demand_s: dict block_id → passenger count for this scenario
        all_blocks: set of blocks that participate in linking/coverage
        cp_set: set of ALL checkpoint block IDs
        cost_mult: km → cost conversion factor
        walk_penalties: dict block_id → penalty for u[b] (walking cost saved
            by detouring = wr * walk_saved_km * demand_s / walk_speed)

    Returns:
        (obj, duals_dict) where duals_dict has keys 'flow', 'link', 'coverage'
        or (None, None) if infeasible.
    """
    import gurobipy as gp
    from gurobipy import GRB

    model = gp.Model("routing_lp")
    model.setParam('OutputFlag', 0)

    # Per-checkpoint demand (for load transitions at checkpoints)
    cp_demand = [demand_s.get(checkpoints[c], 0) for c in range(m)]

    # Compute scenario-dependent load delta per arc.
    # Includes served_blocks demand + destination checkpoint demand.
    arc_load: Dict[int, int] = {}
    for a in arcs:
        arc_load[a.arc_id] = (sum(demand_s.get(b, 0) for b in a.served_blocks)
                              + cp_demand[a.cp_to])

    # Build arc-id-to-arc lookup
    arc_by_id: Dict[int, SubpathArc] = {a.arc_id: a for a in arcs}

    # Pre-compute incidence: which arcs serve each block.
    # For non-skipped checkpoint endpoints, service = flow visits the CP.
    # We express this via outgoing arcs from that checkpoint index.
    cp_index = {checkpoints[c]: c for c in range(m)}
    arcs_serving: Dict[str, List[SubpathArc]] = {}
    # Also track which blocks are checkpoint-served (need special linking)
    cp_served_blocks: Set[str] = set()
    for b in all_blocks:
        # Standard: arcs that have b in served_blocks
        serving = [a for a in arcs if b in a.served_blocks]
        if serving:
            arcs_serving[b] = serving
        elif b in cp_index:
            # Non-skipped checkpoint: served by flow visitation.
            # Use outgoing arcs from this CP as "serving" arcs.
            c = cp_index[b]
            arcs_serving[b] = [a for a in arcs if a.cp_from == c]
            cp_served_blocks.add(b)
        else:
            arcs_serving[b] = []

    # --- Variables ---
    # y[arc_id, load_from] for each arc and valid starting load
    y: Dict[Tuple[int, int], object] = {}
    for a in arcs:
        ld = arc_load[a.arc_id]
        for l in range(max(0, capacity - ld) + 1):
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
                if a.cp_from == c and l <= capacity - ld:
                    outflow += y[(a.arc_id, l)]
            if c == m - 1:
                outflow += term[l]

            # Inflow to (c, l)
            for a in arcs:
                ld = arc_load[a.arc_id]
                if a.cp_to == c:
                    l_from = l - ld
                    if 0 <= l_from <= capacity - ld:
                        inflow += y[(a.arc_id, l_from)]

            rhs = 1.0 if (c == 0 and l == cp_demand[0]) else 0.0
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
        for l in range(max(0, capacity - ld) + 1):
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
        for a in arcs:
            ld = arc_load[a.arc_id]
            for l in range(max(0, capacity - ld) + 1):
                if (a.arc_id, l) in y:
                    route_km += a.cost_km * y[(a.arc_id, l)].X
        u_vals = {b: u[b].X for b in all_blocks}
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

    Reduced cost of arc a from cp_i to cp_j at starting load l:
      rc(a, l) = cost_mult * a.cost_km
                 - (ν[i, l] - ν[j, l + δ_a])
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

    for i in range(m):
        for j in range(i + 1, min(i + K_skip + 2, m)):
            ci_pos = block_pos_km[checkpoints[i]]
            cj_pos = block_pos_km[checkpoints[j]]
            time_budget = _checkpoint_time_budget(
                ci_pos, cj_pos, speed_kmh, alpha_time)

            # Skipped checkpoints (always in served_blocks of any arc on this segment)
            skipped_cps = [checkpoints[k] for k in range(i + 1, j)]
            skipped_set = frozenset(skipped_cps)

            # Corridor blocks (off-checkpoint, within delta of segment)
            corridor: List[str] = []
            if delta > 0:
                for b in block_ids:
                    if b in cp_set:
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

                    for l_start in range(max(0, capacity - load_delta) + 1):
                        nu_i = flow_d.get((i, l_start), 0.0)
                        nu_j = flow_d.get((j, l_start + load_delta), 0.0)
                        rc = (cost_mult * total_km
                              - (nu_i - nu_j)
                              - total_reward)
                        if rc < best_rc_found:
                            best_rc_found = rc
                            best_arc = SubpathArc(
                                arc_id=-1, cp_from=i, cp_to=j,
                                cost_km=total_km,
                                travel_time_h=total_time,
                                served_blocks=frozenset(full_served),
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
                               return_solution=False):
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
            return (None, None, None, None) if return_solution else (None, None)

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
        return_solution=return_solution)

    if return_solution:
        obj, duals, route_km, u_vals = result
        if obj is None:
            return None, None, None, None
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
      - Evaluation via BHH stochastic tour (not subpath LP)

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
        self._name = name or ("Multi-FR-Detour" if delta > 0 else "Multi-FR")

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
        wv: float = 10.0,
        **kwargs,
    ) -> ServiceDesign:
        """
        Two-stage stochastic program solved by SAA multi-cut Benders.

        First stage: select K reference lines from library, choose dispatch
        intervals, assign blocks to lines.

        Second stage: per selected trip, solve subpath routing LP on
        load-expanded checkpoint network with scenario-wise demand.

        Benders cuts: from LP duals of linking constraints (z* → y),
        added per-scenario for multi-cut structure.
        """
        import gurobipy as gp
        from gurobipy import GRB

        block_ids = geodata.short_geoid_list
        N = len(block_ids)
        K = num_districts
        tol = 1e-2
        discount = TSP_TRAVEL_DISCOUNT
        speed = VEHICLE_SPEED_KMH

        # Depot: geographic centroid
        positions_m = np.array([_get_pos(geodata, b) for b in block_ids])
        centroid = positions_m.mean(axis=0)
        depot_idx = int(np.argmin(np.linalg.norm(positions_m - centroid, axis=1)))
        depot_id = block_ids[depot_idx]

        # ---- Step 1: Generate line library ----
        lines, block_pos_km, depot_pos = _generate_line_library(
            geodata, depot_id, block_ids, self.n_angular, self.delta,
            num_districts=K, speed_kmh=speed, alpha_time=self.alpha_time)
        L = len(lines)
        T_vals = self.T_values
        n_T = len(T_vals)

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
        rng = np.random.default_rng(42)
        S = self.n_saa
        U_base = rng.uniform(0.0, 1.0, size=(S, N))

        # ---- Step 3: Build master MIP ----
        def _build_master(hard_coverage):
            """Build master MIP. If hard_coverage fails, caller rebuilds with soft."""
            master = gp.Model("MiND_Benders_Master")
            master.setParam('OutputFlag', 0)

            x_v = {}      # x[ℓ, t] binary: trip operates
            z_v = {}      # z[b, ℓ, t] continuous: block assignment
            theta_v = {}  # θ[ℓ, t] continuous: recourse cost
            slack_low_v = {}
            uncov_v = {}

            for line in lines:
                for ti, T_t in enumerate(T_vals):
                    if (line.idx, ti) in infeasible_pairs:
                        continue
                    x_v[(line.idx, ti)] = master.addVar(
                        vtype=GRB.BINARY, name=f"x_{line.idx}_{ti}")
                    theta_v[(line.idx, ti)] = master.addVar(
                        lb=0.0, name=f"theta_{line.idx}_{ti}")
                    slack_low_v[(line.idx, ti)] = master.addVar(
                        lb=0.0, name=f"slack_{line.idx}_{ti}")
                    for b in line.reachable_blocks:
                        if (b, line.idx, ti) not in z_v:
                            z_v[(b, line.idx, ti)] = master.addVar(
                                lb=0.0, ub=1.0, name=f"z_{b}_{line.idx}_{ti}")

            master.update()

            # Coverage: hard or soft
            penalty_low = 0.1
            penalty_uncov = 100.0

            if not hard_coverage:
                for b in block_ids:
                    uncov_v[b] = master.addVar(lb=0.0, ub=1.0, name=f"uncov_{b}")
                master.update()

            for b in block_ids:
                expr = gp.LinExpr()
                for line in lines:
                    if b in line.reachable_blocks:
                        for ti in range(n_T):
                            if (b, line.idx, ti) in z_v:
                                expr += z_v[(b, line.idx, ti)]
                if hard_coverage:
                    master.addConstr(expr == 1.0, name=f"cov_{b}")
                else:
                    master.addConstr(expr + uncov_v[b] == 1.0, name=f"cov_{b}")

            # Assignment requires operation
            for line in lines:
                for ti in range(n_T):
                    if (line.idx, ti) not in x_v:
                        continue
                    for b in line.reachable_blocks:
                        if (b, line.idx, ti) in z_v:
                            master.addConstr(
                                z_v[(b, line.idx, ti)] <= x_v[(line.idx, ti)],
                                name=f"assign_{b}_{line.idx}_{ti}")

            # One interval per line
            for line in lines:
                active_tis = [ti for ti in range(n_T)
                              if (line.idx, ti) in x_v]
                if active_tis:
                    master.addConstr(
                        gp.quicksum(x_v[(line.idx, ti)] for ti in active_tis) <= 1,
                        name=f"one_T_{line.idx}")

            # K lines selected
            master.addConstr(
                gp.quicksum(x_v[k] for k in x_v) == K,
                name="K_lines")

            # Fleet budget (if specified)
            if self.max_fleet is not None:
                _lbi = {ln.idx: ln for ln in lines}
                fleet_expr = gp.LinExpr()
                for k in x_v:
                    li, ti = k
                    fleet_expr += (_lbi[li].est_tour_km
                                   / (speed * T_vals[ti])) * x_v[k]
                master.addConstr(fleet_expr <= self.max_fleet,
                                 name="fleet_budget")

            # Load bounds: expected-demand based
            # The paper's load factor bounds constrain expected demand per trip.
            # Vehicle capacity is enforced in the second-stage routing LP
            # (scenario-specific, passenger-count load on checkpoint network).
            # Here we only enforce a soft lower bound for balance.
            # Upper bound is NOT in the master — with high Λ, expected demand
            # per trip far exceeds single-vehicle capacity, but multiple vehicles
            # serve the same line (fleet_size = tour_time / (speed * T)).
            L_min_pax = 1.0  # soft lower

            for line in lines:
                for ti, T_t in enumerate(T_vals):
                    if (line.idx, ti) not in x_v:
                        continue
                    expected_pax = gp.quicksum(
                        Lambda * T_t * prob_dict.get(b, 0.0) * z_v[(b, line.idx, ti)]
                        for b in line.reachable_blocks
                        if (b, line.idx, ti) in z_v)
                    # Soft lower only
                    master.addConstr(
                        expected_pax + slack_low_v[(line.idx, ti)]
                        >= L_min_pax * x_v[(line.idx, ti)],
                        name=f"load_lo_{line.idx}_{ti}")

            # θ active only when trip operates
            M_big = max(l.est_tour_km for l in lines) * 100.0 + 1000.0
            for k in x_v:
                master.addConstr(
                    theta_v[k] <= M_big * x_v[k],
                    name=f"theta_link_{k[0]}_{k[1]}")

            # Warm-start cuts: θ ≥ shortest-path cost through checkpoint network
            for line in lines:
                cps = line.checkpoints
                mc = len(cps)
                if mc <= 1:
                    continue
                dp = [float('inf')] * mc
                dp[0] = 0.0
                for j in range(1, mc):
                    for ii in range(max(0, j - self.K_skip - 1), j):
                        d = float(np.linalg.norm(
                            block_pos_km[cps[j]] - block_pos_km[cps[ii]]))
                        dp[j] = min(dp[j], dp[ii] + d)
                sp_km = dp[-1]
                for ti, T_t in enumerate(T_vals):
                    if (line.idx, ti) not in x_v:
                        continue
                    cm = 1.0 / (discount * T_t) + wr / (2.0 * speed)
                    master.addConstr(
                        theta_v[(line.idx, ti)] >= sp_km * cm * x_v[(line.idx, ti)],
                        name=f"theta_ws_{line.idx}_{ti}")

            # Objective (FR-consistent: no linehaul, no ODD)
            #
            # Wait cost is demand-weighted via z variables:
            #   wr * T_t/2 * Σ_b prob[b] * z[b,ℓ,t]
            # This makes the master properly internalize that a line serving
            # more demand incurs more total wait cost.
            #
            # Pre-index z_v keys by (line_idx, ti) for efficient objective build
            z_by_trip: Dict[Tuple[int, int], List[Tuple[str, object]]] = {}
            for (b, li, ti), var in z_v.items():
                z_by_trip.setdefault((li, ti), []).append((b, var))

            obj_expr = gp.LinExpr()
            for k in x_v:
                li, ti = k
                T_t = T_vals[ti]
                # Recourse proxy + slack penalty
                obj_expr += theta_v[k] + penalty_low * slack_low_v[k]
                # Demand-weighted wait: wr * T/2 * Σ prob[b] * z[b,ℓ,t]
                for b, zvar in z_by_trip.get((li, ti), []):
                    coeff = wr * T_t / 2.0 * prob_dict.get(b, 0.0)
                    if coeff > 1e-15:
                        obj_expr += coeff * zvar
            if not hard_coverage:
                obj_expr += penalty_uncov * gp.quicksum(
                    uncov_v[b] for b in block_ids)
            master.setObjective(obj_expr, GRB.MINIMIZE)

            return master, x_v, z_v, theta_v, uncov_v

        # Try hard coverage first; fall back to soft if infeasible
        master, x, z, theta, uncov = _build_master(hard_coverage=True)
        master.optimize()
        if master.status != GRB.OPTIMAL:
            print(f"  [MiND] Hard coverage infeasible with K={K}, "
                  f"falling back to soft coverage")
            master, x, z, theta, uncov = _build_master(hard_coverage=False)
            master.optimize()
            if master.status == GRB.OPTIMAL and uncov:
                uncov_blocks = [b for b in block_ids
                                if b in uncov and uncov[b].X > 0.5]
                if uncov_blocks:
                    print(f"  [MiND] Uncovered blocks in fallback: {uncov_blocks}")
        # Reset for Benders — will re-optimize in the loop
        # (The initial solve was just to check feasibility)

        # ---- Step 4: Benders loop ----
        pending_cuts: List[dict] = []
        cuts_added = 0
        best_ub = float('inf')
        best_solution = None

        # Build line-obj lookup
        line_by_idx = {ln.idx: ln for ln in lines}

        for benders_iter in range(self.max_iters):
            # Add pending cuts
            for cut in pending_cuts[cuts_added:]:
                li, ti = cut['trip']
                if (li, ti) not in theta:
                    continue
                master.addConstr(
                    theta[(li, ti)] >= cut['const'] + gp.quicksum(
                        cut['coeffs'].get(b, 0.0) * z[(b, li, ti)]
                        for b in cut['blocks'] if (b, li, ti) in z
                    ),
                    name=f"cut_{cut['id']}")
            cuts_added = len(pending_cuts)

            # Solve master
            master.optimize()
            if master.status != GRB.OPTIMAL:
                print(f"  [MiND Benders] Master infeasible at iter {benders_iter}")
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
                z_dict = {b: z_star_all.get((b, li, ti), 0.0)
                          for b in line_obj.reachable_blocks}

                # All assigned blocks participate in linking/coverage,
                # including non-skipped checkpoint endpoints.
                cp_set_local = set(line_obj.checkpoints)
                all_blocks = set(b for b in line_obj.reachable_blocks
                                 if z_dict.get(b, 0.0) > 1e-9)

                cost_mult = 1.0 / (discount * T_t) + wr / (2.0 * speed)

                # First-stage cost: demand-weighted wait from z solution
                op_wait = sum(wr * T_t / 2.0 * prob_dict.get(b, 0.0)
                              * z_dict.get(b, 0.0)
                              for b in line_obj.reachable_blocks)

                if mc <= 1 or len(initial_arcs) == 0:
                    ub_iter += op_wait
                    iter_solution[(li, ti)] = (z_dict, T_t, 0.0)
                    continue

                # Generate demand per scenario
                demand_matrix = _crn_demand(
                    U_base, Lambda, T_t, prob_dict, block_ids)
                scenario_costs = []

                for s in range(S):
                    demand_s = {block_ids[j]: int(demand_matrix[s, j])
                                for j in range(N)}

                    # Per-scenario walking penalties and costs
                    s_walk_pen: Dict[str, float] = {}
                    s_walk_det_cost: Dict[str, float] = {}
                    for b in all_blocks:
                        d_b = demand_s.get(b, 0)
                        s_walk_pen[b] = (wr * walk_saved_km.get(b, 0.0)
                                         * d_b / walk_speed)
                        s_walk_det_cost[b] = (wr * walk_det_km.get(b, 0.0)
                                              * d_b / walk_speed)

                    # Solve with column generation
                    lp_obj_s, lp_coeffs_s = _solve_routing_lp_with_cg(
                        initial_arcs, line_obj, block_ids, block_pos_km,
                        mc, self.capacity, z_dict, demand_s, all_blocks,
                        cp_set_local, cost_mult, s_walk_pen,
                        self.delta, speed, self.alpha_time, self.K_skip,
                        max_cg_iters=self.max_cg_iters)

                    if lp_obj_s is None:
                        max_dist = line_obj.est_tour_km + 10.0
                        fallback_pen = 10.0 * max_dist * cost_mult
                        scenario_costs.append(fallback_pen * N)
                        continue

                    # Total scenario cost = LP routing/walk-penalty
                    #   + base walking cost (walk_det for served blocks)
                    base_walk = sum(
                        s_walk_det_cost.get(b, 0.0) * z_dict.get(b, 0.0)
                        for b in all_blocks)
                    total_obj_s = lp_obj_s + base_walk
                    scenario_costs.append(total_obj_s)

                    # Benders cut coefficients
                    full_coeffs: Dict[str, float] = {}
                    for b in all_blocks:
                        lp_coeff = (lp_coeffs_s.get(b, 0.0)
                                    if lp_coeffs_s else 0.0)
                        wdc = s_walk_det_cost.get(b, 0.0)
                        coeff = lp_coeff + wdc
                        if abs(coeff) > 1e-12:
                            full_coeffs[b] = coeff

                    if full_coeffs:
                        const = total_obj_s - sum(
                            full_coeffs.get(b, 0.0) * z_dict.get(b, 0.0)
                            for b in all_blocks)
                        pending_cuts.append({
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
            if gap < tol:
                break
            if n_new_cuts == 0 and benders_iter > 0:
                break

        # ---- Step 5: Map to ServiceDesign ----
        if best_solution is None:
            # Fallback: trivial design
            assignment = np.eye(N)
            return ServiceDesign(
                name=self._name, assignment=assignment,
                depot_id=depot_id,
                district_roots=[block_ids[0]],
                dispatch_intervals={block_ids[0]: 1.0},
                district_routes={})

        _, z_star_final, _, active_trips = best_solution

        assignment = np.zeros((N, N))
        roots_final: List[str] = []
        dispatch_intervals: Dict[str, float] = {}
        district_routes: Dict[str, List[int]] = {}

        used_roots: Set[str] = set()

        for li, ti in active_trips:
            T_t = T_vals[ti]
            line_obj = line_by_idx.get(li)
            if line_obj is None:
                continue

            # Assigned blocks: those with z > 0.5
            assigned_blocks = []
            for b in line_obj.reachable_blocks:
                if z_star_final.get((b, li, ti), 0.0) > 0.5:
                    assigned_blocks.append(b)
            if not assigned_blocks:
                continue

            # District root: pick a unique block not already used as root.
            # Prefer checkpoints (in order), then any assigned block.
            root_id = None
            for cp in line_obj.checkpoints:
                if cp in assigned_blocks and cp not in used_roots:
                    root_id = cp
                    break
            if root_id is None:
                for b in assigned_blocks:
                    if b not in used_roots:
                        root_id = b
                        break
            if root_id is None:
                continue  # all blocks already roots (degenerate)

            if root_id not in assigned_blocks:
                assigned_blocks.append(root_id)
            root_idx = block_ids.index(root_id)
            used_roots.add(root_id)

            # Fill assignment matrix
            for b in assigned_blocks:
                j = block_ids.index(b)
                assignment[j, root_idx] = 1.0

            roots_final.append(root_id)
            dispatch_intervals[root_id] = T_t

            # District route: checkpoint ordering as global block indices
            route_order = []
            for cp in line_obj.checkpoints:
                if cp in assigned_blocks:
                    route_order.append(block_ids.index(cp))
            district_routes[root_id] = route_order

        # Assign uncovered blocks to nearest district root
        if roots_final:
            for j, b in enumerate(block_ids):
                if assignment[j].sum() < 0.5:
                    best_root = min(
                        roots_final,
                        key=lambda r: float(np.linalg.norm(
                            block_pos_km[b] - block_pos_km[r])))
                    ri = block_ids.index(best_root)
                    assignment[j, ri] = 1.0

        design = ServiceDesign(
            name=self._name,
            assignment=assignment,
            depot_id=depot_id,
            district_roots=roots_final,
            dispatch_intervals=dispatch_intervals,
            district_routes=district_routes,
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
    ) -> EvaluationResult:
        """Dispatch to evaluate() so run_baseline uses FR-specific costs."""
        return self.evaluate(
            design, geodata, prob_dict, Omega_dict, J_function,
            Lambda, wr, wv)

    # ------------------------------------------------------------------
    # Evaluate — subpath routing (same model as optimization)
    # ------------------------------------------------------------------

    def evaluate(
        self, design: ServiceDesign, geodata,
        prob_dict: Dict[str, float], Omega_dict,
        J_function: Callable, Lambda: float,
        wr: float, wv: float,
        beta: float = BETA, road_network=None,
    ) -> EvaluationResult:
        """FR-specific evaluation using subpath routing on checkpoint network.

        For each district and demand scenario, solves the subpath routing LP
        (same model as the Benders subproblem) to decide which corridor blocks
        are detoured to vs walked.  This aligns optimization and evaluation
        on the same cost model.

        Cost structure:
          - No separate linehaul (depot travel is in the checkpoint tour)
          - No ODD cost (fixed schedule, not on-demand dispatch)
          - Provider cost = (checkpoint_tour_km + detour_km) / (discount * T)
          - User cost = T/2 (wait) + tour_km/(2*speed) (in-vehicle) + walk
        """
        block_ids = geodata.short_geoid_list
        N = len(block_ids)
        assignment = design.assignment
        depot_id = design.depot_id
        discount = TSP_TRAVEL_DISCOUNT
        speed = VEHICLE_SPEED_KMH
        walk_speed = self.walk_speed_kmh

        # Build block positions in km
        block_pos_km = {}
        for b in block_ids:
            block_pos_km[b] = np.array(_get_pos(geodata, b),
                                       dtype=float) / 1000.0
        depot_pos_km = np.array(_get_pos(geodata, depot_id),
                                dtype=float) / 1000.0

        # Pre-compute walking geometry per block
        walk_saved_km: Dict[str, float] = {}
        walk_det_km: Dict[str, float] = {}
        walk_no_det_km: Dict[str, float] = {}
        for b in block_ids:
            area_b = float(geodata.get_area(b))
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

        # SAA: generate demand scenarios
        rng = np.random.default_rng(42)
        n_eval = self.num_scenarios
        U_base = rng.uniform(0.0, 1.0, size=(n_eval, N))

        acc_in_district = 0.0
        acc_wait = 0.0
        acc_invehicle = 0.0
        acc_walk_riders = 0.0  # walk_km * n_riders accumulated
        acc_riders = 0.0       # total riders accumulated
        acc_fleet = 0.0
        acc_T_weighted = 0.0
        acc_prob = 0.0

        for i, root in enumerate(block_ids):
            if round(assignment[i, i]) != 1:
                continue
            assigned = [j for j in range(N) if round(assignment[j, i]) == 1]
            if not assigned:
                continue

            assigned_blocks = [block_ids[j] for j in assigned]
            T_i = design.dispatch_intervals.get(root, 1.0)
            district_prob = sum(prob_dict.get(b, 0.0) for b in assigned_blocks)

            # Build checkpoint list from district_routes
            route_global = design.district_routes.get(root, [])
            checkpoints = []
            for gi in route_global:
                bid = block_ids[gi]
                if bid in assigned_blocks:
                    checkpoints.append(bid)
            if not checkpoints:
                # All assigned blocks as checkpoints (NN order)
                pts = np.array([block_pos_km[b] for b in assigned_blocks])
                order = _greedy_nn_route(pts, depot_pos_km)
                order = _two_opt_improve(order, pts, depot_pos_km)
                checkpoints = [assigned_blocks[o] for o in order]

            m = len(checkpoints)
            cp_set = set(checkpoints)
            all_blocks_set = set(assigned_blocks)

            # Build subpath arcs for this district's checkpoint route
            # (Wrap as a ReferenceLine-like object for _build_initial_arcs)
            line_obj = ReferenceLine(
                idx=0, checkpoints=checkpoints,
                reachable_blocks=all_blocks_set,
                est_tour_km=_tour_distance(depot_pos_km, checkpoints,
                                           block_pos_km))

            arcs = _build_initial_arcs(
                line_obj, block_ids, block_pos_km, self.delta,
                self.K_skip, speed, self.alpha_time)

            # Checkpoint tour distance (depot → checkpoints → depot)
            cp_tour_km = line_obj.est_tour_km

            # Generate demand scenarios for this district's interval
            demand_matrix = _crn_demand(
                U_base, Lambda, T_i, prob_dict, block_ids)

            # Per-scenario: solve subpath LP to get detour routing + walk
            acc_route_km = 0.0      # checkpoint + detour km
            district_walk_riders = 0.0
            district_riders = 0.0

            for s in range(n_eval):
                demand_s = {block_ids[j]: int(demand_matrix[s, j])
                            for j in range(N)}
                n_total = sum(demand_s.get(b, 0) for b in assigned_blocks)
                if n_total == 0:
                    acc_route_km += cp_tour_km
                    continue

                # z_star = 1 for all assigned blocks (evaluation)
                z_star = {b: 1.0 for b in all_blocks_set}

                # Same cost_mult as optimization so the detour trade-off
                # is identical: cost_mult * detour_km vs walk_pen.
                eval_cost_mult = 1.0 / (discount * T_i) + wr / (2.0 * speed)

                # Walk penalties: same scaling as optimization subproblem
                wp = {}
                for b in all_blocks_set:
                    d_b = demand_s.get(b, 0)
                    wp[b] = wr * walk_saved_km.get(b, 0.0) * d_b / walk_speed

                if m > 1 and arcs:
                    # Solve with CG — same cost model as master subproblem
                    result = _solve_routing_lp_with_cg(
                        arcs, line_obj, block_ids, block_pos_km,
                        m, self.capacity, z_star, demand_s,
                        all_blocks_set, cp_set, eval_cost_mult, wp,
                        self.delta, speed, self.alpha_time, self.K_skip,
                        max_cg_iters=15, return_solution=True)
                    _, _, detour_km, u_vals = result

                    if detour_km is not None:
                        acc_route_km += cp_tour_km + detour_km
                        # Walk: walk_det for served blocks, walk_no_det for unserved
                        for b in assigned_blocks:
                            d_b = demand_s.get(b, 0)
                            if d_b == 0:
                                continue
                            u_b = u_vals.get(b, 1.0) if b not in cp_set else 0.0
                            # Checkpoint blocks: always visited, walk = walk_det
                            # Corridor served (u≈0): walk = walk_det
                            # Corridor unserved (u≈1): walk = walk_no_det
                            w = (walk_det_km.get(b, 0.0) * (1.0 - u_b)
                                 + walk_no_det_km.get(b, 0.0) * u_b)
                            district_walk_riders += w * d_b
                        district_riders += n_total
                    else:
                        # Fallback: checkpoint-only route
                        acc_route_km += cp_tour_km
                        for b in assigned_blocks:
                            d_b = demand_s.get(b, 0)
                            if b in cp_set:
                                district_walk_riders += walk_det_km.get(b, 0.0) * d_b
                            else:
                                district_walk_riders += walk_no_det_km.get(b, 0.0) * d_b
                        district_riders += n_total
                else:
                    # Singleton or no arcs: just checkpoint tour
                    acc_route_km += cp_tour_km
                    for b in assigned_blocks:
                        d_b = demand_s.get(b, 0)
                        if b in cp_set:
                            district_walk_riders += walk_det_km.get(b, 0.0) * d_b
                        else:
                            district_walk_riders += walk_no_det_km.get(b, 0.0) * d_b
                    district_riders += n_total

            avg_tour_km = acc_route_km / n_eval

            # Accumulate into overall metrics
            provider_d = (avg_tour_km / discount / T_i) * district_prob
            acc_in_district += provider_d
            acc_wait += (T_i / 2.0) * district_prob
            acc_invehicle += (avg_tour_km / (2.0 * speed)) * district_prob
            acc_fleet += avg_tour_km / (speed * T_i)
            acc_walk_riders += district_walk_riders
            acc_riders += district_riders
            acc_T_weighted += T_i * district_prob
            acc_prob += district_prob

        acc_prob = max(acc_prob, 1e-12)

        avg_wait = acc_wait / acc_prob
        avg_invehicle = acc_invehicle / acc_prob
        avg_walk_km = acc_walk_riders / max(acc_riders, 1.0)
        avg_walk = avg_walk_km / walk_speed
        total_user_time = avg_wait + avg_invehicle + avg_walk

        in_district_cost = acc_in_district
        fleet_size = acc_fleet
        avg_dispatch_interval = acc_T_weighted / acc_prob

        provider_cost = in_district_cost  # no linehaul, no ODD
        user_cost = wr * total_user_time * acc_prob
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
            avg_dispatch_interval=avg_dispatch_interval,
            odd_cost=0.0,
            provider_cost=provider_cost,
            user_cost=user_cost,
            total_cost=total_cost,
        )
