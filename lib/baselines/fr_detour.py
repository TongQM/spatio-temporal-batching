"""
Multi Fixed-Route with Detour — MiND adaptation (Martin-Iradi et al. 2025)

Adaptation of the MiND (Microtransit Network Design) model from:
  "A Double Decomposition Algorithm for Network Planning and Operations in
  Deviated Fixed-route Microtransit"
  (Martin-Iradi, Schmid, Cummings, Jacquillat 2025, arXiv:2402.01265)

to a steady-state district-based service design setting.

Algorithm structure (faithfully adapted from the paper):
  Double decomposition:
    - Outer: multi-cut Benders with SAA scenario-wise optimality cuts
    - Inner: subpath enumeration (or column generation for large instances)

  First stage — line selection + demand assignment:
    - Generates a library of candidate reference lines from depot
    - Master MIP selects K lines, chooses dispatch intervals, assigns blocks.
    - Variables: x[ℓ,t] (trip selection), z[b,ℓ,t] (assignment), θ[ℓ,t] (recourse)

  Second stage — subpath routing on load-expanded checkpoint network:
    - Per selected trip, builds a (checkpoint, load) state-space network
    - Subpath arcs between checkpoint pairs (with skip ≤ K_skip) serve
      corridor blocks within δ of the line segment
    - Routing LP with flow balance, linking constraints z*→y, demand coverage
    - LP duals from linking constraints produce valid Benders cuts

Deliberate simplifications from the paper:
  - Steady-state operation (fixed dispatch interval) vs finite horizon
  - Block-level demand aggregation vs individual-passenger tracking
  - Arc costs = vehicle travel distance (reduced operational-cost model);
    service-quality penalties computed at evaluation time
  - Line library via angular partition + NN-TSP (not pre-existing network)
  - Load = number of blocks served (not individual passengers)

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
    _get_pos, evaluate_design,
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
    cp_from: int                # checkpoint index (source end)
    cp_to: int                  # checkpoint index (destination end)
    load_delta: int             # number of blocks served
    cost_km: float              # vehicle travel distance (km)
    served_blocks: FrozenSet[str]  # block IDs served by this subpath


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


def _compute_reachable(checkpoints, block_ids, block_pos_km, delta):
    """Blocks within δ of any checkpoint position."""
    reachable = set(checkpoints)
    if delta <= 0:
        return reachable
    for cp in checkpoints:
        cp_pos = block_pos_km[cp]
        for b in block_ids:
            if b not in reachable:
                d = float(np.linalg.norm(block_pos_km[b] - cp_pos))
                if d <= delta + 1e-9:
                    reachable.add(b)
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
                           num_districts=None):
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
        reach = _compute_reachable(cps, block_ids, block_pos_km, delta)
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
        reachable = _compute_reachable([b], block_ids, block_pos_km, delta)
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


def _build_arcs_for_line(line, block_ids, block_pos_km, delta, capacity, K_skip):
    """Build subpath arcs for the load-expanded checkpoint network.

    Returns list of SubpathArc objects (load-independent; LP creates copies
    per valid starting load).

    IMPORTANT: served_blocks only includes OFF-CHECKPOINT corridor blocks.
    Checkpoint blocks are automatically served by the vehicle traversing the
    reference line (flow balance ensures all checkpoints are visited).
    The routing LP's linking/coverage constraints only apply to corridor blocks.
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

            # Corridor blocks: OFF-checkpoint blocks within δ of the segment
            # Skipped checkpoints (between i and j) are ON the line and served
            # automatically — they are NOT included in corridor.
            corridor = []
            if delta > 0:
                for b in block_ids:
                    if b in cp_set:
                        continue  # checkpoint: served automatically
                    d = _point_to_segment_dist(block_pos_km[b], ci_pos, cj_pos)
                    if d <= delta + 1e-9:
                        corridor.append(b)

            # Enumerate subsets of corridor blocks (up to 5 for tractability)
            n_corr = len(corridor)
            max_enum = min(n_corr, 5)

            # Direct arc: no deviation, no corridor blocks served
            direct_cost = float(np.linalg.norm(cj_pos - ci_pos))
            arcs.append(SubpathArc(
                arc_id=arc_id, cp_from=i, cp_to=j,
                load_delta=0, cost_km=direct_cost,
                served_blocks=frozenset(),  # checkpoints served automatically
            ))
            arc_id += 1

            # Subsets of corridor (off-checkpoint) blocks
            for size in range(1, max_enum + 1):
                for subset in combinations(range(n_corr), size):
                    blocks_in_subset = [corridor[s] for s in subset]
                    served = frozenset(blocks_in_subset)
                    served_pos = [block_pos_km[b] for b in blocks_in_subset]
                    cost = _subpath_cost(ci_pos, cj_pos, served_pos)
                    arcs.append(SubpathArc(
                        arc_id=arc_id, cp_from=i, cp_to=j,
                        load_delta=len(blocks_in_subset),
                        cost_km=cost, served_blocks=served,
                    ))
                    arc_id += 1

    return arcs


# ---------------------------------------------------------------------------
# Routing LP on load-expanded checkpoint network (Step 2b)
# ---------------------------------------------------------------------------

def _solve_routing_lp(arcs, m, capacity, z_star, demand_scenario,
                      corridor_blocks, cost_mult, penalty):
    """Solve the subpath routing LP for one scenario on one line.

    Args:
        arcs: list of SubpathArc
        m: number of checkpoints
        capacity: max blocks served (load limit)
        z_star: dict block_id → assignment value (from master)
        demand_scenario: dict block_id → demand count
        corridor_blocks: set of OFF-CHECKPOINT blocks that could be served
            via deviation. Checkpoint blocks are served automatically.
        cost_mult: multiplier to convert km to cost units
        penalty: cost per unserved demand block

    Returns: (objective_value, pi_duals_dict) or (None, None) if infeasible.
    """
    import gurobipy as gp
    from gurobipy import GRB

    model = gp.Model("routing_lp")
    model.setParam('OutputFlag', 0)

    # Pre-compute incidence: which arcs serve each block
    arcs_serving: Dict[str, List[int]] = {}
    for b in corridor_blocks:
        arcs_serving[b] = [a.arc_id for a in arcs if b in a.served_blocks]

    # --- Variables ---
    # y[arc_id, load_from] for each arc and valid starting load
    y = {}
    for a in arcs:
        for l in range(capacity - a.load_delta + 1):
            y[(a.arc_id, l)] = model.addVar(lb=0.0, name=f"y_{a.arc_id}_{l}")

    # Terminal arcs: (last_checkpoint, load) → super-sink
    term = {}
    for l in range(capacity + 1):
        term[l] = model.addVar(lb=0.0, name=f"term_{l}")

    # Unserved slack
    u = {}
    for b in corridor_blocks:
        u[b] = model.addVar(lb=0.0, name=f"u_{b}")

    model.update()

    # --- Flow balance at each (checkpoint, load) node ---
    for c in range(m):
        for l in range(capacity + 1):
            outflow = gp.LinExpr()
            inflow = gp.LinExpr()

            # Outflow from (c, l)
            for a in arcs:
                if a.cp_from == c and l <= capacity - a.load_delta:
                    outflow += y[(a.arc_id, l)]
            if c == m - 1:
                outflow += term[l]

            # Inflow to (c, l)
            for a in arcs:
                if a.cp_to == c:
                    l_from = l - a.load_delta
                    if 0 <= l_from <= capacity - a.load_delta:
                        inflow += y[(a.arc_id, l_from)]

            rhs = 1.0 if (c == 0 and l == 0) else 0.0
            model.addConstr(outflow - inflow == rhs, name=f"flow_{c}_{l}")

    # Super-sink: total terminal flow = 1
    model.addConstr(
        gp.quicksum(term[l] for l in range(capacity + 1)) == 1.0,
        name="sink")

    # --- Linking constraints: can only serve b if assigned ---
    link_constrs: Dict[str, object] = {}
    for b in corridor_blocks:
        aid_list = arcs_serving.get(b, [])
        if not aid_list:
            continue
        zval = z_star.get(b, 0.0)
        expr = gp.LinExpr()
        for aid in aid_list:
            a = arcs[aid] if aid < len(arcs) else None
            if a is None:
                continue
            for l in range(capacity - a.load_delta + 1):
                if (aid, l) in y:
                    expr += y[(aid, l)]
        if expr.size() > 0:
            link_constrs[b] = model.addConstr(expr <= zval, name=f"link_{b}")

    # --- Demand coverage with slack ---
    for b in corridor_blocks:
        if demand_scenario.get(b, 0) <= 0:
            continue
        zval = z_star.get(b, 0.0)
        if zval < 1e-9:
            continue
        aid_list = arcs_serving.get(b, [])
        expr = gp.LinExpr()
        for aid in aid_list:
            a = arcs[aid] if aid < len(arcs) else None
            if a is None:
                continue
            for l in range(capacity - a.load_delta + 1):
                if (aid, l) in y:
                    expr += y[(aid, l)]
        model.addConstr(expr + u[b] >= zval, name=f"cov_{b}")

    # --- Objective ---
    obj = gp.LinExpr()
    for a in arcs:
        for l in range(capacity - a.load_delta + 1):
            obj += cost_mult * a.cost_km * y[(a.arc_id, l)]
    for b in corridor_blocks:
        obj += penalty * u[b]
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        return None, None

    # Extract duals from linking constraints
    pi = {}
    for b, constr in link_constrs.items():
        pi[b] = constr.Pi
    return model.ObjVal, pi


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
    Multi Fixed-Route with Detour — MiND adaptation.

    Double decomposition: Benders master (line selection + assignment) +
    subpath routing LP on load-expanded checkpoint network.

    Parameters
    ----------
    delta          : max vehicle deviation from route (km); 0 = Multi-FR
    max_iters      : max Benders iterations
    num_scenarios  : Monte Carlo scenarios for stochastic evaluation
    max_fleet      : optional fleet budget (max simultaneous vehicles)
    capacity       : max blocks served per trip (load-expanded network)
    n_angular      : number of angular sectors for line library
    K_skip         : max checkpoints that can be skipped between subpath ends
    n_saa          : SAA scenarios for multi-cut Benders
    T_values       : list of candidate dispatch intervals (hours)
    name           : display name override
    """

    def __init__(
        self,
        delta: float = 0.3,
        max_iters: int = 10,
        walk_speed_kmh: float = 5.0,
        num_scenarios: int = 200,
        max_fleet: float | None = None,
        capacity: int = 20,
        n_angular: int = 8,
        K_skip: int = 2,
        n_saa: int = 30,
        T_values: List[float] | None = None,
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

    @staticmethod
    def _odd_cost(Omega_dict, J_function, assigned_blocks):
        if Omega_dict and J_function:
            ref = next(iter(Omega_dict.values()))
            omega = np.zeros_like(ref)
            for b in assigned_blocks:
                omega = np.maximum(omega, Omega_dict.get(b, np.zeros_like(ref)))
            return J_function(omega)
        return 0.0

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
            num_districts=K)
        L = len(lines)
        T_vals = self.T_values
        n_T = len(T_vals)

        # Pre-compute linehaul and ODD for each line
        line_K = {}  # linehaul distance (km)
        line_F = {}  # ODD cost
        for line in lines:
            if line.checkpoints:
                first_cp = block_pos_km[line.checkpoints[0]]
                line_K[line.idx] = float(np.linalg.norm(first_cp - depot_pos))
            else:
                line_K[line.idx] = 0.0
            reachable_list = [b for b in line.reachable_blocks if b in block_ids]
            line_F[line.idx] = self._odd_cost(Omega_dict, J_function, reachable_list)

        # Pre-build subpath arcs for each line
        line_arcs = {}
        for line in lines:
            line_arcs[line.idx] = _build_arcs_for_line(
                line, block_ids, block_pos_km, self.delta,
                self.capacity, self.K_skip)
            # Build arc_id → arc mapping
            for i, a in enumerate(line_arcs[line.idx]):
                a.arc_id = i  # re-index per line

        # ---- SAA: Common random numbers ----
        rng = np.random.default_rng(42)
        S = self.n_saa
        U_base = rng.uniform(0.0, 1.0, size=(S, N))

        # ---- Step 3: Build master MIP ----
        master = gp.Model("MiND_Benders_Master")
        master.setParam('OutputFlag', 0)

        # Variables
        x = {}      # x[ℓ, t] binary: trip operates
        z = {}      # z[b, ℓ, t] continuous: block assignment
        theta = {}  # θ[ℓ, t] continuous: recourse cost
        slack_low = {}  # slack for soft lower load bound

        for line in lines:
            for ti, T_t in enumerate(T_vals):
                x[(line.idx, ti)] = master.addVar(
                    vtype=GRB.BINARY, name=f"x_{line.idx}_{ti}")
                theta[(line.idx, ti)] = master.addVar(
                    lb=0.0, name=f"theta_{line.idx}_{ti}")
                slack_low[(line.idx, ti)] = master.addVar(
                    lb=0.0, name=f"slack_{line.idx}_{ti}")
                for b in line.reachable_blocks:
                    z[(b, line.idx, ti)] = master.addVar(
                        lb=0.0, ub=1.0, name=f"z_{b}_{line.idx}_{ti}")

        master.update()

        # Coverage: every block assigned to at most one trip (soft: penalize uncovered)
        uncov = {}
        penalty_uncov = 100.0  # high penalty to strongly prefer full coverage
        for b in block_ids:
            uncov[b] = master.addVar(lb=0.0, ub=1.0, name=f"uncov_{b}")
        master.update()

        for b in block_ids:
            expr = gp.LinExpr()
            for line in lines:
                if b in line.reachable_blocks:
                    for ti in range(n_T):
                        expr += z[(b, line.idx, ti)]
            master.addConstr(expr + uncov[b] == 1.0, name=f"cov_{b}")

        # Assignment requires operation
        for line in lines:
            for ti in range(n_T):
                for b in line.reachable_blocks:
                    master.addConstr(
                        z[(b, line.idx, ti)] <= x[(line.idx, ti)],
                        name=f"assign_{b}_{line.idx}_{ti}")

        # One interval per line
        for line in lines:
            master.addConstr(
                gp.quicksum(x[(line.idx, ti)] for ti in range(n_T)) <= 1,
                name=f"one_T_{line.idx}")

        # K lines selected
        master.addConstr(
            gp.quicksum(x[(line.idx, ti)]
                        for line in lines for ti in range(n_T)) == K,
            name="K_lines")

        # Fleet budget (if specified)
        if self.max_fleet is not None:
            master.addConstr(
                gp.quicksum(
                    line.est_tour_km / (speed * T_vals[ti]) * x[(line.idx, ti)]
                    for line in lines for ti in range(n_T)
                ) <= self.max_fleet,
                name="fleet_budget")

        # Load bounds (block count): hard upper, soft lower
        L_max = self.capacity   # max blocks assigned to one trip
        L_min = 1.0             # min blocks assigned (soft)
        penalty_low = 0.1       # soft penalty for violating lower bound

        for line in lines:
            for ti, T_t in enumerate(T_vals):
                block_count = gp.quicksum(
                    z[(b, line.idx, ti)]
                    for b in line.reachable_blocks)
                # Hard upper: at most capacity blocks per trip
                master.addConstr(
                    block_count <= L_max * x[(line.idx, ti)],
                    name=f"load_hi_{line.idx}_{ti}")
                # Soft lower
                master.addConstr(
                    block_count + slack_low[(line.idx, ti)]
                    >= L_min * x[(line.idx, ti)],
                    name=f"load_lo_{line.idx}_{ti}")

        # θ active only when trip operates
        M_big = max(l.est_tour_km for l in lines) * 100.0 + 1000.0
        for line in lines:
            for ti in range(n_T):
                master.addConstr(
                    theta[(line.idx, ti)] <= M_big * x[(line.idx, ti)],
                    name=f"theta_link_{line.idx}_{ti}")

        # Warm-start: θ ≥ shortest-path routing cost through checkpoint network.
        # Uses DP to find min-cost flow (may skip up to K_skip checkpoints).
        for line in lines:
            cps = line.checkpoints
            mc = len(cps)
            if mc <= 1:
                continue
            # Shortest path DP: dp[j] = min cost from cp_0 to cp_j
            dp = [float('inf')] * mc
            dp[0] = 0.0
            for j in range(1, mc):
                for i in range(max(0, j - self.K_skip - 1), j):
                    d = float(np.linalg.norm(
                        block_pos_km[cps[j]] - block_pos_km[cps[i]]))
                    dp[j] = min(dp[j], dp[i] + d)
            sp_km = dp[-1]  # shortest path through checkpoint graph
            for ti, T_t in enumerate(T_vals):
                cm = 1.0 / (discount * T_t) + wr / (2.0 * speed)
                master.addConstr(
                    theta[(line.idx, ti)] >= sp_km * cm * x[(line.idx, ti)],
                    name=f"theta_warmstart_{line.idx}_{ti}")

        # Operating cost per trip
        def _op_cost(line_idx, ti):
            T_t = T_vals[ti]
            return (line_K[line_idx] + line_F[line_idx]) / T_t + wr * T_t / 2.0

        # Objective: operating cost + recourse + soft penalties
        master.setObjective(
            gp.quicksum(
                _op_cost(line.idx, ti) * x[(line.idx, ti)]
                + theta[(line.idx, ti)]
                + penalty_low * slack_low[(line.idx, ti)]
                for line in lines for ti in range(n_T)
            ) + penalty_uncov * gp.quicksum(uncov[b] for b in block_ids),
            GRB.MINIMIZE)

        # ---- Step 4: Benders loop ----
        pending_cuts: List[dict] = []
        cuts_added = 0
        best_ub = float('inf')
        best_solution = None

        for benders_iter in range(self.max_iters):
            # Add pending cuts
            for cut in pending_cuts[cuts_added:]:
                li, ti = cut['trip']
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
            for line in lines:
                for ti in range(n_T):
                    x_star[(line.idx, ti)] = round(x[(line.idx, ti)].X)
                    for b in line.reachable_blocks:
                        z_star_all[(b, line.idx, ti)] = z[(b, line.idx, ti)].X

            # Active trips
            active_trips = [(li, ti) for (li, ti), v in x_star.items() if v > 0.5]

            # Solve subproblems
            ub_iter = 0.0
            iter_solution = {}  # (li, ti) → (z_dict, T_t, cost)

            for li, ti in active_trips:
                T_t = T_vals[ti]
                line = lines[li]  # lines indexed by position; find by idx
                # Find actual line object
                line_obj = None
                for ln in lines:
                    if ln.idx == li:
                        line_obj = ln
                        break
                if line_obj is None:
                    continue

                m = len(line_obj.checkpoints)
                arcs = line_arcs[li]
                z_dict = {b: z_star_all.get((b, li, ti), 0.0)
                          for b in line_obj.reachable_blocks}
                # Corridor blocks: reachable but NOT on the checkpoint path
                cp_set = set(line_obj.checkpoints)
                corridor = line_obj.reachable_blocks - cp_set

                # Cost multiplier: convert km to cost units
                cost_mult = 1.0 / (discount * T_t) + wr / (2.0 * speed)
                max_dist = line_obj.est_tour_km + 10.0
                pen = 10.0 * max_dist * cost_mult

                op = _op_cost(li, ti)

                if m <= 1 or len(arcs) == 0:
                    # Singleton line: routing cost = 0
                    ub_iter += op
                    iter_solution[(li, ti)] = (z_dict, T_t, 0.0)
                    # No cuts from singleton (routing cost doesn't depend on z)
                    continue

                # Generate demand and solve routing LP per scenario
                demand_matrix = _crn_demand(U_base, Lambda, T_t, prob_dict, block_ids)
                scenario_costs = []

                for s in range(S):
                    demand_s = {block_ids[j]: int(demand_matrix[s, j])
                                for j in range(N)}
                    obj_s, pi_s = _solve_routing_lp(
                        arcs, m, self.capacity, z_dict, demand_s,
                        corridor, cost_mult, pen)

                    if obj_s is None:
                        obj_s = pen * N  # fallback
                        pi_s = {}

                    scenario_costs.append(obj_s)

                    # Build scenario-wise Benders cut:
                    #   θ[ℓ,t] ≥ obj_s + Σ_b π_b * (z[b,ℓ,t] - z*[b,ℓ,t])
                    # Rearranged:
                    #   θ[ℓ,t] ≥ (obj_s - Σ_b π_b * z*_b) + Σ_b π_b * z[b,ℓ,t]
                    if pi_s:
                        const = obj_s - sum(
                            pi_s.get(b, 0.0) * z_dict.get(b, 0.0)
                            for b in corridor)
                        coeffs = {b: pi_s.get(b, 0.0) for b in corridor
                                  if abs(pi_s.get(b, 0.0)) > 1e-12}
                        if coeffs:
                            pending_cuts.append({
                                'trip': (li, ti),
                                'const': const,
                                'coeffs': coeffs,
                                'blocks': set(coeffs.keys()),
                                'id': len(pending_cuts),
                            })

                avg_cost = float(np.mean(scenario_costs))
                ub_iter += op + avg_cost
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
            # No cuts generated → solution is stable, stop early
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

        for li, ti in active_trips:
            T_t = T_vals[ti]
            line_obj = None
            for ln in lines:
                if ln.idx == li:
                    line_obj = ln
                    break
            if line_obj is None:
                continue

            # Assigned blocks: those with z > 0.5
            assigned_blocks = []
            for b in line_obj.reachable_blocks:
                if z_star_final.get((b, li, ti), 0.0) > 0.5:
                    assigned_blocks.append(b)
            if not assigned_blocks:
                continue

            # District root: first checkpoint, or centroid if no checkpoints
            root_id = line_obj.checkpoints[0] if line_obj.checkpoints else assigned_blocks[0]
            if root_id not in assigned_blocks:
                assigned_blocks.append(root_id)
            root_idx = block_ids.index(root_id)

            # Fill assignment matrix
            for b in assigned_blocks:
                j = block_ids.index(b)
                assignment[j, root_idx] = 1.0

            roots_final.append(root_id)
            dispatch_intervals[root_id] = T_t

            # District route: checkpoint ordering as local indices
            local_blocks = assigned_blocks
            route_order = []
            for cp in line_obj.checkpoints:
                if cp in local_blocks:
                    route_order.append(local_blocks.index(cp))
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

        return ServiceDesign(
            name=self._name,
            assignment=assignment,
            depot_id=depot_id,
            district_roots=roots_final,
            dispatch_intervals=dispatch_intervals,
            district_routes=district_routes,
        )

    # ------------------------------------------------------------------
    # Route cost extraction (for use with simulate_design)
    # ------------------------------------------------------------------

    def compute_route_costs(
        self, design: ServiceDesign, geodata,
        prob_dict: Dict[str, float], Lambda: float, wr: float,
    ) -> tuple:
        """Second-stage evaluation: stochastic tour with optimal deviation.

        Returns (tour_km_overrides, avg_walk_time_h).
        """
        block_ids = geodata.short_geoid_list
        N = len(block_ids)
        assignment = design.assignment
        depot_id = design.depot_id

        tour_km_overrides: Dict[str, float] = {}
        acc_walk_km = 0.0
        acc_prob = 0.0
        rng = np.random.default_rng(42)

        for i, root in enumerate(block_ids):
            if round(assignment[i, i]) != 1:
                continue
            assigned_idx = [j for j in range(N)
                            if round(assignment[j, i]) == 1]
            if not assigned_idx:
                continue
            assigned_blocks = [block_ids[j] for j in assigned_idx]

            depot_pos_km, pts_km, areas_km2, prob_weights, _ = \
                self._district_data(geodata, block_ids, assigned_blocks,
                                    depot_id, prob_dict)

            T_i = design.dispatch_intervals.get(root, 0.5)
            route_order = design.district_routes.get(root)
            if route_order is None:
                route_order = _greedy_nn_route(pts_km, depot_pos_km)

            # Evaluate with deviation (self.delta)
            rng_state = rng.bit_generator.state
            tour_det, walk_det = _fr_detour_tour_stochastic(
                pts_km, depot_pos_km, areas_km2, self.delta, prob_weights,
                Lambda, T_i, route_order,
                wr=wr, num_scenarios=self.num_scenarios, rng=rng)
            # Evaluate without deviation (same scenarios)
            rng.bit_generator.state = rng_state
            tour_no, walk_no = _fr_detour_tour_stochastic(
                pts_km, depot_pos_km, areas_km2, 0.0, prob_weights,
                Lambda, T_i, route_order,
                wr=wr, num_scenarios=self.num_scenarios, rng=rng)

            # Per-district fallback: pick lower cost
            cost_det = (tour_det / (TSP_TRAVEL_DISCOUNT * T_i)
                        + wr * (tour_det / (2 * VEHICLE_SPEED_KMH)
                                + walk_det / self.walk_speed_kmh))
            cost_no = (tour_no / (TSP_TRAVEL_DISCOUNT * T_i)
                       + wr * (tour_no / (2 * VEHICLE_SPEED_KMH)
                               + walk_no / self.walk_speed_kmh))

            if cost_det <= cost_no:
                tour_km_overrides[root] = tour_det
                walk_km = walk_det
            else:
                tour_km_overrides[root] = tour_no
                walk_km = walk_no

            district_prob = float(prob_weights.sum())
            acc_walk_km += walk_km * district_prob
            acc_prob += district_prob

        acc_prob = max(acc_prob, 1e-12)
        avg_walk_time_h = (acc_walk_km / acc_prob) / self.walk_speed_kmh
        return tour_km_overrides, avg_walk_time_h

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self, design: ServiceDesign, geodata,
        prob_dict: Dict[str, float], Omega_dict,
        J_function: Callable, Lambda: float,
        wr: float, wv: float,
        beta: float = BETA, road_network=None,
    ) -> EvaluationResult:
        """Evaluation with stochastic demand and optimal deviations."""
        tour_km_overrides, avg_walk_time_h = self.compute_route_costs(
            design, geodata, prob_dict, Lambda, wr)
        result = evaluate_design(
            design, geodata, prob_dict, Omega_dict, J_function,
            Lambda, wr, wv, beta, road_network,
            walk_time_override=avg_walk_time_h,
            tour_km_overrides=tour_km_overrides)
        return EvaluationResult(
            name=self._name,
            **{k: v for k, v in result.__dict__.items() if k != 'name'})
