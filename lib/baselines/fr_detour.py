"""
Multi Fixed-Route with Detour baseline — adapted from Martin-Iradi et al. (2025)
"A Double Decomposition Algorithm for Network Planning and Operations in
Deviated Fixed-route Microtransit".

Adaptation of the paper's double-decomposition (DD) to last-mile delivery:

  First stage (network design — joint decisions):
    - Partition: K districts via p-median MIP (Gurobi, Benders)
    - Route selection: best fixed-route ordering per district from
      a candidate collection (all permutations for n≤8, heuristic for n>8)
    - Schedule: dispatch interval T* jointly optimised with route
    - Fleet budget: optional max_fleet constraint (scale T if exceeded)

  Second stage (operations — given fixed route and schedule):
    - Vehicle follows the fixed route ordering, skipping blocks with no
      realised demand (checkpoint skipping)
    - Per-block optimal deviation decision: deviate only when walk-time
      savings exceed extra vehicle cost
    - Expected tour and walk computed via Monte Carlo over Poisson demand
    - Routing costs feed back to master as updated cost coefficients

  Benders iteration (DD loop):
    1. Master MIP proposes a partition (cost-optimal block assignment)
    2. Subproblem jointly optimises (route, T*) per district
    3. Per-block marginal costs update the master's objective
    4. Repeat until partition stabilises

When delta=0 this reduces to "Multi-FR" (no deviations, customers walk
full distance to block centroid).

Reference: Martin-Iradi, Schmid, Cummings, Jacquillat (2025, arXiv:2402.01265).
"""
from __future__ import annotations

import itertools
import math
from typing import Callable, Dict, List, Tuple

import numpy as np

from lib.baselines.base import (
    BETA, VEHICLE_SPEED_KMH, BaselineMethod, EvaluationResult, ServiceDesign,
    _get_pos, evaluate_design,
)
from lib.constants import TSP_TRAVEL_DISCOUNT


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _greedy_nn_route(points_km: np.ndarray, depot_km: np.ndarray) -> List[int]:
    """Greedy nearest-neighbour route from depot through all blocks.

    Returns a permutation of range(n) representing the visit order.
    Used for quick route generation (candidate seeding, marginal cost eval).
    """
    n = len(points_km)
    if n == 0:
        return []
    if n == 1:
        return [0]
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


# ---------------------------------------------------------------------------
# Fixed-route tour computation
# ---------------------------------------------------------------------------

def _fixed_route_tour_km(
    route_order: List[int],
    points_km: np.ndarray,
    depot_km: np.ndarray,
    active_mask: np.ndarray,
) -> float:
    """
    Tour distance following a fixed route order, skipping inactive blocks.

    The vehicle visits only blocks with active_mask=True, but in the order
    specified by route_order (checkpoint skipping).  O(n) per call.
    """
    active_stops = [i for i in route_order if active_mask[i]]
    if not active_stops:
        return 0.0
    pts = np.vstack([depot_km[None]]
                    + [points_km[i][None] for i in active_stops]
                    + [depot_km[None]])
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


# ---------------------------------------------------------------------------
# Candidate route generation and selection
# ---------------------------------------------------------------------------

def _two_opt_improve(
    route: List[int],
    points_km: np.ndarray,
    depot_km: np.ndarray,
    max_iters: int = 100,
) -> List[int]:
    """Improve a route ordering via 2-opt on the full (all-blocks-active) tour."""
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


def _generate_candidates(
    n: int,
    points_km: np.ndarray,
    depot_km: np.ndarray,
    max_candidates: int = 50,
) -> List[List[int]]:
    """
    Generate candidate fixed-route orderings for a district with n blocks.

    n <= 8: enumerate all distinct orderings (fix first element, deduplicate
            directional equivalents).
    n > 8:  heuristic candidates via NN from each block + 2-opt.
    """
    if n == 0:
        return [[]]
    if n == 1:
        return [[0]]

    if n <= 8:
        # Fix first element = 0 to remove rotational equivalence.
        # Keep only lexicographically smaller of (perm, reversed(perm))
        # to remove directional equivalence.
        candidates = []
        for perm in itertools.permutations(range(1, n)):
            full = [0] + list(perm)
            rev = [0] + list(reversed(perm))
            if full <= rev:
                candidates.append(full)
            # else the reverse is already (or will be) in the set
        return candidates

    # n > 8: heuristic generation
    seen = set()
    candidates = []

    def _add(route):
        # Canonical form: min of (route, reversed) starting from element 0
        r = list(route)
        # Rotate so element 0 is first
        idx0 = r.index(0)
        r = r[idx0:] + r[:idx0]
        rev = [r[0]] + r[1:][::-1]
        key = tuple(min(r, rev))
        if key not in seen:
            seen.add(key)
            candidates.append(list(key))

    # NN from each starting block
    for s in range(n):
        others = [i for i in range(n) if i != s]
        others_pts = points_km[others]
        nn_order = _greedy_nn_route(others_pts, points_km[s])
        route = [s] + [others[i] for i in nn_order]
        route = _two_opt_improve(route, points_km, depot_km)
        _add(route)

    # Fill with random permutations + 2-opt if needed
    rng = np.random.default_rng(123)
    attempts = 0
    while len(candidates) < max_candidates and attempts < max_candidates * 3:
        perm = list(rng.permutation(n))
        perm = _two_opt_improve(perm, points_km, depot_km)
        _add(perm)
        attempts += 1

    return candidates


# ---------------------------------------------------------------------------
# Per-block geometry
# ---------------------------------------------------------------------------

def _block_geometry(areas_km2: np.ndarray, delta: float):
    """
    Per-block detour cost and walk distances (with and without detour).

      dev_j         = 2 · min(delta, r_j)         extra vehicle km if deviating
      walk_det_j    = max(0, 2·r_j/3 − delta)     customer walk with detour (km)
      walk_no_det_j = 2·r_j/3                     customer walk without detour (km)
    """
    r = np.sqrt(areas_km2 / math.pi)
    dev = 2.0 * np.minimum(delta, r)
    walk_det = np.maximum(0.0, 2.0 * r / 3.0 - delta)
    walk_no_det = 2.0 * r / 3.0
    return dev, walk_det, walk_no_det


# ---------------------------------------------------------------------------
# Per-block optimal deviation decision (second-stage routing optimisation)
# ---------------------------------------------------------------------------

def _deviation_mask(
    dev: np.ndarray,
    walk_det: np.ndarray,
    walk_no_det: np.ndarray,
    prob_weights: np.ndarray,
    district_prob: float,
    wr: float,
    walk_speed: float,
    discount: float,
    speed: float,
    T: float,
) -> np.ndarray:
    """
    Per-block optimal deviation decision (Martin-Iradi et al.'s second-stage).

    Deviate at block j iff the walk-time savings for riders in block j exceed
    the provider cost + in-vehicle cost increase from the extra vehicle km.

    Returns boolean array (True = deviate at block j).
    """
    walk_saved = walk_no_det - walk_det
    benefit = wr * prob_weights * walk_saved / walk_speed
    cost = (dev * district_prob / (discount * T)
            + wr * dev * district_prob / (2.0 * speed))
    return benefit > cost


# ---------------------------------------------------------------------------
# Stochastic tour computation with optimal deviation decisions
# ---------------------------------------------------------------------------

def _fr_detour_tour_stochastic(
    points_km: np.ndarray,
    depot_km: np.ndarray,
    areas_km2: np.ndarray,
    delta: float,
    prob_weights: np.ndarray,
    Lambda: float,
    T: float,
    route_order: List[int],
    wr: float = 1.0,
    discount: float = TSP_TRAVEL_DISCOUNT,
    num_scenarios: int = 200,
    rng: np.random.Generator | None = None,
) -> Tuple[float, float]:
    """
    Stochastic tour under Poisson demand with optimal deviation decisions.

    The vehicle follows ``route_order`` (a fixed ordering of block indices)
    and skips blocks with no realised demand (checkpoint skipping).

    Returns (expected_tour_km, demand_weighted_avg_walk_km).
    """
    n = len(points_km)
    if n == 0:
        return 0.0, 0.0

    dev, walk_det, walk_no_det = _block_geometry(areas_km2, delta)
    district_prob = float(prob_weights.sum())
    rates = Lambda * T * prob_weights

    if rng is None:
        rng = np.random.default_rng(42)
    ns = max(num_scenarios, 200)

    # Per-block deviation decision (first-stage: depends on T, not demand)
    if delta > 0:
        det_mask = _deviation_mask(
            dev, walk_det, walk_no_det, prob_weights, district_prob,
            wr, 5.0, discount, VEHICLE_SPEED_KMH, T,
        )
    else:
        det_mask = np.zeros(n, dtype=bool)

    # Walk depends on deviation decision
    block_walk = np.where(det_mask, walk_det, walk_no_det)
    avg_walk_km = float(np.dot(block_walk, prob_weights)) / max(district_prob, 1e-12)

    # Monte Carlo tour
    acc_tour = 0.0
    for _ in range(ns):
        demand = rng.poisson(rates)
        has_demand = demand > 0
        if not np.any(has_demand):
            continue
        tsp_km = _fixed_route_tour_km(route_order, points_km, depot_km, has_demand)
        active_deviate = det_mask[has_demand]
        acc_tour += tsp_km + float(np.sum(dev[has_demand][active_deviate]))
    tour_km = acc_tour / ns

    return tour_km, avg_walk_km


# ---------------------------------------------------------------------------
# Full district cost evaluator
# ---------------------------------------------------------------------------

def _district_cost(
    K_i: float, F_i: float,
    points_km: np.ndarray, depot_km: np.ndarray,
    areas_km2: np.ndarray, delta: float, prob_weights: np.ndarray,
    Lambda: float, T: float, wr: float,
    route_order: List[int],
    discount: float = TSP_TRAVEL_DISCOUNT,
    num_scenarios: int = 200,
) -> float:
    """
    Total cost for one district at dispatch interval T.

    cost = (K + E[tour]/discount + F) / T  +  wr · (T/2 + E[tour]/(2·speed) + walk)
    """
    tour_km, avg_walk_km = _fr_detour_tour_stochastic(
        points_km, depot_km, areas_km2, delta, prob_weights,
        Lambda, T, route_order,
        wr=wr, discount=discount,
        num_scenarios=num_scenarios,
        rng=np.random.default_rng(42),
    )
    walk_h = avg_walk_km / 5.0
    provider = (K_i + tour_km / discount + F_i) / T
    user = wr * (T / 2.0 + tour_km / (2.0 * VEHICLE_SPEED_KMH) + walk_h)
    return provider + user


# ---------------------------------------------------------------------------
# T* optimisation
# ---------------------------------------------------------------------------

def _stochastic_T_star(
    K_i: float, F_i: float,
    points_km: np.ndarray, depot_km: np.ndarray,
    areas_km2: np.ndarray, delta: float, prob_weights: np.ndarray,
    Lambda: float, wr: float,
    route_order: List[int],
    discount: float = TSP_TRAVEL_DISCOUNT,
    num_scenarios: int = 200,
    T_min: float = 1e-3, T_max: float = 24.0,
) -> float:
    """Optimal T* via bounded search using the full stochastic cost model."""
    from scipy.optimize import minimize_scalar

    def cost(T):
        return _district_cost(
            K_i, F_i, points_km, depot_km, areas_km2, delta,
            prob_weights, Lambda, T, wr, route_order, discount, num_scenarios,
        )

    result = minimize_scalar(cost, bounds=(T_min, T_max), method='bounded')
    return float(result.x)


# ---------------------------------------------------------------------------
# Joint route + schedule (T) optimisation
# ---------------------------------------------------------------------------

def _select_best_route_and_T(
    K_i: float,
    F_i: float,
    points_km: np.ndarray,
    depot_km: np.ndarray,
    areas_km2: np.ndarray,
    delta: float,
    prob_weights: np.ndarray,
    Lambda: float,
    wr: float,
    discount: float = TSP_TRAVEL_DISCOUNT,
    num_scenarios: int = 200,
    max_candidates: int = 50,
    top_k: int = 20,
) -> Tuple[List[int], float, float]:
    """
    Joint first-stage optimisation of fixed route and dispatch interval T.

    1. Generate candidate route orderings.
    2. Pre-screen: evaluate all candidates at a reference T (greedy-NN T*),
       keep top ``top_k`` by expected tour km.
    3. Full optimisation: for each surviving candidate, optimise T* and
       compute full cost.
    4. Return (best_route, T_star, cost).
    """
    n = len(points_km)
    candidates = _generate_candidates(n, points_km, depot_km, max_candidates)

    if len(candidates) <= 1:
        route = candidates[0] if candidates else list(range(n))
        T_star = _stochastic_T_star(
            K_i, F_i, points_km, depot_km, areas_km2, delta,
            prob_weights, Lambda, wr, route, discount, num_scenarios,
        )
        c = _district_cost(
            K_i, F_i, points_km, depot_km, areas_km2, delta,
            prob_weights, Lambda, T_star, wr, route, discount, num_scenarios,
        )
        return route, T_star, c

    # Phase 1: reference T using first candidate (NN from depot) as warm start
    nn_route = _greedy_nn_route(points_km, depot_km)
    T_ref = _stochastic_T_star(
        K_i, F_i, points_km, depot_km, areas_km2, delta,
        prob_weights, Lambda, wr, nn_route, discount, num_scenarios,
    )

    # Phase 2: pre-screen candidates by expected tour km at T_ref
    rng = np.random.default_rng(42)
    ns = max(num_scenarios, 200)
    rates_ref = Lambda * T_ref * prob_weights
    # Pre-sample demand scenarios (common random numbers)
    demand_samples = rng.poisson(np.broadcast_to(rates_ref, (ns, n)))

    tour_scores = []
    for cand in candidates:
        acc = 0.0
        for s in range(ns):
            has_demand = demand_samples[s] > 0
            if not np.any(has_demand):
                continue
            acc += _fixed_route_tour_km(cand, points_km, depot_km, has_demand)
        tour_scores.append(acc / ns)

    # Keep top_k candidates
    ranked = sorted(range(len(candidates)), key=lambda i: tour_scores[i])
    shortlist = [candidates[i] for i in ranked[:min(top_k, len(candidates))]]

    # Phase 3: full T* optimisation for shortlisted candidates
    best_route = shortlist[0]
    best_T = T_ref
    best_cost = float('inf')

    for cand in shortlist:
        T_star = _stochastic_T_star(
            K_i, F_i, points_km, depot_km, areas_km2, delta,
            prob_weights, Lambda, wr, cand, discount, num_scenarios,
        )
        c = _district_cost(
            K_i, F_i, points_km, depot_km, areas_km2, delta,
            prob_weights, Lambda, T_star, wr, cand, discount, num_scenarios,
        )
        if c < best_cost:
            best_cost = c
            best_route = cand
            best_T = T_star

    return best_route, best_T, best_cost


# ---------------------------------------------------------------------------
# Baseline class
# ---------------------------------------------------------------------------

class FRDetour(BaselineMethod):
    """
    Multi Fixed-Route with Detour baseline (Martin-Iradi et al. 2025).

    Adaptation of the double-decomposition algorithm:

      First stage (network design — joint decisions):
        - Partition: K districts via p-median MIP (Gurobi, Benders)
        - Route selection: best fixed-route ordering per district from
          a candidate collection
        - Schedule: dispatch interval T* jointly optimised with route
        - Fleet budget: optional max_fleet constraint

      Second stage (operations — given fixed route and schedule):
        - Vehicle follows the fixed route, skipping empty checkpoints
        - Per-block optimal deviation decision
        - Expected cost via Monte Carlo over Poisson demand

    With delta=0 this is "Multi-FR" (no deviations).
    With delta>0 this is "Multi-FR-Detour".

    Parameters
    ----------
    delta          : max vehicle deviation from reference route (km)
    max_iters      : max Benders iterations for partition refinement
    walk_speed_kmh : pedestrian walking speed (km/h)
    num_scenarios  : Monte Carlo scenarios for stochastic tour estimation
    max_fleet      : optional fleet budget (max simultaneous vehicles)
    name           : display name override
    """

    def __init__(
        self,
        delta: float = 0.3,
        max_iters: int = 10,
        walk_speed_kmh: float = 5.0,
        num_scenarios: int = 200,
        max_fleet: float | None = None,
        name: str | None = None,
    ):
        self.delta = delta
        self.max_iters = max_iters
        self.walk_speed_kmh = walk_speed_kmh
        self.num_scenarios = num_scenarios
        self.max_fleet = max_fleet
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
             for b in assigned_blocks]
        )
        areas_km2 = np.array(
            [float(geodata.get_area(b)) for b in assigned_blocks]
        )
        prob_weights = np.array(
            [float(prob_dict.get(b, 0.0)) for b in assigned_blocks]
        )
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

    def _evaluate_district(self, geodata, block_ids, assigned_blocks,
                           depot_id, prob_dict, Omega_dict, J_function,
                           Lambda, wr, ns=None):
        """Jointly optimise route and T* for one district.

        Returns (cost, T_star, best_route_order).
        """
        ns = ns or self.num_scenarios
        depot_km, pts_km, areas_km2, pw, K_i = self._district_data(
            geodata, block_ids, assigned_blocks, depot_id, prob_dict)
        F_i = self._odd_cost(Omega_dict, J_function, assigned_blocks)

        route, T_star, c_i = _select_best_route_and_T(
            K_i, F_i, pts_km, depot_km, areas_km2, self.delta,
            pw, Lambda, wr, num_scenarios=ns,
        )
        return c_i, T_star, route

    def _evaluate_district_at_T(self, geodata, block_ids, assigned_blocks,
                                depot_id, prob_dict, Omega_dict, J_function,
                                Lambda, wr, T, ns=None):
        """Compute stochastic cost for one district at fixed T (fast, no T* opt).

        Uses a quick NN route (no candidate search) for marginal cost evaluation.
        """
        ns = ns or min(self.num_scenarios, 100)
        depot_km, pts_km, areas_km2, pw, K_i = self._district_data(
            geodata, block_ids, assigned_blocks, depot_id, prob_dict)
        F_i = self._odd_cost(Omega_dict, J_function, assigned_blocks)
        route = _greedy_nn_route(pts_km, depot_km)

        return _district_cost(
            K_i, F_i, pts_km, depot_km, areas_km2, self.delta,
            pw, Lambda, T, wr, route, num_scenarios=ns)

    # ------------------------------------------------------------------
    # Design: Benders decomposition (MIP master + stochastic subproblem)
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
        Adapted double-decomposition (Martin-Iradi et al. 2025):

        1. Benders master: MIP partition (p-median, Gurobi)
        2. Benders subproblem: stochastic routing cost per district
        3. Feedback: marginal block costs update MIP objective
        4. Iterate until partition stabilises or max_iters reached
        5. Final T* per district with full stochastic model
        """
        import gurobipy as gp
        from gurobipy import GRB

        block_ids = geodata.short_geoid_list
        N = len(block_ids)
        K = num_districts

        # Depot: geographic centroid
        positions_m = np.array([_get_pos(geodata, b) for b in block_ids])
        centroid = positions_m.mean(axis=0)
        depot_idx = int(np.argmin(np.linalg.norm(positions_m - centroid, axis=1)))
        depot_id = block_ids[depot_idx]

        # Pairwise distance matrix (km) for initial p-median
        dist_km = np.zeros((N, N))
        for j in range(N):
            for i in range(N):
                dist_km[j, i] = geodata.get_dist(block_ids[j], block_ids[i])

        # ---- MIP partition (Benders master) ----
        model = gp.Model("FRDetour_DD")
        model.setParam('OutputFlag', 0)

        z = model.addVars(N, N, vtype=GRB.BINARY, name="z")
        y = model.addVars(N, vtype=GRB.BINARY, name="y")

        # Partition constraints
        for j in range(N):
            model.addConstr(gp.quicksum(z[j, i] for i in range(N)) == 1)
        for j in range(N):
            for i in range(N):
                model.addConstr(z[j, i] <= y[i])
        for i in range(N):
            model.addConstr(z[i, i] == y[i])
        model.addConstr(gp.quicksum(y[i] for i in range(N)) == K)

        # Cost coefficients (updated each Benders iteration)
        c_coeff = dist_km.copy()

        best_cost = float('inf')
        best_assignment = None
        best_intervals: Dict[str, float] = {}
        prev_assignment = None

        for benders_iter in range(self.max_iters):
            # ---- Master: solve MIP ----
            model.setObjective(
                gp.quicksum(c_coeff[j, i] * z[j, i]
                            for j in range(N) for i in range(N)),
                GRB.MINIMIZE,
            )
            model.optimize()
            if model.status != GRB.OPTIMAL:
                break

            assignment = np.zeros((N, N))
            for j in range(N):
                for i in range(N):
                    assignment[j, i] = round(z[j, i].X)

            # Convergence: same partition as last iteration
            if prev_assignment is not None and np.array_equal(assignment, prev_assignment):
                break
            prev_assignment = assignment.copy()

            roots = [i for i in range(N) if round(assignment[i, i]) == 1]

            # ---- Subproblem: evaluate stochastic routing cost per district ----
            total_cost = 0.0
            intervals: Dict[str, float] = {}
            district_costs: Dict[int, Tuple[float, float, List[int]]] = {}

            for i in roots:
                assigned_idx = [j for j in range(N)
                                if round(assignment[j, i]) == 1]
                assigned_blocks = [block_ids[j] for j in assigned_idx]

                c_i, T_i, route_i = self._evaluate_district(
                    geodata, block_ids, assigned_blocks, depot_id,
                    prob_dict, Omega_dict, J_function, Lambda, wr,
                )
                total_cost += c_i
                intervals[block_ids[i]] = T_i
                district_costs[i] = (c_i, T_i, route_i)

            if total_cost < best_cost:
                best_cost = total_cost
                best_assignment = assignment.copy()
                best_intervals = intervals.copy()

            # ---- Benders feedback: compute per-block marginal costs ----
            new_c = np.full((N, N), 1e6)

            for i in roots:
                c_i, T_i, _ = district_costs[i]
                assigned_idx = [j for j in range(N)
                                if round(assignment[j, i]) == 1]
                assigned_blocks = [block_ids[j] for j in assigned_idx]
                n_i = len(assigned_idx)

                # Blocks currently in district i: uniform share
                per_block = c_i / max(n_i, 1)
                for j in assigned_idx:
                    new_c[j, i] = per_block

                # Blocks NOT in district i: marginal cost of adding
                # Evaluate at fixed T_i (fast, no re-optimisation)
                for j in range(N):
                    if j in assigned_idx:
                        continue
                    hypo_blocks = assigned_blocks + [block_ids[j]]
                    c_hypo = self._evaluate_district_at_T(
                        geodata, block_ids, hypo_blocks, depot_id,
                        prob_dict, Omega_dict, J_function,
                        Lambda, wr, T_i,
                    )
                    new_c[j, i] = per_block + (c_hypo - c_i)

            c_coeff = new_c

        # ---- Final: jointly optimise (route, T*) for best partition ----
        assignment = best_assignment
        roots_final = [block_ids[i] for i in range(N)
                       if round(assignment[i, i]) == 1]
        dispatch_intervals: Dict[str, float] = {}
        district_routes: Dict[str, List[int]] = {}

        for i in range(N):
            if round(assignment[i, i]) != 1:
                continue
            assigned_idx = [j for j in range(N)
                            if round(assignment[j, i]) == 1]
            assigned_blocks = [block_ids[j] for j in assigned_idx]

            _, T_star, route = self._evaluate_district(
                geodata, block_ids, assigned_blocks, depot_id,
                prob_dict, Omega_dict, J_function, Lambda, wr,
            )
            dispatch_intervals[block_ids[i]] = T_star
            district_routes[block_ids[i]] = route

        # ---- Fleet budget enforcement ----
        if self.max_fleet is not None:
            depot_pos_km = np.array(_get_pos(geodata, depot_id), dtype=float) / 1000.0
            total_fleet = 0.0
            tour_kms: Dict[str, float] = {}
            for root_id in roots_final:
                ri = block_ids.index(root_id)
                assigned_idx = [j for j in range(N)
                                if round(assignment[j, ri]) == 1]
                assigned_blocks = [block_ids[j] for j in assigned_idx]
                _, pts_km, areas_km2, pw, _ = self._district_data(
                    geodata, block_ids, assigned_blocks, depot_id, prob_dict)
                T_i = dispatch_intervals[root_id]
                route_order = district_routes.get(root_id)
                if route_order is None:
                    route_order = _greedy_nn_route(pts_km, depot_pos_km)
                tour_km, _ = _fr_detour_tour_stochastic(
                    pts_km, depot_pos_km, areas_km2, self.delta, pw,
                    Lambda, T_i, route_order,
                    wr=1.0, num_scenarios=self.num_scenarios,
                    rng=np.random.default_rng(42),
                )
                tour_kms[root_id] = tour_km
                total_fleet += tour_km / (VEHICLE_SPEED_KMH * T_i)

            if total_fleet > self.max_fleet:
                scale = total_fleet / self.max_fleet
                for root_id in roots_final:
                    dispatch_intervals[root_id] *= scale

        return ServiceDesign(
            name=self._name,
            assignment=assignment,
            depot_id=depot_id,
            district_roots=roots_final,
            dispatch_intervals=dispatch_intervals,
            district_routes=district_routes,
        )

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------

    def evaluate(
        self,
        design: ServiceDesign,
        geodata,
        prob_dict: Dict[str, float],
        Omega_dict,
        J_function: Callable,
        Lambda: float,
        wr: float,
        wv: float,
        beta: float = BETA,
        road_network=None,
    ) -> EvaluationResult:
        """
        Second-stage evaluation with stochastic demand and optimal deviations.
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
            tour_km, avg_walk_km = _fr_detour_tour_stochastic(
                pts_km, depot_pos_km, areas_km2, self.delta, prob_weights,
                Lambda, T_i, route_order,
                wr=wr, num_scenarios=self.num_scenarios, rng=rng,
            )
            tour_km_overrides[root] = tour_km

            district_prob = float(prob_weights.sum())
            acc_walk_km += avg_walk_km * district_prob
            acc_prob += district_prob

        acc_prob = max(acc_prob, 1e-12)
        avg_walk_time_h = (acc_walk_km / acc_prob) / self.walk_speed_kmh

        result = evaluate_design(
            design, geodata, prob_dict, Omega_dict, J_function,
            Lambda, wr, wv, beta, road_network,
            walk_time_override=avg_walk_time_h,
            tour_km_overrides=tour_km_overrides,
        )
        return EvaluationResult(
            name=self._name,
            **{k: v for k, v in result.__dict__.items() if k != 'name'},
        )
