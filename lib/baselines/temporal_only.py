"""
Temporal-only baseline (Liu & Luo 2023): single depot, fixed dispatch interval,
ADP-optimized fleet dispatch, VRP-optimized routing.

Three-layer architecture following the paper:
  Layer 1 (ADP): APT cost-to-go via LP allocation for forward-looking dispatch
  Layer 2 (Benders): Iterative dispatch/routing refinement with cut-inspired
                     sample concentration on promising K values
  Layer 3 (CG + route enum): VRP via set-partitioning + ESPPRC pricing

Demand model (paper-faithful): a finite set of candidate delivery locations
I = {1, ..., I} is a hyperparameter.  Each dispatch period, per-location
demands q_i ~ Poisson arise at a random subset I^n of these locations.

Objective: minimize total expected delivery time (sum of customer arrival
times), faithful to the paper.  No spatial partition.  No BHH approximation.

Evaluation: scenario-based VRP simulation with realized delivery times.
No linehaul cost (single depot, non-partition mode).
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from lib.baselines.base import (
    BETA, VEHICLE_SPEED_KMH, BaselineMethod, EvaluationResult, ServiceDesign,
    _get_pos, _nn_tsp_length,
)
from lib.constants import DEFAULT_FLEET_COST_RATE, TSP_TRAVEL_DISCOUNT


# ---------------------------------------------------------------------------
# Discrete demand sampling at candidate locations (paper-faithful)
# ---------------------------------------------------------------------------

def _sample_discrete_demand(
    probs: np.ndarray,
    positions_km: np.ndarray,
    Lambda_T: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample per-location Poisson demands at candidate locations.

    Following the paper, I = {1,...,I} is the fixed set of candidate
    locations.  Each period, q_i ~ Poisson(Lambda * T * p_i) orders
    arise at location i.  Active locations I^n = {i : q_i > 0}.

    Returns (active_positions_km, demands) where:
      active_positions_km : (n_active, 2)
      demands             : (n_active,) integer demand at each location
    """
    q = rng.poisson(Lambda_T * probs)
    active = q > 0
    return positions_km[active], q[active]


# ---------------------------------------------------------------------------
# Continuous-space demand sampler (kept for backward compatibility)
# ---------------------------------------------------------------------------

class DemandSampler:
    """Samples customer locations from continuous density over the service region."""

    def __init__(self, geodata, prob_dict: Dict[str, float], resolution: int = 100):
        block_ids = geodata.short_geoid_list
        block_pos_m = np.array([_get_pos(geodata, b) for b in block_ids])
        block_pos = block_pos_m / 1000.0  # km

        areas_km2 = np.array([max(float(geodata.get_area(b)), 1e-12)
                              for b in block_ids])
        probs = np.array([prob_dict.get(b, 0.0) for b in block_ids])

        # Fine grid
        span = block_pos.max(axis=0) - block_pos.min(axis=0)
        pad = float(span.max()) * 0.05 + 0.05
        lo = block_pos.min(axis=0) - pad
        hi = block_pos.max(axis=0) + pad

        xs = np.linspace(lo[0], hi[0], resolution)
        ys = np.linspace(lo[1], hi[1], resolution)
        self.cell_dx = float(xs[1] - xs[0]) if len(xs) > 1 else 0.0
        self.cell_dy = float(ys[1] - ys[0]) if len(ys) > 1 else 0.0
        dA = self.cell_dx * self.cell_dy

        xx, yy = np.meshgrid(xs, ys)
        self.grid_pos = np.column_stack([xx.ravel(), yy.ravel()])

        dists_gb = np.linalg.norm(
            self.grid_pos[:, None, :] - block_pos[None, :, :], axis=2)
        nearest_block = np.argmin(dists_gb, axis=1)

        f_vals = probs[nearest_block] / areas_km2[nearest_block]
        f_dA = np.maximum(f_vals, 0.0) * dA
        total_mass = float(f_dA.sum())
        self.cell_probs = (f_dA / total_mass if total_mass > 1e-15
                           else np.ones(len(self.grid_pos)) / len(self.grid_pos))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Return (n, 2) customer positions in km."""
        if n == 0:
            return np.empty((0, 2))
        idx = rng.choice(len(self.grid_pos), size=n, p=self.cell_probs)
        pts = self.grid_pos[idx].copy()
        pts[:, 0] += rng.uniform(-0.5 * self.cell_dx, 0.5 * self.cell_dx, size=n)
        pts[:, 1] += rng.uniform(-0.5 * self.cell_dy, 0.5 * self.cell_dy, size=n)
        return pts


# ---------------------------------------------------------------------------
# VRP solver: CVRP via Miller-Tucker-Zemlin formulation (small instances)
# ---------------------------------------------------------------------------

def _vrp_mtz(depot_km: np.ndarray, customer_km: np.ndarray,
             K: int, capacity: int = 20,
             demands: Optional[np.ndarray] = None,
             time_limit: float = 10.0):
    """
    Exact CVRP via Miller-Tucker-Zemlin formulation with Gurobi.

    Parameters
    ----------
    demands : optional (n,) array of per-customer demand.
              If None, unit demand is assumed.

    Returns (routes, total_km, per_veh_km).
    """
    import gurobipy as gp
    from gurobipy import GRB

    n = len(customer_km)
    if n == 0:
        return [[] for _ in range(K)], 0.0, [0.0] * K
    if n == 1:
        d = float(np.linalg.norm(customer_km[0] - depot_km))
        return [[0]], 2.0 * d, [2.0 * d] + [0.0] * (K - 1)

    if demands is None:
        demands = np.ones(n, dtype=float)
    Q = float(capacity)

    # 0 = depot, 1..n = customers
    all_pts = np.vstack([depot_km[None], customer_km])
    V = n + 1
    dist = np.linalg.norm(all_pts[:, None] - all_pts[None, :], axis=2)

    model = gp.Model("CVRP_MTZ")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", time_limit)

    x = {}
    for i in range(V):
        for j in range(V):
            if i != j:
                x[i, j] = model.addVar(vtype=GRB.BINARY)
    # u[i] = cumulative demand on the route upon arrival at customer i
    u = {i: model.addVar(lb=float(demands[i - 1]), ub=Q,
                         vtype=GRB.CONTINUOUS)
         for i in range(1, V)}

    model.setObjective(
        gp.quicksum(dist[i, j] * x[i, j]
                     for i in range(V) for j in range(V) if i != j),
        GRB.MINIMIZE)

    # K vehicles leave / return to depot
    model.addConstr(gp.quicksum(x[0, j] for j in range(1, V)) == K)
    model.addConstr(gp.quicksum(x[i, 0] for i in range(1, V)) == K)

    # Flow conservation
    for j in range(1, V):
        model.addConstr(
            gp.quicksum(x[i, j] for i in range(V) if i != j) == 1)
        model.addConstr(
            gp.quicksum(x[j, i] for i in range(V) if i != j) == 1)

    # CVRP subtour elimination + capacity
    for i in range(1, V):
        for j in range(1, V):
            if i != j:
                model.addConstr(
                    u[i] - u[j] + Q * x[i, j] <= Q - demands[j - 1])

    model.optimize()

    if model.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT) or model.SolCount == 0:
        tour_km = _nn_tsp_length(depot_km, customer_km)
        return [list(range(n))], tour_km, [tour_km]

    # Extract routes by tracing from depot
    succ = {}
    for i in range(V):
        for j in range(V):
            if i != j and x[i, j].X > 0.5:
                succ.setdefault(i, []).append(j)

    routes, per_veh_km = [], []
    visited = set()
    for j in succ.get(0, []):
        route = []
        cur = j
        while cur != 0 and cur not in visited:
            visited.add(cur)
            route.append(cur - 1)
            nxts = [nx for nx in succ.get(cur, []) if nx == 0 or nx not in visited]
            cur = nxts[0] if nxts else 0
        if route:
            tour = 0.0
            prev = depot_km
            for idx in route:
                tour += float(np.linalg.norm(customer_km[idx] - prev))
                prev = customer_km[idx]
            tour += float(np.linalg.norm(prev - depot_km))
            routes.append(route)
            per_veh_km.append(tour)

    while len(routes) < K:
        routes.append([])
        per_veh_km.append(0.0)

    return routes, sum(per_veh_km), per_veh_km


# ---------------------------------------------------------------------------
# ESPPRC pricing via label-setting DP
# ---------------------------------------------------------------------------

def _espprc_pricing(dist: np.ndarray, n: int,
                    pi: Dict[int, float], mu: float,
                    capacity: int,
                    demands: Optional[np.ndarray] = None,
                    max_labels: int = 200_000):
    """
    Find a route with minimum reduced cost via label-setting.

    dist : (n+1)x(n+1), node 0 = depot, 1..n = customers.
    pi   : dual for customer coverage (0-indexed customer key).
    mu   : dual for fleet constraint.
    demands : optional (n,) per-customer demand; default unit.

    Returns route dict or None.
    """
    import heapq

    if demands is None:
        demands_arr = np.ones(n, dtype=float)
    else:
        demands_arr = np.asarray(demands, dtype=float)
    Q = float(capacity)

    use_bits = n <= 30

    labels_store: list = []

    def _make_visited_empty():
        return 0 if use_bits else frozenset()

    def _add_to_visited(v, j):
        return v | (1 << j) if use_bits else v | {j}

    def _in_visited(v, j):
        return bool(v & (1 << j)) if use_bits else (j in v)

    def _is_subset(v1, v2):
        return (v1 & v2) == v1 if use_bits else v1 <= v2

    node_labels: Dict[int, list] = {j: [] for j in range(n + 1)}

    heap: list = []
    initial_rc = -mu
    lid = 0
    # Label: (rc, cost, node, visited, demand_sum, seq)
    labels_store.append((initial_rc, 0.0, 0, _make_visited_empty(), 0.0, []))
    heapq.heappush(heap, (initial_rc, lid))

    best_route = None
    best_rc = -1e-6

    while heap and lid < max_labels:
        rc_cur, cur_lid = heapq.heappop(heap)
        if rc_cur >= best_rc:
            continue
        rc, cost, node, visited, demand_sum, seq = labels_store[cur_lid]
        if rc != rc_cur:
            continue  # stale

        for j in range(1, n + 1):
            if _in_visited(visited, j):
                continue

            new_demand = demand_sum + demands_arr[j - 1]
            if new_demand > Q:
                continue  # capacity exceeded

            new_cost = cost + dist[node, j]
            new_rc = rc + dist[node, j] - pi.get(j - 1, 0.0)
            new_visited = _add_to_visited(visited, j)
            new_seq = seq + [j - 1]

            # Dominance check at node j
            dominated = False
            for (old_rc, old_v) in node_labels[j]:
                if old_rc <= new_rc + 1e-9 and _is_subset(old_v, new_visited):
                    dominated = True
                    break
            if dominated:
                continue

            node_labels[j] = [
                (lrc, lv) for (lrc, lv) in node_labels[j]
                if not (new_rc <= lrc + 1e-9 and _is_subset(new_visited, lv))
            ]
            node_labels[j].append((new_rc, new_visited))

            # Complete route (return to depot)
            return_rc = new_rc + dist[j, 0]
            if return_rc < best_rc:
                best_rc = return_rc
                best_route = {
                    "customers": set(new_seq),
                    "cost": new_cost + dist[j, 0],
                    "sequence": list(new_seq),
                    "reduced_cost": return_rc,
                }

            lid += 1
            labels_store.append(
                (new_rc, new_cost, j, new_visited, new_demand, new_seq))
            heapq.heappush(heap, (new_rc, lid))

    return best_route


# ---------------------------------------------------------------------------
# VRP via column generation (larger instances)
# ---------------------------------------------------------------------------

def _solve_vrp_cg(depot_km: np.ndarray, customer_km: np.ndarray,
                  K: int, capacity: int = 20,
                  demands: Optional[np.ndarray] = None,
                  time_limit: float = 30.0):
    """
    Solve CVRP for K vehicles via set partitioning with column generation.

    Returns (routes, total_km, per_veh_km).
    """
    import gurobipy as gp
    from gurobipy import GRB

    n = len(customer_km)
    if n == 0:
        return [[] for _ in range(K)], 0.0, [0.0] * K

    if demands is None:
        demands_arr = np.ones(n, dtype=float)
    else:
        demands_arr = np.asarray(demands, dtype=float)
    Q = float(capacity)

    all_pts = np.vstack([depot_km[None], customer_km])
    dist = np.linalg.norm(all_pts[:, None] - all_pts[None, :], axis=2)

    # Initial columns: capacity-feasible groups in angular order
    columns: List[dict] = []

    angles = np.arctan2(customer_km[:, 1] - depot_km[1],
                        customer_km[:, 0] - depot_km[0])
    order = np.argsort(angles)

    # Greedy grouping respecting capacity
    groups: List[List[int]] = []
    grp: List[int] = []
    grp_demand = 0.0
    for i in range(n):
        idx = int(order[i])
        d = demands_arr[idx]
        if grp_demand + d > Q and grp:
            groups.append(grp)
            grp = []
            grp_demand = 0.0
        grp.append(idx)
        grp_demand += d
    if grp:
        groups.append(grp)

    for grp in groups:
        if len(grp) <= 1:
            seq = list(grp)
        else:
            seq = []
            cur = depot_km.copy()
            remaining = list(grp)
            while remaining:
                dists_r = [float(np.linalg.norm(customer_km[j] - cur))
                           for j in remaining]
                best = int(np.argmin(dists_r))
                seq.append(remaining[best])
                cur = customer_km[remaining[best]]
                remaining.pop(best)
        c = float(dist[0, seq[0] + 1])
        for i in range(1, len(seq)):
            c += float(dist[seq[i - 1] + 1, seq[i] + 1])
        c += float(dist[seq[-1] + 1, 0])
        columns.append({
            "customers": set(seq),
            "cost": c,
            "sequence": list(seq),
        })

    # Single-customer routes for CG flexibility
    for j in range(n):
        columns.append({
            "customers": {j},
            "cost": dist[0, j + 1] + dist[j + 1, 0],
            "sequence": [j],
        })

    max_cg_iters = 80

    for _ in range(max_cg_iters):
        R = len(columns)
        model = gp.Model("SPM_LP")
        model.setParam("OutputFlag", 0)

        theta = model.addVars(R, lb=0.0, vtype=GRB.CONTINUOUS)

        cover = {}
        for j in range(n):
            col_idx = [r for r in range(R) if j in columns[r]["customers"]]
            cover[j] = model.addConstr(
                gp.quicksum(theta[r] for r in col_idx) == 1)

        fleet_c = model.addConstr(gp.quicksum(theta[r] for r in range(R)) == K)

        model.setObjective(
            gp.quicksum(columns[r]["cost"] * theta[r] for r in range(R)),
            GRB.MINIMIZE)
        model.optimize()

        if model.status != GRB.OPTIMAL:
            break

        pi = {j: cover[j].Pi for j in range(n)}
        mu_val = fleet_c.Pi

        new_route = _espprc_pricing(dist, n, pi, mu_val, capacity,
                                    demands=demands_arr)
        if new_route is None:
            break
        columns.append(new_route)

    # Integer phase
    R = len(columns)
    model_int = gp.Model("SPM_MIP")
    model_int.setParam("OutputFlag", 0)
    model_int.setParam("TimeLimit", time_limit)

    theta = model_int.addVars(R, vtype=GRB.BINARY)
    for j in range(n):
        col_idx = [r for r in range(R) if j in columns[r]["customers"]]
        model_int.addConstr(gp.quicksum(theta[r] for r in col_idx) == 1)
    model_int.addConstr(gp.quicksum(theta[r] for r in range(R)) == K)
    model_int.setObjective(
        gp.quicksum(columns[r]["cost"] * theta[r] for r in range(R)),
        GRB.MINIMIZE)
    model_int.optimize()

    if model_int.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT) or model_int.SolCount == 0:
        tour_km = _nn_tsp_length(depot_km, customer_km)
        return [list(range(n))], tour_km, [tour_km]

    routes, per_veh_km = [], []
    for r in range(R):
        if theta[r].X > 0.5:
            routes.append(columns[r]["sequence"])
            per_veh_km.append(columns[r]["cost"])

    while len(routes) < K:
        routes.append([])
        per_veh_km.append(0.0)

    return routes, sum(per_veh_km), per_veh_km


# ---------------------------------------------------------------------------
# VRP dispatch: choose solver by instance size
# ---------------------------------------------------------------------------

def _solve_vrp(depot_km: np.ndarray, customer_km: np.ndarray,
               K: int, capacity: int = 20,
               demands: Optional[np.ndarray] = None,
               time_limit: float = 15.0):
    """Route K vehicles from depot to serve all customers (CVRP)."""
    n = len(customer_km)
    if n == 0:
        return [], 0.0, []
    if K >= n:
        routes = [[j] for j in range(n)]
        per_veh = [2.0 * float(np.linalg.norm(customer_km[j] - depot_km))
                   for j in range(n)]
        return routes, sum(per_veh), per_veh
    if n <= 15:
        return _vrp_mtz(depot_km, customer_km, K, capacity,
                        demands=demands, time_limit=time_limit)
    return _solve_vrp_cg(depot_km, customer_km, K, capacity,
                         demands=demands, time_limit=time_limit)


# ---------------------------------------------------------------------------
# Delivery-time computation from VRP routes (paper-faithful objective)
# ---------------------------------------------------------------------------

def _delivery_times_from_routes(
    depot_km: np.ndarray,
    customer_km: np.ndarray,
    routes: List[List[int]],
    speed: float,
    demands: Optional[np.ndarray] = None,
) -> Tuple[float, int]:
    """
    Compute total delivery time = sum of arrival times at each customer.

    When demands is provided, each location j contributes demands[j]
    orders, all sharing the same arrival time (one stop serves all
    orders at that candidate location).

    Returns (sum_of_weighted_arrival_times_h, total_orders_served).
    """
    total_arrival_h = 0.0
    n_served = 0
    for route in routes:
        if not route:
            continue
        cum_dist = 0.0
        prev = depot_km
        for idx in route:
            cum_dist += float(np.linalg.norm(customer_km[idx] - prev))
            arrival_h = cum_dist / speed
            d = int(demands[idx]) if demands is not None else 1
            total_arrival_h += arrival_h * d
            n_served += d
            prev = customer_km[idx]
    return total_arrival_h, n_served


# ---------------------------------------------------------------------------
# TemporalOnly baseline
# ---------------------------------------------------------------------------

class TemporalOnly(BaselineMethod):
    """
    Temporal-only baseline (Liu & Luo 2023).

    Single depot, fixed dispatch interval T, ADP-optimized fleet dispatch,
    VRP-optimized routing.  No spatial partition, no BHH.

    Demand model: a finite set of candidate locations I = {1,...,I} (the
    block centroids) is a hyperparameter.  Each period, per-location
    demands q_i ~ Poisson(Lambda * T * p_i) arise at a random subset.

    Three-layer architecture:
      Layer 1 (ADP): APT cost-to-go via LP allocation for forward-looking
                     dispatch.
      Layer 2 (Benders): Iterative refinement — concentrate MC samples on
                     K values selected by the APT.
      Layer 3 (CG): VRP via set partitioning + ESPPRC pricing.

    Evaluation: custom VRP-based simulation with realized delivery times.
    No linehaul cost, no ODD cost (single depot, non-partition mode).
    """

    def __init__(
        self,
        T_fixed: float = 0.25,
        n_mc: int = 30,
        max_fleet: int = 200,
        horizon: int = 40,
        capacity: int = 20,
        benders_iters: int = 2,
        n_eval_scenarios: int = 500,
        n_candidates: int = 0,
        region_km: float = 10.0,
        region_low: float = 0.0,
    ):
        self.T_fixed = T_fixed
        self.n_mc = n_mc
        self.max_fleet = max_fleet
        self.horizon = horizon
        self.capacity = capacity
        self.benders_iters = benders_iters
        self.n_eval_scenarios = n_eval_scenarios
        self.n_candidates = n_candidates
        self.region_km = region_km
        self.region_low = region_low

        # Populated by design()
        self._cost_table: Dict[int, dict] = {}
        self._fleet_size: int = 1
        self._dispatch_K: int = 1
        self._candidate_pos_km: Optional[np.ndarray] = None
        self._candidate_probs: Optional[np.ndarray] = None
        self._depot_km: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # H^s(K) estimation via Monte Carlo  (Layer 2 + 3)
    # ------------------------------------------------------------------
    def _compute_Hs_samples(
        self,
        K: int,
        n_samples: int,
        candidate_pos_km: np.ndarray,
        candidate_probs: np.ndarray,
        depot_km: np.ndarray,
        Lambda_T: float,
        rng: np.random.Generator,
    ) -> Dict[str, list]:
        """
        Compute MC samples of H^s(K) using discrete candidate locations.

        Each sample: draw per-location Poisson demands, solve CVRP for K
        vehicles, compute sum of arrival times (delivery-time objective).
        """
        speed = VEHICLE_SPEED_KMH
        T = self.T_fixed

        arrival_sums: list = []
        total_kms: list = []
        max_tour_hs: list = []
        n_served_list: list = []

        for _ in range(n_samples):
            active_pos, demands = _sample_discrete_demand(
                candidate_probs, candidate_pos_km, Lambda_T, rng)
            n_active = len(active_pos)
            n_cust = int(demands.sum())

            if n_active == 0:
                arrival_sums.append(0.0)
                total_kms.append(0.0)
                max_tour_hs.append(0.0)
                n_served_list.append(0)
                continue

            # Cap demand to feasible capacity
            total_demand = int(demands.sum())
            if total_demand > K * self.capacity:
                # Truncate: keep locations with highest demand first
                order = np.argsort(-demands)
                cum = np.cumsum(demands[order])
                keep = int(np.searchsorted(cum, K * self.capacity, side="right")) + 1
                keep = min(keep, n_active)
                sel = order[:keep]
                active_pos = active_pos[sel]
                demands = demands[sel]
                n_active = len(active_pos)

            K_eff = min(K, n_active)

            routes, total_km, per_veh = _solve_vrp(
                depot_km, active_pos, K_eff, self.capacity,
                demands=demands)

            sum_arrival, n_served = _delivery_times_from_routes(
                depot_km, active_pos, routes, speed, demands=demands)

            total_kms.append(total_km)
            max_h = max((pv / speed for pv in per_veh), default=0.0)
            max_tour_hs.append(max_h)
            arrival_sums.append(sum_arrival)
            n_served_list.append(n_served)

        return {
            "arrival_sums": arrival_sums,
            "total_kms": total_kms,
            "max_tour_hs": max_tour_hs,
            "n_served": n_served_list,
        }

    @staticmethod
    def _aggregate_Hs(samples: Dict[str, list], T: float) -> dict:
        """Aggregate MC samples into cost_table entry."""
        arrival_sums = samples["arrival_sums"]
        total_kms = samples["total_kms"]
        max_tour_hs = samples["max_tour_hs"]
        n_served = samples["n_served"]

        E_arrival = float(np.mean(arrival_sums))
        E_n = float(np.mean(n_served))
        E_total_km = float(np.mean(total_kms))
        E_max_h = float(np.mean(max_tour_hs))
        D_K = max(1, int(math.ceil(E_max_h / T))) if E_max_h > 0 else 1

        # Paper objective: total delivery time = wait + in-vehicle
        H_s = E_n * T / 2.0 + E_arrival

        avg_inveh = E_arrival / max(E_n, 1e-12)

        return {
            "H_s": H_s,
            "arrival_sum": E_arrival,
            "avg_inveh_h": avg_inveh,
            "total_km": E_total_km,
            "max_tour_h": E_max_h,
            "D": D_K,
            "n_samples": len(arrival_sums),
        }

    # ------------------------------------------------------------------
    # APT allocation via LP  (Layer 1)
    # ------------------------------------------------------------------
    def _solve_apt(
        self,
        M: int,
        cost_table: Dict[int, dict],
        K_max: int,
        horizon: int,
    ) -> Tuple[float, Dict[int, int]]:
        """
        Solve the APT allocation LP for a given fleet size M.

        Variables: y[m, k] = 1 if K_m = k  (binary)
        Constraints: driver-availability carryover.
        Returns (avg_cost_per_period, dispatch_profile).
        """
        import gurobipy as gp
        from gurobipy import GRB

        H = horizon
        all_k = sorted(k for k in cost_table if cost_table[k]["H_s"] < 1e5)
        if 0 not in all_k:
            all_k = [0] + all_k

        model = gp.Model("APT")
        model.setParam("OutputFlag", 0)
        model.setParam("TimeLimit", 30.0)

        y = {}
        for m in range(H):
            for k in all_k:
                y[m, k] = model.addVar(vtype=GRB.BINARY)

        for m in range(H):
            model.addConstr(
                gp.quicksum(y[m, k] for k in all_k) == 1)

        for m in range(H):
            busy = gp.LinExpr()
            for k in all_k:
                busy += k * y[m, k]
            for m_prev in range(m):
                delta = m - m_prev
                for k in all_k:
                    if k == 0:
                        continue
                    D_k = cost_table[k]["D"]
                    if D_k > delta:
                        busy += k * y[m_prev, k]
            model.addConstr(busy <= M)

        obj = gp.LinExpr()
        for m in range(H):
            for k in all_k:
                obj += cost_table[k]["H_s"] * y[m, k]
        model.setObjective(obj, GRB.MINIMIZE)
        model.optimize()

        if model.status not in (GRB.OPTIMAL, GRB.TIME_LIMIT) or model.SolCount == 0:
            return float("inf"), {}

        profile: Dict[int, int] = {}
        for m in range(H):
            for k in all_k:
                if y[m, k].X > 0.5:
                    profile[m] = k
                    break

        avg_cost = model.ObjVal / H
        return avg_cost, profile

    # ------------------------------------------------------------------
    # design
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
        T = self.T_fixed
        speed = VEHICLE_SPEED_KMH
        Lambda_T = Lambda * T

        rng = np.random.default_rng(42)

        if self.n_candidates > 0:
            # Paper-faithful: random uniform candidate locations in a square
            N = self.n_candidates
            candidate_pos_km = rng.uniform(self.region_low, self.region_km, size=(N, 2))
            candidate_probs = np.ones(N) / N  # uniform probability
        else:
            # Fall back to geodata block positions
            block_ids = geodata.short_geoid_list
            N = len(block_ids)
            positions_m = np.array([_get_pos(geodata, b) for b in block_ids])
            candidate_pos_km = positions_m / 1000.0
            candidate_probs = np.array([prob_dict.get(b, 0.0) for b in block_ids])
            total_p = candidate_probs.sum()
            if total_p > 1e-15:
                candidate_probs = candidate_probs / total_p

        self._candidate_pos_km = candidate_pos_km
        self._candidate_probs = candidate_probs

        # Depot: centroid of candidate locations
        centroid_km = candidate_pos_km.mean(axis=0)
        depot_idx = int(np.argmin(np.linalg.norm(candidate_pos_km - centroid_km, axis=1)))
        depot_km = candidate_pos_km[depot_idx]
        self._depot_km = depot_km

        # Expected total demand per period
        demand_rate = Lambda_T  # = Lambda * T (since probs sum to 1)

        # Dispatch-K range
        K_min_feasible = max(1, int(math.ceil(demand_rate / max(self.capacity, 1))))
        K_max = max(K_min_feasible, min(self.max_fleet, K_min_feasible + 10))

        # ---- Layer 2+3: Pre-compute E[H^s(K)] via MC + CVRP ----
        print(f"    Pre-computing H^s(K) for K={K_min_feasible}..{K_max} "
              f"({self.n_mc} MC samples, demand_rate={demand_rate:.1f}, "
              f"{N} candidate locations) ...", flush=True)

        cost_table: Dict[int, dict] = {}
        for K in range(0, K_min_feasible):
            cost_table[K] = {
                "H_s": 1e6, "arrival_sum": 0.0, "avg_inveh_h": 0.0,
                "total_km": 0.0, "max_tour_h": 0.0, "D": 0, "n_samples": 0,
            }
        all_samples: Dict[int, Dict[str, list]] = {}
        for K in range(K_min_feasible, K_max + 1):
            samples = self._compute_Hs_samples(
                K, self.n_mc, candidate_pos_km, candidate_probs,
                depot_km, Lambda_T, rng)
            all_samples[K] = samples
            cost_table[K] = self._aggregate_Hs(samples, T)

        self._cost_table = cost_table

        # ---- Layer 1: APT fleet optimisation ----
        candidate_fleets = set()
        for K in range(K_min_feasible, K_max + 1):
            M_needed = K * cost_table[K]["D"]
            candidate_fleets.add(M_needed)
            candidate_fleets.add(M_needed + K)
        candidate_fleets = sorted(
            m for m in candidate_fleets if 1 <= m <= self.max_fleet)
        if not candidate_fleets:
            candidate_fleets = [K_min_feasible]

        print(f"    Running APT LP for fleet optimisation "
              f"({len(candidate_fleets)} candidates) ...", flush=True)

        best_M = candidate_fleets[0]
        best_cost = float("inf")
        best_profile: Dict[int, int] = {}
        for M in candidate_fleets:
            avg_cost, profile = self._solve_apt(
                M, cost_table, K_max, self.horizon)
            if avg_cost < best_cost:
                best_cost = avg_cost
                best_M = M
                best_profile = profile

        # ---- Layer 2 (Benders): Refine H^s for selected K values ----
        for benders_iter in range(self.benders_iters):
            K_used = set(best_profile.values())
            K_refine = set()
            for K in K_used:
                if K > 0:
                    for dk in range(-1, 2):
                        Kc = K + dk
                        if K_min_feasible <= Kc <= K_max:
                            K_refine.add(Kc)

            if not K_refine:
                break

            changed = False
            for K in K_refine:
                new_samples = self._compute_Hs_samples(
                    K, self.n_mc, candidate_pos_km, candidate_probs,
                    depot_km, Lambda_T, rng)
                old = all_samples.get(K, {
                    "arrival_sums": [], "total_kms": [],
                    "max_tour_hs": [], "n_served": [],
                })
                merged = {
                    key: old.get(key, []) + new_samples[key]
                    for key in new_samples
                }
                all_samples[K] = merged
                new_entry = self._aggregate_Hs(merged, T)
                old_Hs = cost_table[K]["H_s"]
                cost_table[K] = new_entry
                if abs(new_entry["H_s"] - old_Hs) > 0.01 * abs(old_Hs + 1e-12):
                    changed = True

            if not changed:
                break

            for M in candidate_fleets:
                avg_cost, profile = self._solve_apt(
                    M, cost_table, K_max, self.horizon)
                if avg_cost < best_cost:
                    best_cost = avg_cost
                    best_M = M
                    best_profile = profile

        self._cost_table = cost_table
        self._fleet_size = best_M

        # Steady-state dispatch K: most common K in APT profile
        if best_profile:
            from collections import Counter
            K_counts = Counter(
                k for k in best_profile.values() if k > 0)
            self._dispatch_K = (K_counts.most_common(1)[0][0]
                                if K_counts else K_min_feasible)
        else:
            self._dispatch_K = K_min_feasible

        print(f"    APT -> fleet={best_M}, dispatch_K={self._dispatch_K}, "
              f"H^s/period={best_cost:.4f}")

        return ServiceDesign(
            name="TP-Lit",
            assignment=None,
            depot_id=f"candidate_{depot_idx}",
            district_roots=[],
            dispatch_intervals={},
            service_mode="temporal",
            service_metadata={
                "fleet_size": best_M,
                "dispatch_K": self._dispatch_K,
                "T_fixed": T,
                "capacity": self.capacity,
                "demand_rate": demand_rate,
                "n_candidate_locations": N,
                "n_eval_scenarios": self.n_eval_scenarios,
            },
        )

    # ------------------------------------------------------------------
    # Evaluation: custom VRP-based simulation (no BHH, no linehaul)
    # ------------------------------------------------------------------
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
    ) -> EvaluationResult:
        """
        Scenario-based VRP evaluation with realized delivery times.

        Each scenario is an independent dispatch period:
          1. Sample per-location Poisson demands at candidate locations.
          2. Solve CVRP with K vehicles.
          3. Compute per-customer delivery time from route structure.

        No linehaul, no ODD, no walk (single depot, non-partition mode).
        """
        meta = design.service_metadata
        T = float(meta.get("T_fixed", self.T_fixed))
        K = int(meta.get("dispatch_K", self._dispatch_K))
        capacity = int(meta.get("capacity", self.capacity))
        n_scenarios = int(meta.get("n_eval_scenarios", self.n_eval_scenarios))
        Lambda_T = Lambda * T

        # Rebuild candidate arrays if needed (e.g. fresh instance)
        candidate_pos_km = self._candidate_pos_km
        candidate_probs = self._candidate_probs
        depot_km = self._depot_km
        if candidate_pos_km is None:
            block_ids = geodata.short_geoid_list
            positions_m = np.array([_get_pos(geodata, b) for b in block_ids])
            candidate_pos_km = positions_m / 1000.0
            candidate_probs = np.array([prob_dict.get(b, 0.0)
                                        for b in block_ids])
            total_p = candidate_probs.sum()
            if total_p > 1e-15:
                candidate_probs = candidate_probs / total_p
            self._candidate_pos_km = candidate_pos_km
            self._candidate_probs = candidate_probs
        if depot_km is None:
            block_ids = geodata.short_geoid_list
            positions_m = np.array([_get_pos(geodata, b) for b in block_ids])
            centroid = positions_m.mean(axis=0)
            depot_idx = int(np.argmin(
                np.linalg.norm(positions_m - centroid, axis=1)))
            depot_km = positions_m[depot_idx] / 1000.0
            self._depot_km = depot_km

        speed = VEHICLE_SPEED_KMH
        discount = TSP_TRAVEL_DISCOUNT

        rng = np.random.default_rng(123)

        sum_wait_h = 0.0
        sum_invehicle_h = 0.0
        sum_travel_km = 0.0
        sum_customers = 0

        for _ in range(n_scenarios):
            active_pos, demands = _sample_discrete_demand(
                candidate_probs, candidate_pos_km, Lambda_T, rng)
            n_active = len(active_pos)
            if n_active == 0:
                continue

            total_demand = int(demands.sum())
            # Cap to feasible capacity
            if total_demand > K * capacity:
                order = np.argsort(-demands)
                cum = np.cumsum(demands[order])
                keep = int(np.searchsorted(cum, K * capacity, side="right")) + 1
                keep = min(keep, n_active)
                sel = order[:keep]
                active_pos = active_pos[sel]
                demands = demands[sel]
                n_active = len(active_pos)
                total_demand = int(demands.sum())

            K_eff = min(K, n_active)

            routes, total_km, per_veh = _solve_vrp(
                depot_km, active_pos, K_eff, capacity, demands=demands)

            sum_arrival, n_served = _delivery_times_from_routes(
                depot_km, active_pos, routes, speed, demands=demands)

            sum_wait_h += total_demand * T / 2.0
            sum_invehicle_h += sum_arrival
            sum_travel_km += total_km
            sum_customers += total_demand

        if sum_customers == 0:
            raw_fleet_size = int(meta.get("fleet_size", self._fleet_size))
            return EvaluationResult(
                name=design.name,
                avg_wait_time=T / 2.0,
                avg_invehicle_time=0.0,
                avg_walk_time=0.0,
                total_user_time=T / 2.0,
                linehaul_cost=0.0,
                in_district_cost=0.0,
                total_travel_cost=0.0,
                fleet_size=0.0,
                raw_fleet_size=raw_fleet_size,
                avg_dispatch_interval=T,
                odd_cost=0.0,
                provider_cost=fleet_cost_rate * raw_fleet_size,
                user_cost=wr * T / 2.0,
                total_cost=fleet_cost_rate * raw_fleet_size + wr * T / 2.0,
            )

        avg_wait = sum_wait_h / sum_customers
        avg_invehicle = sum_invehicle_h / sum_customers
        avg_walk = 0.0
        total_user_time = avg_wait + avg_invehicle + avg_walk

        avg_travel_km = sum_travel_km / n_scenarios
        in_district_cost = avg_travel_km / (discount * T)
        linehaul_cost = 0.0
        odd_cost = 0.0
        total_travel_cost = in_district_cost
        fleet_size = avg_travel_km / (VEHICLE_SPEED_KMH * T)
        raw_fleet_size = int(meta.get("fleet_size", self._fleet_size))
        avg_dispatch_interval = T

        provider_cost = total_travel_cost + fleet_cost_rate * raw_fleet_size
        user_cost = wr * total_user_time
        total_cost = provider_cost + user_cost

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
            raw_fleet_size=raw_fleet_size,
            avg_dispatch_interval=avg_dispatch_interval,
            odd_cost=odd_cost,
            provider_cost=provider_cost,
            user_cost=user_cost,
            total_cost=total_cost,
        )

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
        fleet_cost_rate: float = DEFAULT_FLEET_COST_RATE,
        beta: float = BETA,
        road_network=None,
    ) -> EvaluationResult:
        return self.custom_simulate(
            design, geodata, prob_dict, Omega_dict, J_function,
            Lambda, wr, wv, fleet_cost_rate,
        )
