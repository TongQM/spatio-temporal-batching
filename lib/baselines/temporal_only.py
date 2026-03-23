"""
Temporal-only baseline (Liu & Luo 2023): single depot, fixed 15-min dispatch,
ADP-optimized fleet dispatch, VRP-optimized routing.

Three-layer architecture following the paper:
  Layer 1 (ADP): APT cost-to-go approximation for forward-looking dispatch
  Layer 2 (Benders): Dispatch/routing separation — evaluate H^s(K) per K
  Layer 3 (CG + route enum): VRP via set-partitioning + ESPPRC pricing

No spatial partition. No BHH approximation. Continuous demand region.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Tuple

import numpy as np

from lib.baselines.base import (
    BETA, VEHICLE_SPEED_KMH, BaselineMethod, EvaluationResult, ServiceDesign,
    _get_pos, _nn_tsp_length,
)
from lib.constants import TSP_TRAVEL_DISCOUNT


# ---------------------------------------------------------------------------
# Continuous-space demand sampler
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
# VRP solver: MTZ formulation (small instances, n <= 15)
# ---------------------------------------------------------------------------

def _vrp_mtz(depot_km: np.ndarray, customer_km: np.ndarray,
             K: int, time_limit: float = 10.0):
    """
    Exact VRP via Miller-Tucker-Zemlin formulation with Gurobi.

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

    # 0 = depot, 1..n = customers
    all_pts = np.vstack([depot_km[None], customer_km])
    V = n + 1
    dist = np.linalg.norm(all_pts[:, None] - all_pts[None, :], axis=2)

    model = gp.Model("VRP_MTZ")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", time_limit)

    x = {}
    for i in range(V):
        for j in range(V):
            if i != j:
                x[i, j] = model.addVar(vtype=GRB.BINARY)
    u = {i: model.addVar(lb=1.0, ub=float(n), vtype=GRB.CONTINUOUS)
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

    # MTZ subtour elimination
    for i in range(1, V):
        for j in range(1, V):
            if i != j:
                model.addConstr(u[i] - u[j] + n * x[i, j] <= n - 1)

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
                    capacity: int, max_labels: int = 200_000):
    """
    Find a route with minimum reduced cost via label-setting.

    dist : (n+1)x(n+1), node 0 = depot, 1..n = customers.
    pi   : dual for customer coverage (0-indexed customer key).
    mu   : dual for fleet constraint.

    Returns route dict or None.
    """
    import heapq

    # Label: (reduced_cost, cost, node, visited_bits, count, prev_label_id)
    # visited_bits: integer bitmask for n <= 30; frozenset otherwise
    use_bits = n <= 30

    # label storage: id -> (rc, cost, node, visited, count, sequence)
    labels_store: list = []

    def _make_visited_empty():
        return 0 if use_bits else frozenset()

    def _add_to_visited(v, j):
        return v | (1 << j) if use_bits else v | {j}

    def _in_visited(v, j):
        return bool(v & (1 << j)) if use_bits else (j in v)

    def _is_subset(v1, v2):
        # v1 ⊆ v2
        return (v1 & v2) == v1 if use_bits else v1 <= v2

    # Best labels per node for dominance: node -> list of (rc, visited)
    node_labels: Dict[int, list] = {j: [] for j in range(n + 1)}

    # Priority queue: (rc, label_id)
    heap: list = []
    initial_rc = -mu
    lid = 0
    labels_store.append((initial_rc, 0.0, 0, _make_visited_empty(), 0, []))
    heapq.heappush(heap, (initial_rc, lid))

    best_route = None
    best_rc = -1e-6

    while heap and lid < max_labels:
        rc_cur, cur_lid = heapq.heappop(heap)
        if rc_cur >= best_rc:
            continue
        rc, cost, node, visited, count, seq = labels_store[cur_lid]
        if rc != rc_cur:
            continue  # stale

        if count >= capacity:
            continue

        for j in range(1, n + 1):
            if _in_visited(visited, j):
                continue

            new_cost = cost + dist[node, j]
            new_rc = rc + dist[node, j] - pi.get(j - 1, 0.0)
            new_visited = _add_to_visited(visited, j)
            new_count = count + 1
            new_seq = seq + [j - 1]

            # Dominance check at node j
            dominated = False
            for (old_rc, old_v) in node_labels[j]:
                if old_rc <= new_rc + 1e-9 and _is_subset(old_v, new_visited):
                    dominated = True
                    break
            if dominated:
                continue

            # Remove dominated labels at node j
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

            # Continue extending
            lid += 1
            labels_store.append(
                (new_rc, new_cost, j, new_visited, new_count, new_seq))
            heapq.heappush(heap, (new_rc, lid))

    return best_route


# ---------------------------------------------------------------------------
# VRP via column generation (larger instances)
# ---------------------------------------------------------------------------

def _solve_vrp_cg(depot_km: np.ndarray, customer_km: np.ndarray,
                  K: int, capacity: int = 20, time_limit: float = 30.0):
    """
    Solve VRP for K vehicles via set partitioning with column generation.

    Returns (routes, total_km, per_veh_km).
    """
    import gurobipy as gp
    from gurobipy import GRB

    n = len(customer_km)
    if n == 0:
        return [[] for _ in range(K)], 0.0, [0.0] * K

    all_pts = np.vstack([depot_km[None], customer_km])
    dist = np.linalg.norm(all_pts[:, None] - all_pts[None, :], axis=2)

    # Initial columns: K disjoint routes covering all customers (angular partition)
    columns: List[dict] = []

    # Angular partition from depot: exactly K groups, each ≤ capacity
    angles = np.arctan2(customer_km[:, 1] - depot_km[1],
                        customer_km[:, 0] - depot_km[0])
    order = np.argsort(angles)
    # Balanced partition: K groups of floor(n/K) or ceil(n/K) customers
    base_size = n // K
    extra = n % K  # first `extra` groups get one more customer
    final_groups: List[List[int]] = []
    start = 0
    for k in range(K):
        size = base_size + (1 if k < extra else 0)
        if start >= n or size == 0:
            break
        end = min(start + size, n)
        grp = [int(order[i]) for i in range(start, end)]
        final_groups.append(grp)
        start = end

    for grp in final_groups:
        # NN-TSP ordering within group
        grp_pts = customer_km[grp]
        if len(grp) <= 1:
            seq = list(grp)
        else:
            visited = set()
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
        # Compute route cost
        c = float(dist[0, seq[0] + 1])
        for i in range(1, len(seq)):
            c += float(dist[seq[i - 1] + 1, seq[i] + 1])
        c += float(dist[seq[-1] + 1, 0])
        columns.append({
            "customers": set(seq),
            "cost": c,
            "sequence": list(seq),
        })

    # Also add single-customer routes for CG flexibility
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

        # Coverage: each customer exactly once
        cover = {}
        for j in range(n):
            col_idx = [r for r in range(R) if j in columns[r]["customers"]]
            cover[j] = model.addConstr(
                gp.quicksum(theta[r] for r in col_idx) == 1)

        # Fleet constraint: exactly K vehicles
        fleet_c = model.addConstr(gp.quicksum(theta[r] for r in range(R)) == K)

        model.setObjective(
            gp.quicksum(columns[r]["cost"] * theta[r] for r in range(R)),
            GRB.MINIMIZE)
        model.optimize()

        if model.status != GRB.OPTIMAL:
            break

        pi = {j: cover[j].Pi for j in range(n)}
        mu_val = fleet_c.Pi

        new_route = _espprc_pricing(dist, n, pi, mu_val, capacity)
        if new_route is None:
            break
        columns.append(new_route)

    # Integer phase: solve MIP with all generated columns
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
               K: int, capacity: int = 20, time_limit: float = 15.0):
    """Route K vehicles from depot to serve all customers."""
    n = len(customer_km)
    if n == 0:
        return [], 0.0, []
    if K >= n:
        # Each vehicle serves at most 1 customer
        routes = [[j] for j in range(n)]
        per_veh = [2.0 * float(np.linalg.norm(customer_km[j] - depot_km))
                   for j in range(n)]
        return routes, sum(per_veh), per_veh
    if n <= 15:
        return _vrp_mtz(depot_km, customer_km, K, time_limit)
    return _solve_vrp_cg(depot_km, customer_km, K, capacity, time_limit)


# ---------------------------------------------------------------------------
# TemporalOnly baseline
# ---------------------------------------------------------------------------

class TemporalOnly(BaselineMethod):
    """
    Temporal-only baseline (Liu & Luo 2023).

    Single depot, fixed dispatch interval T, ADP-optimized fleet dispatch,
    VRP-optimized routing.  No spatial partition, no BHH.

    Three-layer architecture:
      Layer 1 (ADP): APT cost-to-go for forward-looking dispatch
      Layer 2 (Benders): Dispatch/routing separation — evaluate H^s(K) per K
      Layer 3 (CG): VRP via set partitioning + ESPPRC pricing
    """

    def __init__(
        self,
        T_fixed: float = 0.25,
        n_mc: int = 30,
        max_fleet: int = 200,
        horizon: int = 40,
        capacity: int = 20,
    ):
        self.T_fixed = T_fixed
        self.n_mc = n_mc
        self.max_fleet = max_fleet
        self.horizon = horizon
        self.capacity = capacity

        # Populated by design()
        self._cost_table: Dict[int, dict] = {}
        self._fleet_size: int = 1
        self._dispatch_K: int = 1
        self._sampler: DemandSampler | None = None
        self._depot_km: np.ndarray | None = None

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
        block_ids = geodata.short_geoid_list
        N = len(block_ids)
        T = self.T_fixed
        speed = VEHICLE_SPEED_KMH
        discount = TSP_TRAVEL_DISCOUNT
        demand_rate = Lambda * T

        # Depot: geographic centroid block
        positions = np.array([_get_pos(geodata, b) for b in block_ids])
        centroid = positions.mean(axis=0)
        depot_idx = int(np.argmin(np.linalg.norm(positions - centroid, axis=1)))
        depot_id = block_ids[depot_idx]
        depot_km = positions[depot_idx] / 1000.0
        self._depot_km = depot_km

        # Demand sampler
        sampler = DemandSampler(geodata, prob_dict)
        self._sampler = sampler

        rng = np.random.default_rng(42)

        # Upper bound on vehicles per dispatch:
        # need at least ceil(demand / capacity) vehicles; search a range above that
        K_min_needed = max(1, int(math.ceil(demand_rate / max(self.capacity, 1))))
        K_max = max(K_min_needed, min(self.max_fleet, K_min_needed + 10))

        # ---- Pre-compute E[H^s(K)] for K = K_min..K_max  (Layer 2 + 3) ----
        # Only evaluate K that can feasibly serve demand given capacity
        K_min_feasible = K_min_needed  # = ceil(demand / capacity)
        print(f"    Pre-computing VRP costs for K={K_min_feasible}..{K_max} "
              f"({self.n_mc} MC samples, demand_rate={demand_rate:.1f}) …",
              flush=True)

        cost_table: Dict[int, dict] = {}
        for K in range(K_min_feasible, K_max + 1):
            total_veh_kms: list = []
            max_tour_hs: list = []
            avg_inveh_hs: list = []

            for _ in range(self.n_mc):
                n_cust = rng.poisson(demand_rate)
                if n_cust == 0:
                    total_veh_kms.append(0.0)
                    max_tour_hs.append(0.0)
                    avg_inveh_hs.append(0.0)
                    continue

                # Cap at capacity for Poisson tail; keeps VRP feasible
                n_serve = min(n_cust, K * self.capacity)
                customers = sampler.sample(n_serve, rng)
                K_eff = min(K, n_serve)

                routes, total_km, per_veh = _solve_vrp(
                    depot_km, customers, K_eff, self.capacity)

                total_veh_kms.append(total_km)
                max_h = max((pv / speed for pv in per_veh), default=0.0)
                max_tour_hs.append(max_h)

                # Per-customer in-vehicle time ≈ tour_k / (2 * speed)
                tot_inv = 0.0
                for ri, route in enumerate(routes):
                    if route and ri < len(per_veh):
                        tot_inv += per_veh[ri] / (2.0 * speed) * len(route)
                avg_inveh_hs.append(tot_inv / max(n_serve, 1))

            E_total_km = float(np.mean(total_veh_kms))
            E_max_h = float(np.mean(max_tour_hs))
            E_inv = float(np.mean(avg_inveh_hs))
            D_K = max(1, int(math.ceil(E_max_h / T)))

            # Steady-state cost matching evaluate_design convention:
            #   provider = total_veh_km / (discount * T)   [km/hour rate]
            #   user     = wr * (T/2 + avg_inveh)          [per-rider time cost]
            # total_cost = provider + user  (same units as baseline comparison)
            provider_rate = E_total_km / (discount * T)
            user_per_rider = wr * (T / 2.0 + E_inv)
            H_s = provider_rate + user_per_rider

            cost_table[K] = {
                "H_s": H_s,
                "total_km": E_total_km,
                "max_tour_h": E_max_h,
                "avg_inveh_h": E_inv,
                "D": D_K,
            }

        # K < K_min_feasible: infeasible due to capacity — large penalty
        for K in range(0, K_min_feasible):
            cost_table[K] = {
                "H_s": 1e6,
                "total_km": 0.0, "max_tour_h": 0.0, "avg_inveh_h": 0.0, "D": 0,
            }
        self._cost_table = cost_table

        # ---- ADP: optimise fleet M  (Layer 1) ----
        # Fleet upper bound: K_min * max_D (enough to dispatch every period)
        max_D = max((v["D"] for k, v in cost_table.items() if k >= K_min_feasible),
                    default=1)
        fleet_upper = min(self.max_fleet, K_max * max_D + K_max)
        print(f"    Running ADP for fleet optimisation (M=1..{fleet_upper}) …",
              flush=True)

        best_M = 1
        best_cost = float("inf")
        for M in range(1, fleet_upper + 1):
            avg = self._simulate_apt_policy(M, cost_table, K_max)
            if avg < best_cost:
                best_cost = avg
                best_M = M

        self._fleet_size = best_M

        # Steady-state dispatch K: best feasible K given fleet M
        best_K = K_min_feasible
        best_H = float("inf")
        for K in range(K_min_feasible, K_max + 1):
            info = cost_table[K]
            if K * info["D"] <= best_M and info["H_s"] < best_H:
                best_H = info["H_s"]
                best_K = K
        self._dispatch_K = best_K

        print(f"    ADP → fleet={best_M}, dispatch_K={best_K}, "
              f"cost/period={best_cost:.4f}")

        # ServiceDesign: single district
        assignment = np.zeros((N, N))
        assignment[:, 0] = 1.0
        root = block_ids[0]

        return ServiceDesign(
            name="TP-Lit",
            assignment=assignment,
            depot_id=depot_id,
            district_roots=[root],
            dispatch_intervals={root: T},
        )

    # ------------------------------------------------------------------
    # APT simulation
    # ------------------------------------------------------------------
    def _simulate_apt_policy(self, M: int, cost_table: dict, K_max: int) -> float:
        """Forward-simulate APT dispatch policy; return avg cost per period."""
        horizon = self.horizon
        T = self.T_fixed
        return_queue: List[Tuple[int, int]] = []  # (return_period, count)
        z = M
        total_cost = 0.0

        for period in range(horizon):
            # Process returns
            new_q: list = []
            for rp, cnt in return_queue:
                if rp <= period:
                    z = min(z + cnt, M)
                else:
                    new_q.append((rp, cnt))
            return_queue = new_q

            # Dispatch decision: minimise immediate + lookahead
            best_K = 0
            best_val = float("inf")
            for K in range(0, min(z, K_max) + 1):
                if K not in cost_table:
                    continue
                immediate = cost_table[K]["H_s"]
                future = self._apt_lookahead(
                    z - K, return_queue, K, period, cost_table, K_max, M)
                val = immediate + future
                if val < best_val:
                    best_val = val
                    best_K = K

            total_cost += cost_table[best_K]["H_s"]

            D_K = cost_table[best_K]["D"]
            if best_K > 0 and D_K > 0:
                return_queue.append((period + D_K, best_K))
            z -= best_K

        return total_cost / max(horizon, 1)

    def _apt_lookahead(self, z_after: int, return_queue, K_disp: int,
                       period: int, cost_table: dict, K_max: int,
                       M: int, lookahead: int = 10) -> float:
        """APT cost-to-go: greedy forward capacity allocation."""
        future_returns: Dict[int, int] = {}
        for rp, cnt in return_queue:
            if rp > period:
                future_returns[rp] = future_returns.get(rp, 0) + cnt
        D_K = cost_table[K_disp]["D"] if K_disp > 0 else 0
        if K_disp > 0 and D_K > 0:
            rp = period + D_K
            future_returns[rp] = future_returns.get(rp, 0) + K_disp

        z = z_after
        total = 0.0
        for m in range(period + 1, period + 1 + lookahead):
            z = min(z + future_returns.get(m, 0), M)
            best_K = 0
            best_H = cost_table[0]["H_s"]
            for k in range(1, min(z, K_max) + 1):
                if k in cost_table and cost_table[k]["H_s"] < best_H:
                    best_H = cost_table[k]["H_s"]
                    best_K = k
            total += best_H
            z -= best_K
            if best_K > 0:
                D = cost_table[best_K]["D"]
                if D > 0:
                    rp2 = m + D
                    future_returns[rp2] = future_returns.get(rp2, 0) + best_K
        return total

    # ------------------------------------------------------------------
    # Evaluation: delegates to simulate_design for consistency
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
        from lib.baselines.base import simulate_design
        return simulate_design(
            design, geodata, prob_dict, Omega_dict, J_function,
            Lambda, wr, wv,
        )
