"""
VCC (Vehicle-Customer Coordination) baseline — adapted from Zhang et al. 2023
(Management Science) to the last-mile delivery setting.

Adaptation choices:
  - Origins fixed at depot  (all customers board at the depot)
  - No time-window constraints
  - Drop-off flexibility only: vehicle stops at meeting point M_j;
    customer walks from M_j to destination H_j, d(M_j, H_j) ≤ lw.

Algorithm: MSO-VCC (greedy NN initialisation + iterative meeting-point
optimisation, adapted from Algorithm 1 / Section 4 of the paper).

Spatial partition: same random-search as SP-Lit (ε=0, fixed T), with the
partition search scoring partitions at T_fixed (not the unconstrained T*).
The VCC benefit is shorter in-district vehicle routes; user walk time is
non-zero (bounded by lw / walk_speed).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np

from lib.baselines.base import (
    BETA, BaselineMethod, EvaluationResult, ServiceDesign,
    _get_pos, evaluate_design,
)


# ---------------------------------------------------------------------------
# VCC geometry helpers
# ---------------------------------------------------------------------------

def _optimize_meeting_point(
    prev: np.ndarray,
    nxt: np.ndarray,
    dest: np.ndarray,
    lw: float,
) -> np.ndarray:
    """
    Find M on/near segment [prev, nxt] within lw of dest.

    Minimises ||prev-M|| + ||M-nxt|| s.t. ||M-dest|| ≤ lw.

    The unconstrained minimum is any M on segment [prev, nxt] (detour = 0).
    Project dest onto the segment; if within lw return it. Otherwise shift
    lw along the dest→projection direction to land on the ball boundary.
    """
    seg = nxt - prev
    seg_len_sq = float(np.dot(seg, seg))

    if seg_len_sq < 1e-18:
        d = float(np.linalg.norm(dest - prev))
        if d <= lw:
            return prev.copy()
        return dest + lw * (prev - dest) / d

    t = float(np.dot(dest - prev, seg)) / seg_len_sq
    t = max(0.0, min(1.0, t))
    M_proj = prev + t * seg

    diff = M_proj - dest
    diff_len = float(np.linalg.norm(diff))
    if diff_len <= lw:
        return M_proj

    return dest + lw * diff / diff_len


def _greedy_nn_tour(depot: np.ndarray, points: np.ndarray) -> List[int]:
    """Greedy nearest-neighbour from depot; returns permutation of range(n)."""
    n = len(points)
    if n == 0:
        return []
    unvisited = list(range(n))
    route: List[int] = []
    cur = depot.copy()
    while unvisited:
        dists = [float(np.linalg.norm(points[j] - cur)) for j in unvisited]
        idx = int(np.argmin(dists))
        nearest = unvisited[idx]
        route.append(nearest)
        cur = points[nearest]
        unvisited.pop(idx)
    return route


def _vcc_tour(
    points_km: np.ndarray,
    depot_km: np.ndarray,
    lw: float,
    vcc_iters: int = 30,
) -> Tuple[float, float]:
    """
    VCC-optimised tour for one dispatch batch.

    Returns
    -------
    tour_km   : total vehicle route length (km)
    avg_walk  : average customer walk distance (km)
    """
    n = len(points_km)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        dist_to_dest = float(np.linalg.norm(points_km[0] - depot_km))
        walk = min(lw, dist_to_dest)
        return 2.0 * max(0.0, dist_to_dest - walk), walk

    route = _greedy_nn_tour(depot_km, points_km)
    M = points_km[route].copy()

    for _ in range(vcc_iters):
        changed = False
        for i in range(n):
            prev = M[i - 1] if i > 0 else depot_km
            nxt  = M[i + 1] if i < n - 1 else depot_km
            dest = points_km[route[i]]
            M_new = _optimize_meeting_point(prev, nxt, dest, lw)
            if float(np.linalg.norm(M_new - M[i])) > 1e-9:
                M[i] = M_new
                changed = True
        if not changed:
            break

    stops = np.vstack([depot_km[None], M, depot_km[None]])
    tour_km = float(np.sum(np.linalg.norm(np.diff(stops, axis=0), axis=1)))
    walk_dists = [float(np.linalg.norm(M[i] - points_km[route[i]])) for i in range(n)]
    return tour_km, float(np.mean(walk_dists))


# ---------------------------------------------------------------------------
# Baseline class
# ---------------------------------------------------------------------------

class VCC(BaselineMethod):
    """
    VCC (Vehicle-Customer Coordination) baseline.

    K districts via random search (ε=0, partition scored at T_fixed),
    fixed dispatch interval T_fixed, in-district routing via MSO-VCC
    with max walking distance lw.

    Compared to SP-Lit (same partition framework, fixed T, no VCC):
      - shorter vehicle routes (meeting-point shortcutting)
      - non-zero user walk time (≤ lw / walk_speed_kmh hours)

    Reference: Zhang et al. (2023, Management Science).
    """

    def __init__(
        self,
        T_fixed: float = 0.5,
        lw: float = 0.3,
        max_iters: int = 200,
        vcc_iters: int = 30,
        walk_speed_kmh: float = 5.0,
    ):
        """
        Parameters
        ----------
        T_fixed        : fixed dispatch interval (hours)
        lw             : maximum walking distance (km); default 300 m
        max_iters      : random-search iterations for spatial partitioning
        vcc_iters      : VCC meeting-point optimisation iterations per district
        walk_speed_kmh : pedestrian speed (km/h)
        """
        self.T_fixed = T_fixed
        self.lw = lw
        self.max_iters = max_iters
        self.vcc_iters = vcc_iters
        self.walk_speed_kmh = walk_speed_kmh

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
        """K districts via random search scored at T_fixed (ε=0)."""
        from lib.algorithm import Partition

        partition = Partition(geodata, num_districts, prob_dict, epsilon=0.0)
        depot_id, roots, assignment, _, district_info = partition.random_search(
            max_iters=self.max_iters,
            prob_dict=prob_dict,
            Lambda=Lambda,
            wr=wr,
            wv=wv,
            Omega_dict=Omega_dict,
            J_function=J_function,
            T_override=self.T_fixed,
        )
        dispatch_intervals = {r: self.T_fixed for r in roots}
        return ServiceDesign(
            name="VCC",
            assignment=assignment,
            depot_id=depot_id,
            district_roots=list(roots),
            dispatch_intervals=dispatch_intervals,
        )

    def compute_route_costs(
        self,
        design: ServiceDesign,
        geodata,
        prob_dict: Dict[str, float],
        Lambda: float,
        wr: float,
    ) -> tuple:
        """Compute VCC-optimised tour lengths and walk times.

        Returns (tour_km_overrides, avg_walk_time_h).
        """
        block_ids = geodata.short_geoid_list
        N = len(block_ids)
        assignment = design.assignment
        depot_id = design.depot_id

        depot_pos = np.array(_get_pos(geodata, depot_id), dtype=float) / 1000.0

        tour_km_overrides: Dict[str, float] = {}
        acc_walk_km = 0.0
        acc_prob = 0.0

        for i, root in enumerate(block_ids):
            if round(assignment[i, i]) != 1:
                continue
            assigned_idx = [j for j in range(N) if round(assignment[j, i]) == 1]
            if not assigned_idx:
                continue
            assigned_blocks = [block_ids[j] for j in assigned_idx]

            pts_km = np.array(
                [np.array(_get_pos(geodata, b), dtype=float) / 1000.0
                 for b in assigned_blocks]
            )
            tour_km, avg_walk_km = _vcc_tour(pts_km, depot_pos, self.lw, self.vcc_iters)
            tour_km_overrides[root] = tour_km

            district_prob = sum(prob_dict.get(b, 0.0) for b in assigned_blocks)
            acc_walk_km += avg_walk_km * district_prob
            acc_prob += district_prob

        acc_prob = max(acc_prob, 1e-12)
        avg_walk_time_h = (acc_walk_km / acc_prob) / self.walk_speed_kmh
        return tour_km_overrides, avg_walk_time_h

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
        """Evaluate with VCC-optimised in-district tour lengths."""
        tour_km_overrides, avg_walk_time_h = self.compute_route_costs(
            design, geodata, prob_dict, Lambda, wr,
        )
        result = evaluate_design(
            design, geodata, prob_dict, Omega_dict, J_function,
            Lambda, wr, wv, beta, road_network,
            walk_time_override=avg_walk_time_h,
            tour_km_overrides=tour_km_overrides,
        )
        return EvaluationResult(
            name="VCC",
            **{k: v for k, v in result.__dict__.items() if k != 'name'},
        )


# Backward-compatible alias
ZhangEtAl2023 = VCC
