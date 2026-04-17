"""
Balanced partition baseline -- Carlsson et al. (2024)
"Provably Good Region Partitioning for On-Time Last-Mile Delivery"

Implements the ham-sandwich geometric partitioning algorithm from Section 4.
All partitioning operates in continuous space: the demand density f(x) is
approximated on a fine regular grid, measures mu_1 and mu_2 are computed as
numerical integrals, and cuts (lines / three-fans) partition continuous
space.  Blocks are assigned to districts only at the very end based on
which continuous region contains their centroid.

Measures (Theorem 6, Corollary 2 with s_bar = 0):
  mu_1(D) = integral_D sqrt(f(x)) dx           -- BHH / TSP spatial term
  mu_2(D) = integral_D 2||x - x0|| f(x) dx     -- radial travel workload

Fixed dispatch interval T_fixed for all districts (no schedule optimisation).
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Tuple

import numpy as np

from lib.baselines.base import (
    BETA, BaselineMethod, EvaluationResult, ServiceDesign,
    VEHICLE_SPEED_KMH, _effective_fleet_size, _get_pos, _nn_tsp_route,
    _newton_T_star, _bhh_alpha,
)
from lib.constants import DEFAULT_FLEET_COST_RATE


# ---------------------------------------------------------------------------
# Continuous-space grid -- numerical integration of f(x)
# ---------------------------------------------------------------------------

def _build_grid(
    geodata,
    prob_dict: Dict[str, float],
    depot_pos: np.ndarray,
    resolution: int = 100,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate the continuous demand density on a fine regular grid.

    The density f(x) at each grid cell is estimated via nearest-block
    interpolation:  f(x) = p_b / A_b  where b is the closest block.

    Returns
    -------
    grid_pos : (M, 2) cell centre positions
    g1       : (M,)   sqrt(f(x)) * dA           (mu_1 integrand x cell area)
    g2       : (M,)   2 ||x - x0|| f(x) * dA    (mu_2 integrand x cell area)
    """
    block_ids = geodata.short_geoid_list
    block_pos = np.array([_get_pos(geodata, b) for b in block_ids])

    span = block_pos.max(axis=0) - block_pos.min(axis=0)
    pad = float(span.max()) * 0.05 + 50.0
    lo = block_pos.min(axis=0) - pad
    hi = block_pos.max(axis=0) + pad

    xs = np.linspace(lo[0], hi[0], resolution)
    ys = np.linspace(lo[1], hi[1], resolution)
    dA = (xs[1] - xs[0]) * (ys[1] - ys[0]) if resolution > 1 else 1.0

    xx, yy = np.meshgrid(xs, ys)
    grid_pos = np.column_stack([xx.ravel(), yy.ravel()])  # (M, 2)

    # Nearest-block interpolation for density
    dists = np.linalg.norm(
        grid_pos[:, None, :] - block_pos[None, :, :], axis=2,
    )  # (M, N)
    nearest = np.argmin(dists, axis=1)  # (M,)

    areas = np.array([max(float(geodata.get_area(b)), 1e-12)
                       for b in block_ids])
    probs = np.array([prob_dict.get(b, 0.0) for b in block_ids])
    f_vals = probs[nearest] / areas[nearest]  # (M,)

    g1 = np.sqrt(np.maximum(f_vals, 0.0)) * dA
    g2 = (2.0 * np.linalg.norm(grid_pos - depot_pos, axis=1)
          * np.maximum(f_vals, 0.0) * dA)

    return grid_pos, g1, g2


# ---------------------------------------------------------------------------
# Step 1: vertical slivers equalising mu_1
# ---------------------------------------------------------------------------

def _vertical_sliver_cuts(
    grid_pos: np.ndarray, g1: np.ndarray, mask: np.ndarray, m: int,
) -> List[float]:
    """Return m-1 x-coordinates of vertical cuts giving equal-mu_1 slivers."""
    active = np.where(mask)[0]
    if len(active) == 0 or m <= 1:
        return []

    x_active = grid_pos[active, 0]
    x_unique = np.sort(np.unique(np.round(x_active, decimals=6)))
    total = g1[active].sum()
    if total < 1e-15:
        return []

    cuts: List[float] = []
    cum = 0.0
    k = 0
    for i, x in enumerate(x_unique):
        col = active[np.abs(x_active - x) < 1e-4]
        cum += g1[col].sum()
        if k < m - 1 and cum >= (k + 1) / m * total - 1e-12:
            if i + 1 < len(x_unique):
                cuts.append((x + x_unique[i + 1]) / 2.0)
            else:
                cuts.append(x + 1.0)
            k += 1
    return cuts


# ---------------------------------------------------------------------------
# Step 2: half-space signs
# ---------------------------------------------------------------------------

def _compute_signs(
    grid_pos: np.ndarray, g2: np.ndarray, mask: np.ndarray,
    cuts: List[float], m: int,
) -> List[int]:
    """s(i) = sign(mu_2(H_i) - (i/m) mu_2_total)  for i = 1 .. m-1."""
    active = np.where(mask)[0]
    total = g2[active].sum()
    if total < 1e-15:
        return [0] * (m - 1)

    x_active = grid_pos[active, 0]
    signs: List[int] = []
    for i, cx in enumerate(cuts):
        half = active[x_active <= cx]
        diff = g2[half].sum() / total - (i + 1) / m
        signs.append(0 if abs(diff) < 1e-10 else (1 if diff > 0 else -1))
    return signs


# ---------------------------------------------------------------------------
# Step 3: pair / triple decomposition (Bespamyatnikh et al. 2000, Thm 7)
# ---------------------------------------------------------------------------

def _find_decomposition(signs: List[int], m: int) -> Tuple[int, ...]:
    """Return pair (i*, j*) or triple (i*, j*, k*)."""
    for i in range(1, m):
        j = m - i
        if 1 <= j <= m - 1 and (signs[i - 1] == 0
                                or signs[i - 1] == signs[j - 1]):
            return (i, j)
    for i in range(1, m - 1):
        for j in range(1, m - i):
            k = m - i - j
            if 1 <= k <= m - 1 and signs[i-1] == signs[j-1] == signs[k-1]:
                return (i, j, k)
    q, r = divmod(m, 3)
    return tuple(q + (1 if i < r else 0) for i in range(3))


# ---------------------------------------------------------------------------
# Step 4a: ham-sandwich line (Section 4.1)
# ---------------------------------------------------------------------------

def _ham_line_split(
    grid_pos: np.ndarray, g1: np.ndarray, g2: np.ndarray,
    mask: np.ndarray, target_g1: float, target_g2: float,
    n_angles: int = 360,
) -> Tuple[np.ndarray, np.ndarray]:
    """Line  x cos(theta) + y sin(theta) = r  ->  (R_mask, L_mask).

    Grid-search over theta in [0, pi); for each angle, sweep r through
    cell projections and find the split that best matches the targets.
    """
    active = np.where(mask)[0]
    pos = grid_pos[active]
    g1a, g2a = g1[active], g2[active]
    norm1, norm2 = max(g1a.sum(), 1e-12), max(g2a.sum(), 1e-12)

    best_err = float('inf')
    best_theta = 0.0
    best_r = 0.0

    for ai in range(n_angles):
        theta = ai * math.pi / n_angles
        ct, st = math.cos(theta), math.sin(theta)
        proj = pos[:, 0] * ct + pos[:, 1] * st
        order = np.argsort(-proj)  # descending

        cum1 = np.cumsum(g1a[order])
        cum2 = np.cumsum(g2a[order])
        err = (np.abs(cum1[:-1] - target_g1) / norm1
               + np.abs(cum2[:-1] - target_g2) / norm2)
        k = int(np.argmin(err))

        if err[k] < best_err:
            best_err = float(err[k])
            best_theta = theta
            best_r = float(proj[order[k]] + proj[order[k + 1]]) / 2.0

    # Apply the line to all grid cells within the mask
    ct, st = math.cos(best_theta), math.sin(best_theta)
    proj_all = grid_pos[:, 0] * ct + grid_pos[:, 1] * st
    R = mask & (proj_all >= best_r)
    L = mask & (proj_all < best_r)

    if not R.any() or not L.any():
        med = float(np.median(grid_pos[active, 0]))
        R = mask & (grid_pos[:, 0] >= med)
        L = mask & (grid_pos[:, 0] < med)
    return R, L


# ---------------------------------------------------------------------------
# Step 4b: three-fan (Section 4.2, Figure 3)
# ---------------------------------------------------------------------------

def _three_fan_split(
    grid_pos: np.ndarray, g1: np.ndarray, g2: np.ndarray,
    mask: np.ndarray,
    target_R: Tuple[float, float],
    target_L: Tuple[float, float],
    target_U: Tuple[float, float],
    n_centers: int = 60,
    n_bins: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three-fan cut -> (R_mask, L_mask, U_mask).

    Fan geometry (Figure 3): centre (cx, cy), one ray points straight down,
    right ray at clockwise angle alpha, left ray at angle 2pi - beta.
      R = [0, alpha),  U = [alpha, 2pi-beta),  L = [2pi-beta, 2pi).

    Search over candidate centres and angle bins.
    """
    active = np.where(mask)[0]
    pos = grid_pos[active]
    g1a, g2a = g1[active], g2[active]
    n_act = len(active)
    norm1 = max(g1a.sum(), 1e-12)
    norm2 = max(g2a.sum(), 1e-12)
    tR1, tR2 = target_R
    tL1, tL2 = target_L
    tU1, tU2 = target_U

    step = max(1, n_act // n_centers)
    centers = list(range(0, n_act, step))
    bw = 2.0 * math.pi / n_bins
    half = n_bins // 2

    best_err = float('inf')
    best_cxy = pos[0].copy()
    best_alpha = math.pi / 2
    best_beta_bnd = 3.0 * math.pi / 2  # = 2pi - beta

    for ci in centers:
        cx, cy = pos[ci]
        dx = pos[:, 0] - cx
        dy = pos[:, 1] - cy
        ang = np.arctan2(dx, -dy)  # clockwise from down
        ang = np.where(ang < 0, ang + 2 * np.pi, ang)

        bins = np.clip((ang / bw).astype(int), 0, n_bins - 1)
        bg1 = np.zeros(n_bins)
        bg2 = np.zeros(n_bins)
        np.add.at(bg1, bins, g1a)
        np.add.at(bg2, bins, g2a)

        cum1 = np.concatenate([[0.0], np.cumsum(bg1)])
        cum2 = np.concatenate([[0.0], np.cumsum(bg2)])

        # a bins in R, b = start bin of L
        # alpha = a * bw in (0, pi)   -> a in [1, half]
        # beta  = (n_bins-b) * bw in (0, pi) -> b in [half+1, n_bins-1]
        # alpha + beta >= pi  -> b <= a + half
        for a in range(1, half + 1):
            r1, r2 = cum1[a], cum2[a]
            b_lo = max(a + 1, half + 1)
            b_hi = min(n_bins, a + half)
            for b in range(b_lo, b_hi + 1):
                l1 = cum1[n_bins] - cum1[b]
                l2 = cum2[n_bins] - cum2[b]
                u1 = cum1[b] - cum1[a]
                u2 = cum2[b] - cum2[a]
                err = (abs(r1 - tR1) / norm1 + abs(r2 - tR2) / norm2
                       + abs(l1 - tL1) / norm1 + abs(l2 - tL2) / norm2
                       + abs(u1 - tU1) / norm1 + abs(u2 - tU2) / norm2)
                if err < best_err:
                    best_err = err
                    best_cxy = np.array([cx, cy])
                    best_alpha = a * bw
                    best_beta_bnd = b * bw

    # Apply the fan to all grid cells within the mask
    cx, cy = best_cxy
    ang_all = np.arctan2(grid_pos[:, 0] - cx, -(grid_pos[:, 1] - cy))
    ang_all = np.where(ang_all < 0, ang_all + 2 * np.pi, ang_all)

    Rm = mask & (ang_all < best_alpha)
    Lm = mask & (ang_all >= best_beta_bnd)
    Um = mask & (ang_all >= best_alpha) & (ang_all < best_beta_bnd)

    if not Rm.any() or not Lm.any() or not Um.any():
        sx = active[np.argsort(grid_pos[active, 0])]
        n1 = max(1, n_act // 3)
        n2 = max(1, n_act // 3)
        Rm = np.zeros(len(mask), dtype=bool); Rm[sx[:n1]] = True
        Lm = np.zeros(len(mask), dtype=bool); Lm[sx[n1:n1+n2]] = True
        Um = np.zeros(len(mask), dtype=bool); Um[sx[n1+n2:]] = True
    return Rm, Lm, Um


# ---------------------------------------------------------------------------
# Step 5: recursive partition
# ---------------------------------------------------------------------------

def _recursive_partition(
    grid_pos: np.ndarray, g1: np.ndarray, g2: np.ndarray,
    mask: np.ndarray, m: int, n_angles: int = 360,
) -> List[np.ndarray]:
    """Recursively partition the masked region into m districts."""
    if m <= 0:
        return []
    if m == 1 or mask.sum() <= 1:
        return [mask.copy()]

    total1 = g1[mask].sum()
    total2 = g2[mask].sum()

    cuts = _vertical_sliver_cuts(grid_pos, g1, mask, m)
    signs = _compute_signs(grid_pos, g2, mask, cuts, m)
    decomp = _find_decomposition(signs, m)

    if len(decomp) == 2:
        i_s, j_s = decomp
        R, L = _ham_line_split(
            grid_pos, g1, g2, mask,
            i_s / m * total1, i_s / m * total2, n_angles,
        )
        return (_recursive_partition(grid_pos, g1, g2, R, i_s, n_angles)
                + _recursive_partition(grid_pos, g1, g2, L, j_s, n_angles))

    i_s, j_s, k_s = decomp
    R, L, U = _three_fan_split(
        grid_pos, g1, g2, mask,
        (i_s / m * total1, i_s / m * total2),
        (j_s / m * total1, j_s / m * total2),
        (k_s / m * total1, k_s / m * total2),
    )
    return (_recursive_partition(grid_pos, g1, g2, R, i_s, n_angles)
            + _recursive_partition(grid_pos, g1, g2, L, j_s, n_angles)
            + _recursive_partition(grid_pos, g1, g2, U, k_s, n_angles))


# ---------------------------------------------------------------------------
# Continuous-space last-mile simulation for SP-Lit
# ---------------------------------------------------------------------------

def _simulate_carlsson_design(
    design: ServiceDesign,
    geodata,
    prob_dict: Dict[str, float],
    Lambda: float,
    wr: float,
    resolution: int,
    fleet_cost_rate: float = DEFAULT_FLEET_COST_RATE,
    n_scenarios: int = 500,
) -> EvaluationResult:
    """Passenger-level simulation for depot-origin last-mile service.

    In each district and dispatch cycle:
      1. Sample passenger arrivals to the depot via a Poisson process.
      2. Sample each passenger's destination from the continuous density.
      3. Batch all passengers to the next dispatch.
      4. Route a district vehicle from the depot to destinations via NN.
      5. Measure realized wait and in-vehicle time per passenger.
    """
    block_ids = geodata.short_geoid_list
    block_pos_m = np.array([_get_pos(geodata, b) for b in block_ids], dtype=float)
    block_pos_km = block_pos_m / 1000.0
    depot_idx = block_ids.index(design.depot_id)
    depot_pos_km = block_pos_km[depot_idx]

    assignment = design.assignment
    if assignment is None:
        raise ValueError("Carlsson simulation requires a partition assignment.")

    roots = [i for i in range(len(block_ids)) if round(assignment[i, i]) == 1]
    root_to_district = {root_idx: d for d, root_idx in enumerate(roots)}
    block_district = np.full(len(block_ids), -1, dtype=int)
    for root_idx, d in root_to_district.items():
        for j in range(len(block_ids)):
            if round(assignment[j, root_idx]) == 1:
                block_district[j] = d

    depot_pos_m = block_pos_m[depot_idx]
    grid_pos_m, _, _ = _build_grid(geodata, prob_dict, depot_pos_m, resolution)
    grid_pos_km = grid_pos_m / 1000.0

    span_km = block_pos_km.max(axis=0) - block_pos_km.min(axis=0)
    pad_km = float(span_km.max()) * 0.05 + 0.05
    lo_km = block_pos_km.min(axis=0) - pad_km
    hi_km = block_pos_km.max(axis=0) + pad_km
    xs = np.linspace(lo_km[0], hi_km[0], resolution)
    ys = np.linspace(lo_km[1], hi_km[1], resolution)
    dA = float((xs[1] - xs[0]) * (ys[1] - ys[0])) if resolution > 1 else 1.0
    cell_dx = float(xs[1] - xs[0]) if len(xs) > 1 else 0.0
    cell_dy = float(ys[1] - ys[0]) if len(ys) > 1 else 0.0

    dists_gb = np.linalg.norm(
        grid_pos_m[:, None, :] - block_pos_m[None, :, :], axis=2,
    )
    nearest_block = np.argmin(dists_gb, axis=1)
    areas_km2 = np.array([max(float(geodata.get_area(b)), 1e-12) for b in block_ids])
    probs = np.array([prob_dict.get(b, 0.0) for b in block_ids])
    f_dA = np.maximum(probs[nearest_block] / areas_km2[nearest_block], 0.0) * dA

    grid_label = design.service_metadata.get("grid_label")
    if grid_label is None or len(grid_label) != len(grid_pos_km):
        grid_label = block_district[nearest_block]
    else:
        grid_label = np.asarray(grid_label, dtype=int)

    district_cells = []
    T_dist = []
    root_ids = []
    for d, root_idx in enumerate(roots):
        root_id = block_ids[root_idx]
        root_ids.append(root_id)
        T_i = float(design.dispatch_intervals.get(root_id, 1.0))
        T_dist.append(T_i)

        cell_idx = np.where(grid_label == d)[0]
        mass = f_dA[cell_idx]
        total_mass = float(mass.sum())
        if len(cell_idx) > 0 and total_mass > 1e-15:
            cell_prob = mass / total_mass
        else:
            cell_prob = np.ones(max(len(cell_idx), 1), dtype=float) / max(len(cell_idx), 1)
        district_cells.append((cell_idx, grid_pos_km[cell_idx], cell_prob, total_mass))

    rng = np.random.default_rng(42)

    total_wait_h = 0.0
    total_invehicle_h = 0.0
    total_passengers = 0
    route_km_sum = np.zeros(len(roots), dtype=float)

    for s in range(n_scenarios):
        for d, root_id in enumerate(root_ids):
            T_i = T_dist[d]
            cell_idx, cell_pos_km, cell_prob, district_mass = district_cells[d]
            if len(cell_idx) == 0 or district_mass <= 1e-15:
                continue

            n_d = int(rng.poisson(Lambda * T_i * district_mass))
            if n_d == 0:
                continue

            arrivals = rng.uniform(0.0, T_i, size=n_d)
            waits = T_i - arrivals

            sampled = rng.choice(len(cell_idx), size=n_d, p=cell_prob)
            dest_points = cell_pos_km[sampled].copy()
            dest_points[:, 0] += rng.uniform(-0.5 * cell_dx, 0.5 * cell_dx, size=n_d)
            dest_points[:, 1] += rng.uniform(-0.5 * cell_dy, 0.5 * cell_dy, size=n_d)

            _, cumulative_km, route_km = _nn_tsp_route(depot_pos_km, dest_points)
            invehicle_h = cumulative_km / VEHICLE_SPEED_KMH

            total_wait_h += float(waits.sum())
            total_invehicle_h += float(invehicle_h.sum())
            total_passengers += n_d
            route_km_sum[d] += route_km

    avg_route_km = route_km_sum / max(n_scenarios, 1)
    provider_travel_rate = float(sum(
        avg_route_km[d] / max(T_dist[d], 1e-9) for d in range(len(T_dist))
    ))
    fleet_size = float(sum(
        avg_route_km[d] / (VEHICLE_SPEED_KMH * max(T_dist[d], 1e-9))
        for d in range(len(T_dist))
    ))
    raw_fleet_size = int(sum(
        _effective_fleet_size(float(avg_route_km[d]), float(T_dist[d]))
        for d in range(len(T_dist))
    ))

    if total_passengers > 0:
        avg_wait_h = total_wait_h / total_passengers
        avg_invehicle_h = total_invehicle_h / total_passengers
    else:
        avg_wait_h = 0.0
        avg_invehicle_h = 0.0

    total_user_time = avg_wait_h + avg_invehicle_h
    avg_dispatch_interval = float(np.mean(T_dist)) if T_dist else 0.0
    user_cost = wr * total_user_time

    return EvaluationResult(
        name=design.name,
        avg_wait_time=avg_wait_h,
        avg_invehicle_time=avg_invehicle_h,
        avg_walk_time=0.0,
        total_user_time=total_user_time,
        linehaul_cost=0.0,
        in_district_cost=provider_travel_rate,
        total_travel_cost=provider_travel_rate,
        fleet_size=fleet_size,
        raw_fleet_size=raw_fleet_size,
        avg_dispatch_interval=avg_dispatch_interval,
        odd_cost=0.0,
        provider_cost=provider_travel_rate + fleet_cost_rate * raw_fleet_size,
        user_cost=user_cost,
        total_cost=provider_travel_rate + fleet_cost_rate * raw_fleet_size + user_cost,
    )


# ---------------------------------------------------------------------------
# Baseline class
# ---------------------------------------------------------------------------

class CarlssonPartition(BaselineMethod):
    """Carlsson et al. (2024) balanced partition via ham-sandwich cuts.

    Parameters
    ----------
    T_fixed    : fixed dispatch interval for all districts (hours).
                 If None, compute T* per district via Newton's method
                 using the same _newton_T_star formula as Joint baselines,
                 making the design regime-responsive to (Lambda, wr, wv).
    n_angles   : angular resolution for ham-sandwich line search
    resolution : side length of the fine integration grid  (resolution^2 cells)
    """

    def __init__(
        self,
        T_fixed: float | None = 0.5,
        n_angles: int = 360,
        resolution: int = 100,
        name: str | None = None,
    ):
        self.T_fixed = T_fixed
        self.n_angles = n_angles
        self.resolution = resolution
        self._name = name or "Carlsson"

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
        block_ids = geodata.short_geoid_list
        N = len(block_ids)
        K = num_districts
        block_pos = np.array([_get_pos(geodata, b) for b in block_ids])

        # Depot = centroid block
        centroid = block_pos.mean(axis=0)
        depot_idx = int(np.argmin(np.linalg.norm(block_pos - centroid, axis=1)))
        depot_id = block_ids[depot_idx]

        # Build fine grid for continuous integration
        grid_pos, g1, g2 = _build_grid(
            geodata, prob_dict, block_pos[depot_idx], self.resolution,
        )
        M = len(grid_pos)

        # Recursive partition -> list of boolean masks over the grid
        mask0 = np.ones(M, dtype=bool)
        district_masks = _recursive_partition(
            grid_pos, g1, g2, mask0, K, self.n_angles,
        )
        district_masks = [dm for dm in district_masks if dm.any()]

        # Label each grid cell with its district index
        grid_label = np.full(M, -1, dtype=int)
        for d, dm in enumerate(district_masks):
            grid_label[dm] = d

        # Assign blocks to districts via nearest grid cell
        dists_bg = np.linalg.norm(
            block_pos[:, None, :] - grid_pos[None, :, :], axis=2,
        )  # (N, M)
        nearest_grid = np.argmin(dists_bg, axis=1)
        block_label = grid_label[nearest_grid]

        # Fallback for any unassigned block
        unassigned = np.where(block_label < 0)[0]
        if len(unassigned):
            assigned_cells = np.where(grid_label >= 0)[0]
            for j in unassigned:
                d = np.linalg.norm(
                    grid_pos[assigned_cells] - block_pos[j], axis=1,
                )
                block_label[j] = grid_label[assigned_cells[int(np.argmin(d))]]

        # Build N x N assignment matrix
        assignment = np.zeros((N, N))
        roots: List[str] = []

        for d in range(len(district_masks)):
            group = list(np.where(block_label == d)[0])
            if not group:
                continue
            gc = block_pos[group].mean(axis=0)
            root_idx = group[int(np.argmin(
                np.linalg.norm(block_pos[group] - gc, axis=1),
            ))]
            roots.append(block_ids[root_idx])
            for j in group:
                assignment[j, root_idx] = 1.0

        # Compute dispatch intervals: fixed or per-district optimal T*
        if self.T_fixed is not None:
            dispatch_intervals = {r: self.T_fixed for r in roots}
        else:
            # Per-district T* via Newton's method, same formula as Joint
            dispatch_intervals = {}
            for r in roots:
                r_idx = block_ids.index(r)
                assigned_blocks = [block_ids[j] for j in range(N)
                                   if round(assignment[j, r_idx]) == 1]
                K_i = min(
                    float(np.linalg.norm(block_pos[depot_idx] - block_pos[block_ids.index(b)]))
                    / 1000.0  # metres → km
                    for b in assigned_blocks
                )
                alpha_i = _bhh_alpha(geodata, assigned_blocks, prob_dict)
                F_i = 0.0
                if Omega_dict and J_function:
                    ref = next(iter(Omega_dict.values()))
                    d_omega = np.zeros_like(ref)
                    for b in assigned_blocks:
                        d_omega = np.maximum(d_omega, Omega_dict.get(b, np.zeros_like(ref)))
                    F_i = J_function(d_omega)
                dispatch_intervals[r] = _newton_T_star(K_i, F_i, alpha_i, wr, wv)

        return ServiceDesign(
            name=self._name,
            assignment=assignment,
            depot_id=depot_id,
            district_roots=roots,
            dispatch_intervals=dispatch_intervals,
            service_metadata={
                "grid_label": grid_label.copy(),
                "resolution": self.resolution,
            },
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
    ) -> EvaluationResult:
        return _simulate_carlsson_design(
            design=design,
            geodata=geodata,
            prob_dict=prob_dict,
            Lambda=Lambda,
            wr=wr,
            fleet_cost_rate=fleet_cost_rate,
            resolution=int(design.service_metadata.get("resolution", self.resolution)),
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
        result = self.custom_simulate(
            design, geodata, prob_dict, Omega_dict, J_function,
            Lambda, wr, wv, fleet_cost_rate,
        )
        return EvaluationResult(
            name=self._name,
            **{k: v for k, v in result.__dict__.items() if k != 'name'},
        )
