"""Visualise service designs for all baselines with mode-appropriate rendering.

Each service mode gets a visualization matching its operational model:
  - Fixed-route (Multi-FR, Multi-FR-Detour):
        closed depot→checkpoint→depot loops + shaded service corridors
  - Carlsson partition (SP-Lit):
        continuous-space partition + TSP tours
  - Partition-based (Joint-Nom, Joint-DRO):
        colored districts + depot-originating TSP tours
  - Temporal-only (TP-Lit):
        one dispatch wave with sampled demand + serving tour
  - VCC:  sampled OD requests + coordinated non-partition fleet
  - On-demand (OD):  sample direct-service trips from depot
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.colors import ListedColormap
from scipy.spatial import cKDTree
import networkx as nx

# ── Project imports ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from scripts.experiments.baseline_comparison import (
    make_toy_geodata, build_prob_dict, build_omega_dict, identity_J,
)
from lib.baselines import (
    FullyOD, TemporalOnly,
    JointNom, JointDRO, VCC, FRDetour, CarlssonPartition,
)
from scripts.plots.fr_detour_comparison import (
    build_synthetic_instance,
    find_first_served_stop_scenario,
    make_uniform_positions,
    plot_fixed_route_panel,
    solve_design_scenario,
)

# ── Settings ──────────────────────────────────────────────────────────────
GRID_SIZE = 5
K = 3
LAMBDA = 500.0
WR, WV = 1.0, 10.0
T_FIXED = 0.5
RS_ITERS = 100
EPS = 0.001

# ── Build data (grid — for non-FR baselines) ─────────────────────────────
geodata = make_toy_geodata(GRID_SIZE)
prob_dict = build_prob_dict(geodata)
Omega_dict = build_omega_dict(geodata)
J_function = identity_J

block_ids = geodata.short_geoid_list
N = len(block_ids)
block_pos_km = np.array([
    np.array(geodata.pos[b], dtype=float) / 1000.0 for b in block_ids
])

# ── Uniform synthetic geodata for FR / FR-Detour ─────────────────────────
_fr_positions_km = make_uniform_positions()
fr_geodata, fr_block_ids, fr_block_pos_km, fr_prob_dict, fr_Omega_dict = (
    build_synthetic_instance(_fr_positions_km)
)

DELTA_DETOUR = 0.8
FR_K = 2
FR_LAMBDA = 8.0
FR_WR = 20.0


# ── Run all baselines ────────────────────────────────────────────────────
# FR: run FR-Detour once, then overlay the same selected design under δ=0 and δ>0
print(f"  Designing FR-Detour (uniform, K={FR_K}, δ={DELTA_DETOUR}) ...",
      flush=True)
_fr_shared_builder = FRDetour(delta=DELTA_DETOUR, max_iters=RS_ITERS)
_fr_shared_inputs = _fr_shared_builder._prepare_shared_inputs(fr_geodata, FR_K)
_fr_design = FRDetour(delta=DELTA_DETOUR, max_iters=RS_ITERS).design(
    fr_geodata, fr_prob_dict, fr_Omega_dict, identity_J, FR_K,
    Lambda=FR_LAMBDA, wr=FR_WR, wv=WV, **_fr_shared_inputs,
)
_fr_method = FRDetour(delta=0.0, max_iters=RS_ITERS, name="Multi-FR")
_fr_det_method = FRDetour(delta=DELTA_DETOUR, max_iters=RS_ITERS, name="Multi-FR-Detour")
FR_SCENARIO_IDX, _fr_detour_services = find_first_served_stop_scenario(
    _fr_design,
    _fr_det_method,
    fr_geodata,
    fr_block_ids,
    fr_block_pos_km,
    fr_prob_dict,
    _fr_shared_inputs["crn_uniforms"],
    FR_LAMBDA,
    FR_WR,
)

# Both panels share the same selected design; only the operational delta changes.
designs = {
    "Multi-FR":        _fr_design,
    "Multi-FR-Detour": _fr_design,
}

# Other baselines use regular grid
grid_baselines = [
    ("SP-Lit",          CarlssonPartition(T_fixed=T_FIXED)),
    ("TP-Lit",          TemporalOnly(n_candidates=25, region_low=-0.5, region_km=4.5)),
    ("Joint-Nom",       JointNom(max_iters=RS_ITERS)),
    ("Joint-DRO",       JointDRO(epsilon=EPS, max_iters=RS_ITERS)),
    ("VCC",             VCC()),
    ("OD",              FullyOD()),
]

methods = {}
for name, method in grid_baselines:
    print(f"  Designing {name} ...", flush=True)
    design = method.design(
        geodata, prob_dict, Omega_dict, J_function, K,
        Lambda=LAMBDA, wr=WR, wv=WV,
    )
    designs[name] = design
    methods[name] = method


# ── Colors and constants ─────────────────────────────────────────────────
PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
NEUTRAL = "#b8d4e8"
CELL = 0.9  # km (grid spacing = 1 km)


# ── Mode detection ───────────────────────────────────────────────────────
def _mode_type(name: str) -> str:
    n = name.lower()
    if "fr" in n:
        return "fr"
    if "sp-lit" in n or "carlsson" in n:
        return "carlsson"
    if "tp" in n or "temporal" in n:
        return "tp"
    if n == "od" or "on-demand" in n:
        return "od"
    if "vcc" in n:
        return "vcc"
    return "partition"


# ── Shared helpers ───────────────────────────────────────────────────────
def _get_roots_and_districts(design):
    """Return (roots_global_idx, block_district_array) from assignment matrix."""
    if design.assignment is None:
        raise ValueError(f"{design.name} has no district assignment.")
    assignment = design.assignment
    n_local = assignment.shape[0]
    roots = [i for i in range(n_local) if round(assignment[i, i]) == 1]
    block_district = np.full(n_local, -1, dtype=int)
    for d, ri in enumerate(roots):
        for j in range(n_local):
            if round(assignment[j, ri]) == 1:
                block_district[j] = d
    return roots, block_district


def _draw_block(ax, j, color, alpha=0.4):
    x, y = block_pos_km[j]
    rect = plt.Rectangle(
        (x - CELL / 2, y - CELL / 2), CELL, CELL,
        facecolor=color, edgecolor="white", linewidth=1.5, alpha=alpha,
    )
    ax.add_patch(rect)
    ax.text(x, y, block_ids[j][-3:], ha="center", va="center",
            fontsize=5.5, color="black", alpha=0.7)


def _draw_depot(ax, depot_pos):
    ax.plot(*depot_pos, "*", color="red", markersize=14,
            markeredgecolor="black", markeredgewidth=0.8, zorder=6)


def _partition_plot_context(design):
    meta = design.service_metadata or {}
    local_block_ids = meta.get("plot_block_ids", block_ids)
    local_block_pos = meta.get("plot_block_pos_km", block_pos_km)
    local_cell = float(meta.get("plot_cell_size_km", CELL))
    local_grid = int(meta.get("plot_grid_size", GRID_SIZE))
    return local_block_ids, np.asarray(local_block_pos), local_cell, local_grid


def _nn_tour_order(positions, depot_pos):
    """Greedy NN starting from nearest-to-depot; returns index permutation."""
    n = len(positions)
    if n == 0:
        return []
    dists = np.linalg.norm(positions - depot_pos, axis=1)
    start = int(np.argmin(dists))
    visited = [start]
    remaining = set(range(n)) - {start}
    while remaining:
        cur = positions[visited[-1]]
        d = {r: float(np.linalg.norm(positions[r] - cur)) for r in remaining}
        nxt = min(d, key=d.get)
        visited.append(nxt)
        remaining.discard(nxt)
    return visited


def _two_opt(positions, depot_pos):
    """Return 2-opt improved tour order (indices into positions array)."""
    n = len(positions)
    if n <= 2:
        return _nn_tour_order(positions, depot_pos)

    order = _nn_tour_order(positions, depot_pos)

    def _dist(o):
        total = np.linalg.norm(positions[o[0]] - depot_pos)
        for i in range(len(o) - 1):
            total += np.linalg.norm(positions[o[i + 1]] - positions[o[i]])
        total += np.linalg.norm(depot_pos - positions[o[-1]])
        return total

    improved = True
    while improved:
        improved = False
        best = _dist(order)
        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = order[:i] + order[i:j + 1][::-1] + order[j + 1:]
                d = _dist(cand)
                if d < best - 1e-10:
                    order = cand
                    best = d
                    improved = True
    return order


def _t_str(design):
    T_info = design.dispatch_intervals
    if not T_info:
        return ""
    T_vals = list(T_info.values())
    if len(set(f"{t:.2f}" for t in T_vals)) == 1:
        return f"T={T_vals[0]:.2f}h"
    return f"T={min(T_vals):.2f}\u2013{max(T_vals):.2f}h"


def _setup_axes(ax, title, subtitle=""):
    full = f"{title}\n{subtitle}" if subtitle else title
    ax.set_title(full, fontsize=9)
    ax.set_xlim(-0.6, GRID_SIZE - 0.4)
    ax.set_ylim(-0.6, GRID_SIZE - 0.4)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)


def _draw_continuous_partition(ax, centroids, n_districts):
    """Draw continuous partition via Voronoi of district centroids.

    Parameters
    ----------
    centroids : (K, 2) array of district centroid positions in km.
    n_districts : int

    Uses K centroid positions as Voronoi generators so that boundaries
    are smooth straight lines (approximating ham-sandwich cuts).
    """
    res = 300
    xs = np.linspace(-0.5, GRID_SIZE - 0.5, res)
    ys = np.linspace(-0.5, GRID_SIZE - 0.5, res)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel()])

    tree = cKDTree(centroids)
    _, nearest = tree.query(pts)
    districts = nearest.reshape(XX.shape)

    colors = [PALETTE[d % len(PALETTE)] for d in range(n_districts)]
    cmap = ListedColormap(colors)

    ax.pcolormesh(XX, YY, districts, cmap=cmap, alpha=0.25, shading="auto",
                  zorder=0, vmin=-0.5, vmax=n_districts - 0.5)


# ── Fixed-route: Multi-FR, Multi-FR-Detour ──────────────────────────────
def _pt_seg_dist(p, a, b):
    """Distance from point p to segment a–b."""
    ab = b - a
    ab2 = np.dot(ab, ab)
    if ab2 < 1e-12:
        return float(np.linalg.norm(p - a))
    t = np.clip(np.dot(p - a, ab) / ab2, 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def _in_corridor(j, route_global, bpos, delta):
    """True if block j is within delta of any route segment (CP-to-CP)."""
    pt = bpos[j]
    for k in range(len(route_global) - 1):
        if _pt_seg_dist(pt, bpos[route_global[k]], bpos[route_global[k + 1]]) <= delta + 1e-9:
            return True
    return False


def _plot_fixed_route(ax, design, title):
    """Scattered-demand FR with scenario-based served-stop overlay."""
    delta = DELTA_DETOUR if "detour" in title.lower() else 0.0
    method = _fr_det_method if delta > 0 else _fr_method
    services = (
        _fr_detour_services if delta > 0 else
        solve_design_scenario(
            design,
            method,
            fr_geodata,
            fr_block_ids,
            fr_block_pos_km,
            fr_prob_dict,
            _fr_shared_inputs["crn_uniforms"],
            FR_LAMBDA,
            FR_WR,
            FR_SCENARIO_IDX,
        )
    )
    plot_fixed_route_panel(
        ax,
        design,
        None,
        f"{title}\nfixed selected design; scenario {FR_SCENARIO_IDX}",
        delta,
        fr_geodata,
        fr_block_ids,
        fr_block_pos_km,
        services=services,
        skip_info=None,
    )
    ax.tick_params(labelsize=6)


# ── Carlsson: continuous partition + TSP tours ───────────────────────────
def _plot_carlsson(ax, design, title):
    """SP-Lit: continuous-space partition background + 2-opt TSP tours.

    Carlsson's partition divides continuous 2D space via ham-sandwich cuts.
    Demand arises anywhere in the region (continuous density), not at
    block centroids. Show only the partition regions and illustrative
    TSP tours through scattered demand samples.
    """
    roots, bd = _get_roots_and_districts(design)
    n_d = len(roots)
    depot_idx = block_ids.index(design.depot_id)
    depot_pos = block_pos_km[depot_idx]

    # Compute district centroids from block positions
    centroids = np.zeros((n_d, 2))
    for d in range(n_d):
        members = [j for j in range(N) if bd[j] == d]
        if members:
            centroids[d] = block_pos_km[members].mean(axis=0)

    # Continuous partition background (Voronoi of centroids)
    _draw_continuous_partition(ax, centroids, n_d)

    # Sample scattered demand points within each region (continuous demand)
    rng = np.random.RandomState(42)
    n_samples_per_region = 8
    region_samples = {}  # d → (n_samples, 2) array
    for d in range(n_d):
        members = [j for j in range(N) if bd[j] == d]
        if not members:
            continue
        # Sample random positions within the convex hull of region members
        mpos = block_pos_km[members]
        x_lo, y_lo = mpos.min(axis=0) - 0.3
        x_hi, y_hi = mpos.max(axis=0) + 0.3
        # Clip to service area
        x_lo, y_lo = max(x_lo, -0.4), max(y_lo, -0.4)
        x_hi, y_hi = min(x_hi, GRID_SIZE - 0.6), min(y_hi, GRID_SIZE - 0.6)
        pts_d = []
        while len(pts_d) < n_samples_per_region:
            px = rng.uniform(x_lo, x_hi)
            py = rng.uniform(y_lo, y_hi)
            # Accept if nearest centroid is this district
            dists = np.linalg.norm(centroids - [px, py], axis=1)
            if np.argmin(dists) == d:
                pts_d.append([px, py])
        region_samples[d] = np.array(pts_d)
        color = PALETTE[d % len(PALETTE)]
        ax.scatter(region_samples[d][:, 0], region_samples[d][:, 1],
                   s=12, color=color, edgecolors="white", linewidths=0.3,
                   alpha=0.7, zorder=3)

    # Per-district 2-opt TSP tour through sampled demand
    for d in range(n_d):
        if d not in region_samples:
            continue
        color = PALETTE[d % len(PALETTE)]
        pos = region_samples[d]
        order = _two_opt(pos, depot_pos)
        pts = np.vstack([[depot_pos], pos[order], [depot_pos]])
        ax.plot(pts[:, 0], pts[:, 1], "-", color=color, linewidth=1.5,
                alpha=0.7, zorder=2)
        if len(pts) >= 2:
            ax.annotate("", xy=pts[1], xytext=pts[0],
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=1.5, mutation_scale=10), zorder=3)

    _draw_depot(ax, depot_pos)
    _setup_axes(ax, title, f"{n_d} regions, {_t_str(design)}")


# ── Temporal-only: TP-Lit ────────────────────────────────────────────────
def _plot_temporal(ax, design, title):
    """TP-Lit: random candidate locations + dispatch waves with demand + VRP routes.

    Following the paper, the set of potential customer locations
    I = {1, ..., I} are uniformly distributed random points in a square
    (hyperparameter). Each dispatch wave, a random subset receives orders
    with random quantities.  Show:
      - all candidate locations (sized by demand probability)
      - per-wave active subset with order quantities
      - per-wave VRP routes from the depot
    """
    # Use the TP-Lit method's own candidate locations (random point cloud)
    tp_method = methods.get("TP-Lit")
    if tp_method is not None and tp_method._candidate_pos_km is not None:
        cand_pos = tp_method._candidate_pos_km
        cand_probs = tp_method._candidate_probs
        depot_pos = tp_method._depot_km
    else:
        # Fallback to grid positions
        depot_idx = block_ids.index(design.depot_id)
        depot_pos = block_pos_km[depot_idx]
        cand_pos = block_pos_km
        cand_probs = np.array([prob_dict.get(b, 0.0) for b in block_ids])
        cand_probs = cand_probs / (cand_probs.sum() + 1e-15)

    n_cand = len(cand_pos)
    meta = design.service_metadata or {}
    T_val = float(meta.get("T_fixed",
                           list(design.dispatch_intervals.values())[0]
                           if design.dispatch_intervals else 0.25))
    K_dispatch = int(meta.get("dispatch_K", 3))

    # --- Candidate-location probabilities ---
    probs_norm = cand_probs / (cand_probs.max() + 1e-15)

    # All candidate locations: light circles sized by probability
    for j in range(n_cand):
        sz = 5 + probs_norm[j] * 14
        ax.plot(*cand_pos[j], "o", color="#d9d9d9",
                markersize=sz, markeredgecolor="#aaaaaa",
                markeredgewidth=0.5, alpha=0.50, zorder=1)
        ax.text(cand_pos[j, 0], cand_pos[j, 1] - 0.18,
                f"{j}", ha="center", va="top",
                fontsize=4.5, color="#888888", zorder=1)

    # --- Dispatch waves ---
    rng = np.random.default_rng(42)
    n_waves = 3
    wave_colors = ["#1f77b4", "#2ca02c", "#9467bd"]
    wave_markers = ["o", "s", "^"]

    # Moderate demand rate for visualization: ~10 active locations per wave
    demand_rate = 10.0

    for w in range(n_waves):
        color = wave_colors[w % len(wave_colors)]
        marker = wave_markers[w % len(wave_markers)]

        # Per-location Poisson orders: n_j ~ Poisson(Lambda*T * p_j)
        order_qty = rng.poisson(demand_rate * cand_probs)
        active_idx = [j for j in range(n_cand) if order_qty[j] > 0]

        if not active_idx:
            continue

        # Active locations: coloured markers sized by quantity
        active_pos = cand_pos[active_idx]
        for j in active_idx:
            qty = order_qty[j]
            sz = 5 + qty * 3
            ax.plot(*cand_pos[j], marker, color=color, markersize=sz,
                    markeredgecolor="white", markeredgewidth=0.6,
                    alpha=0.75, zorder=4)
            ax.text(cand_pos[j, 0] + 0.15, cand_pos[j, 1] + 0.15,
                    str(qty), fontsize=5, fontweight="bold", color=color,
                    alpha=0.85, zorder=5)

        # VRP routes via actual CVRP solver from the implementation
        # Cap at 3 vehicles for visual clarity
        from lib.baselines.temporal_only import _solve_vrp
        demands_w = np.array([order_qty[j] for j in active_idx])
        K_show = min(K_dispatch, 3, max(1, len(active_idx) // 2))
        if K_show == 0:
            continue

        routes, _, _ = _solve_vrp(
            depot_pos, active_pos, K_show, capacity=20,
            demands=demands_w, time_limit=5.0,
        )

        veh_styles = ["-", "--", "-."]
        for k, route in enumerate(routes):
            if not route:
                continue
            route_pts = active_pos[route]
            pts = np.vstack([[depot_pos], route_pts, [depot_pos]])
            ls = veh_styles[k % len(veh_styles)]
            ax.plot(pts[:, 0], pts[:, 1], ls, color=color,
                    linewidth=1.3, alpha=0.60, zorder=2)
            if len(pts) >= 2:
                ax.annotate("", xy=pts[1], xytext=pts[0],
                            arrowprops=dict(arrowstyle="-|>", color=color,
                                            lw=1.3, alpha=0.60,
                                            mutation_scale=8),
                            zorder=3)

        ax.scatter([], [], color=color, marker=marker, s=30,
                   label=f"Wave {w + 1} ({len(active_idx)} loc, "
                         f"{len(routes)} veh)")

    _draw_depot(ax, depot_pos)
    ax.legend(fontsize=5.5, loc="upper right", framealpha=0.85,
              markerscale=0.9, handletextpad=0.3)
    _setup_axes(ax, title,
                f"{n_waves} waves, K={K_dispatch}, T={T_val:.2f}h")


# ── Partition-based: Joint-Nom, Joint-DRO ────────────────────────────────
def _plot_partition(ax, design, title):
    """Colored districts + 2-opt TSP tours from depot."""
    roots, bd = _get_roots_and_districts(design)
    n_d = len(roots)
    local_block_ids, local_block_pos, local_cell, local_grid = _partition_plot_context(design)
    depot_idx = local_block_ids.index(design.depot_id)
    depot_pos = local_block_pos[depot_idx]

    # Draw actual blocks at the refined resolution.
    for j in range(len(local_block_ids)):
        d = bd[j]
        c = PALETTE[d % len(PALETTE)] if d >= 0 else "#dddddd"
        x, y = local_block_pos[j]
        rect = plt.Rectangle(
            (x - local_cell / 2, y - local_cell / 2),
            local_cell,
            local_cell,
            facecolor=c,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.28,
            zorder=0,
        )
        ax.add_patch(rect)
        if len(local_block_ids) <= 36:
            ax.text(x, y, local_block_ids[j][-3:], ha="center", va="center",
                    fontsize=5.0, color="black", alpha=0.55)

    # Per-district 2-opt TSP tour: depot → blocks → depot
    for d, ri in enumerate(roots):
        color = PALETTE[d % len(PALETTE)]
        idx_list = [j for j in range(len(local_block_ids)) if bd[j] == d]
        if not idx_list:
            continue
        pos = local_block_pos[idx_list]
        order = _two_opt(pos, depot_pos)
        pts = np.vstack([[depot_pos], pos[order], [depot_pos]])
        ax.plot(pts[:, 0], pts[:, 1], "-", color=color, linewidth=1.5,
                alpha=0.8, zorder=2)
        if len(pts) >= 2:
            ax.annotate("", xy=pts[1], xytext=pts[0],
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=1.5, mutation_scale=10), zorder=3)
        ax.plot(*local_block_pos[ri], "s", color=color, markersize=8,
                markeredgecolor="black", markeredgewidth=1.0, zorder=5)

    _draw_depot(ax, depot_pos)
    ax.set_xlim(-0.6, GRID_SIZE - 0.4)
    ax.set_ylim(-0.6, GRID_SIZE - 0.4)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)
    _setup_axes(ax, title, f"{n_d} districts, {_t_str(design)}")


# ── VCC: meeting-point coordination ──────────────────────────────────────
def _plot_vcc(ax, design, title):
    """VCC: sampled OD requests and a shared non-partition coordinated fleet."""
    depot_idx = block_ids.index(design.depot_id)
    depot_pos = block_pos_km[depot_idx]
    meta = design.service_metadata or {}
    lw = float(meta.get("max_walk_km", 0.3))
    fleet_size = int(meta.get("fleet_size", K))
    capacity = int(meta.get("vehicle_capacity", 3))

    # Neutral background (no partition coloring)
    for j in range(N):
        _draw_block(ax, j, NEUTRAL, alpha=0.12)

    rng = np.random.RandomState(42)
    probs = np.array([prob_dict.get(b, 0.0) for b in block_ids], dtype=float)
    probs = probs / probs.sum()
    n_req = 6
    req_orig = rng.choice(N, size=n_req, p=probs)
    req_dest = rng.choice(N, size=n_req, p=probs)
    for i in range(n_req):
        if req_dest[i] == req_orig[i]:
            req_dest[i] = (req_dest[i] + 1) % N

    # Group requests to vehicles just for illustration.
    groups = [[] for _ in range(max(fleet_size, 1))]
    for r in range(n_req):
        groups[r % max(fleet_size, 1)].append(r)

    for d, req_ids in enumerate(groups):
        color = PALETTE[d % len(PALETTE)]
        if not req_ids:
            continue
        stops = [depot_pos]
        for rid in req_ids:
            origin = block_pos_km[req_orig[rid]]
            dest = block_pos_km[req_dest[rid]]
            meet_pick = origin + 0.6 * (depot_pos - origin)
            if np.linalg.norm(meet_pick - origin) > lw:
                direction = depot_pos - origin
                meet_pick = origin + lw * direction / np.linalg.norm(direction)
            meet_drop = dest + 0.4 * (depot_pos - dest)
            if np.linalg.norm(meet_drop - dest) > lw:
                direction = depot_pos - dest
                meet_drop = dest + lw * direction / np.linalg.norm(direction)
            stops.extend([meet_pick, meet_drop])

            ax.plot(*origin, "o", color=color, markersize=4.5,
                    markeredgecolor="white", markeredgewidth=0.5, zorder=4)
            ax.plot(*dest, "x", color=color, markersize=5.5,
                    markeredgewidth=1.0, zorder=4)
            ax.add_patch(Circle(origin, lw, facecolor=color, edgecolor=color,
                                alpha=0.06, linewidth=0.8, linestyle="--", zorder=1))
            ax.add_patch(Circle(dest, lw, facecolor=color, edgecolor=color,
                                alpha=0.06, linewidth=0.8, linestyle="--", zorder=1))
            ax.plot(*meet_pick, "D", color=color, markersize=4.5,
                    markeredgecolor="black", markeredgewidth=0.4, zorder=5)
            ax.plot(*meet_drop, "s", color=color, markersize=4.5,
                    markeredgecolor="black", markeredgewidth=0.4, zorder=5)

        stops.append(depot_pos)
        pts = np.array(stops)
        ax.plot(pts[:, 0], pts[:, 1], "--", color=color, linewidth=1.5,
                alpha=0.7, zorder=2)

    _draw_depot(ax, depot_pos)
    _setup_axes(ax, title,
                f"{fleet_size} vehicles, cap={capacity}, lw={lw}km")


# ── On-demand: OD ────────────────────────────────────────────────────────
def _plot_on_demand(ax, design, title):
    """OD: sample direct-service trips from depot.

    Demand can arise at any location in the service region (continuous).
    Show arrows to random locations, not block centroids.
    """
    depot_idx = block_ids.index(design.depot_id)
    depot_pos = block_pos_km[depot_idx]

    # Light service-region boundary (no block grid)
    region = plt.Rectangle(
        (-0.5, -0.5), GRID_SIZE, GRID_SIZE,
        facecolor=NEUTRAL, edgecolor="#999999", linewidth=1.0,
        alpha=0.12, zorder=0,
    )
    ax.add_patch(region)

    # Sample random customer locations (continuous demand)
    rng = np.random.RandomState(42)
    n_trips = 6
    trip_locs = rng.uniform(-0.3, GRID_SIZE - 0.7, size=(n_trips, 2))

    for i in range(n_trips):
        color = PALETTE[i % len(PALETTE)]
        ax.annotate("", xy=trip_locs[i], xytext=depot_pos,
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=1.5, alpha=0.7, mutation_scale=10),
                    zorder=2)
        ax.plot(*trip_locs[i], "o", color=color, markersize=5,
                markeredgecolor="black", markeredgewidth=0.4, zorder=3)

    _draw_depot(ax, depot_pos)
    _setup_axes(ax, title, f"direct service, {_t_str(design)}")


# ── Dispatcher ───────────────────────────────────────────────────────────
def plot_design(ax, design, title):
    mode = _mode_type(title)
    if mode == "fr":
        _plot_fixed_route(ax, design, title)
    elif mode == "carlsson":
        _plot_carlsson(ax, design, title)
    elif mode == "tp":
        _plot_temporal(ax, design, title)
    elif mode == "od":
        _plot_on_demand(ax, design, title)
    elif mode == "vcc":
        _plot_vcc(ax, design, title)
    else:
        _plot_partition(ax, design, title)


# ── Create figure ─────────────────────────────────────────────────────────
n_methods = len(designs)
ncols = 4
nrows = (n_methods + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows))
axes = axes.flatten()

for idx, (name, design) in enumerate(designs.items()):
    plot_design(axes[idx], design, name)

for idx in range(n_methods, len(axes)):
    axes[idx].set_visible(False)

fig.suptitle(f"Service Designs \u2014 K={K}, \u039b={LAMBDA}",
             fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()

out_dir = "results/baseline_comparison/figures"
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "service_designs.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
