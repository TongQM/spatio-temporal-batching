"""Visualise service designs for all baselines with mode-appropriate rendering.

Each service mode gets a visualization matching its operational model:
  - Partition-based (SP-Lit, Joint-Nom, Joint-DRO):
        colored districts + depot-originating tours
  - Fixed-route (Multi-FR, Multi-FR-Detour):
        route lines through checkpoints + shaded service areas
  - VCC:  meeting-point coordination, walking-distance circles
  - Temporal-only (TP-Lit):  uniform service area, no spatial structure
  - On-demand (OD):  uniform service area, direct-service illustration
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ── Project imports ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from scripts.experiments.baseline_comparison import (
    make_toy_geodata, build_prob_dict, build_omega_dict, identity_J,
)
from lib.baselines import (
    FullyOD, TemporalOnly,
    JointNom, JointDRO, VCC, FRDetour, CarlssonPartition,
)

# ── Settings ──────────────────────────────────────────────────────────────
GRID_SIZE = 5
K = 3
LAMBDA = 1000.0
WR, WV = 1.0, 10.0
T_FIXED = 0.5
RS_ITERS = 100
EPS = 0.001

# ── Build data ────────────────────────────────────────────────────────────
geodata = make_toy_geodata(GRID_SIZE)
prob_dict = build_prob_dict(geodata)
Omega_dict = build_omega_dict(geodata)
J_function = identity_J

block_ids = geodata.short_geoid_list
N = len(block_ids)
block_pos_km = np.array([
    np.array(geodata.pos[b], dtype=float) / 1000.0 for b in block_ids
])

# ── Run all baselines ────────────────────────────────────────────────────
baselines = [
    ("Multi-FR",        FRDetour(delta=0.0, max_iters=RS_ITERS, name="Multi-FR")),
    ("Multi-FR-Detour", FRDetour(delta=0.3, max_iters=RS_ITERS)),
    ("SP-Lit",          CarlssonPartition(T_fixed=T_FIXED)),
    ("TP-Lit",          TemporalOnly()),
    ("Joint-Nom",       JointNom(max_iters=RS_ITERS)),
    ("Joint-DRO",       JointDRO(epsilon=EPS, max_iters=RS_ITERS)),
    ("VCC",             VCC()),
    ("OD",              FullyOD()),
]

designs = {}
for name, method in baselines:
    print(f"  Designing {name} ...", flush=True)
    design = method.design(
        geodata, prob_dict, Omega_dict, J_function, K,
        Lambda=LAMBDA, wr=WR, wv=WV,
    )
    designs[name] = design


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
    assignment = design.assignment
    roots = [i for i in range(N) if round(assignment[i, i]) == 1]
    block_district = np.full(N, -1, dtype=int)
    for d, ri in enumerate(roots):
        for j in range(N):
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


# ── Partition-based: SP-Lit, Joint-Nom, Joint-DRO ───────────────────────
def _plot_partition(ax, design, title):
    roots, bd = _get_roots_and_districts(design)
    n_d = len(roots)
    depot_idx = block_ids.index(design.depot_id)
    depot_pos = block_pos_km[depot_idx]

    # Colored blocks
    for j in range(N):
        d = bd[j]
        c = PALETTE[d % len(PALETTE)] if d >= 0 else "#dddddd"
        _draw_block(ax, j, c)

    # Per-district: depot -> NN tour -> depot
    for d, ri in enumerate(roots):
        color = PALETTE[d % len(PALETTE)]
        idx_list = [j for j in range(N) if bd[j] == d]
        if not idx_list:
            continue
        pos = block_pos_km[idx_list]
        order = _nn_tour_order(pos, depot_pos)
        pts = np.array([depot_pos] + [pos[k] for k in order] + [depot_pos])
        ax.plot(pts[:, 0], pts[:, 1], '-', color=color, linewidth=1.5,
                alpha=0.8, zorder=2)
        if len(pts) >= 2:
            ax.annotate("", xy=pts[1], xytext=pts[0],
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=1.5, mutation_scale=10), zorder=3)
        ax.plot(*block_pos_km[ri], "s", color=color, markersize=8,
                markeredgecolor="black", markeredgewidth=1.0, zorder=5)

    _draw_depot(ax, depot_pos)
    _setup_axes(ax, title, f"{n_d} districts, {_t_str(design)}")


# ── Fixed-route: Multi-FR, Multi-FR-Detour ──────────────────────────────
def _plot_fixed_route(ax, design, title):
    roots, bd = _get_roots_and_districts(design)
    n_lines = len(roots)
    depot_idx = block_ids.index(design.depot_id)
    depot_pos = block_pos_km[depot_idx]

    for d, ri in enumerate(roots):
        root_id = block_ids[ri]
        color = PALETTE[d % len(PALETTE)]
        route_global = design.district_routes.get(root_id, [])
        cp_set = set(route_global)  # global indices of checkpoints

        # Draw blocks: checkpoints darker, service-area blocks lighter
        for j in range(N):
            if bd[j] != d:
                continue
            alpha = 0.5 if j in cp_set else 0.2
            _draw_block(ax, j, color, alpha=alpha)

        # Fixed route: line through checkpoints in order
        if route_global:
            cp_pos = np.array([block_pos_km[gi] for gi in route_global])
            ax.plot(cp_pos[:, 0], cp_pos[:, 1], '-o', color=color,
                    linewidth=2.2, alpha=0.85, markersize=6,
                    markeredgecolor="black", markeredgewidth=0.6, zorder=3)
            if len(cp_pos) >= 2:
                ax.annotate("", xy=cp_pos[1], xytext=cp_pos[0],
                            arrowprops=dict(arrowstyle="-|>", color=color,
                                            lw=2.0, mutation_scale=12),
                            zorder=4)

    # Unassigned blocks (shouldn't happen but handle gracefully)
    for j in range(N):
        if bd[j] < 0:
            _draw_block(ax, j, "#dddddd", alpha=0.2)

    _draw_depot(ax, depot_pos)
    _setup_axes(ax, title, f"{n_lines} routes, {_t_str(design)}")


# ── VCC: coordination with meeting points ────────────────────────────────
def _plot_vcc(ax, design, title):
    """VCC uses partitions in our adaptation but operationally coordinates
    vehicles and riders via meeting points. Show districts + walking circles."""
    roots, bd = _get_roots_and_districts(design)
    n_d = len(roots)
    depot_idx = block_ids.index(design.depot_id)
    depot_pos = block_pos_km[depot_idx]

    # Walking distance (from VCC default)
    lw = 0.3  # km

    # Colored blocks
    for j in range(N):
        d = bd[j]
        c = PALETTE[d % len(PALETTE)] if d >= 0 else "#dddddd"
        _draw_block(ax, j, c)

    # Per-district: show NN route (dashed) + walking circles at stops
    for d, ri in enumerate(roots):
        color = PALETTE[d % len(PALETTE)]
        idx_list = [j for j in range(N) if bd[j] == d]
        if not idx_list:
            continue
        pos = block_pos_km[idx_list]
        order = _nn_tour_order(pos, depot_pos)
        pts = np.array([depot_pos] + [pos[k] for k in order] + [depot_pos])
        # Dashed route to indicate vehicle-customer coordination
        ax.plot(pts[:, 0], pts[:, 1], '--', color=color, linewidth=1.3,
                alpha=0.7, zorder=2)
        # Walking-distance circles at each stop
        for k in order:
            circ = Circle(pos[k], lw, facecolor=color, edgecolor=color,
                          alpha=0.08, linewidth=0.5, linestyle=":", zorder=1)
            ax.add_patch(circ)
        ax.plot(*block_pos_km[ri], "s", color=color, markersize=8,
                markeredgecolor="black", markeredgewidth=1.0, zorder=5)

    _draw_depot(ax, depot_pos)
    _setup_axes(ax, title, f"{n_d} zones, {_t_str(design)}, lw={lw}km")


# ── Temporal-only: TP-Lit ────────────────────────────────────────────────
def _plot_temporal(ax, design, title):
    """TP-Lit: no spatial partition. Uniform service area."""
    depot_idx = block_ids.index(design.depot_id)
    depot_pos = block_pos_km[depot_idx]

    for j in range(N):
        _draw_block(ax, j, NEUTRAL, alpha=0.35)

    _draw_depot(ax, depot_pos)
    _setup_axes(ax, title, f"no partition, {_t_str(design)}")


# ── On-demand: OD ────────────────────────────────────────────────────────
def _plot_on_demand(ax, design, title):
    """OD: each request served individually. Show radial direct-service lines."""
    depot_idx = block_ids.index(design.depot_id)
    depot_pos = block_pos_km[depot_idx]

    for j in range(N):
        _draw_block(ax, j, NEUTRAL, alpha=0.35)

    # Radial lines from depot to each block (illustrating direct service)
    for j in range(N):
        ax.plot([depot_pos[0], block_pos_km[j, 0]],
                [depot_pos[1], block_pos_km[j, 1]],
                '-', color="#888888", linewidth=0.4, alpha=0.35, zorder=1)

    _draw_depot(ax, depot_pos)
    _setup_axes(ax, title, f"direct service, {_t_str(design)}")


# ── Dispatcher ───────────────────────────────────────────────────────────
def plot_design(ax, design, title):
    mode = _mode_type(title)
    if mode == "fr":
        _plot_fixed_route(ax, design, title)
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

fig.suptitle(f"Service Designs \u2014 {GRID_SIZE}x{GRID_SIZE} grid, K={K}, "
             f"\u039b={LAMBDA}",
             fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()

out_dir = "results/baseline_comparison/figures"
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "service_designs.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
