"""Visualise service designs (district assignments + routing) for all baselines."""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.colors as mcolors

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
    print(f"  Designing {name} …", flush=True)
    design = method.design(
        geodata, prob_dict, Omega_dict, J_function, K,
        Lambda=LAMBDA, wr=WR, wv=WV,
    )
    designs[name] = design


# ── Plotting helper ───────────────────────────────────────────────────────
DISTRICT_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
]


def plot_design(ax, design, title):
    """Plot one service design on the given axes."""
    assignment = design.assignment
    depot_id = design.depot_id

    # Identify districts
    roots = []
    for i in range(N):
        if round(assignment[i, i]) == 1:
            roots.append(i)

    # Assign block → district
    block_district = np.full(N, -1, dtype=int)
    for d, root_idx in enumerate(roots):
        for j in range(N):
            if round(assignment[j, root_idx]) == 1:
                block_district[j] = d

    n_districts = len(roots)

    # Draw blocks as colored squares
    cell_size = 0.9  # km (grid spacing is 1 km)
    for j in range(N):
        d = block_district[j]
        color = DISTRICT_COLORS[d % len(DISTRICT_COLORS)] if d >= 0 else "#dddddd"
        x, y = block_pos_km[j]
        rect = plt.Rectangle(
            (x - cell_size / 2, y - cell_size / 2), cell_size, cell_size,
            facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.4,
        )
        ax.add_patch(rect)
        ax.text(x, y, block_ids[j][-3:], ha="center", va="center",
                fontsize=5.5, color="black", alpha=0.7)

    # Draw routes per district
    depot_idx = block_ids.index(depot_id)
    depot_pos = block_pos_km[depot_idx]

    for d, root_idx in enumerate(roots):
        root_id = block_ids[root_idx]
        color = DISTRICT_COLORS[d % len(DISTRICT_COLORS)]

        assigned_idx = [j for j in range(N) if block_district[j] == d]
        if not assigned_idx:
            continue

        assigned_pos = block_pos_km[assigned_idx]

        # Get route order if available
        route_order = design.district_routes.get(root_id)
        if route_order is not None and len(route_order) > 0:
            # Route is in local indices within the district
            ordered_pos = [assigned_pos[k] for k in route_order
                           if k < len(assigned_pos)]
        else:
            # NN route from depot through assigned blocks
            dists_from_depot = np.linalg.norm(assigned_pos - depot_pos, axis=1)
            start = int(np.argmin(dists_from_depot))
            visited = [start]
            remaining = set(range(len(assigned_idx))) - {start}
            while remaining:
                cur = assigned_pos[visited[-1]]
                dists = {r: float(np.linalg.norm(assigned_pos[r] - cur))
                         for r in remaining}
                nxt = min(dists, key=dists.get)
                visited.append(nxt)
                remaining.discard(nxt)
            ordered_pos = [assigned_pos[k] for k in visited]

        # Draw depot → route → depot
        route_pts = np.array([depot_pos] + ordered_pos + [depot_pos])
        ax.plot(route_pts[:, 0], route_pts[:, 1],
                '-', color=color, linewidth=1.5, alpha=0.8, zorder=2)

        # Arrows for direction on first segment
        if len(route_pts) >= 2:
            ax.annotate("", xy=route_pts[1], xytext=route_pts[0],
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=1.5, mutation_scale=10),
                        zorder=3)

        # Mark district root
        root_pos = block_pos_km[root_idx]
        ax.plot(*root_pos, "s", color=color, markersize=8, markeredgecolor="black",
                markeredgewidth=1.0, zorder=5)

    # Mark depot
    ax.plot(*depot_pos, "*", color="red", markersize=14, markeredgecolor="black",
            markeredgewidth=0.8, zorder=6)

    T_info = design.dispatch_intervals
    if T_info:
        T_vals = list(T_info.values())
        if len(set(f"{t:.2f}" for t in T_vals)) == 1:
            t_str = f"T={T_vals[0]:.2f}h"
        else:
            t_str = f"T={min(T_vals):.2f}-{max(T_vals):.2f}h"
    else:
        t_str = ""

    ax.set_title(f"{title}\n{n_districts} districts, {t_str}", fontsize=9)
    ax.set_xlim(-0.6, GRID_SIZE - 0.4)
    ax.set_ylim(-0.6, GRID_SIZE - 0.4)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)


# ── Create figure ─────────────────────────────────────────────────────────
n_methods = len(designs)
ncols = 4
nrows = (n_methods + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows))
axes = axes.flatten()

for idx, (name, design) in enumerate(designs.items()):
    plot_design(axes[idx], design, name)

# Hide unused subplots
for idx in range(n_methods, len(axes)):
    axes[idx].set_visible(False)

fig.suptitle(f"Service Designs — {GRID_SIZE}x{GRID_SIZE} grid, K={K}, Λ={LAMBDA}",
             fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()

out_dir = "results/baseline_comparison/figures"
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "service_designs.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
