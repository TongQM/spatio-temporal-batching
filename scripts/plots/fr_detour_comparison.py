"""Side-by-side FR vs FR-Detour with scattered (non-grid) demand.

On a regular grid, every block sits at a grid intersection, so the
optimizer can always make it a checkpoint.  With randomly scattered
demand, some blocks naturally fall *near* a route but not *on* it —
exactly where detour capability matters.

Each panel runs its own design (delta=0 vs delta>0) so the optimizer
can exploit corridor reachability differently.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import networkx as nx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from lib.baselines import FRDetour
from lib.baselines.base import ServiceDesign


# ── Custom geodata with arbitrary block positions ────────────────────────
class ScatteredGeoData:
    """Minimal GeoData with user-specified block positions."""

    def __init__(self, positions_km):
        self.short_geoid_list = list(positions_km.keys())
        self.n_blocks = len(self.short_geoid_list)
        self.level = "block_group"
        self.gdf = None
        # pos in METRES (GeoData convention)
        self.pos = {b: (x * 1000.0, y * 1000.0)
                    for b, (x, y) in positions_km.items()}
        # Fully-connected contiguity graph
        self.G = nx.complete_graph(self.short_geoid_list)
        self.block_graph = self.G
        self.arc_list = []
        for u, v in self.G.edges():
            self.arc_list.append((u, v))
            self.arc_list.append((v, u))

    def get_dist(self, b1, b2):
        p1, p2 = np.array(self.pos[b1]), np.array(self.pos[b2])
        return float(np.linalg.norm(p1 - p2))

    def get_area(self, _):
        return 250_000.0  # 0.25 km² per block

    def get_K(self, _):
        return 2.0

    def get_F(self, _):
        return 0.0

    def get_arc_list(self):
        return self.arc_list


# ── Scattered demand layout ──────────────────────────────────────────────
rng = np.random.RandomState(7)
n_blocks = 25
positions_km = {}
for i in range(n_blocks):
    positions_km[f"B{i:02d}"] = (
        round(rng.uniform(0.2, 4.8), 2),
        round(rng.uniform(0.2, 4.8), 2),
    )

geodata = ScatteredGeoData(positions_km)
block_ids = geodata.short_geoid_list
N = len(block_ids)
block_pos_km = np.array([
    np.array(geodata.pos[b], dtype=float) / 1000.0 for b in block_ids
])
prob_dict = {b: 1.0 / N for b in block_ids}
Omega_dict = {b: np.array([1.0]) for b in block_ids}

def identity_J(omega):
    return float(np.sum(omega))

# ── Settings ─────────────────────────────────────────────────────────────
K = 2
LAMBDA = 1000.0
WR, WV = 1.0, 10.0
RS_ITERS = 100
DELTA_DETOUR = 0.8

# ── Run separate designs for delta=0 and delta>0 ────────────────────────
print(f"  Designing Multi-FR (K={K}, delta=0) ...", flush=True)
fr0 = FRDetour(delta=0.0, max_iters=RS_ITERS, name="Multi-FR")
design_fr = fr0.design(
    geodata, prob_dict, Omega_dict, identity_J, K,
    Lambda=LAMBDA, wr=WR, wv=WV,
)

print(f"  Designing Multi-FR-Detour (K={K}, delta={DELTA_DETOUR}) ...",
      flush=True)
fr_det = FRDetour(delta=DELTA_DETOUR, max_iters=RS_ITERS)
design_det = fr_det.design(
    geodata, prob_dict, Omega_dict, identity_J, K,
    Lambda=LAMBDA, wr=WR, wv=WV,
)

# ── Evaluate both designs ────────────────────────────────────────────────
print("  Evaluating ...", flush=True)
eval_fr = fr0.evaluate(
    design_fr, geodata, prob_dict, Omega_dict, identity_J,
    LAMBDA, WR, WV,
)
eval_det = fr_det.evaluate(
    design_det, geodata, prob_dict, Omega_dict, identity_J,
    LAMBDA, WR, WV,
)

designs = {
    "Multi-FR (delta=0)": (design_fr, eval_fr, 0.0),
    f"Multi-FR-Detour (delta={DELTA_DETOUR})": (design_det, eval_det, DELTA_DETOUR),
}

for name, (d, ev, _) in designs.items():
    routes = d.district_routes or {}
    for root_id, route in routes.items():
        cp_names = [block_ids[gi] for gi in route]
        ri = block_ids.index(root_id)
        assigned = [block_ids[j] for j in range(N)
                    if round(d.assignment[j, ri]) == 1]
        corridor_only = [b for b in assigned if block_ids.index(b) not in route]
        print(f"  [{name}] Route {root_id}: {len(route)} CPs -> {cp_names}")
        print(f"      Corridor-only: {corridor_only}")
    print(f"  [{name}] total_cost={ev.total_cost:.4f}  "
          f"user_time={ev.total_user_time:.4f}h  "
          f"provider={ev.provider_cost:.4f}")
print()


# ── Visualization ────────────────────────────────────────────────────────
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]


def _pt_seg_dist(p, a, b):
    """Distance from point p to segment a-b."""
    ab = b - a
    ab2 = np.dot(ab, ab)
    if ab2 < 1e-12:
        return float(np.linalg.norm(p - a))
    t = np.clip(np.dot(p - a, ab) / ab2, 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def _in_corridor(j, route_global, delta):
    """True if block j is within delta of any route segment (CP-to-CP)."""
    pt = block_pos_km[j]
    for k in range(len(route_global) - 1):
        if _pt_seg_dist(pt, block_pos_km[route_global[k]],
                        block_pos_km[route_global[k + 1]]) <= delta + 1e-9:
            return True
    return False


def _plot_fr(ax, design, eval_result, title, delta):
    depot_idx = block_ids.index(design.depot_id)
    depot_pos = block_pos_km[depot_idx]

    routes = design.district_routes or {}
    assignment = design.assignment

    active = [(rid, r) for rid, r in routes.items() if r]
    n_lines = len(active)

    # Collect all checkpoint indices
    all_cps = set()
    for _, route in active:
        all_cps.update(route)

    # Map blocks -> active routes, with geometric corridor check
    block_route = np.full(N, -1, dtype=int)
    for d, (root_id, route_global) in enumerate(active):
        ri = block_ids.index(root_id)
        for j in range(N):
            if round(assignment[j, ri]) != 1:
                continue
            if j in all_cps:
                block_route[j] = d
            elif delta > 0 and _in_corridor(j, route_global, delta):
                block_route[j] = d

    # Draw blocks
    for j in range(N):
        x, y = block_pos_km[j]
        if block_route[j] >= 0:
            d = block_route[j]
            color = PALETTE[d % len(PALETTE)]
            is_cp = j in all_cps
            marker = "s" if is_cp else "o"
            size = 10 if is_cp else 7
            edge = "black" if is_cp else "white"
            alpha = 0.9 if is_cp else 0.6
            ax.plot(x, y, marker, color=color, markersize=size,
                    markeredgecolor=edge, markeredgewidth=0.7,
                    alpha=alpha, zorder=4)
        else:
            ax.plot(x, y, "o", color="#bbbbbb", markersize=6,
                    markeredgecolor="black", markeredgewidth=0.4,
                    alpha=0.5, zorder=3)
        ax.text(x + 0.12, y + 0.12, block_ids[j][-3:], fontsize=5,
                color="black", alpha=0.6, zorder=7)

    # Draw closed route loops
    for d, (root_id, route_global) in enumerate(active):
        color = PALETTE[d % len(PALETTE)]
        cp_pos = np.array([block_pos_km[gi] for gi in route_global])
        if len(cp_pos) < 1:
            continue
        loop = np.vstack([[depot_pos], cp_pos, [depot_pos]])
        ax.plot(loop[:, 0], loop[:, 1], "-", color=color,
                linewidth=2.0, alpha=0.7, zorder=2)
        if len(loop) >= 2:
            ax.annotate("", xy=loop[1], xytext=loop[0],
                        arrowprops=dict(arrowstyle="-|>", color=color,
                                        lw=2.0, mutation_scale=12), zorder=3)

        # Corridor outline (FR-Detour only)
        if delta > 0 and len(cp_pos) >= 2:
            for k in range(len(cp_pos) - 1):
                p1, p2 = cp_pos[k], cp_pos[k + 1]
                seg = p2 - p1
                seg_len = np.linalg.norm(seg)
                if seg_len < 1e-6:
                    continue
                perp = np.array([-seg[1], seg[0]]) / seg_len * delta
                corners = np.array([p1 + perp, p2 + perp, p2 - perp, p1 - perp])
                poly = Polygon(corners, facecolor=color, edgecolor=color,
                               alpha=0.06, linewidth=0.6, linestyle="--", zorder=0)
                ax.add_patch(poly)
            for cp in cp_pos:
                circ = Circle(cp, delta, facecolor=color, edgecolor=color,
                              alpha=0.06, linewidth=0.6, linestyle="--", zorder=0)
                ax.add_patch(circ)

    # Depot
    ax.plot(*depot_pos, "*", color="red", markersize=16,
            markeredgecolor="black", markeredgewidth=0.8, zorder=6)

    # Stats
    n_cp = len(all_cps)
    n_corr = sum(1 for j in range(N) if block_route[j] >= 0
                 and j not in all_cps)
    n_uncov = sum(1 for j in range(N) if block_route[j] < 0)

    T_info = design.dispatch_intervals
    T_vals = list(T_info.values()) if T_info else []
    t_str = (f"T={T_vals[0]:.2f}h"
             if T_vals and len(set(f"{t:.2f}" for t in T_vals)) == 1
             else f"T={min(T_vals):.2f}-{max(T_vals):.2f}h" if T_vals
             else "")

    # Title with route stats
    ax.set_title(f"{title}\n{n_lines} routes, {t_str}\n"
                 f"{n_cp} CPs  {n_corr} corridor  {n_uncov} uncov",
                 fontsize=9)

    # Evaluation metrics text box
    ev = eval_result
    textstr = (f"User: wait={ev.avg_wait_time:.3f} "
               f"ride={ev.avg_invehicle_time:.3f} "
               f"walk={ev.avg_walk_time:.3f}h\n"
               f"Provider: {ev.provider_cost:.3f}  "
               f"Fleet: {ev.fleet_size:.1f}\n"
               f"Total cost: {ev.total_cost:.3f}")
    props = dict(boxstyle="round,pad=0.3", facecolor="white",
                 edgecolor="#cccccc", alpha=0.85)
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=6.5,
            verticalalignment="bottom", bbox=props, zorder=10)

    margin = 0.3
    ax.set_xlim(block_pos_km[:, 0].min() - margin,
                block_pos_km[:, 0].max() + margin)
    ax.set_ylim(block_pos_km[:, 1].min() - margin,
                block_pos_km[:, 1].max() + margin)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=6)
    ax.set_xlabel("x (km)", fontsize=8)
    ax.set_ylabel("y (km)", fontsize=8)


# ── Figure ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))

for i, (name, (design, ev, delta)) in enumerate(designs.items()):
    _plot_fr(axes[i], design, ev, name, delta)

fig.suptitle(f"FR vs FR-Detour — scattered demand, K={K}, "
             f"\u039b={LAMBDA:.0f}",
             fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()

out_dir = "results/baseline_comparison/figures"
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "fr_detour_comparison.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
