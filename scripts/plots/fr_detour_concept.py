"""Conceptual illustration: FR vs FR-Detour service mechanism.

Hand-crafted example with a fixed route and off-route demand points
to clearly show how detour expands coverage without adding checkpoints.
This is a pedagogical diagram, not an optimization output.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

# ── Route and demand geometry ────────────────────────────────────────────
# Fixed route: 4 checkpoints along a path
checkpoints = np.array([
    [0.5, 0.0],  # CP 0
    [1.5, 1.5],  # CP 1
    [3.0, 2.5],  # CP 2
    [4.5, 1.0],  # CP 3
])
depot = np.array([0.0, -0.5])

# Demand points: some on-route, some off-route but nearby, some far
demands = {
    # (position, label, type)
    # type: "checkpoint" = at a checkpoint,
    #        "corridor"  = within delta of route,
    #        "far"       = outside corridor
    "A": (np.array([0.5, 0.0]),  "checkpoint"),   # at CP 0
    "B": (np.array([1.5, 1.5]),  "checkpoint"),   # at CP 1
    "C": (np.array([3.0, 2.5]),  "checkpoint"),   # at CP 2
    "D": (np.array([4.5, 1.0]),  "checkpoint"),   # at CP 3
    "E": (np.array([1.0, 2.2]),  "corridor"),     # near segment CP1-CP2
    "F": (np.array([2.2, 2.5]),  "corridor"),     # near segment CP1-CP2
    "G": (np.array([3.8, 1.8]),  "corridor"),     # near segment CP2-CP3
    "H": (np.array([0.0, 1.0]),  "corridor"),     # near segment CP0-CP1
    "I": (np.array([4.8, 2.8]),  "far"),          # far from route
    "J": (np.array([0.5, 3.5]),  "far"),          # far from route
    "K": (np.array([2.5, 0.0]),  "far"),          # far from route
}

DELTA = 1.0  # detour corridor width

COLORS = {
    "checkpoint": "#2ca02c",  # green — served by both
    "corridor":   "#1f77b4",  # blue  — served by FR-Detour only
    "far":        "#d62728",  # red   — not served
}

ROUTE_COLOR = "#444444"


def _draw_corridor(ax, p1, p2, delta, color="#1f77b4", alpha=0.08):
    """Draw corridor rectangle around a route segment."""
    seg = p2 - p1
    seg_len = np.linalg.norm(seg)
    if seg_len < 1e-6:
        return
    perp = np.array([-seg[1], seg[0]]) / seg_len * delta
    corners = np.array([p1 + perp, p2 + perp, p2 - perp, p1 - perp, p1 + perp])
    from matplotlib.patches import Polygon
    poly = Polygon(corners[:4], facecolor=color, edgecolor=color,
                   alpha=alpha, linewidth=0.8, linestyle="--", zorder=0)
    ax.add_patch(poly)
    ax.plot(corners[:, 0], corners[:, 1], "--", color=color,
            linewidth=0.6, alpha=0.3, zorder=0)


def _draw_panel(ax, title, show_corridor, show_detour_service):
    """Draw one panel of the comparison."""
    # Service region background
    ax.add_patch(Rectangle((-0.5, -1.2), 6.0, 5.5,
                           facecolor="#f5f5f5", edgecolor="#cccccc",
                           linewidth=1.0, zorder=-1))

    # Corridor (FR-Detour only)
    if show_corridor:
        for i in range(len(checkpoints) - 1):
            _draw_corridor(ax, checkpoints[i], checkpoints[i + 1], DELTA)

    # Route line
    full_route = np.vstack([depot, checkpoints, depot])
    ax.plot(full_route[:, 0], full_route[:, 1], "-", color=ROUTE_COLOR,
            linewidth=2.5, alpha=0.7, zorder=2)
    ax.annotate("", xy=full_route[1], xytext=full_route[0],
                arrowprops=dict(arrowstyle="-|>", color=ROUTE_COLOR,
                                lw=2.5, mutation_scale=14), zorder=3)

    # Checkpoint markers
    ax.plot(checkpoints[:, 0], checkpoints[:, 1], "s", color=ROUTE_COLOR,
            markersize=10, markeredgecolor="black", markeredgewidth=0.8,
            zorder=4, label="Checkpoint stops")

    # Demand points
    for label, (pos, dtype) in demands.items():
        if dtype == "checkpoint":
            continue  # drawn as checkpoint marker above

        if dtype == "corridor":
            if show_detour_service:
                # Served: show detour path
                color = COLORS["corridor"]
                ax.plot(*pos, "o", color=color, markersize=9,
                        markeredgecolor="black", markeredgewidth=0.6,
                        zorder=5, alpha=0.9)
                # Dashed line: vehicle detours to this point
                # Find nearest point on route
                ax.plot([pos[0]], [pos[1]], "o", color=color, markersize=9,
                        markeredgecolor="black", markeredgewidth=0.6, zorder=5)
                # Detour arrow
                nearest_cp = checkpoints[np.argmin(
                    np.linalg.norm(checkpoints - pos, axis=1))]
                ax.annotate("", xy=pos, xytext=nearest_cp,
                            arrowprops=dict(arrowstyle="-|>", color=color,
                                            lw=1.0, linestyle="dashed",
                                            alpha=0.5, mutation_scale=8),
                            zorder=3)
            else:
                # Not served: customer must walk (shown with X)
                color = "#999999"
                ax.plot(*pos, "o", color=color, markersize=9,
                        markeredgecolor="black", markeredgewidth=0.6,
                        zorder=5, alpha=0.5)
                ax.plot(*pos, "x", color="#d62728", markersize=8,
                        markeredgewidth=2.0, zorder=6, alpha=0.7)
        else:  # far
            ax.plot(*pos, "o", color="#999999", markersize=9,
                    markeredgecolor="black", markeredgewidth=0.6,
                    zorder=5, alpha=0.5)
            ax.plot(*pos, "x", color="#d62728", markersize=8,
                    markeredgewidth=2.0, zorder=6, alpha=0.7)

        ax.text(pos[0] + 0.15, pos[1] + 0.15, label, fontsize=8,
                fontweight="bold", color="black", zorder=7)

    # Label checkpoints
    for i, cp in enumerate(checkpoints):
        ax.text(cp[0] + 0.15, cp[1] - 0.25, f"CP{i}", fontsize=7,
                color=ROUTE_COLOR, zorder=7)

    # Depot
    ax.plot(*depot, "*", color="red", markersize=16,
            markeredgecolor="black", markeredgewidth=0.8, zorder=6)
    ax.text(depot[0] + 0.2, depot[1] - 0.15, "Depot", fontsize=7,
            color="red", zorder=7)

    ax.set_xlim(-1.0, 5.8)
    ax.set_ylim(-1.5, 4.2)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.tick_params(labelsize=7)
    ax.set_xlabel("x (km)", fontsize=8)
    ax.set_ylabel("y (km)", fontsize=8)


# ── Left panel: FR (no detour) ──────────────────────────────────────────
_draw_panel(axes[0], "Multi-FR (δ = 0)\nOnly checkpoint stops served",
            show_corridor=False, show_detour_service=False)

# ── Right panel: FR-Detour ───────────────────────────────────────────────
_draw_panel(axes[1], f"Multi-FR-Detour (δ = {DELTA} km)\nCorridor blocks served by deviation",
            show_corridor=True, show_detour_service=True)

# ── Legend ────────────────────────────────────────────────────────────────
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker="s", color="w", markerfacecolor=ROUTE_COLOR,
           markeredgecolor="black", markersize=10, label="Checkpoint (served by both)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=COLORS["corridor"],
           markeredgecolor="black", markersize=9, label="Corridor demand (served by detour)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#999999",
           markeredgecolor="black", markersize=9, label="Demand location"),
    Line2D([0], [0], marker="x", color="#d62728", markersize=8,
           markeredgewidth=2.0, linestyle="None", label="Not served (must walk)"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=4, fontsize=9,
           frameon=True, bbox_to_anchor=(0.5, -0.02))

fig.suptitle("Fixed-Route Service: How Detour Expands Coverage",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout()

out_dir = "results/baseline_comparison/figures"
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "fr_detour_concept.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
