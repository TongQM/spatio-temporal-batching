"""Scatter plot: user time vs provider cost across service modes.

This script does not recompute evaluation metrics. It reads the method-specific
outputs stored in `results.json` by `baseline_comparison.py` and plots:

  - x-axis: provider cost reported by each method's evaluation
  - y-axis: total user time (hours per rider)
  - labels: total cost from the same stored evaluation

The underlying evaluation semantics can differ by method. For example,
partition-based methods may use Monte Carlo dispatch-cycle simulation,
while other baselines may use custom simulators. This figure is therefore
a comparison of reported outputs, not a reimplementation of a single common
cost formula inside the plotting script.
"""
import json
import sys
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Load results ──────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    dirs = sorted(glob.glob("results/baseline_comparison/synthetic_*"))
    if not dirs:
        sys.exit("No results found. Run baseline_comparison.py first.")
    path = os.path.join(dirs[-1], "results.json")

print(f"Loading: {path}")
with open(path) as f:
    data = json.load(f)

results = data["results"]
meta = data.get("experiment", {})
wr = meta.get("wr", 1.0)

# ── Extract data ──────────────────────────────────────────────────────────
names = [r["name"] for r in results]
user_time = np.array([r["user"]["total_user_time_h"] for r in results])
provider_cost = np.array([r["provider"]["provider_cost"] for r in results])
total_cost = np.array([r["total_cost"] for r in results])

def _short(n):
    if "Multi-FR-Detour" in n: return "FR-Detour"
    if "Multi-FR" in n:        return "Multi-FR"
    if "SP-Lit" in n:          return "SP-Lit"
    if "TP-Lit" in n:          return "TP-Lit"
    return n.split("(")[0].strip()

short_names = [_short(n) for n in names]

# ── Style ─────────────────────────────────────────────────────────────────
style_map = {
    "Multi-FR (Jacquillat)":         dict(marker="s",  color="#1f77b4", ms=10),
    "Multi-FR-Detour (Jacquillat)":  dict(marker="D",  color="#2ca02c", ms=10),
    "SP-Lit (Carlsson)":             dict(marker="^",  color="#ff7f0e", ms=11),
    "TP-Lit (Liu)":                  dict(marker="v",  color="#d62728", ms=11),
    "Joint-Nom":                     dict(marker="o",  color="#9467bd", ms=10),
    "Joint-DRO":                     dict(marker="p",  color="#8c564b", ms=10),
    "VCC":                           dict(marker="*",  color="#e377c2", ms=13),
    "OD":                            dict(marker="X",  color="#7f7f7f", ms=10),
}
default_sty = dict(marker="o", color="gray", ms=9)

# ── Estimate avg total_prob for iso-lines ─────────────────────────────────
total_probs = []
for r in results:
    ut = r["user"]["total_user_time_h"]
    uc = r["user"]["user_cost"]
    if ut > 1e-9 and wr > 0:
        total_probs.append(uc / (wr * ut))
avg_prob = np.mean(total_probs) if total_probs else 1.0

# ── Label offsets (dx, dy) in points ──────────────────────────────────────
# Tuned for log-x single panel.  Positive dx = right, positive dy = up.
offsets = {
    "Multi-FR":  (8,  10),
    "FR-Detour": (8,  -16),
    "VCC":       (8,  10),
    "Joint-Nom": (-8, 10),
    "Joint-DRO": (8,  10),
    "SP-Lit":    (8,  -14),
    "TP-Lit":    (8,  10),
    "OD":        (-8, 10),
}

# ── Figure ────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))

# Iso-cost curves: total_cost = provider + wr * user_time * prob = C
# => user_time = (C - provider) / (wr * prob)
pc_line = np.geomspace(0.5, 500, 500)
non_od = [i for i in range(len(names)) if "OD" not in names[i].upper()]
tc_vals = sorted(total_cost[non_od])
iso_levels = np.linspace(tc_vals[0] * 0.7, tc_vals[-1] * 1.2, 5)
for C in iso_levels:
    ut_line = (C - pc_line) / (wr * avg_prob)
    mask = (ut_line > -0.05) & (ut_line < 2.5) & (pc_line < 300)
    if not mask.any():
        continue
    ax.plot(pc_line[mask], ut_line[mask], "-", color="#e8e8e8",
            linewidth=0.7, zorder=0)
    # Label near the right end of visible segment
    vis = np.where(mask)[0]
    li = vis[-1]
    if ut_line[li] > 0.05:
        ax.text(pc_line[li], ut_line[li] + 0.03, f"TC={C:.0f}",
                fontsize=5.5, color="#c0c0c0", ha="right", va="bottom")

# Plot points
for i in range(len(names)):
    s = style_map.get(names[i], default_sty)
    ax.plot(provider_cost[i], user_time[i], s["marker"],
            color=s["color"], markersize=s["ms"],
            markeredgecolor="white", markeredgewidth=0.6, zorder=5)

# Annotate
for i in range(len(names)):
    sn = short_names[i]
    s = style_map.get(names[i], default_sty)
    dx, dy = offsets.get(sn, (8, 8))
    ha = "left" if dx >= 0 else "right"
    va = "bottom" if dy >= 0 else "top"
    label = f"{sn} (TC={total_cost[i]:.1f})"
    ax.annotate(
        label, (provider_cost[i], user_time[i]),
        xytext=(dx, dy), textcoords="offset points",
        fontsize=7.5, color=s["color"], fontweight="semibold",
        ha=ha, va=va,
        arrowprops=dict(arrowstyle="-", color="#cccccc", lw=0.3,
                        shrinkA=0, shrinkB=4),
    )

# Axes
ax.set_xscale("log")
ax.set_xlim(max(0.5, provider_cost.min() * 0.5), provider_cost.max() * 1.5)
ax.set_ylim(-0.02, max(user_time) * 1.15)
ax.set_xlabel("Provider cost rate", fontsize=12)
ax.set_ylabel("User time (hours per rider)", fontsize=12)
ax.grid(True, alpha=0.15, linestyle="--", which="both")

# Legend
handles = []
labels = []
for name in names:
    s = style_map.get(name, default_sty)
    h, = ax.plot([], [], s["marker"], color=s["color"], markersize=7,
                 markeredgecolor="white", markeredgewidth=0.4)
    handles.append(h)
    labels.append(name)
ax.legend(handles, labels, fontsize=7, loc="upper left",
          framealpha=0.92, borderpad=0.6, handletextpad=0.4)

# Title + experiment info
odd_label = "on" if meta.get("use_odd", False) else "off"
info = (f"K={meta.get('districts', '?')}, "
        f"\u039b={meta.get('Lambda', '?')}, "
        f"wr={meta.get('wr', '?')}, wv={meta.get('wv', '?')}, "
        f"ODD={odd_label}")
ax.set_title(f"User vs Provider Performance by Service Mode\n"
             f"{info}", fontsize=12, fontweight="bold", pad=12)

fig.tight_layout()
out_dir = "results/baseline_comparison/figures"
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "scatter_user_provider.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
