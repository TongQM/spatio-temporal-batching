"""Scatter plot: user performance vs provider performance across service modes."""
import json
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Load results ──────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    path = sys.argv[1]
else:
    import glob, os
    dirs = sorted(glob.glob("results/baseline_comparison/synthetic_*"))
    if not dirs:
        sys.exit("No results found")
    path = os.path.join(dirs[-1], "results.json")

with open(path) as f:
    data = json.load(f)

results = data["results"]

# ── Extract data ──────────────────────────────────────────────────────────
names = [r["name"] for r in results]
user_cost = [r["user"]["total_user_time_h"] for r in results]
provider_cost = [r["provider"]["provider_cost"] for r in results]

# ── Style: markers & colours per method ───────────────────────────────────
style = {
    "Multi-FR (Jacquillat)":         dict(marker="s", color="#1f77b4"),
    "Multi-FR-Detour (Jacquillat)":  dict(marker="D", color="#2ca02c"),
    "SP-Lit (Carlsson)":             dict(marker="^", color="#ff7f0e"),
    "TP-Lit (Liu)":                  dict(marker="v", color="#d62728"),
    "Joint-Nom":                     dict(marker="o", color="#9467bd"),
    "Joint-DRO":                     dict(marker="p", color="#8c564b"),
    "VCC":                           dict(marker="*", color="#e377c2"),
    "OD":                            dict(marker="X", color="#7f7f7f"),
}

# ── Plot ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

for name, uc, pc in zip(names, user_cost, provider_cost):
    s = style.get(name, dict(marker="o", color="gray"))
    ax.scatter(pc, uc, s=120, zorder=3, label=name, **s)

# Annotate each point (offset to avoid overlap)
for name, uc, pc in zip(names, user_cost, provider_cost):
    # Short label
    short = name.split("(")[0].strip()
    ax.annotate(short, (pc, uc),
                textcoords="offset points", xytext=(8, 5),
                fontsize=8, alpha=0.85)

ax.set_xlabel("Provider cost (per period)", fontsize=12)
ax.set_ylabel("User time (hours per rider)", fontsize=12)
ax.set_title("User vs Provider Performance by Service Mode", fontsize=13)
ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
ax.grid(True, alpha=0.3)

# Log scale helps since OD is an extreme outlier
ax.set_xscale("log")
ax.set_xlim(left=1)

fig.tight_layout()
import os
out_dir = "results/baseline_comparison/figures"
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, "scatter_user_provider.png")
fig.savefig(out, dpi=150)
print(f"Saved: {out}")
