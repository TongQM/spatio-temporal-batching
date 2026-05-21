"""Plot the wr trajectory at Pittsburgh Lambda=150 for Joint-Nom, Joint-DRO,
Multi-FR, extended to wr in {30, 100} to test whether high wr closes the
gap to Multi-FR."""
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
from scripts.experiments.baseline_cloud_sweep import SERVICE_STYLES


# 1. Load existing wr ∈ {1, 3, 10} from main Pittsburgh aggregate
main = pd.read_csv("results/baseline_clouds/cities_road/pittsburgh/aggregate_summary.csv")
main_l150 = main[(main["Lambda"] == 150) & (main["baseline"].isin(
    ["Joint-Nom", "Joint-DRO", "Multi-FR (Jacquillat)"]))].copy()

# 2. Load extended wr ∈ {30, 100} rows from per-setting JSONs
ext_rows = []
for p in sorted(Path("results/baseline_clouds/cities_road/pittsburgh_wr_extension").glob("*/setting_*.json")):
    d = json.load(open(p))
    r = d.get("row", {})
    if r:
        ext_rows.append(r)
ext = pd.DataFrame(ext_rows)

# 3. Merge
combined = pd.concat([main_l150, ext], ignore_index=True)
combined = combined.sort_values(["baseline", "wr"]).reset_index(drop=True)

print("=== Combined wr trajectory (Pittsburgh Λ=150) ===")
print(combined[["baseline", "wr", "Provider cost", "User Total (h)",
                "Wait (h)", "Raw fleet"]].to_string(index=False))

# 4. Plot
fig, ax = plt.subplots(figsize=(10, 7))
for bn in ["Joint-Nom", "Joint-DRO", "Multi-FR (Jacquillat)"]:
    g = combined[combined["baseline"] == bn].sort_values("wr")
    st = SERVICE_STYLES.get(bn, {"color": "#333", "marker": "o"})
    # Exclude the (0,0) degenerate Multi-FR wr=100 point from the line, but mark separately
    valid = (g["Provider cost"] > 0) & (g["User Total (h)"] > 0)
    g_valid = g[valid]
    g_invalid = g[~valid]
    ax.plot(g_valid["Provider cost"], g_valid["User Total (h)"], "-",
            color=st["color"], alpha=0.4, linewidth=1.5, zorder=2)
    for _, row in g_valid.iterrows():
        ax.scatter(row["Provider cost"], row["User Total (h)"],
                   s=80, color=st["color"], marker=st["marker"],
                   edgecolors="white", linewidths=0.8, zorder=4)
        ax.annotate(f" wr={row['wr']:g}",
                    (row["Provider cost"], row["User Total (h)"]),
                    fontsize=8, color=st["color"],
                    xytext=(6, 0), textcoords="offset points")
    # Mark degenerate points with an X
    for _, row in g_invalid.iterrows():
        ax.scatter([1.5], [0.02], s=120, marker="X", color=st["color"],
                   edgecolors="black", linewidths=1.0, zorder=5)
        ax.annotate(f"{bn} wr={row['wr']:g}: DEGENERATE (fleet=0)",
                    (1.5, 0.02), fontsize=8, color=st["color"],
                    xytext=(10, 0), textcoords="offset points")

# Legend
from matplotlib.lines import Line2D
handles = []
for bn in ["Joint-Nom", "Joint-DRO", "Multi-FR (Jacquillat)"]:
    st = SERVICE_STYLES.get(bn, {"color": "#333", "marker": "o"})
    handles.append(Line2D([0], [0], marker=st["marker"], color="w",
                          markerfacecolor=st["color"],
                          markeredgecolor=st["color"], markersize=9, label=bn))
ax.legend(handles=handles, loc="upper right", fontsize=10, framealpha=0.92)

ax.set_xscale("log")
ax.set_xlabel("Provider cost rate", fontsize=11)
ax.set_ylabel("User time (h per rider)", fontsize=11)
ax.set_title("Extended-$w_r$ trajectory on Pittsburgh ($\\Lambda$=150)\n"
             "Joint-Nom/DRO keep responding to $w_r$; Multi-FR locked at $T_{\\max}$ then degenerates",
             fontsize=11, fontweight="bold")
ax.grid(True, alpha=0.2, ls="--", which="both")
ax.set_xlim(left=10)
ax.set_ylim(bottom=0)

out = "docs/figures/external baselines/cloud_pittsburgh_wr_extension.png"
fig.tight_layout()
fig.savefig(out, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out}")
