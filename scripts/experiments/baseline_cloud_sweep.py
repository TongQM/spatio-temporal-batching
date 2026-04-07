#!/usr/bin/env python3
"""Sweep baseline hyperparameters and plot service-mode performance clouds.

This script reuses the existing baseline-comparison evaluation pipeline so each
service is still evaluated with its own intended design/simulation logic.

Outputs
-------
results/baseline_clouds/<experiment_name>/
  aggregate_results.json
  aggregate_summary.csv
  scatter_user_provider_cloud.png
  <baseline_slug>/
    settings.json
    summary.csv
    setting_01.json
    ...
    setting_10.json

Each per-setting JSON stores the hyperparameter setting, the common experiment
parameters, and the reported EvaluationResult for that induced baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from scripts.experiments.baseline_comparison import (
    build_geodata,
    build_omega_dict,
    build_prob_dict,
    build_road_network,
    identity_J,
    run_baseline,
    zero_J,
)
from lib.baselines import (
    CarlssonPartition,
    FullyOD,
    FRDetour,
    JointDRO,
    JointNom,
    TemporalOnly,
    VCC,
)


SERVICE_STYLES = {
    "Multi-FR (Jacquillat)": dict(color="#1f77b4", marker="s"),
    "Multi-FR-Detour (Jacquillat)": dict(color="#2ca02c", marker="D"),
    "SP-Lit (Carlsson)": dict(color="#ff7f0e", marker="^"),
    "TP-Lit (Liu)": dict(color="#d62728", marker="v"),
    "Joint-Nom": dict(color="#9467bd", marker="o"),
    "Joint-DRO": dict(color="#8c564b", marker="p"),
    "VCC": dict(color="#e377c2", marker="*"),
    "OD": dict(color="#7f7f7f", marker="X"),
}


def _slugify(name: str) -> str:
    out = []
    for ch in name.lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "_":
            out.append("_")
    return "".join(out).strip("_")


def _label_value(value: Any) -> str:
    if isinstance(value, float):
        if value == 0.0:
            return "0"
        if abs(value) >= 1e-2 and abs(value) < 1e3:
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return f"{value:.2e}"
    return str(value)


def _primary_settings(primary_name: str, primary_values: List[Any]) -> List[Dict[str, Any]]:
    return [
        {
            primary_name: primary_value,
            "setting_label": f"{primary_name}={_label_value(primary_value)}",
        }
        for primary_value in primary_values
    ]


def _cartesian_settings(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(param_grid.keys())
    settings: List[Dict[str, Any]] = []
    for combo in product(*(param_grid[key] for key in keys)):
        setting = {key: value for key, value in zip(keys, combo)}
        setting["setting_label"] = ", ".join(
            f"{key}={_label_value(value)}" for key, value in zip(keys, combo)
        )
        settings.append(setting)
    return settings


def _spread_int_values(max_value: int, count: int) -> List[int]:
    if max_value <= 0:
        return [1]
    if max_value <= count:
        return list(range(1, max_value + 1))
    return sorted(set(int(round(v)) for v in np.linspace(1, max_value, count)))


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Monotone chain convex hull for 2D points."""
    if len(points) <= 1:
        return points
    pts = np.unique(points, axis=0)
    if len(pts) <= 2:
        return pts

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.array(lower[:-1] + upper[:-1], dtype=float)
    return hull


def _filter_cloud_outliers(group: pd.DataFrame) -> pd.DataFrame:
    """Drop extreme cloud outliers per baseline for plotting only."""
    if len(group) < 5:
        return group

    x = np.log10(group["Provider cost"].to_numpy(dtype=float))
    y = group["User Total (h)"].to_numpy(dtype=float)

    def mod_z(arr: np.ndarray) -> np.ndarray:
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        if mad < 1e-12:
            return np.zeros_like(arr)
        return 0.6745 * (arr - med) / mad

    zx = np.abs(mod_z(x))
    zy = np.abs(mod_z(y))
    keep = (zx <= 2.5) & (zy <= 2.5)
    if keep.sum() >= max(3, len(group) - 2):
        return group.loc[group.index[keep]]

    # Fallback: trim the most extreme high-user-time point if it is clearly
    # separated from the rest. This keeps a single runaway point from
    # dominating the cloud hull.
    y_vals = group["User Total (h)"].to_numpy(dtype=float)
    order = np.argsort(y_vals)
    if len(order) >= 4:
        y_sorted = y_vals[order]
        top_gap = y_sorted[-1] - y_sorted[-2]
        baseline_gap = y_sorted[-2] - y_sorted[-3]
        if top_gap > max(0.15, 2.0 * baseline_gap):
            return group.drop(group.index[order[-1]])
    return group


def _result_row(
    baseline_name: str,
    setting_index: int,
    lambda_value: float,
    wr_value: float,
    setting: Dict[str, Any],
    result,
) -> Dict[str, Any]:
    row = {
        "baseline": baseline_name,
        "setting_index": setting_index,
        "setting_label": setting["setting_label"],
        "Lambda": lambda_value,
        "wr": wr_value,
        "Wait (h)": result.avg_wait_time,
        "In-vehicle (h)": result.avg_invehicle_time,
        "Walk (h)": result.avg_walk_time,
        "User Total (h)": result.total_user_time,
        "Linehaul": result.linehaul_cost,
        "In-district": result.in_district_cost,
        "Total travel": result.total_travel_cost,
        "Fleet": result.fleet_size,
        "Dispatch interval (h)": result.avg_dispatch_interval,
        "ODD cost": result.odd_cost,
        "Provider cost": result.provider_cost,
        "User cost": result.user_cost,
        "Total cost": result.total_cost,
    }
    for key, value in setting.items():
        if key == "setting_label":
            continue
        row[key] = value
    return row


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def _build_output_dir(args: argparse.Namespace) -> Path:
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    descriptor = f"grid{args.grid_size}x{args.grid_size}" if args.mode == "synthetic" else "real"
    folder = (
        f"{args.mode}_{descriptor}"
        f"_K{args.districts}"
        f"_L{int(args.Lambda)}"
        f"_wr{args.wr}_wv{args.wv}"
        f"_cloud_{timestamp}"
    )
    out_dir = Path("results") / "baseline_clouds" / folder
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--grid_size", type=int, default=5)
    parser.add_argument("--districts", type=int, default=3)
    parser.add_argument("--shapefile", type=str, default=None)
    parser.add_argument("--geoid_list", type=str, default=None)
    parser.add_argument("--bbox", type=float, nargs=4, metavar=("NORTH", "SOUTH", "EAST", "WEST"))
    parser.add_argument("--road_network", action="store_true")
    parser.add_argument("--Lambda", type=float, default=300.0)
    parser.add_argument("--wr", type=float, default=1.0)
    parser.add_argument("--wv", type=float, default=10.0)
    parser.add_argument("--fleet_cost_rate", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=1e-2)
    parser.add_argument("--use_odd", action="store_true")
    parser.add_argument("--rs_iters", type=int, default=100)
    parser.add_argument("--tp_candidates", type=int, default=25)
    parser.add_argument("--tp_region_km", type=float, default=4.5)
    parser.add_argument("--tp_region_low", type=float, default=-0.5)
    parser.add_argument("--services", nargs="*", default=None,
                        choices=[
                            "multi_fr", "multi_fr_detour",
                            "sp_lit", "tp_lit", "joint_nom", "joint_dro", "vcc", "od",
                        ])
    parser.add_argument("--settings_per_baseline", type=int, default=20)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fr_delta", type=float, default=1.0)
    return parser.parse_args()


def _build_data_context(args: argparse.Namespace, use_odd: bool) -> Dict[str, Any]:
    road_network = build_road_network(args) if args.mode == "real" else None

    grid_geodata = build_geodata(args)
    grid_prob = build_prob_dict(grid_geodata)
    grid_omega = build_omega_dict(grid_geodata, use_odd=use_odd)

    return {
        "road_network": road_network,
        "fr": (grid_geodata, grid_prob, grid_omega),
        "grid": (grid_geodata, grid_prob, grid_omega),
        "tp": (grid_geodata, grid_prob, grid_omega),
        "od": (grid_geodata, grid_prob, grid_omega),
    }


def _build_service_specs(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    max_joint_k = min(20, args.grid_size * args.grid_size) if args.mode == "synthetic" else 20
    joint_k_values = list(range(1, max_joint_k + 1))
    paired_k_values = _spread_int_values(max_joint_k, 4)
    fr_fleet_values = list(range(1, 21))
    fr_detour_fleet_values = _spread_int_values(20, 5)
    sp_t_values = [0.10, 0.20, 0.30, 0.50, 0.80]
    tp_t_values = [0.05, 0.10, 0.15, 0.25, 0.40]
    tp_max_fleet_values = [20, 40, 80, 120]
    vcc_fleet_values = [5, 10, 20, 35, 50]
    vcc_walk_values = [0.10, 0.20, 0.30, 0.50]
    dro_eps_values = [1e-3, 3e-3, 1e-2, 3e-2]
    fr_delta_values = [0.25, 0.50, 1.00, 1.50]
    od_tmin_values = list(np.geomspace(1.0 / 300.0, 1.0 / 30.0, 20))

    specs: Dict[str, Dict[str, Any]] = {
        "multi_fr": {
            "display_name": "Multi-FR (Jacquillat)",
            "data_group": "fr",
            "base_lambda": args.Lambda,
            "base_wr": args.wr,
            "varying_note": "vary route/fleet budget (delta=0)",
            "settings": _primary_settings(
                "max_fleet", fr_fleet_values
            ),
            "make_method": lambda s: FRDetour(
                delta=0.0,
                max_iters=args.rs_iters,
                max_fleet=float(s["max_fleet"]),
                name="Multi-FR (Jacquillat)",
            ),
        },
        "multi_fr_detour": {
            "display_name": "Multi-FR-Detour (Jacquillat)",
            "data_group": "fr",
            "base_lambda": args.Lambda,
            "base_wr": args.wr,
            "varying_note": "vary route/fleet budget and detour allowance",
            "settings": _cartesian_settings(
                {
                    "max_fleet": fr_detour_fleet_values,
                    "delta": fr_delta_values,
                }
            ),
            "make_method": lambda s: FRDetour(
                delta=float(s["delta"]),
                max_iters=args.rs_iters,
                max_fleet=float(s["max_fleet"]),
            ),
        },
        "sp_lit": {
            "display_name": "SP-Lit (Carlsson)",
            "data_group": "grid",
            "base_lambda": args.Lambda,
            "base_wr": args.wr,
            "varying_note": "vary T_fixed and num_districts",
            "settings": _cartesian_settings(
                {
                    "num_districts": paired_k_values,
                    "T_fixed": sp_t_values,
                }
            ),
            "make_method": lambda s: CarlssonPartition(T_fixed=float(s["T_fixed"])),
        },
        "tp_lit": {
            "display_name": "TP-Lit (Liu)",
            "data_group": "tp",
            "base_lambda": args.Lambda,
            "base_wr": args.wr,
            "varying_note": "vary T_fixed and max_fleet",
            "settings": _cartesian_settings(
                {
                    "T_fixed": tp_t_values,
                    "max_fleet": tp_max_fleet_values,
                }
            ),
            "make_method": lambda s: TemporalOnly(
                T_fixed=float(s["T_fixed"]),
                max_fleet=int(s["max_fleet"]),
                n_candidates=args.tp_candidates,
                region_km=args.tp_region_km,
                region_low=args.tp_region_low,
            ),
        },
        "joint_nom": {
            "display_name": "Joint-Nom",
            "data_group": "grid",
            "base_lambda": args.Lambda,
            "base_wr": args.wr,
            "varying_note": "vary num_districts",
            "settings": _primary_settings(
                "num_districts", joint_k_values
            ),
            "make_method": lambda s: JointNom(
                max_iters=args.rs_iters,
            ),
        },
        "joint_dro": {
            "display_name": "Joint-DRO",
            "data_group": "grid",
            "base_lambda": args.Lambda,
            "base_wr": args.wr,
            "varying_note": "vary num_districts and epsilon",
            "settings": _cartesian_settings(
                {
                    "num_districts": [v for v in _spread_int_values(max_joint_k, 5) if v >= 1],
                    "epsilon": dro_eps_values,
                }
            ),
            "make_method": lambda s: JointDRO(
                epsilon=float(s["epsilon"]),
                max_iters=args.rs_iters,
            ),
        },
        "vcc": {
            "display_name": "VCC",
            "data_group": "od",
            "base_lambda": args.Lambda,
            "base_wr": args.wr,
            "varying_note": "vary fleet_size and max_walk_km",
            "settings": _cartesian_settings(
                {
                    "fleet_size": vcc_fleet_values,
                    "max_walk_km": vcc_walk_values,
                }
            ),
            "make_method": lambda s: VCC(
                fleet_size=int(s["fleet_size"]),
                max_walk_km=float(s["max_walk_km"]),
            ),
        },
        "od": {
            "display_name": "OD",
            "data_group": "od",
            "base_lambda": args.Lambda,
            "base_wr": args.wr,
            "varying_note": "vary T_min",
            "settings": _primary_settings(
                "T_min", od_tmin_values
            ),
            "make_method": lambda s: FullyOD(T_min=float(s["T_min"])),
        },
    }

    selected = args.services or [
        "multi_fr", "multi_fr_detour",
        "sp_lit", "tp_lit", "joint_nom", "joint_dro", "vcc", "od"
    ]
    for key in selected:
        specs[key]["settings"] = specs[key]["settings"][:args.settings_per_baseline]
    return {key: specs[key] for key in selected}


def _plot_cloud(df: pd.DataFrame, out_path: Path, meta: Dict[str, Any]) -> None:
    fig = plt.figure(figsize=(10.5, 8.8))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[12, 1.8], hspace=0.18)
    ax = fig.add_subplot(gs[0, 0])
    notes_ax = fig.add_subplot(gs[1, 0])
    notes_ax.axis("off")

    for baseline_name, group in df.groupby("baseline", sort=False):
        group = _filter_cloud_outliers(group)
        style = SERVICE_STYLES.get(baseline_name, {"color": "#333333", "marker": "o"})
        x = group["Provider cost"].to_numpy(dtype=float)
        y = group["User Total (h)"].to_numpy(dtype=float)
        valid = x > 0
        x = x[valid]
        y = y[valid]
        if len(x) >= 3:
            hull_xy = np.column_stack([np.log10(x), y])
            hull = _convex_hull(hull_xy)
            if len(hull) >= 3:
                ax.fill(
                    10 ** hull[:, 0],
                    hull[:, 1],
                    facecolor=style["color"],
                    edgecolor=style["color"],
                    alpha=0.08,
                    linewidth=1.0,
                    zorder=1,
                )
        elif len(x) == 2:
            ax.plot(x, y, "-", color=style["color"], alpha=0.20, linewidth=2.0, zorder=1)

        ax.scatter(
            group["Provider cost"],
            group["User Total (h)"],
            s=52,
            alpha=0.78,
            color=style["color"],
            marker=style["marker"],
            edgecolors="white",
            linewidths=0.6,
            label=baseline_name,
            zorder=3,
        )
        cx = float(group["Provider cost"].median())
        cy = float(group["User Total (h)"].median())
        ax.annotate(
            baseline_name,
            (cx, cy),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            color=style["color"],
            fontweight="semibold",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Provider cost rate")
    ax.set_ylabel("User time (hours per rider)")
    ax.set_title(
        "Baseline Performance Clouds\n"
        f"base K={meta['districts']}, Λ={meta['Lambda']}, wr={meta['wr']}, wv={meta['wv']}",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.2, linestyle="--", which="both")
    ax.legend(loc="best", framealpha=0.92, fontsize=8)

    notes = [
        f"{name}: {meta['varying_notes'].get(name, 'vary setting')}"
        for name in meta["display_services"]
    ]
    footer = textwrap.fill(
        "Clouds use light convex hulls in log-provider-cost space. "
        + "Varying by baseline: " + " | ".join(notes),
        width=115,
        break_long_words=False,
        break_on_hyphens=False,
    )
    notes_ax.text(
        0.0, 0.95, footer,
        ha="left", va="top", fontsize=8.2, color="#444444",
        transform=notes_ax.transAxes,
    )
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    out_dir = _build_output_dir(args)
    J_function = identity_J if args.use_odd else zero_J
    data_ctx = _build_data_context(args, use_odd=args.use_odd)
    service_specs = _build_service_specs(args)

    aggregate_rows: List[Dict[str, Any]] = []
    experiment_meta = {
        "timestamp": datetime.now().isoformat(),
        "mode": args.mode,
        "districts": args.districts,
        "grid_size": args.grid_size if args.mode == "synthetic" else None,
        "Lambda": args.Lambda,
        "wr": args.wr,
        "wv": args.wv,
        "fleet_cost_rate": args.fleet_cost_rate,
        "epsilon": args.epsilon,
        "use_odd": args.use_odd,
        "services": list(service_specs.keys()),
        "display_services": [spec["display_name"] for spec in service_specs.values()],
        "varying_notes": {
            spec["display_name"]: spec.get("varying_note", "vary setting")
            for spec in service_specs.values()
        },
        "settings_per_baseline": args.settings_per_baseline,
        "fixed_lambda": True,
        "tp_candidates": args.tp_candidates,
        "tp_region_km": args.tp_region_km,
        "tp_region_low": args.tp_region_low,
        "fr_delta": args.fr_delta,
    }

    _save_json(out_dir / "experiment.json", experiment_meta)

    for service_key, spec in service_specs.items():
        baseline_dir = out_dir / _slugify(spec["display_name"])
        baseline_dir.mkdir(parents=True, exist_ok=True)
        _save_json(baseline_dir / "settings.json", {"settings": spec["settings"]})

        data_group = spec["data_group"]
        geodata, prob_dict, omega_dict = data_ctx[data_group]
        baseline_rows: List[Dict[str, Any]] = []

        print(f"\n=== {spec['display_name']} — {len(spec['settings'])} settings ===")
        for setting_index, setting in enumerate(spec["settings"], start=1):
            lambda_value = float(spec["base_lambda"])
            wr_value = spec["base_wr"]
            setting_path = baseline_dir / f"setting_{setting_index:02d}.json"

            if args.resume and setting_path.exists():
                with setting_path.open() as f:
                    payload = json.load(f)
                baseline_rows.append(payload["row"])
                aggregate_rows.append(payload["row"])
                print(f"  [{setting_index:02d}] resumed {setting['setting_label']}")
                continue

            print(f"  [{setting_index:02d}] {setting['setting_label']}")
            method = spec["make_method"](setting)
            num_districts = int(setting.get("num_districts", args.districts))
            result = run_baseline(
                method=method,
                name=spec["display_name"],
                geodata=geodata,
                prob_dict=prob_dict,
                Omega_dict=omega_dict,
                J_function=J_function,
                num_districts=num_districts,
                Lambda=lambda_value,
                wr=wr_value,
                wv=args.wv,
                road_network=data_ctx["road_network"],
                fleet_cost_rate=args.fleet_cost_rate,
            )
            if result is None:
                continue

            row = _result_row(
                baseline_name=spec["display_name"],
                setting_index=setting_index,
                lambda_value=lambda_value,
                wr_value=wr_value,
                setting=setting,
                result=result,
            )
            payload = {
                "experiment": experiment_meta,
                "baseline": spec["display_name"],
                "service_key": service_key,
                "setting_index": setting_index,
                "setting": setting,
                "row": row,
                "result": result.to_dict(),
            }
            _save_json(setting_path, payload)
            baseline_rows.append(row)
            aggregate_rows.append(row)

        if baseline_rows:
            pd.DataFrame(baseline_rows).to_csv(baseline_dir / "summary.csv", index=False)

    if not aggregate_rows:
        raise RuntimeError("No sweep results were produced.")

    aggregate_df = pd.DataFrame(aggregate_rows)
    aggregate_df.to_csv(out_dir / "aggregate_summary.csv", index=False)
    _save_json(
        out_dir / "aggregate_results.json",
        {"experiment": experiment_meta, "rows": aggregate_rows},
    )
    _plot_cloud(aggregate_df, out_dir / "scatter_user_provider_cloud.png", experiment_meta)

    print("\nSaved baseline cloud outputs to:")
    print(f"  {out_dir}")
    print("Artifacts:")
    print("  experiment.json")
    print("  aggregate_results.json")
    print("  aggregate_summary.csv")
    print("  scatter_user_provider_cloud.png")


if __name__ == "__main__":
    main()
