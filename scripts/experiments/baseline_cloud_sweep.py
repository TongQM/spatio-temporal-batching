#!/usr/bin/env python3
"""Baseline comparison sweep — two views of the service-mode performance cloud.

This script reuses the baseline-comparison evaluation pipeline so each service
is still evaluated with its own intended design/simulation logic. It supports
two sweep views:

Regime view (``--view regime``, the default, main comparison)
--------------------------------------------------------------
Sweep the shared system inputs ``(Λ, wr)`` on a small grid and let each
baseline re-optimize at its library defaults. Every cloud point means
"method M's best design at system regime ``(Λ, wr)``" — same axes, same
interpretation, directly comparable across baselines.

Hyperparameter view (``--view hyper``, diagnostic)
--------------------------------------------------
Sweep per-baseline hyperparameters (fleet for FR, T for SP-Lit, K for Joint,
etc.). Useful to see each method's own Pareto shape, but NOT apples-to-apples
across baselines: different knobs span different axes of the cost plane.

Shared parameters (used by both views)
--------------------------------------
``Λ`` — total arrival rate [passengers / hour]
    Poisson arrival rate summed over the whole service area. Spatial
    distribution is fixed by ``prob_dict`` so the per-block rate is
    ``Λ · p_b``. A system input, not a design choice. Every baseline
    receives the same ``Λ`` and must design a service that handles it.

``wr`` — rider time cost weight [cost / rider-hour]
    Weight on total user time (``wait + in-vehicle + walk``) in the design
    objective and in the reported ``EvaluationResult.user_cost =
    wr · total_user_time_per_rider · expected demand``.

``wv`` — vehicle operating cost weight [cost / vehicle-km]
    Weight on provider travel distance (linehaul + in-district + detour).
    **Pinned to 1.0 in regime view** because ``evaluate_design`` /
    ``simulate_design`` build ``provider_cost`` in km/hour without multiplying
    by ``wv``, while the optimization side (e.g. ``_newton_T_star``, the
    FR-Detour master) uses ``wv`` explicitly. Fixing ``wv = 1`` keeps the two
    sides on the same scale without touching the shared evaluation helpers.

Regime-responsiveness of each baseline at design time
-----------------------------------------------------
Not every baseline re-optimizes its design when ``(Λ, wr)`` changes:

  Multi-FR / Multi-FR-Detour   : Λ ✓   wr/wv ✓   (Benders master)
  SP-Lit (Carlsson)            : Λ ✗   wr/wv ✗   (uses fixed T_fixed)
  TP-Lit (Liu)                 : Λ ✓   wr/wv ~   (Λ·T drives fleet sizing)
  Joint-Nom / Joint-DRO        : Λ ✓   wr/wv ✓   (Newton T* uses wr/wv)
  VCC                          : Λ ✓   wr/wv ✗   (Λ drives planning horizon)
  OD (FullyOD)                 : Λ ✗   wr/wv ✗   (fixed T_min)

For regime-insensitive designs the cloud shows "how does a fixed design
behave as the world around it changes" — a legitimate and informative data
point, not a bug.

Outputs
-------
``results/baseline_clouds/<view>_<descriptor>/``
    ``experiment.json``           — sweep metadata including view, grid, pinned wv
    ``aggregate_results.json``    — all rows + metadata
    ``aggregate_summary.csv``     — flat CSV with Lambda, wr columns populated
    ``scatter_user_provider_cloud.png``
    ``<baseline_slug>/``
        ``settings.json``         — the sweep settings for this baseline
        ``summary.csv``
        ``setting_01.json`` … setting_NN.json   — per-setting cache (resumable)

See ``.claude/rules/baseline_sweep.md`` for the full methodology reference.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import textwrap
import traceback
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


_WORKER_ARGS: argparse.Namespace | None = None
_WORKER_DATA_CTX: Dict[str, Any] | None = None
_WORKER_SERVICE_SPECS: Dict[str, Dict[str, Any]] | None = None
_WORKER_J_FUNCTION = None


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
        "Raw fleet": result.raw_fleet_size,
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


def _worker_init(args_dict: Dict[str, Any]) -> None:
    global _WORKER_ARGS, _WORKER_DATA_CTX, _WORKER_SERVICE_SPECS, _WORKER_J_FUNCTION
    _WORKER_ARGS = argparse.Namespace(**args_dict)
    _WORKER_J_FUNCTION = identity_J if _WORKER_ARGS.use_odd else zero_J
    _WORKER_DATA_CTX = _build_data_context(_WORKER_ARGS, use_odd=_WORKER_ARGS.use_odd)
    _WORKER_SERVICE_SPECS = _build_specs_for_view(_WORKER_ARGS)


def _build_specs_for_view(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    """Dispatch to the regime or hyper spec builder based on `args.view`."""
    if getattr(args, "view", "regime") == "regime":
        return _build_regime_specs(args)
    return _build_service_specs(args)


def _worker_state(
    args_dict: Dict[str, Any],
) -> Tuple[argparse.Namespace, Dict[str, Any], Dict[str, Dict[str, Any]], Any]:
    global _WORKER_ARGS, _WORKER_DATA_CTX, _WORKER_SERVICE_SPECS, _WORKER_J_FUNCTION
    if _WORKER_ARGS is None:
        _worker_init(args_dict)
    assert _WORKER_ARGS is not None
    assert _WORKER_DATA_CTX is not None
    assert _WORKER_SERVICE_SPECS is not None
    assert _WORKER_J_FUNCTION is not None
    return _WORKER_ARGS, _WORKER_DATA_CTX, _WORKER_SERVICE_SPECS, _WORKER_J_FUNCTION


def _run_setting_task(task: Dict[str, Any]) -> Dict[str, Any]:
    args, data_ctx, service_specs, J_function = _worker_state(task["args"])

    service_key = task["service_key"]
    setting_index = int(task["setting_index"])
    spec = service_specs[service_key]
    setting = spec["settings"][setting_index - 1]
    baseline_dir = Path(task["baseline_dir"])
    setting_path = baseline_dir / f"setting_{setting_index:02d}.json"

    if task["resume"] and setting_path.exists():
        with setting_path.open() as f:
            payload = json.load(f)
        return {
            "status": "resumed",
            "service_key": service_key,
            "baseline_name": spec["display_name"],
            "setting_index": setting_index,
            "setting_label": setting["setting_label"],
            "row": payload["row"],
        }

    geodata, prob_dict, omega_dict = data_ctx[spec["data_group"]]
    method = spec["make_method"](setting)
    # Per-setting Lambda / wr overrides let regime view sweep the shared
    # (Lambda, wr) grid through the same infrastructure as hyper view.
    lambda_value = float(setting.get("Lambda", spec["base_lambda"]))
    wr_value = float(setting.get("wr", spec["base_wr"]))
    result = run_baseline(
        method=method,
        name=spec["display_name"],
        geodata=geodata,
        prob_dict=prob_dict,
        Omega_dict=omega_dict,
        J_function=J_function,
        num_districts=int(setting.get("num_districts", args.districts)),
        Lambda=lambda_value,
        wr=wr_value,
        wv=args.wv,
        road_network=data_ctx["road_network"],
        fleet_cost_rate=args.fleet_cost_rate,
    )
    if result is None:
        return {
            "status": "skipped",
            "service_key": service_key,
            "baseline_name": spec["display_name"],
            "setting_index": setting_index,
            "setting_label": setting["setting_label"],
            "row": None,
        }

    row = _result_row(
        baseline_name=spec["display_name"],
        setting_index=setting_index,
        lambda_value=lambda_value,
        wr_value=wr_value,
        setting=setting,
        result=result,
    )
    payload = {
        "experiment": task["experiment_meta"],
        "baseline": spec["display_name"],
        "service_key": service_key,
        "setting_index": setting_index,
        "setting": setting,
        "row": row,
        "result": result.to_dict(),
    }
    _save_json(setting_path, payload)
    return {
        "status": "completed",
        "service_key": service_key,
        "baseline_name": spec["display_name"],
        "setting_index": setting_index,
        "setting_label": setting["setting_label"],
        "row": row,
    }


def _build_output_dir(args: argparse.Namespace) -> Path:
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    descriptor = f"grid{args.grid_size}x{args.grid_size}" if args.mode == "synthetic" else "real"
    folder = (
        f"{args.view}"
        f"_{args.mode}_{descriptor}"
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
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fr_delta", type=float, default=1.0)
    parser.add_argument(
        "--view",
        choices=["regime", "hyper"],
        default="regime",
        help=(
            "regime: sweep shared (Lambda, wr) grid at method defaults "
            "(main comparison view). "
            "hyper: sweep per-baseline hyperparameters (diagnostic)."
        ),
    )
    parser.add_argument(
        "--regime_lambdas",
        type=float,
        nargs="+",
        default=[50.0, 150.0, 300.0],
        help="Arrival rates (passengers/hour) for regime view.",
    )
    parser.add_argument(
        "--regime_wrs",
        type=float,
        nargs="+",
        default=[1.0, 3.0, 10.0],
        help="Rider-time weights for regime view. wv is pinned to 1.0.",
    )
    parser.add_argument(
        "--fr_t_max",
        type=float,
        default=3.0,
        help=(
            "Max dispatch interval (hours) in Multi-FR / Multi-FR-Detour's "
            "internal T_values grid. Default 3.0 extends the grid to "
            "[0.1, ..., 1.0, 1.5, 2.0, 3.0] so FR can explore longer "
            "headways comparable to partition methods' T*."
        ),
    )
    parser.add_argument(
        "--fr_t_dense",
        action="store_true",
        help=(
            "Use a denser T grid for Multi-FR / Multi-FR-Detour by inserting "
            "midpoints between every pair of adjacent values. Roughly doubles "
            "the number of grid points and lets FR approximate a continuous "
            "T* more precisely, at the cost of ~2x Benders runtime."
        ),
    )
    parser.add_argument(
        "--regime_fr_max_fleet",
        type=float,
        default=None,
        help=(
            "Override max_fleet for Multi-FR / Multi-FR-Detour in regime view. "
            "Default (None) lets FRDetour fall back to num_districts for "
            "fleet-parity with partition baselines. Set explicitly (e.g. 30) "
            "when comparing FR's routing flexibility under a larger fleet."
        ),
    )
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
            # epsilon (Wasserstein radius) is a robustness knob, not a
            # provider-vs-user Pareto knob — sweeping it mixes a different
            # effect into the cloud. Keep Joint-DRO at the CLI --epsilon and
            # only sweep num_districts here. Put epsilon in a dedicated
            # robustness plot instead.
            "varying_note": "vary num_districts (epsilon fixed at --epsilon; not a Pareto knob)",
            "settings": _primary_settings(
                "num_districts",
                [v for v in _spread_int_values(max_joint_k, 5) if v >= 1],
            ),
            "make_method": lambda s: JointDRO(
                epsilon=float(args.epsilon),
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


def _od_fleet_grid(Lambda: float) -> List[int]:
    """Lambda-scaled fleet-size candidates for OD's queueing optimization.

    Sizes the grid so the minimum vehicle count keeps utilisation below
    ~85% at the given arrival rate. Rough calculation:
        required_fleet ≈ ceil(Lambda · round_trip_h / 0.85)
    where round_trip ≈ 4 km at 30 km/h = 0.133 h. Candidates span from
    near-saturation (cheap, moderate queue) to large-fleet (near-zero queue).
    """
    round_trip_h = 4.0 / 30.0
    min_fleet = max(3, int(np.ceil(Lambda * round_trip_h / 0.85)))
    raw = [
        min_fleet,
        int(np.ceil(min_fleet * 1.25)),
        int(np.ceil(min_fleet * 1.5)),
        int(np.ceil(min_fleet * 2.0)),
        int(np.ceil(min_fleet * 3.0)),
    ]
    return sorted(set(raw))


def _regime_settings(lambdas: List[float], wrs: List[float]) -> List[Dict[str, Any]]:
    """Shared (Lambda, wr) grid reused across every baseline in regime view."""
    settings: List[Dict[str, Any]] = []
    for lam in lambdas:
        for wr in wrs:
            settings.append(
                {
                    "Lambda": float(lam),
                    "wr": float(wr),
                    "setting_label": f"Λ={_label_value(lam)}, wr={_label_value(wr)}",
                }
            )
    return settings


def _build_regime_specs(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    """Service specs for the regime view.

    Each baseline sweeps the same shared (Lambda, wr) grid at its own
    library defaults. `make_method` ignores the setting dict (Lambda / wr
    are applied by `_run_setting_task` as per-setting overrides passed into
    `run_baseline`). Every cloud point therefore represents "method M's best
    design at system regime (Λ, wr)" — same axes, same interpretation.
    """
    shared_settings = _regime_settings(args.regime_lambdas, args.regime_wrs)
    first_lambda = float(args.regime_lambdas[0]) if args.regime_lambdas else float(args.Lambda)
    first_wr = float(args.regime_wrs[0]) if args.regime_wrs else float(args.wr)
    regime_note = (
        "shared (Λ, wr) regime sweep at method library defaults; "
        "wv pinned to 1.0"
    )

    fr_max_fleet = (
        float(args.regime_fr_max_fleet)
        if getattr(args, "regime_fr_max_fleet", None) is not None
        else None
    )
    fr_fleet_note = (
        f" (max_fleet={fr_max_fleet:g})"
        if fr_max_fleet is not None
        else " (max_fleet falls back to num_districts)"
    )

    # Build extended T_values grid for FR so it can explore longer
    # dispatch intervals comparable to partition methods' Newton T*.
    # Denser grid in [0.1, 1.0] so FR can approximate Joint's continuous T*
    # (which lands near 0.11, 0.33, 0.88 in our regime sweep).
    fr_t_max = float(getattr(args, "fr_t_max", 3.0))
    base_t = [0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4,
              0.5, 0.625, 0.75, 0.875, 1.0]
    extended_t = [1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    fr_t_values = sorted(set(t for t in base_t + extended_t if t <= fr_t_max + 1e-9))

    if getattr(args, "fr_t_dense", False):
        # Insert midpoints between every adjacent pair → roughly doubles density.
        midpoints = [(fr_t_values[i] + fr_t_values[i + 1]) / 2.0
                     for i in range(len(fr_t_values) - 1)]
        fr_t_values = sorted(set(fr_t_values + [round(m, 6) for m in midpoints]))

    fr_t_note = f" (T_values up to {fr_t_max:g}h, {len(fr_t_values)} values)"

    specs: Dict[str, Dict[str, Any]] = {
        "multi_fr": {
            "display_name": "Multi-FR (Jacquillat)",
            "data_group": "fr",
            "base_lambda": first_lambda,
            "base_wr": first_wr,
            "varying_note": regime_note + fr_fleet_note + fr_t_note,
            "settings": list(shared_settings),
            "make_method": lambda s: FRDetour(
                delta=0.0,
                max_iters=args.rs_iters,
                max_fleet=fr_max_fleet,
                T_values=fr_t_values,
                name="Multi-FR (Jacquillat)",
            ),
        },
        "multi_fr_detour": {
            "display_name": "Multi-FR-Detour (Jacquillat)",
            "data_group": "fr",
            "base_lambda": first_lambda,
            "base_wr": first_wr,
            "varying_note": regime_note + fr_fleet_note + fr_t_note,
            "settings": list(shared_settings),
            "make_method": lambda s: FRDetour(
                delta=float(args.fr_delta),
                max_iters=args.rs_iters,
                max_fleet=fr_max_fleet,
                T_values=fr_t_values,
            ),
        },
        "sp_lit": {
            "display_name": "SP-Lit (Carlsson)",
            "data_group": "grid",
            "base_lambda": first_lambda,
            "base_wr": first_wr,
            "varying_note": regime_note + " (T* optimized per district via Newton, same as Joint)",
            "settings": list(shared_settings),
            "make_method": lambda s: CarlssonPartition(T_fixed=None),
        },
        "tp_lit": {
            "display_name": "TP-Lit (Liu)",
            "data_group": "tp",
            "base_lambda": first_lambda,
            "base_wr": first_wr,
            "varying_note": regime_note,
            "settings": list(shared_settings),
            "make_method": lambda s: TemporalOnly(
                n_candidates=args.tp_candidates,
                region_km=args.tp_region_km,
                region_low=args.tp_region_low,
            ),
        },
        "joint_nom": {
            "display_name": "Joint-Nom",
            "data_group": "grid",
            "base_lambda": first_lambda,
            "base_wr": first_wr,
            "varying_note": regime_note,
            "settings": list(shared_settings),
            "make_method": lambda s: JointNom(max_iters=args.rs_iters),
        },
        "joint_dro": {
            "display_name": "Joint-DRO",
            "data_group": "grid",
            "base_lambda": first_lambda,
            "base_wr": first_wr,
            "varying_note": regime_note + f" (epsilon={args.epsilon})",
            "settings": list(shared_settings),
            "make_method": lambda s: JointDRO(
                epsilon=float(args.epsilon),
                max_iters=args.rs_iters,
            ),
        },
        "vcc": {
            "display_name": "VCC",
            "data_group": "od",
            "base_lambda": first_lambda,
            "base_wr": first_wr,
            "varying_note": regime_note + " (VCC uses constructor fleet_size; design is regime-insensitive in wr)",
            "settings": list(shared_settings),
            "make_method": lambda s: VCC(fleet_size=30),
        },
        "od": {
            "display_name": "OD",
            "data_group": "od",
            "base_lambda": first_lambda,
            "base_wr": first_wr,
            "varying_note": regime_note + " (queueing, Lambda-scaled fleet grid; best picked per regime)",
            "settings": list(shared_settings),
            # Lambda-scaled fleet grid: minimum fleet sized to keep utilisation
            # below ~85% (round-trip 4 km @ 30 km/h → 0.133 h per trip).
            # Avoids the unrealistic "fleet=3 at Λ=300" case where OD queues
            # grow unboundedly.
            "make_method": lambda s: FullyOD(
                fleet_size_grid=_od_fleet_grid(float(s["Lambda"])),
            ),
        },
    }

    selected = args.services or [
        "multi_fr", "multi_fr_detour",
        "sp_lit", "tp_lit", "joint_nom", "joint_dro", "vcc", "od",
    ]
    return {key: specs[key] for key in selected if key in specs}


def _plot_cloud(df: pd.DataFrame, out_path: Path, meta: Dict[str, Any]) -> None:
    fig = plt.figure(figsize=(10.5, 8.8))
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[12, 1.8], hspace=0.18)
    ax = fig.add_subplot(gs[0, 0])
    notes_ax = fig.add_subplot(gs[1, 0])
    notes_ax.axis("off")

    is_regime = meta.get("view") == "regime"
    regime_lambdas = sorted(set(meta.get("regime_lambdas") or [])) if is_regime else []
    # Marker size per Lambda (small / medium / large) so within-cloud
    # structure is visible in regime view.
    size_for_lambda: Dict[float, float] = {}
    if regime_lambdas:
        min_s, max_s = 28.0, 110.0
        if len(regime_lambdas) == 1:
            size_for_lambda[regime_lambdas[0]] = 0.5 * (min_s + max_s)
        else:
            for i, lam in enumerate(regime_lambdas):
                frac = i / (len(regime_lambdas) - 1)
                size_for_lambda[float(lam)] = min_s + frac * (max_s - min_s)

    def _sizes_for_group(group: pd.DataFrame) -> np.ndarray:
        if not is_regime or "Lambda" not in group.columns or not size_for_lambda:
            return np.full(len(group), 52.0)
        return np.array(
            [size_for_lambda.get(float(lam), 52.0) for lam in group["Lambda"].to_numpy()],
            dtype=float,
        )

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
            s=_sizes_for_group(group),
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
    if is_regime:
        lambdas_txt = ", ".join(f"{l:g}" for l in regime_lambdas) if regime_lambdas else "-"
        wrs_txt = ", ".join(f"{w:g}" for w in (meta.get("regime_wrs") or [])) or "-"
        title = (
            "Baseline Performance Clouds — shared regime sweep\n"
            f"K={meta['districts']}, Λ∈{{{lambdas_txt}}}, wr∈{{{wrs_txt}}}, "
            f"wv={meta['wv']} (pinned); marker size ∝ Λ"
        )
    else:
        title = (
            "Baseline Performance Clouds — per-method hyperparameter sweep\n"
            f"base K={meta['districts']}, Λ={meta['Lambda']}, wr={meta['wr']}, wv={meta['wv']}"
        )
    ax.set_title(title, fontsize=12, fontweight="bold")
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

    # Regime view pins wv=1 so the optimization side (uses wv explicitly)
    # matches the evaluation side (builds provider_cost in km/hour with
    # implicit wv=1). See module docstring for the full rationale.
    if args.view == "regime" and args.wv != 1.0:
        print(
            f"[regime view] pinning wv=1.0 (was {args.wv}) so design-time and "
            f"evaluation-time provider costing are on the same scale."
        )
        args.wv = 1.0

    out_dir = _build_output_dir(args)
    service_specs = _build_specs_for_view(args)

    aggregate_rows: List[Dict[str, Any]] = []
    experiment_meta = {
        "timestamp": datetime.now().isoformat(),
        "view": args.view,
        "mode": args.mode,
        "districts": args.districts,
        "grid_size": args.grid_size if args.mode == "synthetic" else None,
        "Lambda": args.Lambda,
        "wr": args.wr,
        "wv": args.wv,
        "regime_lambdas": list(args.regime_lambdas) if args.view == "regime" else None,
        "regime_wrs": list(args.regime_wrs) if args.view == "regime" else None,
        "regime_fr_max_fleet": (
            float(args.regime_fr_max_fleet)
            if args.view == "regime" and args.regime_fr_max_fleet is not None
            else None
        ),
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
        "jobs": args.jobs,
        "fixed_lambda": args.view != "regime",
        "tp_candidates": args.tp_candidates,
        "tp_region_km": args.tp_region_km,
        "tp_region_low": args.tp_region_low,
        "fr_delta": args.fr_delta,
    }

    _save_json(out_dir / "experiment.json", experiment_meta)

    baseline_rows_map: Dict[str, List[Dict[str, Any]]] = {
        service_key: [] for service_key in service_specs
    }
    tasks: List[Dict[str, Any]] = []
    args_dict = vars(args).copy()

    for service_key, spec in service_specs.items():
        baseline_dir = out_dir / _slugify(spec["display_name"])
        baseline_dir.mkdir(parents=True, exist_ok=True)
        _save_json(baseline_dir / "settings.json", {"settings": spec["settings"]})

        print(f"\n=== {spec['display_name']} — {len(spec['settings'])} settings ===")
        for setting_index, setting in enumerate(spec["settings"], start=1):
            setting_path = baseline_dir / f"setting_{setting_index:02d}.json"

            if args.resume and setting_path.exists():
                with setting_path.open() as f:
                    payload = json.load(f)
                baseline_rows_map[service_key].append(payload["row"])
                aggregate_rows.append(payload["row"])
                print(f"  [{setting_index:02d}] resumed {setting['setting_label']}")
                continue

            print(f"  [{setting_index:02d}] queued {setting['setting_label']}")
            tasks.append(
                {
                    "args": args_dict,
                    "experiment_meta": experiment_meta,
                    "service_key": service_key,
                    "setting_index": setting_index,
                    "baseline_dir": str(baseline_dir),
                    "resume": args.resume,
                }
            )

    if args.jobs == 1:
        import time as _time
        n_tasks = len(tasks)
        t0 = _time.monotonic()
        for task_i, task in enumerate(tasks):
            setting_index = task["setting_index"]
            service_key = task["service_key"]
            spec = service_specs[service_key]
            setting = spec["settings"][setting_index - 1]
            elapsed = _time.monotonic() - t0
            if task_i > 0:
                avg = elapsed / task_i
                eta = avg * (n_tasks - task_i)
                eta_str = f"ETA {eta / 60:.0f}m"
            else:
                eta_str = "ETA --"
            print(
                f"  [{task_i + 1}/{n_tasks}] "
                f"{spec['display_name']} #{setting_index:02d} "
                f"| {setting['setting_label']} "
                f"| elapsed {elapsed / 60:.1f}m, {eta_str}",
                flush=True,
            )
            task_result = _run_setting_task(task)
            if task_result["status"] == "completed":
                baseline_rows_map[service_key].append(task_result["row"])
                aggregate_rows.append(task_result["row"])
            elif task_result["status"] == "resumed":
                baseline_rows_map[service_key].append(task_result["row"])
                aggregate_rows.append(task_result["row"])
    else:
        print(f"\nRunning {len(tasks)} sweep settings with {args.jobs} worker processes.")
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.jobs,
            initializer=_worker_init,
            initargs=(args_dict,),
        ) as executor:
            future_map = {executor.submit(_run_setting_task, task): task for task in tasks}
            for future in concurrent.futures.as_completed(future_map):
                task = future_map[future]
                service_key = task["service_key"]
                spec = service_specs[service_key]
                setting_index = task["setting_index"]
                setting = spec["settings"][setting_index - 1]
                try:
                    task_result = future.result()
                except Exception as exc:
                    print(
                        f"  [{spec['display_name']} #{setting_index:02d}] "
                        f"ERROR — {setting['setting_label']} — {exc}"
                    )
                    traceback.print_exc()
                    continue

                status = task_result["status"]
                if status == "completed":
                    baseline_rows_map[service_key].append(task_result["row"])
                    aggregate_rows.append(task_result["row"])
                    print(
                        f"  [{spec['display_name']} #{setting_index:02d}] "
                        f"done {setting['setting_label']}"
                    )
                elif status == "resumed":
                    baseline_rows_map[service_key].append(task_result["row"])
                    aggregate_rows.append(task_result["row"])
                    print(
                        f"  [{spec['display_name']} #{setting_index:02d}] "
                        f"resumed {setting['setting_label']}"
                    )
                else:
                    print(
                        f"  [{spec['display_name']} #{setting_index:02d}] "
                        f"skipped {setting['setting_label']}"
                    )

    for service_key, spec in service_specs.items():
        baseline_rows = sorted(
            baseline_rows_map[service_key],
            key=lambda row: int(row["setting_index"]),
        )
        if baseline_rows:
            baseline_dir = out_dir / _slugify(spec["display_name"])
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
