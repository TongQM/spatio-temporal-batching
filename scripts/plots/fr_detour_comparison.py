"""FR vs FR-Detour comparison and diagnostics.

This module provides:
- a scattered-demand comparison on the same selected reference design
- a uniform synthetic diagnostic with scenario-based served stops and skip arcs

The helper functions are also imported by `visualize_designs.py` so the
fixed-route panels use the same second-stage service semantics.

Interpretation note:
- these FR / FR-Detour plots use the repo's fixed-common-origin,
  stochastic-destination last-mile semantics
- earlier discussion sometimes used the reverse first-mile wording because the
  simplified geometry is nearly symmetric, but captions and labels here should
  be read as destination-demand service from the depot to realized destinations
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import networkx as nx
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from lib.baselines import FRDetour
from lib.baselines.base import EvaluationResult, VEHICLE_SPEED_KMH
from lib.baselines.fr_detour import (
    ReferenceLine,
    _build_service_arrival_times,
    _build_initial_arcs,
    _crn_demand,
    _solve_projected_service_scenario,
    _solve_routing_lp_with_cg,
    _tour_distance,
)
from lib.constants import TSP_TRAVEL_DISCOUNT
from lib.synthetic import (
    SyntheticGeoData,
    make_uniform_positions,
    make_scattered_positions,
    build_synthetic_instance,
    positions_array as _positions_array,
)


PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b", "#17becf"]


def identity_J(omega):
    return float(np.sum(omega))


def _pt_seg_dist(p, a, b):
    ab = b - a
    ab2 = np.dot(ab, ab)
    if ab2 < 1e-12:
        return float(np.linalg.norm(p - a))
    t = np.clip(np.dot(p - a, ab) / ab2, 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def _in_corridor(block_idx, route_global, block_pos_km, delta):
    if delta <= 0.0 or len(route_global) < 2:
        return False
    pt = block_pos_km[block_idx]
    for k in range(len(route_global) - 1):
        a = block_pos_km[route_global[k]]
        b = block_pos_km[route_global[k + 1]]
        if _pt_seg_dist(pt, a, b) <= delta + 1e-9:
            return True
    return False


def _walk_geometry(geodata, block_ids, delta):
    walk_saved = {}
    for b in block_ids:
        area_b = float(geodata.get_area(b))
        r_b = np.sqrt(area_b / np.pi)
        walk_no_det = 2.0 * r_b / 3.0
        if delta < max(r_b, 1e-12):
            walk_det = walk_no_det - delta + delta ** 3 / (3.0 * max(r_b, 1e-12) ** 2)
        else:
            walk_det = 0.0
        walk_saved[b] = max(walk_no_det - max(walk_det, 0.0), 0.0)
    return walk_saved


def assigned_locations_by_root(design, block_ids):
    meta = design.service_metadata.get("fr_detour", {})
    assigned_meta = meta.get(
        "assigned_locations_by_root",
        meta.get("assigned_blocks_by_root", {}),
    )
    assigned = {}
    for root in design.district_roots:
        blocks = list(assigned_meta.get(root, []))
        if not blocks and design.assignment is not None:
            ri = block_ids.index(root)
            blocks = [
                block_ids[j] for j in range(len(block_ids))
                if round(design.assignment[j, ri]) == 1
            ]
        assigned[root] = blocks
    return assigned


def reachable_locations_by_root(design, block_ids):
    """Geometric reachable set per root (wider than assigned partition)."""
    meta = design.service_metadata.get("fr_detour", {})
    reachable_meta = meta.get(
        "reachable_locations_by_root",
        meta.get("reachable_blocks_by_root", {}),
    )
    result = {}
    for root in design.district_roots:
        blocks = list(reachable_meta.get(root, []))
        if not blocks:
            # Fallback: use assigned blocks
            assigned = assigned_locations_by_root(design, block_ids)
            blocks = assigned.get(root, [])
        result[root] = blocks
    return result


def assigned_stop_points_by_root(design, block_ids):
    meta = design.service_metadata.get("fr_detour", {})
    assigned_stop_meta = meta.get("assigned_stop_points_by_root", {})
    if assigned_stop_meta:
        return {root: list(assigned_stop_meta.get(root, [])) for root in design.district_roots}
    assigned = assigned_locations_by_root(design, block_ids)
    checkpoint_candidates = set(
        meta.get("checkpoint_candidate_locations", [])
    )
    if not checkpoint_candidates:
        checkpoints = selected_checkpoints_by_root(design, block_ids)
        checkpoint_candidates = {
            cp for cps in checkpoints.values() for cp in cps
        }
    return {
        root: [b for b in assigned.get(root, []) if b not in checkpoint_candidates]
        for root in design.district_roots
    }


def selected_checkpoints_by_root(design, block_ids):
    meta = design.service_metadata.get("fr_detour", {})
    selected = meta.get(
        "selected_checkpoints_by_root",
        meta.get("checkpoint_sequence_by_root", {}),
    )
    result = {root: list(selected.get(root, [])) for root in design.district_roots}
    for root in design.district_roots:
        if not result[root]:
            result[root] = [block_ids[gi] for gi in design.district_routes.get(root, [])]
    return result


def solve_design_scenario(
    design,
    method,
    geodata,
    block_ids,
    block_pos_km,
    prob_dict,
    crn_uniforms,
    Lambda,
    wr,
    wv,
    scenario_idx,
):
    """Solve one scenario and recover realized service points plus global access."""
    rider_time_weight = float(wr)
    vehicle_cost_weight = float(wv)
    depot_pos = np.array(geodata.pos[design.depot_id], dtype=float) / 1000.0
    walk_saved_km = _walk_geometry(geodata, block_ids, method.delta)
    block_pos_map = {b: block_pos_km[i] for i, b in enumerate(block_ids)}
    demand_support_locations = list(
        design.service_metadata.get("fr_detour", {}).get(
            "demand_support_locations",
            [b for b in block_ids if prob_dict.get(b, 0.0) > 0.0],
        )
    )
    areas_km2 = {b: float(geodata.get_area(b)) for b in block_ids}
    services = {}
    route_T = []
    for root in design.district_roots:
        route_T.append(float(design.dispatch_intervals.get(root, 1.0)))
    T_global = float(np.mean(route_T)) if route_T else 1.0
    demand_matrix = _crn_demand(crn_uniforms, Lambda, T_global, prob_dict, block_ids)
    demand_s = {block_ids[j]: int(demand_matrix[scenario_idx, j]) for j in range(len(block_ids))}
    scenario_result = _solve_projected_service_scenario(
        design,
        method,
        geodata,
        prob_dict,
        block_ids,
        block_pos_map,
        demand_s,
        depot_pos,
        areas_km2,
        walk_saved_km,
        rider_time_weight,
        vehicle_cost_weight,
        np.random.default_rng(42 + scenario_idx),
        return_assignments=True,
    )
    for root, info in scenario_result["route_services"].items():
        services[root] = {
            **info,
            "assigned_blocks": list(info.get("assigned_locations", [])),
            "reachable_blocks": list(info.get("assigned_stop_points", [])),
            "considered_blocks": list(set(info.get("checkpoints", [])) | set(info.get("active_stop_points", []))),
            "demand_s": demand_s,
        }
    services["_global"] = {
        "demand_s": demand_s,
        "demand_outcomes": scenario_result["demand_outcomes"],
        "service_counts": scenario_result["service_counts"],
        "service_point_info": scenario_result["service_point_info"],
        "assignment_records": scenario_result["assignment_records"],
        "projected_counts": scenario_result["projected_counts"],
    }
    return services


def find_first_skip_scenario(
    design,
    method,
    geodata,
    block_ids,
    block_pos_km,
    prob_dict,
    crn_uniforms,
    Lambda,
    wr,
    wv,
):
    for s in range(int(crn_uniforms.shape[0])):
        services = solve_design_scenario(
            design, method, geodata, block_ids, block_pos_km, prob_dict, crn_uniforms, Lambda, wr, wv, s
        )
        for root in design.district_roots:
            info = services.get(root)
            if info and info["skip_arcs"]:
                return s, info
    return 0, None


def find_first_served_stop_scenario(
    design,
    method,
    geodata,
    block_ids,
    block_pos_km,
    prob_dict,
    crn_uniforms,
    Lambda,
    wr,
    wv,
):
    for s in range(int(crn_uniforms.shape[0])):
        services = solve_design_scenario(
            design, method, geodata, block_ids, block_pos_km, prob_dict, crn_uniforms, Lambda, wr, wv, s
        )
        for info in services.values():
            if info.get("served_stops"):
                return s, services
    return 0, solve_design_scenario(
        design, method, geodata, block_ids, block_pos_km, prob_dict, crn_uniforms, Lambda, wr, wv, 0
    )


def _route_color_by_root(design):
    active_roots = [root for root in design.district_roots if design.district_routes.get(root, [])]
    return {root: PALETTE[i % len(PALETTE)] for i, root in enumerate(active_roots)}


def plot_fixed_route_panel(
    ax,
    design,
    eval_result: Optional[EvaluationResult],
    title,
    delta,
    geodata,
    block_ids,
    block_pos_km,
    services=None,
    skip_info=None,
):
    depot_idx = block_ids.index(design.depot_id)
    depot_pos = block_pos_km[depot_idx]
    routes = design.district_routes or {}
    active = [(rid, r) for rid, r in routes.items() if r]
    route_colors = _route_color_by_root(design)

    all_cps = set()
    for _, route in active:
        all_cps.update(route)

    # Per-route checkpoint sets (global indices) for per-block CP/corridor distinction
    route_cp_sets = {}
    for d, (root_id, route_global) in enumerate(active):
        route_cp_sets[d] = set(route_global)

    block_route = np.full(len(block_ids), -1, dtype=int)
    block_kind = np.full(len(block_ids), "none", dtype=object)  # checkpoint / corridor / served
    demand_count = np.zeros(len(block_ids), dtype=int)
    walk_origin_count = np.zeros(len(block_ids), dtype=int)
    service_load_cp = np.zeros(len(block_ids), dtype=int)
    service_load_stop = np.zeros(len(block_ids), dtype=int)
    if services is not None:
        # With shared stops, a block may be a checkpoint on one route and
        # a corridor stop on another.  Assign each block to its "primary"
        # route: prefer showing it as a corridor stop (more informative
        # for the detour visualization) over a checkpoint, and prefer
        # actually served detour stops over merely corridor-reachable stops.
        route_order = [root_id for root_id, _ in active]
        # First pass: realized served stops.
        for d, root_id in enumerate(route_order):
            info = services.get(root_id, {})
            for b in info.get("served_stops", set()):
                ji = block_ids.index(b)
                if block_route[ji] < 0:
                    block_route[ji] = d
                    block_kind[ji] = "served"
        # Second pass: corridor-reachable stops that were not served.
        for d, root_id in enumerate(route_order):
            info = services.get(root_id, {})
            cp_set = set(info.get("checkpoints", []))
            for b in info.get("assigned_stop_points", info.get("reachable_blocks", [])):
                if b in cp_set or b in info.get("served_stops", set()):
                    continue
                ji = block_ids.index(b)
                if block_route[ji] < 0:
                    block_route[ji] = d
                    block_kind[ji] = "corridor"
        # Third pass: checkpoints (fill in remaining unclaimed blocks)
        for d, root_id in enumerate(route_order):
            info = services.get(root_id, {})
            for cp in info.get("checkpoints", []):
                ji = block_ids.index(cp)
                if block_route[ji] < 0:
                    block_route[ji] = d
                    block_kind[ji] = "checkpoint"

        global_info = services.get("_global", {})
        global_demand = global_info.get("demand_s", {})
        global_outcomes = global_info.get("demand_outcomes", {})
        service_point_info = global_info.get("service_point_info", {})
        service_counts = global_info.get("service_counts", {})
        for loc, count in service_counts.items():
            ji = block_ids.index(loc)
            info = service_point_info.get(loc, {})
            if info.get("kind") == "detour_stop":
                service_load_stop[ji] = int(count)
            elif info.get("kind") == "checkpoint":
                service_load_cp[ji] = int(count)
        for b, n_b in global_demand.items():
            if n_b <= 0:
                continue
            ji = block_ids.index(b)
            outcome = global_outcomes.get(b, {})
            demand_count[ji] = int(n_b)
            walk_origin_count[ji] = int(outcome.get("reassigned_service", 0))
    else:
        assignment = design.assignment
        for d, (root_id, route_global) in enumerate(active):
            ri = block_ids.index(root_id)
            for j in range(len(block_ids)):
                if round(assignment[j, ri]) != 1:
                    continue
                if j in all_cps:
                    block_route[j] = d
                    block_kind[j] = "checkpoint"
                elif delta > 0 and _in_corridor(j, route_global, block_pos_km, delta):
                    block_route[j] = d
                    block_kind[j] = "corridor"

    if services is not None:
        assignment_records = services.get("_global", {}).get("assignment_records", [])
        for rec in assignment_records:
            origin = np.array(rec["origin_pos"], dtype=float)
            target = np.array(rec["service_pos"], dtype=float)
            ax.plot(
                [origin[0], target[0]],
                [origin[1], target[1]],
                linestyle="--",
                color="#9ca3af",
                linewidth=0.55,
                alpha=0.45,
                zorder=1.2,
            )
            ax.plot(
                origin[0],
                origin[1],
                "o",
                color="#6b7280",
                markersize=2.0,
                alpha=0.65,
                markeredgewidth=0.0,
                zorder=1.6,
            )

    for j, bid in enumerate(block_ids):
        x, y = block_pos_km[j]
        ax.plot(x, y, "o", color="#bbbbbb", markersize=6,
                markeredgecolor="black", markeredgewidth=0.4,
                alpha=0.35, zorder=2.5)
        if block_route[j] >= 0:
            root_id = active[block_route[j]][0]
            color = route_colors[root_id]
            kind = block_kind[j]
            if kind == "checkpoint":
                ax.plot(x, y, "s", color=color, markersize=10,
                        markeredgecolor="black", markeredgewidth=0.7,
                        alpha=0.9, zorder=4)
            elif kind == "served":
                ax.plot(x, y, "o", color=color, markersize=7,
                        markeredgecolor="white", markeredgewidth=0.7,
                        alpha=0.85, zorder=4)
            elif kind == "corridor":
                ax.plot(x, y, "o", markerfacecolor="none", markersize=7,
                        markeredgecolor=color, markeredgewidth=1.2,
                        alpha=0.9, zorder=3.8)
        ax.text(x + 0.12, y + 0.12, block_ids[j][-3:], fontsize=5,
                color="black", alpha=0.6, zorder=7)

    for root_id, route_global in active:
        color = route_colors[root_id]
        cp_pos = np.array([block_pos_km[gi] for gi in route_global])
        if len(cp_pos) < 1:
            continue
        loop = np.vstack([[depot_pos], cp_pos, [depot_pos]])
        ax.plot(loop[:, 0], loop[:, 1], "-", color=color, linewidth=2.0, alpha=0.7, zorder=2)
        if len(loop) >= 2:
            ax.annotate("", xy=loop[1], xytext=loop[0],
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.0, mutation_scale=12), zorder=3)
        if delta > 0 and len(cp_pos) >= 2:
            for k in range(len(cp_pos) - 1):
                p1, p2 = cp_pos[k], cp_pos[k + 1]
                seg = p2 - p1
                seg_len = np.linalg.norm(seg)
                if seg_len < 1e-6:
                    continue
                perp = np.array([-seg[1], seg[0]]) / seg_len * delta
                corners = np.array([p1 + perp, p2 + perp, p2 - perp, p1 - perp])
                ax.add_patch(Polygon(corners, facecolor=color, edgecolor=color,
                                     alpha=0.06, linewidth=0.6, linestyle="--", zorder=0))
            for cp in cp_pos:
                ax.add_patch(Circle(cp, delta, facecolor=color, edgecolor=color,
                                    alpha=0.06, linewidth=0.6, linestyle="--", zorder=0))

    if skip_info is not None:
        checkpoints = skip_info["checkpoints"]
        for arc, _flow in skip_info["skip_arcs"]:
            p1 = block_pos_km[block_ids.index(checkpoints[arc.cp_from])]
            p2 = block_pos_km[block_ids.index(checkpoints[arc.cp_to])]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "--", color="black", lw=3.0, alpha=0.9, zorder=4)

    ax.plot(*depot_pos, "*", color="red", markersize=16,
            markeredgecolor="black", markeredgewidth=0.8, zorder=6)

    n_lines = len(active)
    n_cp = sum(1 for j in range(len(block_ids)) if block_kind[j] == "checkpoint")
    n_corr = sum(1 for j in range(len(block_ids)) if block_kind[j] == "corridor")
    n_served = sum(1 for j in range(len(block_ids)) if block_kind[j] == "served")
    n_uncov = sum(1 for j in range(len(block_ids)) if block_route[j] < 0)
    n_dem_cp = int(service_load_cp.sum())
    n_dem_det = int(service_load_stop.sum())
    n_dem_walk = int(walk_origin_count.sum())

    T_info = design.dispatch_intervals
    T_vals = list(T_info.values()) if T_info else []
    t_str = (f"T={T_vals[0]:.2f}h"
             if T_vals and len(set(f"{t:.2f}" for t in T_vals)) == 1
             else f"T={min(T_vals):.2f}-{max(T_vals):.2f}h" if T_vals else "")
    title_line = f"{n_cp} CPs  {n_corr} corridor"
    if services is not None:
        title_line += f"  {n_served} detour-served"
        title_line += f"\nDemand units: C={n_dem_cp}  D={n_dem_det}  W={n_dem_walk}"
    title_line += f"  {n_uncov} uncov"
    ax.set_title(f"{title}\n{n_lines} routes, {t_str}\n{title_line}",
                 fontsize=9)

    if eval_result is not None:
        ev = eval_result
        textstr = (f"User: wait={ev.avg_wait_time:.3f} "
                   f"ride={ev.avg_invehicle_time:.3f} "
                   f"walk={ev.avg_walk_time:.3f}h\n"
                   f"Provider: {ev.provider_cost:.3f}  Fleet: {ev.fleet_size:.1f}\n"
                   f"Total cost: {ev.total_cost:.3f}")
        props = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cccccc", alpha=0.85)
        ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=6.5,
                verticalalignment="bottom", bbox=props, zorder=10)

    margin = 0.3
    ax.set_xlim(block_pos_km[:, 0].min() - margin, block_pos_km[:, 0].max() + margin)
    ax.set_ylim(block_pos_km[:, 1].min() - margin, block_pos_km[:, 1].max() + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")


def run_uniform_comparison():
    positions_km, checkpoint_ids = make_uniform_positions()
    geodata, block_ids, block_pos_km, prob_dict, Omega_dict = build_synthetic_instance(
        positions_km,
        checkpoint_ids,
        demand_support="all_locations",
    )

    budget = 3
    Lambda = 50.0
    rider_time_weight = 10.0
    vehicle_cost_weight = 1.0
    rs_iters = 12
    n_saa = 8
    n_eval = 12
    max_cg_iters = 10
    t_values = [0.25, 0.5, 1.0]
    delta_detour = 0.8

    shared_builder = FRDetour(
        delta=delta_detour,
        max_iters=rs_iters,
        n_saa=n_saa,
        num_scenarios=n_eval,
        max_cg_iters=max_cg_iters,
        T_values=t_values,
        show_progress=True,
    )
    shared_inputs = shared_builder._prepare_shared_inputs(geodata, budget)
    common_kwargs = dict(
        candidate_lines=shared_inputs["candidate_lines"],
        block_pos_km=shared_inputs["block_pos_km"],
        depot_id=shared_inputs["depot_id"],
        crn_uniforms=shared_inputs["crn_uniforms"],
    )

    fr0 = FRDetour(
        delta=0.0,
        max_iters=rs_iters,
        n_saa=n_saa,
        num_scenarios=n_eval,
        max_cg_iters=max_cg_iters,
        T_values=t_values,
        name="Multi-FR",
        show_progress=True,
    )
    fr_det = FRDetour(
        delta=delta_detour,
        max_iters=rs_iters,
        n_saa=n_saa,
        num_scenarios=n_eval,
        max_cg_iters=max_cg_iters,
        T_values=t_values,
        show_progress=True,
    )

    print(f"  Designing FR (budget={budget}, delta=0.0) ...", flush=True)
    fr_design = fr0.design(
        geodata, prob_dict, Omega_dict, identity_J, budget,
        Lambda=Lambda, wr=rider_time_weight, wv=vehicle_cost_weight, **common_kwargs,
    )
    print(f"  Designing FR-Detour (budget={budget}, delta={delta_detour}) ...", flush=True)
    det_design = fr_det.design(
        geodata, prob_dict, Omega_dict, identity_J, budget,
        Lambda=Lambda, wr=rider_time_weight, wv=vehicle_cost_weight, **common_kwargs,
    )

    print("  Evaluating ...", flush=True)
    eval_fr_opt = fr0.evaluate(
        fr_design, geodata, prob_dict, Omega_dict, identity_J,
        Lambda, rider_time_weight, vehicle_cost_weight, crn_uniforms=shared_inputs["crn_uniforms"],
    )
    eval_det_opt = fr_det.evaluate(
        det_design, geodata, prob_dict, Omega_dict, identity_J,
        Lambda, rider_time_weight, vehicle_cost_weight, crn_uniforms=shared_inputs["crn_uniforms"],
    )

    scenario_fr_opt, services_fr_opt = find_first_served_stop_scenario(
        fr_design, fr0, geodata, block_ids, block_pos_km, prob_dict,
        shared_inputs["crn_uniforms"], Lambda, rider_time_weight, vehicle_cost_weight,
    )
    scenario_det_opt, services_det_opt = find_first_served_stop_scenario(
        det_design, fr_det, geodata, block_ids, block_pos_km, prob_dict,
        shared_inputs["crn_uniforms"], Lambda, rider_time_weight, vehicle_cost_weight,
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    plot_fixed_route_panel(
        axes[0], fr_design, eval_fr_opt, "Optimized FR", 0.0,
        geodata, block_ids, block_pos_km, services=services_fr_opt,
    )
    plot_fixed_route_panel(
        axes[1], det_design, eval_det_opt, f"Optimized FR-Detour (delta={delta_detour})", delta_detour,
        geodata, block_ids, block_pos_km, services=services_det_opt,
    )
    fig.suptitle(
        "FR vs FR-Detour comparison\n"
        f"Direct comparison on separately optimized designs.\n"
        f"budget={budget}, Λ={Lambda}, shared CRN, scenarios=FR:{scenario_fr_opt} DET:{scenario_det_opt}",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    out_dir = "results/baseline_comparison/figures"
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "fr_detour_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


def run_uniform_diagnostic(wr=500.0):
    budget = 3
    Lambda = 50.0
    rider_time_weight = wr
    vehicle_cost_weight = 1.0
    delta = 0.8
    n_saa = 8
    n_eval = 12
    rs_iters = 12
    max_cg_iters = 10
    t_values = [0.25, 0.5, 1.0]

    positions_km, checkpoint_ids = make_uniform_positions()
    geodata, block_ids, block_pos_km, prob_dict, Omega_dict = build_synthetic_instance(
        positions_km,
        checkpoint_ids,
        demand_support="all_locations",
    )
    n_blocks = len(block_ids)

    shared_crn = np.random.default_rng(123).uniform(0.0, 1.0, size=(n_saa, n_blocks))
    shared_builder = FRDetour(
        delta=delta,
        max_iters=rs_iters,
        n_saa=n_saa,
        num_scenarios=n_eval,
        K_skip=2,
        max_cg_iters=max_cg_iters,
        T_values=t_values,
        show_progress=True,
    )
    shared_inputs = shared_builder._prepare_shared_inputs(geodata, budget, crn_uniforms=shared_crn)

    fr0 = FRDetour(
        delta=0.0,
        max_iters=rs_iters,
        n_saa=n_saa,
        num_scenarios=n_eval,
        K_skip=2,
        max_cg_iters=max_cg_iters,
        T_values=t_values,
        name="Multi-FR",
        show_progress=True,
    )
    fr_det = FRDetour(
        delta=delta,
        max_iters=rs_iters,
        n_saa=n_saa,
        num_scenarios=n_eval,
        K_skip=2,
        max_cg_iters=max_cg_iters,
        T_values=t_values,
        name="Multi-FR-Detour",
        show_progress=True,
    )

    print(f"Designing shared reference network (budget={budget}, delta={delta}, Lambda={Lambda}, wr={rider_time_weight}) ...",
          flush=True)
    shared_design = fr_det.design(
        geodata, prob_dict, Omega_dict, identity_J, budget,
        Lambda=Lambda, wr=rider_time_weight, wv=vehicle_cost_weight, **shared_inputs,
    )

    scenario_fr, skip_fr = find_first_skip_scenario(
        shared_design, fr0, geodata, block_ids, block_pos_km, prob_dict, shared_inputs["crn_uniforms"], Lambda, rider_time_weight, vehicle_cost_weight
    )
    scenario_det, skip_det = find_first_skip_scenario(
        shared_design, fr_det, geodata, block_ids, block_pos_km, prob_dict, shared_inputs["crn_uniforms"], Lambda, rider_time_weight, vehicle_cost_weight
    )
    scenario_to_plot = min(scenario_fr, scenario_det)
    services_fr = solve_design_scenario(
        shared_design, fr0, geodata, block_ids, block_pos_km, prob_dict, shared_inputs["crn_uniforms"], Lambda, rider_time_weight, vehicle_cost_weight, scenario_to_plot
    )
    services_det = solve_design_scenario(
        shared_design, fr_det, geodata, block_ids, block_pos_km, prob_dict, shared_inputs["crn_uniforms"], Lambda, rider_time_weight, vehicle_cost_weight, scenario_to_plot
    )

    print("\n[FR] selected routes")
    for root in shared_design.district_roots:
        route = [block_ids[gi] for gi in shared_design.district_routes.get(root, [])]
        print(f"  root={root}  T={shared_design.dispatch_intervals.get(root, 1.0):.2f}h  checkpoints={route}")
    if skip_fr is not None:
        print(f"  first skip observed on route {skip_fr['root']} in scenario {scenario_fr}")
        for arc, flow in skip_fr["skip_arcs"]:
            skipped = skip_fr["checkpoints"][arc.cp_from + 1:arc.cp_to]
            print(f"    arc {skip_fr['checkpoints'][arc.cp_from]} -> {skip_fr['checkpoints'][arc.cp_to]} (flow={flow:.2f}) skips {skipped}")

    print("\n[FR-Detour] selected routes")
    for root in shared_design.district_roots:
        route = [block_ids[gi] for gi in shared_design.district_routes.get(root, [])]
        print(f"  root={root}  T={shared_design.dispatch_intervals.get(root, 1.0):.2f}h  checkpoints={route}")
    if skip_det is not None:
        print(f"  first skip observed on route {skip_det['root']} in scenario {scenario_det}")
        for arc, flow in skip_det["skip_arcs"]:
            skipped = skip_det["checkpoints"][arc.cp_from + 1:arc.cp_to]
            print(f"    arc {skip_det['checkpoints'][arc.cp_from]} -> {skip_det['checkpoints'][arc.cp_to]} (flow={flow:.2f}) skips {skipped}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    plot_fixed_route_panel(
        axes[0], shared_design, None, "FR (delta=0)", 0.0,
        geodata, block_ids, block_pos_km, services=services_fr, skip_info=skip_fr,
    )
    plot_fixed_route_panel(
        axes[1], shared_design, None, f"FR-Detour (delta={delta})", delta,
        geodata, block_ids, block_pos_km, services=services_det, skip_info=skip_det,
    )
    fig.suptitle(
        f"Uniform stopping locations, budget={budget}, Lambda={Lambda}, wr={rider_time_weight}, scenario={scenario_to_plot}\n"
        "Same selected reference design; colored circles = second-stage served stops",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    out_dir = "results/baseline_comparison/figures"
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "fr_detour_uniform_skip.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


def main():
    run_uniform_comparison()


if __name__ == "__main__":
    main()
