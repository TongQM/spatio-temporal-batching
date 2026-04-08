"""FR vs FR-Detour comparison and diagnostics.

This module provides:
- a scattered-demand comparison on the same selected reference design
- a uniform synthetic diagnostic with scenario-based served stops and skip arcs

The helper functions are also imported by `visualize_designs.py` so the
fixed-route panels use the same second-stage service semantics.
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
    _sample_global_service_assignments,
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
    checkpoints = selected_checkpoints_by_root(design, block_ids)
    return {
        root: [b for b in assigned.get(root, []) if b not in set(checkpoints.get(root, []))]
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
    depot_pos = np.array(geodata.pos[design.depot_id], dtype=float) / 1000.0
    walk_saved_km = _walk_geometry(geodata, block_ids, method.delta)
    assigned_by_root = assigned_locations_by_root(design, block_ids)
    assigned_stops_by_root = assigned_stop_points_by_root(design, block_ids)
    selected_cps_by_root = selected_checkpoints_by_root(design, block_ids)
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
    service_point_info = {}

    for root in design.district_roots:
        checkpoints = list(selected_cps_by_root.get(root, []))
        assigned_blocks = list(assigned_by_root.get(root, []))
        assigned_stop_points = list(assigned_stops_by_root.get(root, []))
        route_global = design.district_routes.get(root, [])
        T_i = design.dispatch_intervals[root]

        all_blocks = set(checkpoints) | set(assigned_stop_points)
        line_obj = ReferenceLine(
            idx=0,
            checkpoints=checkpoints,
            reachable_locations=all_blocks,
            est_tour_km=_tour_distance(depot_pos, checkpoints, block_pos_map),
        )
        arcs = _build_initial_arcs(
            line_obj, block_ids, block_pos_map,
            method.delta, method.K_skip, VEHICLE_SPEED_KMH, method.alpha_time,
        )

        z_star = {b: 1.0 for b in all_blocks}
        cp_set = set(checkpoints)
        cost_mult = 1.0 / (TSP_TRAVEL_DISCOUNT * T_i) + wv / (2.0 * VEHICLE_SPEED_KMH)
        walk_penalties = {
            b: (
                method.coverage_penalty_factor
                * method.walk_time_weight
                * wv
                * walk_saved_km.get(b, 0.0)
                * demand_s.get(b, 0)
                / method.walk_speed_kmh
            )
            for b in all_blocks
        }

        u_vals = {b: (0.0 if b in cp_set else 1.0) for b in all_blocks}
        active_arcs = []
        if len(checkpoints) > 1 and arcs:
            result = _solve_routing_lp_with_cg(
                arcs,
                line_obj,
                block_ids,
                block_pos_map,
                len(checkpoints),
                method.capacity,
                z_star,
                demand_s,
                all_blocks,
                cp_set,
                cost_mult,
                walk_penalties,
                method.delta,
                VEHICLE_SPEED_KMH,
                method.alpha_time,
                method.K_skip,
                max_cg_iters=method.max_cg_iters,
                return_solution=True,
                return_active_arcs=True,
            )
            if result[0] is not None:
                _, _, _, u_vals, active_arcs = result

        served_stops = {
            b for b in assigned_stop_points
            if u_vals.get(b, 1.0) < 0.5
        }
        arrival_h = _build_service_arrival_times(
            depot_pos, checkpoints, block_pos_map, active_arcs, VEHICLE_SPEED_KMH
        )
        for cp in checkpoints:
            cp_info = {
                "kind": "checkpoint",
                "arrival_h": float(arrival_h.get(cp, 0.0)),
                "headway_h": float(T_i),
                "root": root,
            }
            prev = service_point_info.get(cp)
            if prev is None or cp_info["arrival_h"] < prev.get("arrival_h", 1e18):
                service_point_info[cp] = cp_info
        for stop in served_stops:
            stop_info = {
                "kind": "detour_stop",
                "arrival_h": float(arrival_h.get(stop, 0.0)),
                "headway_h": float(T_i),
                "root": root,
            }
            prev = service_point_info.get(stop)
            if prev is None or stop_info["arrival_h"] < prev.get("arrival_h", 1e18):
                service_point_info[stop] = stop_info

        skip_arcs = [(arc, flow) for arc, flow in active_arcs if arc.cp_to > arc.cp_from + 1]
        services[root] = {
            "root": root,
            "checkpoints": checkpoints,
            "assigned_blocks": assigned_blocks,
            "assigned_stop_points": assigned_stop_points,
            "reachable_blocks": assigned_stop_points,
            "considered_blocks": list(all_blocks),
            "served_stops": served_stops,
            "skip_arcs": skip_arcs,
            "demand_s": demand_s,
            "route_global": route_global,
        }

    (
        _walk_km,
        _wait_h,
        _inveh_h,
        service_counts,
        _total_riders,
        demand_outcomes,
    ) = _sample_global_service_assignments(
        demand_s,
        demand_support_locations,
        block_pos_map,
        areas_km2,
        service_point_info,
        np.random.default_rng(42 + scenario_idx),
    )
    services["_global"] = {
        "demand_s": demand_s,
        "demand_outcomes": demand_outcomes,
        "service_counts": service_counts,
        "service_point_info": service_point_info,
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
    demand_route = np.full(len(block_ids), -1, dtype=int)
    demand_count = np.zeros(len(block_ids), dtype=int)
    demand_cp_served = np.zeros(len(block_ids), dtype=int)
    demand_detour_served = np.zeros(len(block_ids), dtype=int)
    demand_cp_fallback = np.zeros(len(block_ids), dtype=int)
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
        selected_cp_to_route = {}
        for d, root_id in enumerate(route_order):
            for cp in services.get(root_id, {}).get("checkpoints", []):
                selected_cp_to_route[cp] = d
        service_point_info = global_info.get("service_point_info", {})
        for b, n_b in global_demand.items():
            if n_b <= 0:
                continue
            ji = block_ids.index(b)
            outcome = global_outcomes.get(b, {})
            demand_count[ji] = int(n_b)
            demand_cp_served[ji] = int(outcome.get("checkpoint_served", 0))
            demand_detour_served[ji] = int(outcome.get("detour_stop_served", 0))
            demand_cp_fallback[ji] = int(outcome.get("checkpoint_fallback", 0))
            if b in selected_cp_to_route:
                demand_route[ji] = selected_cp_to_route[b]
            else:
                chosen_route = None
                for loc, info in service_point_info.items():
                    if info.get("kind") == "detour_stop" and demand_detour_served[ji] > 0:
                        chosen_route = route_order.index(info.get("root"))
                        break
                if chosen_route is not None:
                    demand_route[ji] = chosen_route
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
        if demand_count[j] > 0:
            parts = []
            if demand_cp_served[j] > 0:
                parts.append(f"C{demand_cp_served[j]}")
            if demand_detour_served[j] > 0:
                parts.append(f"D{demand_detour_served[j]}")
            if demand_cp_fallback[j] > 0:
                parts.append(f"W{demand_cp_fallback[j]}")
            label = "/".join(parts) if parts else f"N{demand_count[j]}"
            if demand_detour_served[j] > 0 and demand_cp_served[j] == 0 and demand_cp_fallback[j] == 0:
                text_color = "#166534"
                bbox_fc = "#dcfce7"
                bbox_ec = "#16a34a"
            elif demand_cp_fallback[j] > 0 and demand_detour_served[j] == 0 and demand_cp_served[j] == 0:
                text_color = "#92400e"
                bbox_fc = "#fef3c7"
                bbox_ec = "#d97706"
            else:
                text_color = "#1d4ed8"
                bbox_fc = "#dbeafe"
                bbox_ec = "#2563eb"
            ax.text(
                x - 0.10, y - 0.16, label, fontsize=5.5, color=text_color,
                zorder=8,
                bbox=dict(boxstyle="round,pad=0.12", facecolor=bbox_fc,
                          edgecolor=bbox_ec, linewidth=0.5, alpha=0.95),
            )
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
    n_dem_cp = int(demand_cp_served.sum())
    n_dem_det = int(demand_detour_served.sum())
    n_dem_fb = int(demand_cp_fallback.sum())

    T_info = design.dispatch_intervals
    T_vals = list(T_info.values()) if T_info else []
    t_str = (f"T={T_vals[0]:.2f}h"
             if T_vals and len(set(f"{t:.2f}" for t in T_vals)) == 1
             else f"T={min(T_vals):.2f}-{max(T_vals):.2f}h" if T_vals else "")
    title_line = f"{n_cp} CPs  {n_corr} corridor"
    if services is not None:
        title_line += f"  {n_served} detour-served"
        title_line += f"\nDemand units: {n_dem_cp} checkpoint  {n_dem_det} stop  {n_dem_fb} fallback"
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
    Lambda = 20.0
    wr, wv = 2.0, 1.0
    rs_iters = 100
    delta_detour = 0.8

    shared_builder = FRDetour(delta=delta_detour, max_iters=rs_iters)
    shared_inputs = shared_builder._prepare_shared_inputs(geodata, budget)
    common_kwargs = dict(
        candidate_lines=shared_inputs["candidate_lines"],
        block_pos_km=shared_inputs["block_pos_km"],
        depot_id=shared_inputs["depot_id"],
        crn_uniforms=shared_inputs["crn_uniforms"],
    )

    fr0 = FRDetour(delta=0.0, max_iters=rs_iters, name="Multi-FR")
    fr_det = FRDetour(delta=delta_detour, max_iters=rs_iters)

    print(f"  Designing FR (budget={budget}, delta=0.0) ...", flush=True)
    fr_design = fr0.design(
        geodata, prob_dict, Omega_dict, identity_J, budget,
        Lambda=Lambda, wr=wr, wv=wv, **common_kwargs,
    )
    print(f"  Designing FR-Detour (budget={budget}, delta={delta_detour}) ...", flush=True)
    det_design = fr_det.design(
        geodata, prob_dict, Omega_dict, identity_J, budget,
        Lambda=Lambda, wr=wr, wv=wv, **common_kwargs,
    )

    print("  Evaluating ...", flush=True)
    eval_fr_opt = fr0.evaluate(
        fr_design, geodata, prob_dict, Omega_dict, identity_J,
        Lambda, wr, wv, crn_uniforms=shared_inputs["crn_uniforms"],
    )
    eval_det_opt = fr_det.evaluate(
        det_design, geodata, prob_dict, Omega_dict, identity_J,
        Lambda, wr, wv, crn_uniforms=shared_inputs["crn_uniforms"],
    )
    eval_fr_shared = fr0.evaluate(
        det_design, geodata, prob_dict, Omega_dict, identity_J,
        Lambda, wr, wv, crn_uniforms=shared_inputs["crn_uniforms"],
    )
    eval_det_shared = fr_det.evaluate(
        det_design, geodata, prob_dict, Omega_dict, identity_J,
        Lambda, wr, wv, crn_uniforms=shared_inputs["crn_uniforms"],
    )

    scenario_fr_opt, services_fr_opt = find_first_served_stop_scenario(
        fr_design, fr0, geodata, block_ids, block_pos_km, prob_dict,
        shared_inputs["crn_uniforms"], Lambda, wr, wv,
    )
    scenario_det_opt, services_det_opt = find_first_served_stop_scenario(
        det_design, fr_det, geodata, block_ids, block_pos_km, prob_dict,
        shared_inputs["crn_uniforms"], Lambda, wr, wv,
    )
    scenario_shared, services_det_shared = find_first_served_stop_scenario(
        det_design, fr_det, geodata, block_ids, block_pos_km, prob_dict,
        shared_inputs["crn_uniforms"], Lambda, wr, wv,
    )
    services_fr_shared = solve_design_scenario(
        det_design, fr0, geodata, block_ids, block_pos_km, prob_dict,
        shared_inputs["crn_uniforms"], Lambda, wr, wv, scenario_shared,
    )

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    plot_fixed_route_panel(
        axes[0, 0], fr_design, eval_fr_opt, "Optimized FR", 0.0,
        geodata, block_ids, block_pos_km, services=services_fr_opt,
    )
    plot_fixed_route_panel(
        axes[0, 1], det_design, eval_det_opt, f"Optimized FR-Detour (delta={delta_detour})", delta_detour,
        geodata, block_ids, block_pos_km, services=services_det_opt,
    )
    plot_fixed_route_panel(
        axes[1, 0], det_design, eval_fr_shared, "Same design diagnostic: FR evaluation", 0.0,
        geodata, block_ids, block_pos_km, services=services_fr_shared,
    )
    plot_fixed_route_panel(
        axes[1, 1], det_design, eval_det_shared, f"Same design diagnostic: FR-Detour evaluation (delta={delta_detour})", delta_detour,
        geodata, block_ids, block_pos_km, services=services_det_shared,
    )
    fig.suptitle(
        "FR vs FR-Detour comparison\n"
        f"Top row: optimized separately. Bottom row: same-design diagnostic on FR-Detour design.\n"
        f"budget={budget}, Λ={Lambda}, shared CRN, scenarios=FR:{scenario_fr_opt} DET:{scenario_det_opt} shared:{scenario_shared}",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()

    out_dir = "results/baseline_comparison/figures"
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "fr_detour_comparison.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


def run_uniform_diagnostic(wr=500.0):
    budget = 5
    Lambda = 300.0
    wv = 1.0
    delta = 0.8
    n_saa = 20
    rs_iters = 30

    positions_km, checkpoint_ids = make_uniform_positions()
    geodata, block_ids, block_pos_km, prob_dict, Omega_dict = build_synthetic_instance(
        positions_km,
        checkpoint_ids,
        demand_support="all_locations",
    )
    n_blocks = len(block_ids)

    shared_crn = np.random.default_rng(123).uniform(0.0, 1.0, size=(n_saa, n_blocks))
    shared_builder = FRDetour(
        delta=delta, max_iters=rs_iters, n_saa=n_saa, K_skip=2, T_values=[0.25, 0.5, 0.75, 1.0]
    )
    shared_inputs = shared_builder._prepare_shared_inputs(geodata, budget, crn_uniforms=shared_crn)

    fr0 = FRDetour(delta=0.0, max_iters=rs_iters, n_saa=n_saa, K_skip=2,
                   T_values=[0.25, 0.5, 0.75, 1.0], name="Multi-FR")
    fr_det = FRDetour(delta=delta, max_iters=rs_iters, n_saa=n_saa, K_skip=2,
                      T_values=[0.25, 0.5, 0.75, 1.0], name="Multi-FR-Detour")

    print(f"Designing shared reference network (budget={budget}, delta={delta}, Lambda={Lambda}, wr={wr}) ...",
          flush=True)
    shared_design = fr_det.design(
        geodata, prob_dict, Omega_dict, identity_J, budget,
        Lambda=Lambda, wr=wr, wv=wv, **shared_inputs,
    )

    scenario_fr, skip_fr = find_first_skip_scenario(
        shared_design, fr0, geodata, block_ids, block_pos_km, prob_dict, shared_inputs["crn_uniforms"], Lambda, wr, wv
    )
    scenario_det, skip_det = find_first_skip_scenario(
        shared_design, fr_det, geodata, block_ids, block_pos_km, prob_dict, shared_inputs["crn_uniforms"], Lambda, wr, wv
    )
    scenario_to_plot = min(scenario_fr, scenario_det)
    services_fr = solve_design_scenario(
        shared_design, fr0, geodata, block_ids, block_pos_km, prob_dict, shared_inputs["crn_uniforms"], Lambda, wr, wv, scenario_to_plot
    )
    services_det = solve_design_scenario(
        shared_design, fr_det, geodata, block_ids, block_pos_km, prob_dict, shared_inputs["crn_uniforms"], Lambda, wr, wv, scenario_to_plot
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
        f"Uniform stopping locations, budget={budget}, Lambda={Lambda}, wr={wr}, scenario={scenario_to_plot}\n"
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
