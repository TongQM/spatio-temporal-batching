"""
Run the real-case optimization and compare against a baseline route-based partition.

Usage (from repo root):
  conda run -n optimization python -m real_case.experiment --method LBBD --districts 3

Outputs:
  - Prints summary metrics for baseline vs optimized
  - Saves a JSON summary under results/lbbd/
"""

from __future__ import annotations

import os
import json
import argparse
from typing import Dict, List, Tuple
import numpy as np
from datetime import datetime

from real_case.inputs import (
    load_geodata,
    load_probability_from_population,
    load_probability_from_commuting,
    load_real_odd,
    load_baseline_assignment,
    set_geodata_speeds,
)
from lib.algorithm import Partition
from lib.data import RouteData
from lib.evaluate import Evaluate
from lib.constants import TSP_TRAVEL_DISCOUNT
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from shapely.geometry import LineString
import polyline
import os


def J_function(omega):
    """Aggregate ODD penalty from feature vector or scalar cost."""
    if hasattr(omega, '__len__'):
        return float(np.nansum(omega))
    return float(omega)


def expand_to_square(partition: Partition, small_assign: np.ndarray, centers: list[str]) -> np.ndarray:
    # Map center list to a square N×N assignment using helper
    if small_assign.shape[1] == len(partition.short_geoid_list):
        return small_assign
    return partition._expand_assignment_matrix(small_assign, centers)


def _aggregate_prob_by_roots(assignment: np.ndarray, block_ids: List[str], center_list: List[str], prob: Dict[str, float]) -> Dict[str, float]:
    """Sum probability mass per district root given a 0/1 assignment (N x N)."""
    root_mass = {root: 0.0 for root in center_list}
    idx_of = {b: i for i, b in enumerate(block_ids)}
    for root in center_list:
        j = idx_of[root]
        # all i assigned to column j
        for i, bi in enumerate(block_ids):
            if round(assignment[i, j]) == 1:
                root_mass[root] += prob.get(bi, 0.0)
    return root_mass


def _district_blocks(assignment: np.ndarray, block_ids: List[str], root_id: str) -> List[str]:
    j = block_ids.index(root_id)
    res = []
    for i, bi in enumerate(block_ids):
        if round(assignment[i, j]) == 1:
            res.append(bi)
    return res


def evaluate_design(partition: Partition, assignment: np.ndarray, prob_dict, Omega_dict):
    # Determine a reasonable depot for this partition
    # First, identify root ids from current assignment
    block_ids = partition.short_geoid_list
    N = len(block_ids)
    roots = [block_ids[i] for i in range(N) if round(assignment[i, i]) == 1]
    if not roots:
        # pick columns with any assigned rows as roots
        cols = np.where(assignment.sum(axis=0) > 0)[0]
        roots = [block_ids[i] for i in cols]
    depot_id = partition.optimize_depot_location(assignment, roots, prob_dict)
    obj, district_info = partition.evaluate_partition_objective(
        assignment, depot_id, prob_dict,
        Lambda=1.0, wr=0.0, wv=10.0, beta=0.7120,
        Omega_dict=Omega_dict, J_function=J_function
    )
    # Summaries
    K_sum = float(sum(info[2] for info in district_info))
    F_sum = float(sum(info[3] for info in district_info))
    return {
        'obj': float(obj),
        'depot': depot_id,
        'K_sum': K_sum,
        'F_sum': F_sum,
        'num_districts': len(district_info)
    }


def _avg_block_diameter_km(geodata) -> float:
    """Approximate average block-group diameter in km using equal-area circle: d=2*sqrt(A/pi)."""
    import math
    areas_km2 = geodata.gdf.loc[geodata.short_geoid_list, 'area'].astype(float)  # km^2
    diam_km = 2.0 * (areas_km2.apply(lambda a: math.sqrt(a / math.pi)))
    return float(diam_km.mean()) if len(diam_km) else 0.1


def _evaluate_fixed_route_full(geodata, prob: Dict[str, float], epsilon_km: float,
                               Omega: Dict[str, np.ndarray] | None = None,
                               J_function_ref=None) -> Dict:
    """Evaluate Fixed-Route (FR) rider metrics using nearest stops and SimPy wait/transit simulation.
    Provider metrics are left as None (route operations treated as sunk in this framework).
    Returns a dict with expected and worst-P rider metrics.
    """
    rd = RouteData('data/hct_routes.json', geodata)
    ev = Evaluate(rd, geodata, epsilon=epsilon_km)

    # Nearest stop per node for walk times
    nearest = rd.find_nearest_stops()
    nodes = list(geodata.short_geoid_list)

    # Expected walk (m/s speed 1.4)
    walking_speed = 1.4
    expected_walk_time = 0.0
    max_walk_time = 0.0
    for n in nodes:
        d = float(nearest[n]['distance'])  # meters
        t = d / walking_speed
        p = float(prob.get(n, 0.0))
        expected_walk_time += p * t
        max_walk_time = max(max_walk_time, t)

    # Worst-P walk using Wasserstein-ball maximization
    try:
        worst_prob, _ = ev.get_fixed_route_worst_distribution(prob)
        worstP_walk_time = sum((worst_prob[n] * float(nearest[n]['distance']) / walking_speed) for n in nodes)
    except Exception:
        worst_prob = None
        worstP_walk_time = None

    # Expected wait/transit using SimPy
    try:
        wait_summary, transit_summary = ev.simulate_wait_and_transit_fixed_route_simpy(prob)
        # Mass-weighted averages across routes
        total_mass = sum(v[0] for v in wait_summary.values()) or 1.0
        expected_wait_time = sum(m * w for (m, w) in wait_summary.values()) / total_mass
        expected_transit_time = sum(m * tr for (m, tr) in transit_summary.values()) / total_mass
    except Exception:
        expected_wait_time = None
        expected_transit_time = None

    # Provider metrics for Fixed-Route (FR)
    # Treat linehaul as 0 (no depot shuttling) and compute an aggregate travel rate
    # from total route distance over total cycle time across routes.
    # Use Evaluate to gather route lengths and travel times without simulation side-effects.
    try:
        fr_eval_basic = ev.evaluate_fixed_route(prob_dict=prob, simulate=False)
        # route_lengths are in projected meters (EPSG:2163), travel_times in seconds
        route_lengths_m = fr_eval_basic.get('route_lengths', {})
        route_times_s = fr_eval_basic.get('travel_times', {})
        # Sum per-route speeds: Σ (length_i/time_i) in km/h
        provider_travel_kmph = 0.0
        for rname, length_m in route_lengths_m.items():
            secs = float(route_times_s.get(rname, 0.0))
            if secs > 0:
                kmph = (float(length_m) / 1000.0) / (secs / 3600.0)
                provider_travel_kmph += kmph
    except Exception:
        provider_travel_kmph = None

    # Compute ODD provider cost for FR by forming FR districts via nearest-stop assignment
    # and taking J(max Omega in district) divided by the route headway (hours), summed over routes
    provider_odd_cost = None
    try:
        assignment_fr, centers_fr = rd.build_assignment_matrix(visualize=False)
        # Map each center to its route name and headway
        route_headway_h = {}
        for route in rd.routes_info:
            name = route.get('Description', f"Route {route.get('RouteID')}")
            secs = [s.get('SecondsToNextStop', 0) for s in route.get('Stops', [])]
            route_headway_h[name] = (sum(secs) / 3600.0) if secs else None
        # rd.center_nodes: district name (route Description) -> center short_GEOID
        center_name_by_id = {center: name for name, center in getattr(rd, 'center_nodes', {}).items()}
        block_ids = list(geodata.short_geoid_list)
        provider_odd_cost = 0.0
        if Omega is not None and (J_function_ref is not None):
            for j, center in enumerate(centers_fr):
                # Determine headway for this center's route
                rname = center_name_by_id.get(center)
                H = route_headway_h.get(rname) if rname in route_headway_h else None
                # Aggregate element-wise max Omega over assigned blocks
                omega_vec = None
                for i, bid in enumerate(block_ids):
                    if round(assignment_fr[i, j]) == 1 and bid in Omega:
                        ov = np.array(Omega[bid], dtype=float)
                        omega_vec = ov if omega_vec is None else np.maximum(omega_vec, ov)
                if omega_vec is not None and H and H > 0:
                    provider_odd_cost += float(J_function_ref(omega_vec)) / H
    except Exception:
        provider_odd_cost = None

    provider_dict = {
        'expected': {
            'linehaul_kmph': 0.0 if provider_travel_kmph is not None else None,
            'travel_kmph': provider_travel_kmph,
            'odd_cost': provider_odd_cost,
        },
        'worst': {
            'linehaul_kmph': 0.0 if provider_travel_kmph is not None else None,
            'travel_kmph': provider_travel_kmph,
            'odd_cost': provider_odd_cost,
        }
    }

    # Worst-user wait/transit for FR using route headways and stop-to-destination times
    try:
        # route headways
        route_headways = {}
        for route in rd.routes_info:
            name = route.get('Description', f"Route {route.get('RouteID')}")
            secs = [s.get('SecondsToNextStop', 0) for s in route.get('Stops', [])]
            route_headways[name] = sum(secs)
        # For a rider waiting at a stop, worst-user wait is the full headway (just missed the bus)
        worst_user_wait_time = max(route_headways.values()) if route_headways else None

        # Max transit time to destination over nodes (board at nearest stop, ride to dest stop)
        dest_name = 'Walmart (Garden Center)'
        worst_user_transit_time = 0.0
        for n in nodes:
            info = nearest[n]
            rname = info['route']
            bstop = info['stop']
            route = next((rt for rt in rd.routes_info if rt.get('Description') == rname), None)
            if not route:
                continue
            stops = route.get('Stops', [])
            if not stops:
                continue
            secs = [s.get('SecondsToNextStop', 0) for s in stops]
            names = [s.get('Description', '') for s in stops]
            try:
                i_board = next(i for i, s in enumerate(stops) if s == bstop)
            except StopIteration:
                continue
            dest_idx = next((i for i, nm in enumerate(names) if nm == dest_name), None)
            if dest_idx is None:
                continue
            if dest_idx >= i_board:
                t_iv = sum(secs[i_board:dest_idx])
            else:
                t_iv = sum(secs[i_board:]) + sum(secs[:dest_idx])
            worst_user_transit_time = max(worst_user_transit_time, t_iv)
    except Exception:
        worst_user_wait_time = None
        worst_user_transit_time = None

    rider_block = {
        'expected': {
            'walk_s': expected_walk_time,
            'wait_s': expected_wait_time,
            'transit_s': expected_transit_time,
        },
        'worstP': {
            'walk_s': worstP_walk_time,
            'wait_s': expected_wait_time,   # sunk for FR
            'transit_s': expected_transit_time,  # sunk for FR
        },
        'worst_rider': {
            'walk_s': max_walk_time,
        },
        'worst_user': {
            'walk_s': max_walk_time,
            'wait_s': worst_user_wait_time,
            'transit_s': worst_user_transit_time,
        }
    }
    # Add minutes variants for user metrics for convenience
    try:
        rider_block['expected_min'] = {
            'walk_min': (expected_walk_time or 0.0) / 60.0,
            'wait_min': (expected_wait_time or 0.0) / 60.0,
            'transit_min': (expected_transit_time or 0.0) / 60.0,
        }
        rider_block['worstP_min'] = {
            'walk_min': (worstP_walk_time or 0.0) / 60.0 if worstP_walk_time is not None else None,
            'wait_min': (expected_wait_time or 0.0) / 60.0,
            'transit_min': (expected_transit_time or 0.0) / 60.0,
        }
        rider_block['worst_rider_min'] = {
            'walk_min': (max_walk_time or 0.0) / 60.0,
        }
        rider_block['worst_user_min'] = {
            'walk_min': (max_walk_time or 0.0) / 60.0,
            'wait_min': (worst_user_wait_time or 0.0) / 60.0 if worst_user_wait_time is not None else None,
            'transit_min': (worst_user_transit_time or 0.0) / 60.0 if worst_user_transit_time is not None else None,
        }
    except Exception:
        pass

    return {
        'rider': rider_block,
        'provider': provider_dict,
    }


def _evaluate_tsp_partition(
    geodata,
    assignment: np.ndarray,
    center_list: List[str],
    prob: Dict[str, float],
    Omega: Dict[str, np.ndarray],
    depot_id: str,
    overall_arrival_rate: float,
    use_evaluate: bool = True,
    worst_case: bool = True,
    max_dispatch_interval: float = 24.0,
    override_T_by_partition: Dict[str, float] | None = None,
    epsilon_km: float | None = None,
) -> Dict:
    """
    Evaluate TSP last‑mile metrics for a given partition.

    - Rider metrics: mean wait per rider (T/2), mean transit per rider based on evaluate's
      mean_transit_distance_per_interval divided by expected riders per interval and wv.
    - Provider metrics: linehaul distance rate (2*dist(depot,root)/T) [km/h],
      ODD (J(max Omega in district)/T), travel distance rate (amt_transit_distance) [km/h].
    Returns dict with 'per_district' and 'aggregate' for expected vs worst-P.
    """
    # Constants for conversion
    # Use km/h consistently
    wv = getattr(geodata, 'wv_kmh', getattr(geodata, 'wv', 18.710765208297367 / 1.60934) * 1.60934)
    travel_discount = getattr(geodata, 'tsp_travel_discount', TSP_TRAVEL_DISCOUNT)
    # build evaluator only if needed
    rd = RouteData('data/hct_routes.json', geodata)
    ev = Evaluate(rd, geodata, epsilon=epsilon_km if epsilon_km is not None else 1e-3)

    # Ensure square assignment and centers
    block_ids = list(geodata.short_geoid_list)
    if set(center_list) != set(block_ids):
        # expand small to square for consistency
        # center_list already maps to root ids present in columns of small assignment
        pass

    # Mass per district (for aggregation); this is with nominal prob
    mass_by_root = _aggregate_prob_by_roots(assignment, block_ids, center_list, prob)

    def build_metrics(eval_dict):
        per_district = {}
        for root in center_list:
            dres = eval_dict[root]
            T = float(dres['dispatch_interval'])
            if override_T_by_partition and root in override_T_by_partition:
                T = float(override_T_by_partition[root])

            wait_per_rider_h = T / 2.0

            dist_interval_service = float(dres['mean_transit_distance_per_interval'])
            provider_service_distance = float(
                dres.get(
                    'provider_mean_transit_distance_per_interval',
                    dist_interval_service / travel_discount if travel_discount else dist_interval_service,
                )
            )
            # District mass for aggregation; prefer evaluation's distribution (worst/expected)
            district_mass = float(dres.get('district_prob', mass_by_root.get(root, 0.0)))
            # Average in-vehicle service time per rider uses the shared-travel identity:
            # E[time per rider] = (service distance per interval) / (2 * vehicle speed)
            # (independent of number of riders), under random drop order.
            transit_time_per_rider_h = (dist_interval_service / (2.0 * wv))

            # Note: worst-user metric will be finalized after dr_min is computed below
            rider_worst_user_h = None

            assigned_blocks = _district_blocks(assignment, block_ids, root)
            # Debug: basic assignment/depot checks
            try:
                print(f"[DEBUG] root={root} | T={T:.4f} | assigned={len(assigned_blocks)} | depot_in={depot_id in assigned_blocks}")
                if assigned_blocks:
                    sample = assigned_blocks[: min(5, len(assigned_blocks))]
                    dists_sample = [(b, geodata.get_dist(depot_id, b)) for b in sample]
                    print(f"[DEBUG] sample depot→block distances (km): {dists_sample}")
            except Exception:
                pass
            if assigned_blocks:
                dr_min = min(geodata.get_dist(depot_id, b) for b in assigned_blocks)
            else:
                dr_min = geodata.get_dist(depot_id, root)
            linehaul_kmph = (2.0 * dr_min / T) if T > 0 else 0.0
            if linehaul_kmph == 0.0 and depot_id not in assigned_blocks:
                print(f"[WARN] linehaul_kmph=0 for root={root}, dr_min={dr_min:.6f}, T={T:.6f}, depot not in district")

            omega_vec = None  # lazily initialize to preserve Omega vector shape
            for b in assigned_blocks:
                if b in Omega:
                    ov = np.array(Omega[b], dtype=float)
                    omega_vec = ov if omega_vec is None else np.maximum(omega_vec, ov)
            if omega_vec is None:
                odd_cost = 0.0
            else:
                odd_cost = float(J_function(omega_vec)) / T

            dist_interval_provider_total = provider_service_distance + 2.0 * dr_min
            travel_kmph = (dist_interval_provider_total / T) if T > 0 else 0.0

            # Add on-board half-linehaul time to average rider transit
            transit_time_per_rider_h += (dr_min / wv)

            # Worst-user (per district) in TSP mode:
            # definition: subinterval T plus TSP service time and the outbound half-linehaul time;
            # do not average by rider count.
            rider_worst_user_h = T + (dr_min / wv) + (dist_interval_service / wv)

            per_district[root] = {
                'dispatch_interval_h': T,
                'rider_wait_h': wait_per_rider_h,
                'rider_transit_h': transit_time_per_rider_h,
                'rider_worst_user_h': rider_worst_user_h,
                'rider_wait_min': wait_per_rider_h * 60.0,
                'rider_transit_min': transit_time_per_rider_h * 60.0,
                'rider_worst_user_min': rider_worst_user_h * 60.0,
                'provider_linehaul_kmph': linehaul_kmph,
                'provider_odd_cost': odd_cost,
                'provider_travel_kmph': travel_kmph,
                'mass': district_mass,
            }
        tot_mass = sum(per_district[r]['mass'] for r in center_list) or 1.0
        rider_wait = sum(per_district[r]['rider_wait_h'] * per_district[r]['mass'] for r in center_list) / tot_mass
        rider_transit = sum(per_district[r]['rider_transit_h'] * per_district[r]['mass'] for r in center_list) / tot_mass
        provider_linehaul = sum(per_district[r]['provider_linehaul_kmph'] for r in center_list)
        provider_odd = sum(per_district[r]['provider_odd_cost'] for r in center_list)
        provider_travel = sum(per_district[r]['provider_travel_kmph'] for r in center_list)
        # Aggregate across districts: mass-weighted means for averages, max for worst-user
        worst_user_max = max(per_district[r]['rider_worst_user_h'] for r in center_list) if center_list else 0.0
        return per_district, {
            'rider_wait_h': rider_wait,
            'rider_transit_h': rider_transit,
            'rider_worst_user_h': worst_user_max,
            'rider_wait_min': rider_wait * 60.0,
            'rider_transit_min': rider_transit * 60.0,
            'rider_worst_user_min': worst_user_max * 60.0,
            'provider_linehaul_kmph': provider_linehaul,
            'provider_odd_cost': provider_odd,
            'provider_travel_kmph': provider_travel,
        }

    # Worst-case (let T be re-optimized under worst-P)
    eval_worst = ev.evaluate_tsp_mode(
        prob_dict=prob,
        node_assignment=assignment,
        center_list=center_list,
        unit_wait_cost=1.0,
        overall_arrival_rate=overall_arrival_rate,
        worst_case=True,
        max_dipatch_interval=max_dispatch_interval,
        fixed_T_by_district=None,
    )
    per_worst, agg_worst = build_metrics(eval_worst)

    # Expected (nominal)
    eval_exp = ev.evaluate_tsp_mode(
        prob_dict=prob,
        node_assignment=assignment,
        center_list=center_list,
        unit_wait_cost=1.0,
        overall_arrival_rate=overall_arrival_rate,
        worst_case=False,
        max_dipatch_interval=max_dispatch_interval,
        fixed_T_by_district=override_T_by_partition,
    )
    per_exp, agg_exp = build_metrics(eval_exp)

    return {
        'per_district_worst': per_worst,
        'aggregate_worst': agg_worst,
        'per_district_expected': per_exp,
        'aggregate_expected': agg_exp,
    }


def run(method: str, num_districts: int, visualize_baseline: bool = False, max_iters: int = 200,
        overall_arrival_rate: float = 1000.0, max_dispatch_cap: float = 1.5, epsilon_km_arg: float | None = None,
        demand_source: str = 'population'):
    geodata = load_geodata()
    # Set speeds (km/h). Defaults: wr=5.04 km/h (1.4 m/s), wv=18.7108 km/h.
    # Use constants by default to match notebook; toggle compute_from_routes=True to recompute.
    wv_kmh, wr_kmh = set_geodata_speeds(
        geodata,
        'data/hct_routes.json',
        walk_kmh=5.04,
        vehicle_kmh=18.710765208297367,
        compute_from_routes=False,
    )
    # Nominal distribution: match notebook if requested
    if str(demand_source).lower().startswith('commut'):
        prob = load_probability_from_commuting(geodata)
    else:
        prob = load_probability_from_population(geodata)
    Omega = load_real_odd()
    # Wasserstein radius (km): user-provided or average block diameter (km)
    epsilon_km = epsilon_km_arg if epsilon_km_arg is not None else _avg_block_diameter_km(geodata)

    # Baseline assignment from routes
    base_assign_small, centers = load_baseline_assignment(geodata, visualize=visualize_baseline)

    part = Partition(geodata, num_districts=num_districts, prob_dict=prob, epsilon=epsilon_km)
    base_assign = expand_to_square(part, base_assign_small, centers)
    base_metrics = evaluate_design(part, base_assign, prob, Omega)

    # Optimized design
    district_info = None  # ensure defined for both branches
    if method.upper() == 'LBBD':
        result = part.benders_decomposition(
            max_iterations=50, tolerance=1e-3, verbose=True,
            Omega_dict=Omega, J_function=J_function,
            Lambda=1.0, wr=0.0, wv=10.0, beta=0.7120
        )
        # Support both dict and tuple legacy returns
        if isinstance(result, dict):
            opt_assign = result.get('best_partition')
            opt_depot = result.get('best_depot')
            district_info = result.get('best_district_info')
        else:
            # legacy: (best_partition, best_cost, history)
            opt_assign = result[0]
            opt_depot = None
            district_info = None
        if opt_assign is None:
            raise RuntimeError('LBBD did not return a partition')
        if opt_depot is None:
            # fallback evaluation to compute depot
            metrics = evaluate_design(part, opt_assign, prob, Omega)
            opt_depot = metrics['depot']
        else:
            obj, district_info2 = part.evaluate_partition_objective(
                opt_assign, opt_depot, prob, 1.0, 1.0, 10.0, 0.7120, Omega, J_function
            )
            district_info = district_info or district_info2
            metrics = {
                'obj': float(obj),
                'depot': opt_depot,
                'K_sum': float(sum(info[2] for info in district_info)) if district_info else None,
                'F_sum': float(sum(info[3] for info in district_info)) if district_info else None,
                'num_districts': len(district_info) if district_info else None
            }

        # Relocate optimized roots to be closest-to-depot to align linehaul with root distance
        try:
            opt_assign, _opt_roots_rel = part.relocate_roots_to_depot_closest(opt_assign, opt_depot)
            # Recompute district_info with relocated roots so T* maps correctly by root id
            obj_post, district_info_post = part.evaluate_partition_objective(
                opt_assign, opt_depot, prob, 1.0, 1.0, 10.0, 0.7120, Omega, J_function
            )
            district_info = district_info_post
            metrics['obj'] = float(obj_post)
        except Exception:
            pass
    else:
        depot, roots, assign, obj_val, rs_district_info = part.random_search(
            max_iters=max_iters, prob_dict=prob, Lambda=1.0, wr=0.0, wv=10.0, beta=0.7120,
            Omega_dict=Omega, J_function=J_function
        )
        opt_assign = assign
        district_info = rs_district_info
        metrics = {
            'obj': float(obj_val),
            'depot': depot,
            'K_sum': None,
            'F_sum': None,
            'num_districts': int((assign.sum(axis=0) > 0).sum())
        }

    # ---------- Extended Evaluation Scenarios ----------
    # 1) Fixed-route under original (route-based) partition
    fr_eval = _evaluate_fixed_route_full(geodata, prob, epsilon_km, Omega=Omega, J_function_ref=J_function)

    # Helper: extract center list for partitions (columns with any assignment)
    block_ids = list(geodata.short_geoid_list)
    base_roots = [block_ids[i] for i in range(len(block_ids)) if round(base_assign[i, i]) == 1]
    if not base_roots:
        cols = np.where(base_assign.sum(axis=0) > 0)[0]
        base_roots = [block_ids[i] for i in cols]

    # Compute depot for baseline partition
    base_depot = part.optimize_depot_location(base_assign, base_roots, prob)

    # 2) TSP under original partition (unconstrained and constrained)
    #    Relocate baseline roots to be depot-closest so dist(depot, root) equals the
    #    shortest distance to the district, per modeling requirement.
    try:
        base_assign, base_roots = part.relocate_roots_to_depot_closest(base_assign, base_depot)
    except Exception:
        pass
    # Relocate baseline roots to be depot-closest to align root IDs and distances
    try:
        base_assign, base_roots = part.relocate_roots_to_depot_closest(base_assign, base_depot)
    except Exception:
        pass

    # Compute realistic T* for baseline partition AFTER relocation so root IDs match
    try:
        _obj_base, _district_info_base = part.evaluate_partition_objective(
            base_assign, base_depot, prob,
            Lambda=1.0, wr=0.0, wv=10.0, beta=0.7120,
            Omega_dict=Omega, J_function=J_function
        )
        T_override_base = {str(info[1]): float(info[4]) for info in _district_info_base}
    except Exception:
        T_override_base = None

    tsp_base_unconstrained = _evaluate_tsp_partition(
        geodata, base_assign, base_roots, prob, Omega, base_depot,
        overall_arrival_rate=overall_arrival_rate,
        worst_case=True, max_dispatch_interval=24.0,
        override_T_by_partition=T_override_base, epsilon_km=epsilon_km,
    )
    tsp_base_constrained = _evaluate_tsp_partition(
        geodata, base_assign, base_roots, prob, Omega, base_depot,
        overall_arrival_rate=overall_arrival_rate,
        worst_case=True, max_dispatch_interval=max_dispatch_cap,
        override_T_by_partition=T_override_base, epsilon_km=epsilon_km,
    )

    # 3) TSP under optimal partition (use T* returned by optimization if available; unconstrained + constrained)
    opt_roots = [block_ids[i] for i in range(len(block_ids)) if round(opt_assign[i, i]) == 1]
    if not opt_roots:
        cols = np.where(opt_assign.sum(axis=0) > 0)[0]
        opt_roots = [block_ids[i] for i in cols]

    # Map T* per root from district_info, if available (hours)
    T_override = None
    if district_info:
        try:
            # district_info entries: [cost, root, K_i, F_i, T_star, ...]
            T_override = {str(info[1]): float(info[4]) for info in district_info}
        except Exception:
            T_override = None

    opt_depot = metrics['depot']
    tsp_opt_unconstrained = _evaluate_tsp_partition(
        geodata, opt_assign, opt_roots, prob, Omega, opt_depot,
        overall_arrival_rate=overall_arrival_rate,
        worst_case=True, max_dispatch_interval=24.0,
        override_T_by_partition=T_override, epsilon_km=epsilon_km,
    )
    tsp_opt_constrained = _evaluate_tsp_partition(
        geodata, opt_assign, opt_roots, prob, Omega, opt_depot,
        overall_arrival_rate=overall_arrival_rate,
        worst_case=True, max_dispatch_interval=max_dispatch_cap,
        override_T_by_partition=T_override, epsilon_km=epsilon_km,
    )

    # Report & save
    summary = {
        'method': method,
        'num_districts': num_districts,
        'baseline': base_metrics,
        'optimized': metrics,
        'improvement': float(base_metrics['obj'] - metrics['obj']),
        'evaluations': {
            'FR_original': fr_eval,
            'TSP_original_unconstrained': tsp_base_unconstrained,
            'TSP_original_constrained': tsp_base_constrained,
            'TSP_opt_unconstrained': tsp_opt_unconstrained,
            'TSP_opt_constrained': tsp_opt_constrained,
        }
    }
    print(json.dumps(summary, indent=2))

    os.makedirs('results', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_path = os.path.join('results', f'real_case_summary_{ts}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Also emit a compact CSV-like table for quick comparison
    table_lines = []
    table_lines.append('Design,Perspective,Metric,WorstP_min,Expected_min,ProviderWorst_total_kmph,ProviderExpected_total_kmph')
    # FR
    fr_r = fr_eval['rider']
    fr_p = fr_eval['provider']
    # Convert FR rider seconds to minutes
    fr_walk_worstP_min = (fr_r['worstP']['walk_s'] or 0.0) / 60.0 if fr_r['worstP']['walk_s'] is not None else None
    fr_walk_exp_min = (fr_r['expected']['walk_s'] or 0.0) / 60.0
    fr_wait_exp_min = (fr_r['expected']['wait_s'] or 0.0) / 60.0
    fr_transit_exp_min = (fr_r['expected']['transit_s'] or 0.0) / 60.0
    table_lines.append(f"P0 FR,Rider,walk,{fr_walk_worstP_min},{fr_walk_exp_min},,")
    table_lines.append(f"P0 FR,Rider,wait,,{fr_wait_exp_min},,")
    table_lines.append(f"P0 FR,Rider,transit,,{fr_transit_exp_min},,")
    # FR provider (identical for worst/expected); use travel_kmph aggregate, no linehaul
    fr_provider = fr_eval.get('provider', {})
    try:
        fr_provider_worst = fr_provider.get('worst', {})
        fr_provider_exp = fr_provider.get('expected', {})
        fr_travel_worst = fr_provider_worst.get('travel_kmph')
        fr_travel_exp = fr_provider_exp.get('travel_kmph')
        if fr_travel_worst is not None and fr_travel_exp is not None:
            table_lines.append(f"P0 FR,Provider,travel,,,{fr_travel_worst},{fr_travel_exp}")
    except Exception:
        pass
    # TSP original
    for tag, res in [('P0 TSP unconstrained', tsp_base_unconstrained), ('P0 TSP constrained', tsp_base_constrained)]:
        aggW = res['aggregate_worst']; aggE = res['aggregate_expected']
        table_lines.append(f"{tag},Rider,wait,{aggW['rider_wait_min']},{aggE['rider_wait_min']},{aggW['provider_travel_kmph']},{aggE['provider_travel_kmph']}")
        table_lines.append(f"{tag},Rider,transit,{aggW['rider_transit_min']},{aggE['rider_transit_min']},,")
    # TSP optimal
    for tag, res in [('P* TSP unconstrained', tsp_opt_unconstrained), ('P* TSP constrained', tsp_opt_constrained)]:
        aggW = res['aggregate_worst']; aggE = res['aggregate_expected']
        table_lines.append(f"{tag},Rider,wait,{aggW['rider_wait_min']},{aggE['rider_wait_min']},{aggW['provider_travel_kmph']},{aggE['provider_travel_kmph']}")
        table_lines.append(f"{tag},Rider,transit,{aggW['rider_transit_min']},{aggE['rider_transit_min']},,")

    os.makedirs('results/lbbd', exist_ok=True)
    table_path = os.path.join('results/lbbd', f'real_case_table_{ts}.csv')
    with open(table_path, 'w') as f:
        f.write('\n'.join(table_lines))
    # --- Also emit a LaTeX table in the target format ---
    def _fmt(x):
        return ("" if x is None else f"{x:.2f}")

    # FR user mins
    fr_worst_user = fr_r.get('worst_user_min', {})
    fr_worstP = fr_r.get('worstP_min', {})
    fr_expected = fr_r.get('expected_min', {})

    fr_walk_wu = fr_worst_user.get('walk_min')
    fr_wait_wu = fr_worst_user.get('wait_min')
    fr_trans_wu = fr_worst_user.get('transit_min')
    fr_total_wu = sum(v for v in [fr_walk_wu, fr_wait_wu, fr_trans_wu] if v is not None) if any(
        v is not None for v in [fr_walk_wu, fr_wait_wu, fr_trans_wu]
    ) else None

    fr_walk_wp = fr_worstP.get('walk_min')
    fr_wait_wp = fr_worstP.get('wait_min')
    fr_trans_wp = fr_worstP.get('transit_min')
    fr_total_wp = sum(v for v in [fr_walk_wp, fr_wait_wp, fr_trans_wp] if v is not None) if any(
        v is not None for v in [fr_walk_wp, fr_wait_wp, fr_trans_wp]
    ) else None

    fr_walk_ex = fr_expected.get('walk_min')
    fr_wait_ex = fr_expected.get('wait_min')
    fr_trans_ex = fr_expected.get('transit_min')
    fr_total_ex = sum(v for v in [fr_walk_ex, fr_wait_ex, fr_trans_ex] if v is not None) if any(
        v is not None for v in [fr_walk_ex, fr_wait_ex, fr_trans_ex]
    ) else None

    # TSP aggregates (unconstrained only for the paper-style table)
    bW, bE = tsp_base_unconstrained['aggregate_worst'], tsp_base_unconstrained['aggregate_expected']
    oW, oE = tsp_opt_unconstrained['aggregate_worst'], tsp_opt_unconstrained['aggregate_expected']

    # Build LaTeX
    latex_lines = []
    latex_lines.append("\\begin{table*}[t]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Comparison of Travel Time (minutes) and Provider Distance Rate (km/h)}")
    latex_lines.append("\\begin{tabular}{l l r r r r r}")
    latex_lines.append("\\toprule")
    latex_lines.append("\\textbf{Design} & \\textbf{Metric} & ")
    latex_lines.append("\\multicolumn{3}{c}{\\textbf{User Perspective (min)}} & ")
    latex_lines.append("\\multicolumn{2}{c}{\\textbf{Provider (km/h)}} " + "\\\\")
    latex_lines.append("\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}")
    latex_lines.append("& & \\textbf{Worst-user} & \\textbf{Worst-P} & \\textbf{Expected} & \\textbf{Worst} & \\textbf{Expected} " + "\\\\")
    latex_lines.append("\\midrule")
    # FR rows
    latex_lines.append(f"$\\mathcal{{P}}_0$ FR & walk    & {_fmt(fr_walk_wu)} & {_fmt(fr_walk_wp)} & {_fmt(fr_walk_ex)} &        &        " + "\\\\")
    latex_lines.append(f"                   & wait    & {_fmt(fr_wait_wu)} & {_fmt(fr_wait_wp)} & {_fmt(fr_wait_ex)} & {_fmt(fr_travel_worst)}  & {_fmt(fr_travel_exp)}  " + "\\\\")
    latex_lines.append(f"                   & transit & {_fmt(fr_trans_wu)} & {_fmt(fr_trans_wp)} & {_fmt(fr_trans_ex)} &        &        " + "\\\\")
    latex_lines.append(f"                   & total   & {_fmt(fr_total_wu)} & {_fmt(fr_total_wp)} & {_fmt(fr_total_ex)} &       &        " + "\\\\")
    # ODD line (per hour)
    fr_odd = fr_provider_exp.get('odd_cost') if fr_provider else None
    latex_lines.append(f"                   & ODD (per h) &  &  &  & \\multicolumn{{2}}{{c}}{{{_fmt(fr_odd)}}} " + "\\\\")
    latex_lines.append("\\midrule")
    # P0 TSP (unconstrained)
    latex_lines.append(
        f"$\\mathcal{{P}}_0$ TSP (unconstrained) & wait    &  & {_fmt(bW['rider_wait_min'])} & {_fmt(bE['rider_wait_min'])} & {_fmt(bW['provider_travel_kmph'])} & {_fmt(bE['provider_travel_kmph'])} " + "\\\\")
    latex_lines.append(
        f"                                    & transit &  & {_fmt(bW['rider_transit_min'])} & {_fmt(bE['rider_transit_min'])} &        &       " + "\\\\")
    latex_lines.append(
        f"                                    & total   & {_fmt(bW['rider_worst_user_min'])} & {_fmt(bW['rider_wait_min'] + bW['rider_transit_min'])} & {_fmt(bE['rider_wait_min'] + bE['rider_transit_min'])} &        &       " + "\\\\")
    latex_lines.append("\\midrule")
    # P* TSP (unconstrained)
    latex_lines.append(
        f"$\\mathcal{{P}}^*$ TSP (unconstrained) & wait    &  & {_fmt(oW['rider_wait_min'])} & {_fmt(oE['rider_wait_min'])} & {_fmt(oW['provider_travel_kmph'])} & {_fmt(oE['provider_travel_kmph'])} " + "\\\\")
    latex_lines.append(
        f"                                    & transit &  & {_fmt(oW['rider_transit_min'])} & {_fmt(oE['rider_transit_min'])} &        &       " + "\\\\")
    latex_lines.append(
        f"                                    & total   & {_fmt(oW['rider_worst_user_min'])} & {_fmt(oW['rider_wait_min'] + oW['rider_transit_min'])} & {_fmt(oE['rider_wait_min'] + oE['rider_transit_min'])} &        &       " + "\\\\")
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("")
    latex_lines.append("\\vspace{0.5ex}")
    latex_lines.append("\\small")
    latex_lines.append(
        f"Notes: FR Worst-P uses worstP\\_min; provider values are aggregate travel km/h; ODD from odd\\_cost. Source: {summary_path}."
    )
    latex_lines.append("\\end{table*}")

    latex_path = os.path.join('results', f'real_case_table_{ts}.tex')
    with open(latex_path, 'w') as f:
        f.write("\n".join(latex_lines))

    print(f"Saved summary to {summary_path}, table to {table_path}, and LaTeX to {latex_path}")

    # ---------- Partition plots (before vs after optimization) ----------
    try:
        fig_dir = 'figures'
        os.makedirs(fig_dir, exist_ok=True)
        out_fig = os.path.join(fig_dir, f'partitions_before_after_{ts}.pdf')

        def _colors_by_relative_location(assign_matrix):
            """Return mapping district_label -> color based on centroid order (top-left to bottom-right)."""
            blocks = list(geodata.short_geoid_list)
            klabels = []
            for j in range(assign_matrix.shape[1]):
                if assign_matrix[:, j].sum() > 0:
                    klabels.append(j)
            # Compute simple centroid per district using polygon centroids
            centroids = {}
            cent = geodata.gdf.geometry.centroid
            for j in klabels:
                idxs = [i for i in range(len(blocks)) if round(assign_matrix[i, j]) == 1]
                if not idxs:
                    continue
                xs = [cent.iloc[i].x for i in idxs]
                ys = [cent.iloc[i].y for i in idxs]
                cx, cy = (np.mean(xs), np.mean(ys))
                centroids[j] = (cx, cy)
            # Order: top-left (max y, min x) to bottom-right
            ordered = sorted(centroids.items(), key=lambda kv: (-kv[1][1], kv[1][0]))
            cmap = plt.get_cmap('tab10', max(1, len(ordered)))
            color_map = {}
            for rank, (lab, _) in enumerate(ordered):
                color_map[lab] = cmap(rank)
            return color_map

        def _plot_assignment(ax, assign_matrix, title, depot_id=None):
            """Generic plot for a partition assignment with consistent color ordering."""
            blocks = list(geodata.short_geoid_list)
            labels = [int(np.argmax(assign_matrix[i, :])) if assign_matrix[i, :].sum() > 0 else -1 for i in range(len(blocks))]
            gdfp = geodata.gdf.copy()
            gdfp['district'] = labels
            districts = sorted([d for d in set(labels) if d >= 0])
            color_map = _colors_by_relative_location(assign_matrix)
            for d in districts:
                col = color_map.get(d, '#cccccc')
                try:
                    gdfp[gdfp['district'] == d].plot(ax=ax, color=col, edgecolor='white', linewidth=0.3)
                except Exception:
                    continue
            if depot_id is not None and depot_id in gdfp.index:
                try:
                    gdfp.loc[[depot_id]].boundary.plot(ax=ax, color='red', linewidth=2.0, zorder=6)
                except Exception:
                    pass
            ax.set_title(title)
            ax.set_axis_off()

        def _left_panel(ax):
            # Plot baseline (route-based) assignment using consistent color ordering
            _plot_assignment(ax, base_assign, 'Original partition (baseline)', depot_id=base_depot)

        def _right_panel(ax):
            # Plot optimized partition using the same color ordering rule (relative location)
            _plot_assignment(ax, opt_assign, 'Optimized partition (robust TSP)', depot_id=metrics['depot'])

        fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
        _left_panel(axes[0])
        _right_panel(axes[1])
        fig.savefig(out_fig, dpi=200)
        plt.close(fig)
        print(f"Saved partition plot to {out_fig}")
    except Exception as e:
        print(f"Partition plotting skipped due to error: {e}")


def main():
    ap = argparse.ArgumentParser(description='Run real-case optimization experiment and compare to baseline')
    ap.add_argument('--method', choices=['LBBD', 'RS'], default='LBBD')
    ap.add_argument('--districts', type=int, default=3)
    ap.add_argument('--visualize-baseline', action='store_true')
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--arrival-rate', type=float, default=100.0, help='Overall arrival rate per hour for TSP evaluation')
    ap.add_argument('--Tmax', type=float, default=1.5, help='Constrained dispatch subinterval cap (hours)')
    ap.add_argument('--epsilon-km', type=float, default=None, help='Wasserstein radius in km (default: avg block diameter)')
    ap.add_argument('--demand-source', choices=['population', 'commuting'], default='population', help='Nominal mass source')
    args = ap.parse_args()
    run(args.method, args.districts, args.visualize_baseline, args.iters,
        overall_arrival_rate=args.arrival_rate, max_dispatch_cap=args.Tmax, epsilon_km_arg=args.epsilon_km,
        demand_source=args.demand_source)


if __name__ == '__main__':
    main()
