"""
Depot-origin offline VCC baseline for last-mile service.

This baseline models a non-partition, offline vehicle-customer coordination
service where:
  - all passengers originate at the depot
  - only destinations are sampled from the spatial demand distribution
  - vehicles coordinate destination-side dropoff meeting points
  - explicit fleet size and vehicle capacity are enforced
  - deadline-style service feasibility is checked scenario by scenario

It remains adapted to this repository's comparison framework:
  - results are reported through EvaluationResult
  - linehaul and ODD are zero
  - provider cost is derived from executed vehicle travel
  - no dispatch-interval concept is exposed in the reported metrics
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from lib.baselines.base import (
    BETA,
    VEHICLE_SPEED_KMH,
    BaselineMethod,
    EvaluationResult,
    ServiceDesign,
    _centroid_block,
    _get_pos,
)
from lib.constants import DEFAULT_FLEET_COST_RATE


@dataclass(frozen=True)
class _Request:
    req_id: int
    dest_block: str
    dest_pos: np.ndarray
    release_h: float
    deadline_h: float
    direct_km: float
    direct_time_h: float


@dataclass(frozen=True)
class _Event:
    req_id: int


@dataclass
class _RouteEval:
    objective: float
    total_distance_km: float
    finish_time_h: float
    events: List[_Event]
    positions: List[np.ndarray]
    depot_wait_h: Dict[int, float]
    drop_walk_h: Dict[int, float]
    invehicle_h: Dict[int, float]


def _project_to_ball(point: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """Closest point in the radius-ball around center to point."""
    diff = point - center
    dist = float(np.linalg.norm(diff))
    if dist <= radius or dist < 1e-12:
        return point.copy()
    return center + radius * diff / dist


def _optimize_stop_on_segment(
    prev: np.ndarray,
    nxt: np.ndarray,
    anchor: np.ndarray,
    radius: float,
) -> np.ndarray:
    """
    Best coordinated stop near `anchor` between predecessor and successor.

    The stop can deviate at most `radius` from the passenger's anchor point
    (origin for pickup, destination for dropoff). For interior stops, project
    the anchor onto the predecessor-successor segment and clip to the feasible
    ball. For terminal stops, fall back to the closest point in the ball.
    """
    seg = nxt - prev
    seg_len_sq = float(np.dot(seg, seg))
    if seg_len_sq < 1e-18:
        return _project_to_ball(prev, anchor, radius)

    t = float(np.dot(anchor - prev, seg)) / seg_len_sq
    t = max(0.0, min(1.0, t))
    proj = prev + t * seg
    return _project_to_ball(proj, anchor, radius)


def _route_anchors(events: Sequence[_Event], request_map: Dict[int, _Request]) -> List[np.ndarray]:
    return [request_map[event.req_id].dest_pos for event in events]


def _optimize_route_positions(
    events: Sequence[_Event],
    request_map: Dict[int, _Request],
    start_pos: np.ndarray,
    max_walk_km: float,
    max_iters: int = 20,
) -> List[np.ndarray]:
    """Coordinate-descent stop-position update for a fixed event sequence."""
    if not events:
        return []

    positions = [p.copy() for p in _route_anchors(events, request_map)]
    for _ in range(max_iters):
        changed = False
        for i, event in enumerate(events):
            req = request_map[event.req_id]
            anchor = req.dest_pos
            if len(events) == 1:
                new_pos = _project_to_ball(start_pos, anchor, max_walk_km)
            elif i == len(events) - 1:
                prev = positions[i - 1] if i > 0 else start_pos
                new_pos = _optimize_stop_on_segment(prev, start_pos, anchor, max_walk_km)
            else:
                prev = positions[i - 1] if i > 0 else start_pos
                nxt = positions[i + 1]
                new_pos = _optimize_stop_on_segment(prev, nxt, anchor, max_walk_km)
            if float(np.linalg.norm(new_pos - positions[i])) > 1e-9:
                positions[i] = new_pos
                changed = True
        if not changed:
            break
    return positions


def _evaluate_route(
    events: Sequence[_Event],
    request_map: Dict[int, _Request],
    start_pos: np.ndarray,
    vehicle_capacity: int,
    max_walk_km: float,
    walk_speed_kmh: float,
    vehicle_speed_kmh: float,
    wr: float,
    planning_horizon_h: Optional[float] = None,
) -> Optional[_RouteEval]:
    """
    Evaluate a depot-origin route with coordinated dropoff meeting points.

    Feasibility rules:
      - the route serves at most vehicle_capacity requests
      - dropoff arrival + final walk must meet the request deadline
      - vehicle must be able to return to depot before the planning horizon
    """
    if not events:
        return _RouteEval(
            objective=0.0,
            total_distance_km=0.0,
            finish_time_h=0.0,
            events=[],
            positions=[],
            depot_wait_h={},
            drop_walk_h={},
            invehicle_h={},
        )

    if len(events) > vehicle_capacity:
        return None

    positions = _optimize_route_positions(events, request_map, start_pos, max_walk_km)

    departure_h = max(request_map[event.req_id].release_h for event in events)
    current_pos = start_pos.copy()
    current_time = departure_h
    total_distance = 0.0

    depot_wait: Dict[int, float] = {}
    drop_walk: Dict[int, float] = {}
    invehicle: Dict[int, float] = {}

    for event, stop_pos in zip(events, positions):
        req = request_map[event.req_id]
        travel_km = float(np.linalg.norm(stop_pos - current_pos))
        arrival_h = current_time + travel_km / vehicle_speed_kmh
        total_distance += travel_km

        walk_km = float(np.linalg.norm(stop_pos - req.dest_pos))
        service_h = arrival_h
        arrival_at_dest_h = service_h + walk_km / walk_speed_kmh
        if arrival_at_dest_h > req.deadline_h + 1e-9:
            return None
        depot_wait[event.req_id] = max(0.0, departure_h - req.release_h)
        invehicle[event.req_id] = max(0.0, service_h - departure_h)
        drop_walk[event.req_id] = walk_km / walk_speed_kmh

        current_time = service_h
        current_pos = stop_pos

    return_km = float(np.linalg.norm(start_pos - current_pos))
    return_time_h = return_km / vehicle_speed_kmh
    if planning_horizon_h is not None and current_time + return_time_h > planning_horizon_h + 1e-9:
        return None
    total_distance += return_km
    current_time += return_time_h

    user_time = (
        sum(depot_wait.values())
        + sum(drop_walk.values())
        + sum(invehicle.values())
    )
    objective = total_distance + wr * user_time
    return _RouteEval(
        objective=objective,
        total_distance_km=total_distance,
        finish_time_h=current_time,
        events=list(events),
        positions=positions,
        depot_wait_h=depot_wait,
        drop_walk_h=drop_walk,
        invehicle_h=invehicle,
    )


def _insert_request_events(events: Sequence[_Event], req_id: int) -> List[List[_Event]]:
    """All insertion positions for one depot-origin destination request."""
    base = list(events)
    out: List[List[_Event]] = []
    m = len(base)
    for pick_idx in range(m + 1):
        out.append(base[:pick_idx] + [_Event(req_id)] + base[pick_idx:])
    return out


def _overflow_request_metrics(
    req: _Request,
    start_pos: np.ndarray,
    planning_horizon_h: float,
) -> Dict[str, float]:
    """Overflow direct trip used when the coordinated fleet cannot serve a request.

    Requests rejected by the coordinated insertion heuristic are treated as an
    explicit overflow service dispatched at the end of the offline batch. This
    preserves service completion while making overload visible in the reported
    wait metric instead of silently assigning zero-wait direct service.
    """
    deadhead_out = float(np.linalg.norm(req.dest_pos - start_pos))
    deadhead_back = deadhead_out
    overflow_wait_h = max(0.0, planning_horizon_h - req.release_h)
    return {
        "travel_km": deadhead_out + deadhead_back,
        "wait_h": overflow_wait_h,
        "drop_walk_h": 0.0,
        "invehicle_h": req.direct_time_h,
    }


def _choose_best_insertion(
    requests: Sequence[_Request],
    fleet_size: int,
    start_pos: np.ndarray,
    vehicle_capacity: int,
    max_walk_km: float,
    walk_speed_kmh: float,
    vehicle_speed_kmh: float,
    wr: float,
    exact_limit: int,
    planning_horizon_h: Optional[float],
) -> Tuple[List[_RouteEval], List[_Request]]:
    """
    Solve one offline batch.

    For small batches, do an exhaustive insertion search over request order.
    For larger batches, use best-feasible insertion in release-time order.
    """
    request_map = {req.req_id: req for req in requests}
    ordered = sorted(requests, key=lambda r: (r.release_h, r.req_id))
    empty_eval = _evaluate_route(
        [],
        request_map,
        start_pos,
        vehicle_capacity,
        max_walk_km,
        walk_speed_kmh,
        vehicle_speed_kmh,
        wr,
        planning_horizon_h,
    )
    assert empty_eval is not None

    def candidate_insertions(route_eval: _RouteEval, req: _Request) -> List[_RouteEval]:
        candidates: List[_RouteEval] = []
        for cand_events in _insert_request_events(route_eval.events, req.req_id):
            cand_eval = _evaluate_route(
                cand_events,
                request_map,
                start_pos,
                vehicle_capacity,
                max_walk_km,
                walk_speed_kmh,
                vehicle_speed_kmh,
                wr,
                planning_horizon_h,
            )
            if cand_eval is not None:
                candidates.append(cand_eval)
        return candidates

    if len(ordered) <= exact_limit:
        best_routes: Optional[List[_RouteEval]] = None
        best_score: Optional[Tuple[int, float]] = None
        best_rejected: List[_Request] = []

        def dfs(idx: int, routes: List[_RouteEval], rejected: List[_Request]) -> None:
            nonlocal best_routes, best_score, best_rejected
            if idx == len(ordered):
                served = len(ordered) - len(rejected)
                total_obj = sum(r.objective for r in routes)
                total_obj += sum(
                    (
                        overflow["travel_km"]
                        + wr * (
                            overflow["wait_h"]
                            + overflow["drop_walk_h"]
                            + overflow["invehicle_h"]
                        )
                    )
                    for req in rejected
                    for overflow in [
                        _overflow_request_metrics(req, start_pos, planning_horizon_h or 0.0)
                    ]
                )
                score = (served, -total_obj)
                if best_score is None or score > best_score:
                    best_score = score
                    best_routes = [
                        _evaluate_route(
                            route.events,
                            request_map,
                            start_pos,
                            vehicle_capacity,
                            max_walk_km,
                            walk_speed_kmh,
                            vehicle_speed_kmh,
                            wr,
                            planning_horizon_h,
                        )
                        for route in routes
                    ]
                    best_routes = [r if r is not None else empty_eval for r in best_routes]
                    best_rejected = list(rejected)
                return

            req = ordered[idx]
            inserted = False
            seen_empty = False
            for vidx, route in enumerate(routes):
                if not route.events:
                    if seen_empty:
                        continue
                    seen_empty = True
                for cand in candidate_insertions(route, req):
                    inserted = True
                    new_routes = list(routes)
                    new_routes[vidx] = cand
                    dfs(idx + 1, new_routes, rejected)
            if not inserted:
                dfs(idx + 1, routes, rejected + [req])

        dfs(0, [empty_eval for _ in range(fleet_size)], [])
        return best_routes or [empty_eval for _ in range(fleet_size)], best_rejected

    routes = [empty_eval for _ in range(fleet_size)]
    rejected: List[_Request] = []
    for req in ordered:
        best_vehicle: Optional[int] = None
        best_eval: Optional[_RouteEval] = None
        best_delta = float("inf")
        seen_empty = False
        for vidx, route in enumerate(routes):
            if not route.events:
                if seen_empty:
                    continue
                seen_empty = True
            for cand in candidate_insertions(route, req):
                delta = cand.objective - route.objective
                if delta < best_delta - 1e-9:
                    best_delta = delta
                    best_vehicle = vidx
                    best_eval = cand
        if best_vehicle is None or best_eval is None:
            rejected.append(req)
        else:
            routes[best_vehicle] = best_eval
    return routes, rejected


def _sample_requests(
    geodata,
    prob_dict: Dict[str, float],
    Lambda: float,
    planning_horizon_h: float,
    deadline_slack_h: float,
    depot_id: str,
    rng: np.random.Generator,
) -> List[_Request]:
    """Sample one offline batch of depot-origin last-mile requests."""
    block_ids = geodata.short_geoid_list
    probs = np.array([prob_dict.get(b, 0.0) for b in block_ids], dtype=float)
    probs = probs / max(float(probs.sum()), 1e-12)
    n_requests = int(rng.poisson(Lambda * planning_horizon_h))
    if n_requests <= 0:
        return []

    veh_speed = VEHICLE_SPEED_KMH
    requests: List[_Request] = []
    depot_pos = np.array(_get_pos(geodata, depot_id), dtype=float) / 1000.0
    for req_id in range(n_requests):
        dest_idx = int(rng.choice(len(block_ids), p=probs))
        dest_block = block_ids[dest_idx]
        dest_pos = np.array(_get_pos(geodata, dest_block), dtype=float) / 1000.0
        direct_km = float(np.linalg.norm(dest_pos - depot_pos))
        direct_time_h = direct_km / veh_speed
        release_h = float(rng.uniform(0.0, planning_horizon_h))
        deadline_h = release_h + direct_time_h + deadline_slack_h
        requests.append(
            _Request(
                req_id=req_id,
                dest_block=dest_block,
                dest_pos=dest_pos,
                release_h=release_h,
                deadline_h=deadline_h,
                direct_km=direct_km,
                direct_time_h=direct_time_h,
            )
        )
    return requests


def _scenario_metrics(
    route_evals: Sequence[_RouteEval],
    rejected: Sequence[_Request],
    requests: Sequence[_Request],
    start_pos: np.ndarray,
    planning_horizon_h: float,
) -> Dict[str, float]:
    """Aggregate one scenario into shared evaluation metrics."""
    n_requests = len(requests)
    if n_requests == 0:
        return {
            "wait_h": 0.0,
            "walk_h": 0.0,
            "invehicle_h": 0.0,
            "travel_km": 0.0,
        }

    req_by_id = {req.req_id: req for req in requests}
    total_wait = 0.0
    total_walk = 0.0
    total_invehicle = 0.0
    total_travel_km = 0.0

    for route in route_evals:
        total_travel_km += route.total_distance_km
        for req_id in route.depot_wait_h:
            total_wait += route.depot_wait_h.get(req_id, 0.0)
            total_walk += route.drop_walk_h.get(req_id, 0.0)
            total_invehicle += route.invehicle_h.get(req_id, 0.0)

    for req in rejected:
        overflow = _overflow_request_metrics(req, start_pos, planning_horizon_h)
        total_travel_km += overflow["travel_km"]
        total_wait += overflow["wait_h"]
        total_walk += overflow["drop_walk_h"]
        total_invehicle += overflow["invehicle_h"]

    # Defensive accounting if a request somehow never appears in either set.
    accounted = set()
    for route in route_evals:
        accounted.update(route.depot_wait_h.keys())
    accounted.update(req.req_id for req in rejected)
    for req_id, req in req_by_id.items():
        if req_id in accounted:
            continue
        overflow = _overflow_request_metrics(req, start_pos, planning_horizon_h)
        total_travel_km += overflow["travel_km"]
        total_wait += overflow["wait_h"]
        total_walk += overflow["drop_walk_h"]
        total_invehicle += overflow["invehicle_h"]

    return {
        "wait_h": total_wait / n_requests,
        "walk_h": total_walk / n_requests,
        "invehicle_h": total_invehicle / n_requests,
        "travel_km": total_travel_km,
    }


class VCC(BaselineMethod):
    """
    Depot-origin last-mile VCC baseline adapted from Zhang et al. (2023).

    The design step specifies service parameters only. Actual assignment and
    destination-side coordination and routing are solved per scenario in
    custom_simulate().
    """

    def __init__(
        self,
        fleet_size: Optional[int] = None,
        vehicle_capacity: int = 3,
        max_walk_km: float = 0.3,
        deadline_slack_h: float = 0.5,
        walk_speed_kmh: float = 5.0,
        vehicle_speed_kmh: float = VEHICLE_SPEED_KMH,
        planning_horizon_h: Optional[float] = None,
        n_scenarios: int = 40,
        max_exact_requests: int = 4,
        random_seed: int = 42,
    ):
        self.fleet_size = None if fleet_size is None else int(fleet_size)
        self.vehicle_capacity = int(vehicle_capacity)
        self.max_walk_km = float(max_walk_km)
        self.deadline_slack_h = float(deadline_slack_h)
        self.walk_speed_kmh = float(walk_speed_kmh)
        self.vehicle_speed_kmh = float(vehicle_speed_kmh)
        self.planning_horizon_h = planning_horizon_h
        self.n_scenarios = int(n_scenarios)
        self.max_exact_requests = int(max_exact_requests)
        self.random_seed = int(random_seed)

    def _planning_horizon(self, Lambda: float, fleet_size: int) -> float:
        if self.planning_horizon_h is not None:
            return float(self.planning_horizon_h)
        # Size the offline batch so expected requests are on the order of the
        # coordinated fleet's instantaneous carrying capacity.
        return max(
            1e-3,
            (fleet_size * self.vehicle_capacity) / max(Lambda, 1e-9),
        )

    def design(
        self,
        geodata,
        prob_dict: Dict[str, float],
        Omega_dict,
        J_function: Callable,
        num_districts: int,
        Lambda: float = 1.0,
        wr: float = 1.0,
        wv: float = 10.0,
        **kwargs,
    ) -> ServiceDesign:
        """Create a non-partition DAR-VCC service template."""
        block_ids = geodata.short_geoid_list
        depot_id = _centroid_block(geodata, block_ids)
        fleet_size = (
            int(self.fleet_size)
            if self.fleet_size is not None
            else int(num_districts)
        )
        horizon_h = self._planning_horizon(Lambda, fleet_size)
        metadata = {
            "fleet_size": fleet_size,
            "vehicle_capacity": self.vehicle_capacity,
            "max_walk_km": self.max_walk_km,
            "deadline_slack_h": self.deadline_slack_h,
            "walk_speed_kmh": self.walk_speed_kmh,
            "vehicle_speed_kmh": self.vehicle_speed_kmh,
            "planning_horizon_h": horizon_h,
            "n_scenarios": self.n_scenarios,
            "max_exact_requests": self.max_exact_requests,
            "random_seed": self.random_seed,
            "overflow_mode": "direct_at_horizon",
        }
        return ServiceDesign(
            name="VCC",
            assignment=None,
            depot_id=depot_id,
            district_roots=[],
            dispatch_intervals={},
            district_routes={},
            service_mode="vcc",
            service_metadata=metadata,
        )

    def _simulate_one_scenario(
        self,
        design: ServiceDesign,
        geodata,
        prob_dict: Dict[str, float],
        Lambda: float,
        wr: float,
        rng: np.random.Generator,
    ) -> Dict[str, float]:
        meta = design.service_metadata
        fleet_size = int(meta["fleet_size"])
        horizon_h = float(meta["planning_horizon_h"])
        requests = _sample_requests(
            geodata,
            prob_dict,
            Lambda,
            horizon_h,
            float(meta["deadline_slack_h"]),
            design.depot_id,
            rng,
        )
        if not requests:
            return {
                "wait_h": 0.0,
                "walk_h": 0.0,
                "invehicle_h": 0.0,
                "travel_km": 0.0,
            }

        start_pos = np.array(_get_pos(geodata, design.depot_id), dtype=float) / 1000.0
        routes, rejected = _choose_best_insertion(
            requests=requests,
            fleet_size=fleet_size,
            start_pos=start_pos,
            vehicle_capacity=int(meta["vehicle_capacity"]),
            max_walk_km=float(meta["max_walk_km"]),
            walk_speed_kmh=float(meta["walk_speed_kmh"]),
            vehicle_speed_kmh=float(meta["vehicle_speed_kmh"]),
            wr=wr,
            exact_limit=int(meta["max_exact_requests"]),
            planning_horizon_h=horizon_h,
        )
        return _scenario_metrics(routes, rejected, requests, start_pos, horizon_h)

    def custom_simulate(
        self,
        design: ServiceDesign,
        geodata,
        prob_dict: Dict[str, float],
        Omega_dict,
        J_function: Callable,
        Lambda: float,
        wr: float,
        wv: float,
        fleet_cost_rate: float = DEFAULT_FLEET_COST_RATE,
    ) -> EvaluationResult:
        """Scenario-by-scenario offline DAR-VCC simulation."""
        if design.service_mode != "vcc":
            raise ValueError("VCC.custom_simulate requires a VCC service design.")

        meta = design.service_metadata
        rng = np.random.default_rng(int(meta.get("random_seed", self.random_seed)))
        scenario_stats = [
            self._simulate_one_scenario(design, geodata, prob_dict, Lambda, wr, rng)
            for _ in range(int(meta.get("n_scenarios", self.n_scenarios)))
        ]

        avg_wait = float(np.mean([s["wait_h"] for s in scenario_stats]))
        avg_walk = float(np.mean([s["walk_h"] for s in scenario_stats]))
        avg_invehicle = float(np.mean([s["invehicle_h"] for s in scenario_stats]))
        total_user_time = avg_wait + avg_walk + avg_invehicle

        raw_avg_travel_km = float(np.mean([s["travel_km"] for s in scenario_stats]))
        horizon_h = float(meta["planning_horizon_h"])
        in_district_cost = raw_avg_travel_km / max(horizon_h, 1e-9)
        linehaul_cost = 0.0
        odd_cost = 0.0
        total_travel_cost = in_district_cost
        fleet_size = raw_avg_travel_km / (
            float(meta["vehicle_speed_kmh"]) * max(horizon_h, 1e-9)
        )
        avg_dispatch_interval = horizon_h

        provider_cost = total_travel_cost + fleet_cost_rate * fleet_size
        user_cost = wr * total_user_time
        total_cost = provider_cost + user_cost

        return EvaluationResult(
            name="VCC",
            avg_wait_time=avg_wait,
            avg_invehicle_time=avg_invehicle,
            avg_walk_time=avg_walk,
            total_user_time=total_user_time,
            linehaul_cost=linehaul_cost,
            in_district_cost=in_district_cost,
            total_travel_cost=total_travel_cost,
            fleet_size=fleet_size,
            avg_dispatch_interval=avg_dispatch_interval,
            odd_cost=odd_cost,
            provider_cost=provider_cost,
            user_cost=user_cost,
            total_cost=total_cost,
        )

    def evaluate(
        self,
        design: ServiceDesign,
        geodata,
        prob_dict: Dict[str, float],
        Omega_dict,
        J_function: Callable,
        Lambda: float,
        wr: float,
        wv: float,
        fleet_cost_rate: float = DEFAULT_FLEET_COST_RATE,
        beta: float = BETA,
        road_network=None,
    ) -> EvaluationResult:
        """Keep evaluate() consistent with the experiment-time simulation."""
        return self.custom_simulate(
            design,
            geodata,
            prob_dict,
            Omega_dict,
            J_function,
            Lambda,
            wr,
            wv,
            fleet_cost_rate,
        )


ZhangEtAl2023 = VCC
