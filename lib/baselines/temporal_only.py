"""
Temporal-only baseline (Liu & Luo 2023): one district, optimal T*.
Represents pure temporal batching with no spatial partitioning.
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from lib.baselines.base import (
    BETA, BaselineMethod, EvaluationResult, ServiceDesign, _bhh_alpha,
    _get_pos, _newton_T_star, evaluate_design,
)
from lib.constants import TSP_TRAVEL_DISCOUNT


class TemporalOnly(BaselineMethod):
    """Single district, Newton-optimal dispatch interval T*.

    Corresponds to Liu & Luo (2023): no spatial partition, but the dispatch
    interval is optimised jointly over provider and user costs.
    """

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
        block_ids = geodata.short_geoid_list
        N = len(block_ids)

        # All blocks → single district rooted at block 0
        assignment = np.zeros((N, N))
        assignment[:, 0] = 1.0
        root = block_ids[0]

        # Depot = geographic centroid block
        positions = np.array([_get_pos(geodata, b) for b in block_ids])
        centroid = positions.mean(axis=0)
        dist_to_centroid = np.linalg.norm(positions - centroid, axis=1)
        depot_id = block_ids[int(np.argmin(dist_to_centroid))]

        # Compute K and F for the single district
        K_i = min(geodata.get_dist(depot_id, b) for b in block_ids)
        if Omega_dict and J_function:
            district_omega = np.zeros_like(next(iter(Omega_dict.values())))
            for b in block_ids:
                district_omega = np.maximum(district_omega, Omega_dict.get(b, np.zeros_like(district_omega)))
            F_i = J_function(district_omega)
        else:
            F_i = 0.0

        alpha_i = _bhh_alpha(geodata, block_ids, prob_dict)
        T_star = _newton_T_star(K_i, F_i, alpha_i, wr, wv, TSP_TRAVEL_DISCOUNT)

        return ServiceDesign(
            name="TP-Lit",
            assignment=assignment,
            depot_id=depot_id,
            district_roots=[root],
            dispatch_intervals={root: T_star},
        )

    def evaluate(
        self,
        design: ServiceDesign,
        geodata,
        prob_dict,
        Omega_dict,
        J_function,
        Lambda: float,
        wr: float,
        wv: float,
        beta: float = BETA,
        road_network=None,
    ) -> EvaluationResult:
        result = evaluate_design(
            design, geodata, prob_dict, Omega_dict, J_function,
            Lambda, wr, wv, beta, road_network,
        )
        # Override name
        return EvaluationResult(
            name="TP-Lit",
            **{k: v for k, v in result.__dict__.items() if k != 'name'},
        )
