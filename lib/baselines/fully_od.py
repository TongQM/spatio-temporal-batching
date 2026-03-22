"""
Fully on-demand (OD) baseline: one district, dispatch every T → 0.
Represents maximum route flexibility, no spatial or temporal batching.
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from lib.baselines.base import (
    BETA, BaselineMethod, EvaluationResult, ServiceDesign, _get_pos, evaluate_design,
)


class FullyOD(BaselineMethod):
    """Single district, T ≈ 0 (T_min = 1 min) dispatch interval.

    Represents the on-demand extreme: every request is served individually
    (or nearly so), minimising wait time at the expense of high travel cost.
    """

    def __init__(self, T_min: float = 1.0 / 60.0):
        """
        Parameters
        ----------
        T_min : float
            Dispatch interval in hours (default: 1 minute = 1/60 h).
        """
        self.T_min = T_min

    def design(
        self,
        geodata,
        prob_dict: Dict[str, float],
        Omega_dict,
        J_function: Callable,
        num_districts: int,
        **kwargs,
    ) -> ServiceDesign:
        block_ids = geodata.short_geoid_list
        N = len(block_ids)

        # One district: assign every block to the first block as root
        # Root = block whose index is 0; all blocks assigned to column 0
        assignment = np.zeros((N, N))
        assignment[:, 0] = 1.0
        assignment[0, 0] = 1.0  # root self-assigns
        root = block_ids[0]

        # Depot = block closest to area centroid (approximated by geographic mean)
        positions = np.array([_get_pos(geodata, b) for b in block_ids])
        centroid = positions.mean(axis=0)
        dist_to_centroid = np.linalg.norm(positions - centroid, axis=1)
        depot_idx = int(np.argmin(dist_to_centroid))
        depot_id = block_ids[depot_idx]

        return ServiceDesign(
            name="OD",
            assignment=assignment,
            depot_id=depot_id,
            district_roots=[root],
            dispatch_intervals={root: self.T_min},
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
        return evaluate_design(
            design, geodata, prob_dict, Omega_dict, J_function,
            Lambda, wr, wv, beta, road_network,
        )
