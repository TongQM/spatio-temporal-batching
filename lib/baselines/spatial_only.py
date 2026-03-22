"""
Spatial-only baseline (Carlsson et al. 2024): K districts, fixed T.
Represents pure spatial partitioning without temporal dispatch optimisation.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from lib.baselines.base import (
    BETA, BaselineMethod, EvaluationResult, ServiceDesign, evaluate_design,
)


class SpatialOnly(BaselineMethod):
    """K districts via random search (ε=0), fixed global dispatch interval T.

    Corresponds to Carlsson et al. (2024): optimal spatial partitioning,
    but no per-district dispatch optimisation.
    """

    def __init__(self, T_fixed: float = 0.5, max_iters: int = 200):
        """
        Parameters
        ----------
        T_fixed : float
            Fixed dispatch interval in hours (default: 0.5 h = 30 min).
        max_iters : int
            Number of random-search iterations for the partitioning step.
        """
        self.T_fixed = T_fixed
        self.max_iters = max_iters

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
        from lib.algorithm import Partition

        # Use ε=0 (no distributionally robust margin) — pure spatial optimisation
        partition = Partition(geodata, num_districts, prob_dict, epsilon=0.0)
        depot_id, roots, assignment, _, district_info = partition.random_search(
            max_iters=self.max_iters,
            prob_dict=prob_dict,
            Lambda=Lambda,
            wr=wr,
            wv=wv,
            Omega_dict=Omega_dict,
            J_function=J_function,
            T_override=self.T_fixed,
        )

        dispatch_intervals = {r: self.T_fixed for r in roots}

        return ServiceDesign(
            name="SP-Lit",
            assignment=assignment,
            depot_id=depot_id,
            district_roots=list(roots),
            dispatch_intervals=dispatch_intervals,
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
        return EvaluationResult(
            name="SP-Lit",
            **{k: v for k, v in result.__dict__.items() if k != 'name'},
        )
