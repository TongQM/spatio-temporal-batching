"""
Joint optimisation baselines (Ours): K districts, per-district optimal T*.

Two variants:
  JointNom — optimised under nominal demand (ε = 0)
  JointDRO — optimised under worst-case Wasserstein ball (ε > 0)

Both are evaluated under the same fixed demand pattern (prob_dict).
The only difference is the epsilon used *during optimisation* (design phase).
"""
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from lib.baselines.base import (
    BETA, BaselineMethod, EvaluationResult, ServiceDesign, evaluate_design,
)


def _joint_design(
    name: str,
    epsilon: float,
    max_iters: int,
    geodata,
    prob_dict: Dict[str, float],
    Omega_dict,
    J_function: Callable,
    num_districts: int,
    Lambda: float,
    wr: float,
    wv: float,
) -> ServiceDesign:
    from lib.algorithm import Partition

    partition = Partition(geodata, num_districts, prob_dict, epsilon=epsilon)
    depot_id, roots, assignment, _, district_info = partition.random_search(
        max_iters=max_iters,
        prob_dict=prob_dict,
        Lambda=Lambda,
        wr=wr,
        wv=wv,
        Omega_dict=Omega_dict,
        J_function=J_function,
    )
    # district_info: list of (cost, root, K_i, F_i, T_star, x_star)
    dispatch_intervals = {info[1]: info[4] for info in district_info}
    return ServiceDesign(
        name=name,
        assignment=assignment,
        depot_id=depot_id,
        district_roots=list(roots),
        dispatch_intervals=dispatch_intervals,
    )


class JointNom(BaselineMethod):
    """Joint spatial+temporal optimisation under nominal demand (ε = 0).

    Optimises partition and T* treating the empirical distribution as exact.
    Corresponds to 'Joint-Nom' in the comparison table.
    """

    def __init__(self, max_iters: int = 200):
        self.max_iters = max_iters

    def design(self, geodata, prob_dict, Omega_dict, J_function,
               num_districts, Lambda=1.0, wr=1.0, wv=10.0, **kwargs) -> ServiceDesign:
        return _joint_design(
            "Joint-Nom", epsilon=0.0, max_iters=self.max_iters,
            geodata=geodata, prob_dict=prob_dict, Omega_dict=Omega_dict,
            J_function=J_function, num_districts=num_districts,
            Lambda=Lambda, wr=wr, wv=wv,
        )

    def evaluate(self, design, geodata, prob_dict, Omega_dict, J_function,
                 Lambda, wr, wv, beta=BETA, road_network=None) -> EvaluationResult:
        result = evaluate_design(design, geodata, prob_dict, Omega_dict, J_function,
                                 Lambda, wr, wv, beta, road_network)
        return EvaluationResult(name="Joint-Nom",
                                **{k: v for k, v in result.__dict__.items() if k != 'name'})


class JointDRO(BaselineMethod):
    """Joint spatial+temporal optimisation under Wasserstein DRO (ε > 0).

    Optimises partition and T* against the worst-case distribution in a
    Wasserstein ball of radius ε around the empirical distribution.
    Corresponds to 'Joint-DRO' in the comparison table.
    """

    def __init__(self, epsilon: float = 1e-3, max_iters: int = 200):
        self.epsilon = epsilon
        self.max_iters = max_iters

    def design(self, geodata, prob_dict, Omega_dict, J_function,
               num_districts, Lambda=1.0, wr=1.0, wv=10.0, **kwargs) -> ServiceDesign:
        return _joint_design(
            "Joint-DRO", epsilon=self.epsilon, max_iters=self.max_iters,
            geodata=geodata, prob_dict=prob_dict, Omega_dict=Omega_dict,
            J_function=J_function, num_districts=num_districts,
            Lambda=Lambda, wr=wr, wv=wv,
        )

    def evaluate(self, design, geodata, prob_dict, Omega_dict, J_function,
                 Lambda, wr, wv, beta=BETA, road_network=None) -> EvaluationResult:
        result = evaluate_design(design, geodata, prob_dict, Omega_dict, J_function,
                                 Lambda, wr, wv, beta, road_network)
        return EvaluationResult(name="Joint-DRO",
                                **{k: v for k, v in result.__dict__.items() if k != 'name'})


# Keep old alias for backward compatibility
SpatioTemporal = JointDRO
