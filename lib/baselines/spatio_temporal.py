"""
Joint optimisation baselines (Ours): K districts, per-district optimal T*.

Two variants:
  JointNom — optimised under nominal demand (ε = 0)
  JointDRO — optimised under worst-case Wasserstein ball (ε > 0)

Both are evaluated under the same fixed demand pattern (prob_dict).
The only difference is the epsilon used *during optimisation* (design phase).
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

from lib.baselines.base import (
    BETA, BaselineMethod, EvaluationResult, ServiceDesign, evaluate_design, simulate_design,
)


class _RefinedToyGeoData:
    """Synthetic grid with finer block resolution over the same spatial extent."""

    def __init__(self, coarse_geodata, factor: int):
        import networkx as nx

        coarse_gs = int(coarse_geodata.grid_size)
        self.factor = int(factor)
        self.grid_size = coarse_gs * self.factor
        self.n_blocks = self.grid_size * self.grid_size
        self.level = coarse_geodata.level
        self.gdf = None
        self.short_geoid_list = [f"BLK{i:03d}" for i in range(self.n_blocks)]
        self.block_to_coord = {i: (i // self.grid_size, i % self.grid_size) for i in range(self.n_blocks)}
        self._cell_size_km = 1.0 / self.factor
        self.pos = {}
        self.parent_block = {}
        for i, bid in enumerate(self.short_geoid_list):
            row, col = self.block_to_coord[i]
            x_km = (col + 0.5) / self.factor - 0.5
            y_km = (row + 0.5) / self.factor - 0.5
            self.pos[bid] = (x_km * 1000.0, y_km * 1000.0)
            parent_row = row // self.factor
            parent_col = col // self.factor
            parent_idx = parent_row * coarse_gs + parent_col
            self.parent_block[bid] = coarse_geodata.short_geoid_list[parent_idx]

        self.G = nx.Graph()
        for bid in self.short_geoid_list:
            self.G.add_node(bid)
        for i in range(self.n_blocks):
            r1, c1 = self.block_to_coord[i]
            for dr, dc in ((1, 0), (0, 1)):
                rr, cc = r1 + dr, c1 + dc
                if rr >= self.grid_size or cc >= self.grid_size:
                    continue
                j = rr * self.grid_size + cc
                self.G.add_edge(self.short_geoid_list[i], self.short_geoid_list[j])
        self.block_graph = self.G
        self.arc_list = []
        for e in self.G.edges():
            self.arc_list.append(e)
            self.arc_list.append((e[1], e[0]))

    def get_dist(self, b1, b2):
        p1 = np.array(self.pos[b1], dtype=float) / 1000.0
        p2 = np.array(self.pos[b2], dtype=float) / 1000.0
        return float(np.linalg.norm(p1 - p2))

    def get_area(self, _):
        return self._cell_size_km ** 2

    def get_K(self, _):
        return 2.0

    def get_F(self, _):
        return 0.0

    def get_arc_list(self):
        return self.arc_list


def _refine_synthetic_inputs(
    geodata,
    prob_dict: Dict[str, float],
    Omega_dict,
    factor: int,
) -> Tuple[object, Dict[str, float], Dict[str, np.ndarray], Dict[str, object]]:
    refined = _RefinedToyGeoData(geodata, factor)
    refined_prob: Dict[str, float] = {}
    refined_omega: Dict[str, np.ndarray] = {}
    scale = factor ** 2
    for bid in refined.short_geoid_list:
        parent = refined.parent_block[bid]
        refined_prob[bid] = prob_dict.get(parent, 0.0) / scale
        if Omega_dict:
            refined_omega[bid] = np.array(Omega_dict.get(parent, np.array([0.0])), copy=True)
    metadata = {
        "refined_geodata": refined,
        "refined_prob_dict": refined_prob,
        "refined_Omega_dict": refined_omega,
        "plot_block_ids": refined.short_geoid_list,
        "plot_block_pos_km": np.array(
            [np.array(refined.pos[b], dtype=float) / 1000.0 for b in refined.short_geoid_list]
        ),
        "plot_cell_size_km": refined._cell_size_km,
        "plot_grid_size": refined.grid_size,
        "synthetic_refine_factor": factor,
    }
    return refined, refined_prob, refined_omega, metadata


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
    synthetic_refine_factor: int,
) -> ServiceDesign:
    from lib.algorithm import Partition

    opt_geodata = geodata
    opt_prob = prob_dict
    opt_omega = Omega_dict
    service_metadata = {}
    if synthetic_refine_factor > 1 and hasattr(geodata, "grid_size") and hasattr(geodata, "short_geoid_list"):
        opt_geodata, opt_prob, opt_omega, service_metadata = _refine_synthetic_inputs(
            geodata, prob_dict, Omega_dict, synthetic_refine_factor
        )

    partition = Partition(opt_geodata, num_districts, opt_prob, epsilon=epsilon)
    depot_id, roots, assignment, _, district_info = partition.random_search(
        max_iters=max_iters,
        prob_dict=opt_prob,
        Lambda=Lambda,
        wr=wr,
        wv=wv,
        Omega_dict=opt_omega,
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
        service_mode="partition",
        service_metadata=service_metadata,
    )


class JointNom(BaselineMethod):
    """Joint spatial+temporal optimisation under nominal demand (ε = 0).

    Optimises partition and T* treating the empirical distribution as exact.
    Corresponds to 'Joint-Nom' in the comparison table.
    """

    def __init__(self, max_iters: int = 200, synthetic_refine_factor: int = 2):
        self.max_iters = max_iters
        self.synthetic_refine_factor = synthetic_refine_factor

    def design(self, geodata, prob_dict, Omega_dict, J_function,
               num_districts, Lambda=1.0, wr=1.0, wv=10.0, **kwargs) -> ServiceDesign:
        return _joint_design(
            "Joint-Nom", epsilon=0.0, max_iters=self.max_iters,
            geodata=geodata, prob_dict=prob_dict, Omega_dict=Omega_dict,
            J_function=J_function, num_districts=num_districts,
            Lambda=Lambda, wr=wr, wv=wv,
            synthetic_refine_factor=self.synthetic_refine_factor,
        )

    def custom_simulate(self, design, geodata, prob_dict, Omega_dict, J_function,
                        Lambda, wr, wv):
        sim_geodata = design.service_metadata.get("refined_geodata", geodata)
        sim_prob = design.service_metadata.get("refined_prob_dict", prob_dict)
        sim_omega = design.service_metadata.get("refined_Omega_dict", Omega_dict)
        result = simulate_design(
            design, sim_geodata, sim_prob, sim_omega, J_function, Lambda, wr, wv
        )
        return EvaluationResult(name="Joint-Nom",
                                **{k: v for k, v in result.__dict__.items() if k != 'name'})

    def evaluate(self, design, geodata, prob_dict, Omega_dict, J_function,
                 Lambda, wr, wv, beta=BETA, road_network=None) -> EvaluationResult:
        sim_geodata = design.service_metadata.get("refined_geodata", geodata)
        sim_prob = design.service_metadata.get("refined_prob_dict", prob_dict)
        sim_omega = design.service_metadata.get("refined_Omega_dict", Omega_dict)
        result = evaluate_design(design, sim_geodata, sim_prob, sim_omega, J_function,
                                 Lambda, wr, wv, beta, road_network)
        return EvaluationResult(name="Joint-Nom",
                                **{k: v for k, v in result.__dict__.items() if k != 'name'})


class JointDRO(BaselineMethod):
    """Joint spatial+temporal optimisation under Wasserstein DRO (ε > 0).

    Optimises partition and T* against the worst-case distribution in a
    Wasserstein ball of radius ε around the empirical distribution.
    Corresponds to 'Joint-DRO' in the comparison table.
    """

    def __init__(self, epsilon: float = 1e-3, max_iters: int = 200, synthetic_refine_factor: int = 2):
        self.epsilon = epsilon
        self.max_iters = max_iters
        self.synthetic_refine_factor = synthetic_refine_factor

    def design(self, geodata, prob_dict, Omega_dict, J_function,
               num_districts, Lambda=1.0, wr=1.0, wv=10.0, **kwargs) -> ServiceDesign:
        return _joint_design(
            "Joint-DRO", epsilon=self.epsilon, max_iters=self.max_iters,
            geodata=geodata, prob_dict=prob_dict, Omega_dict=Omega_dict,
            J_function=J_function, num_districts=num_districts,
            Lambda=Lambda, wr=wr, wv=wv,
            synthetic_refine_factor=self.synthetic_refine_factor,
        )

    def custom_simulate(self, design, geodata, prob_dict, Omega_dict, J_function,
                        Lambda, wr, wv):
        sim_geodata = design.service_metadata.get("refined_geodata", geodata)
        sim_prob = design.service_metadata.get("refined_prob_dict", prob_dict)
        sim_omega = design.service_metadata.get("refined_Omega_dict", Omega_dict)
        result = simulate_design(
            design, sim_geodata, sim_prob, sim_omega, J_function, Lambda, wr, wv
        )
        return EvaluationResult(name="Joint-DRO",
                                **{k: v for k, v in result.__dict__.items() if k != 'name'})

    def evaluate(self, design, geodata, prob_dict, Omega_dict, J_function,
                 Lambda, wr, wv, beta=BETA, road_network=None) -> EvaluationResult:
        sim_geodata = design.service_metadata.get("refined_geodata", geodata)
        sim_prob = design.service_metadata.get("refined_prob_dict", prob_dict)
        sim_omega = design.service_metadata.get("refined_Omega_dict", Omega_dict)
        result = evaluate_design(design, sim_geodata, sim_prob, sim_omega, J_function,
                                 Lambda, wr, wv, beta, road_network)
        return EvaluationResult(name="Joint-DRO",
                                **{k: v for k, v in result.__dict__.items() if k != 'name'})


# Keep old alias for backward compatibility
SpatioTemporal = JointDRO
