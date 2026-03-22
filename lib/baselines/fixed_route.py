"""
Fixed-Route (FR) baseline: fixed routes with fixed headway T_fixed.
Represents pure temporal batching (no spatial optimisation).

For synthetic grids: K vertical strips as routes, blocks assigned to nearest strip.
For real cities: accepts an external assignment matrix (e.g. from inputs.py).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from lib.baselines.base import (
    BETA, BaselineMethod, EvaluationResult, ServiceDesign, _get_pos, evaluate_design,
)


class FixedRoute(BaselineMethod):
    """Fixed routes, fixed headway.

    Parameters
    ----------
    T_fixed : float
        Fixed dispatch headway in hours (default: 0.5 h = 30 min).
    assignment : np.ndarray or None
        Pre-computed N×N assignment matrix.  If None, vertical-strip
        heuristic is used (suitable for synthetic grids).
    depot_id : str or None
        Pre-specified depot block ID.  If None, the geographic centroid
        block is used.
    num_strips : int
        Number of strips for the synthetic heuristic (default = num_districts).
    """

    def __init__(
        self,
        T_fixed: float = 0.5,
        assignment: Optional[np.ndarray] = None,
        depot_id: Optional[str] = None,
        num_strips: Optional[int] = None,
    ):
        self.T_fixed = T_fixed
        self._preset_assignment = assignment
        self._preset_depot = depot_id
        self.num_strips = num_strips

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
        n_strips = self.num_strips if self.num_strips is not None else num_districts

        if self._preset_assignment is not None:
            assignment = self._preset_assignment
        else:
            assignment = _vertical_strip_assignment(geodata, block_ids, n_strips)

        # Identify roots (blocks whose column has assignment[i,i]=1)
        roots: List[str] = []
        for i in range(N):
            if round(assignment[i, i]) == 1:
                roots.append(block_ids[i])

        if self._preset_depot is not None:
            depot_id = self._preset_depot
        else:
            positions = np.array([_get_pos(geodata, b) for b in block_ids])
            centroid = positions.mean(axis=0)
            dist_to_centroid = np.linalg.norm(positions - centroid, axis=1)
            depot_id = block_ids[int(np.argmin(dist_to_centroid))]

        dispatch_intervals = {r: self.T_fixed for r in roots}

        return ServiceDesign(
            name="FR",
            assignment=assignment,
            depot_id=depot_id,
            district_roots=roots,
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
        return evaluate_design(
            design, geodata, prob_dict, Omega_dict, J_function,
            Lambda, wr, wv, beta, road_network,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vertical_strip_assignment(geodata, block_ids: List[str], n_strips: int) -> np.ndarray:
    """Assign blocks to vertical strips by x-coordinate; root = leftmost block per strip."""
    N = len(block_ids)
    xs = np.array([_get_pos(geodata, b)[0] for b in block_ids])
    x_min, x_max = xs.min(), xs.max()
    boundaries = np.linspace(x_min, x_max, n_strips + 1)

    # Strip index for each block
    strip_idx = np.searchsorted(boundaries[1:], xs)  # 0 … n_strips-1

    # One root per strip = block with median x in the strip
    roots_col = {}
    for s in range(n_strips):
        members = np.where(strip_idx == s)[0]
        if len(members) == 0:
            # empty strip: absorb into previous
            continue
        median_member = members[len(members) // 2]
        roots_col[s] = median_member

    assignment = np.zeros((N, N))
    for s, root_col in roots_col.items():
        members = np.where(strip_idx == s)[0]
        for m in members:
            assignment[m, root_col] = 1.0
        assignment[root_col, root_col] = 1.0  # ensure self-assignment

    return assignment
