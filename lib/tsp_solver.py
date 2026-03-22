"""
TSP tour length computation on arbitrary distance matrices.
Uses python_tsp when available; falls back to 2 × MST.
"""

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from typing import List, Tuple

try:
    from python_tsp.heuristics import solve_tsp_local_search, solve_tsp_simulated_annealing
    from python_tsp.exact import solve_tsp_dynamic_programming
    _PYTHON_TSP_AVAILABLE = True
except Exception:
    _PYTHON_TSP_AVAILABLE = False


def tsp_tour_length(dist_matrix: np.ndarray) -> float:
    """Return TSP tour length given an N×N distance matrix (any units).

    Solver selection:
      n ≤ 1  : 0.0
      n == 2 : 2 × dist[0,1]  (out-and-back)
      n ≤ 12 : exact dynamic programming
      n ≤ 30 : simulated annealing (2 s budget)
      n > 30 : local search
    Falls back to 2 × MST if python_tsp is unavailable or raises.
    """
    n = len(dist_matrix)
    if n <= 1:
        return 0.0
    if n == 2:
        return 2.0 * float(dist_matrix[0, 1])

    if _PYTHON_TSP_AVAILABLE:
        try:
            if n <= 12:
                _, dist = solve_tsp_dynamic_programming(dist_matrix)
            elif n <= 30:
                _, dist = solve_tsp_simulated_annealing(dist_matrix, max_processing_time=2.0)
            else:
                _, dist = solve_tsp_local_search(dist_matrix)
            return float(dist)
        except Exception:
            pass  # fall through to MST

    mst = minimum_spanning_tree(dist_matrix)
    return 2.0 * float(mst.sum())


def euclidean_dist_matrix(points: List[Tuple[float, float]]) -> np.ndarray:
    """Compute N×N Euclidean distance matrix from list of (x, y) coordinates."""
    if len(points) == 0:
        return np.zeros((0, 0))
    coords = np.array(points)
    return squareform(pdist(coords, 'euclidean'))
