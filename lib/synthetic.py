"""Synthetic geodata utilities for experiments and visualization.

Provides lightweight GeoData-compatible objects with arbitrary point
locations (uniform random, scattered, etc.) for synthetic experiments
where a real shapefile is not available.
"""

import networkx as nx
import numpy as np


class SyntheticGeoData:
    """Minimal GeoData with arbitrary point locations.

    Parameters
    ----------
    positions_km : dict
        All location IDs → (x_km, y_km).
    checkpoint_ids : list, optional
        Subset of IDs that serve as route checkpoints.  If None, all
        locations are treated as checkpoints (backward compatible).
    """

    def __init__(self, positions_km, checkpoint_ids=None):
        self.short_geoid_list = list(positions_km.keys())
        self.n_blocks = len(self.short_geoid_list)
        self.level = "block_group"
        self.gdf = None
        self.pos = {
            b: (x * 1000.0, y * 1000.0) for b, (x, y) in positions_km.items()
        }
        self.checkpoint_ids = (
            list(checkpoint_ids) if checkpoint_ids is not None
            else list(self.short_geoid_list)
        )
        self.G = nx.complete_graph(self.short_geoid_list)
        self.block_graph = self.G
        self.arc_list = []
        for u, v in self.G.edges():
            self.arc_list.append((u, v))
            self.arc_list.append((v, u))

    def get_dist(self, b1, b2):
        p1 = np.array(self.pos[b1], dtype=float)
        p2 = np.array(self.pos[b2], dtype=float)
        return float(np.linalg.norm(p1 - p2))

    def get_area(self, _):
        return 0.25

    def get_K(self, _):
        return 2.0

    def get_F(self, _):
        return 0.0

    def get_arc_list(self):
        return self.arc_list


def make_uniform_positions(seed=17, n_checkpoints=10, n_stops=30,
                           low=0.3, high=4.7):
    """Generate uniform random checkpoint and stopping-location positions.

    Returns (all_positions, checkpoint_ids) where all_positions is a dict
    of all location IDs → (x_km, y_km) and checkpoint_ids is the subset
    of IDs that define route geometry.
    """
    rng = np.random.default_rng(seed)
    positions = {}
    checkpoint_ids = []
    for i in range(n_checkpoints):
        bid = f"C{i:02d}"
        positions[bid] = (float(rng.uniform(low, high)),
                          float(rng.uniform(low, high)))
        checkpoint_ids.append(bid)
    for i in range(n_stops):
        bid = f"S{i:02d}"
        positions[bid] = (float(rng.uniform(low, high)),
                          float(rng.uniform(low, high)))
    return positions, checkpoint_ids


def make_scattered_positions(seed=7, n_blocks=25, low=0.2, high=4.8):
    """Generate scattered random positions in a square region."""
    rng = np.random.RandomState(seed)
    return {
        f"B{i:02d}": (round(rng.uniform(low, high), 2), round(rng.uniform(low, high), 2))
        for i in range(n_blocks)
    }


def positions_array(geodata, block_ids):
    """Convert geodata positions (meters) to km numpy array."""
    return np.array([np.array(geodata.pos[b], dtype=float) / 1000.0 for b in block_ids])


def build_synthetic_instance(
    positions_km,
    checkpoint_ids=None,
    demand_support="stops_only",
):
    """Build a complete synthetic instance from a positions dict.

    Parameters
    ----------
    positions_km : dict
        All location IDs → (x_km, y_km).
    checkpoint_ids : list, optional
        Subset of IDs that serve as route checkpoints.  If None, all
        locations are checkpoints (backward compatible).
    demand_support : {"stops_only", "all_locations"}, optional
        Support of the synthetic demand distribution. ``"stops_only"``
        preserves the historical behavior of placing demand only on
        non-checkpoint locations. ``"all_locations"`` places demand on
        every candidate service location, including checkpoint candidates.

    Returns (geodata, block_ids, block_pos_km, prob_dict, Omega_dict).
    """
    geodata = SyntheticGeoData(positions_km, checkpoint_ids=checkpoint_ids)
    block_ids = geodata.short_geoid_list
    block_pos_km = positions_array(geodata, block_ids)

    checkpoint_set = set(geodata.checkpoint_ids)
    if demand_support == "stops_only":
        support_ids = [b for b in block_ids if b not in checkpoint_set]
        if not support_ids:
            support_ids = block_ids  # fallback: all are stops
    elif demand_support == "all_locations":
        support_ids = list(block_ids)
    else:
        raise ValueError(
            "demand_support must be one of {'stops_only', 'all_locations'}"
        )

    prob_dict = {b: 1.0 / len(support_ids) if b in set(support_ids) else 0.0
                 for b in block_ids}
    Omega_dict = {b: np.array([1.0]) for b in block_ids}
    return geodata, block_ids, block_pos_km, prob_dict, Omega_dict
