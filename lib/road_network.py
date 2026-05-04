"""
Road network integration via OSMnx.
Fetches street graphs, computes road-distance matrices, and caches results.
"""

import os
import hashlib
import pickle
import logging
from typing import List, Tuple, Optional

import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)

try:
    import osmnx as ox
    _OSMNX_AVAILABLE = True
except ImportError:
    _OSMNX_AVAILABLE = False
    logger.warning("osmnx not installed. RoadNetwork will be unavailable. "
                   "Install with: pip install osmnx")


def _cache_path(key: str, cache_dir: str, suffix: str) -> str:
    h = hashlib.sha1(key.encode()).hexdigest()
    return os.path.join(cache_dir, f"road_{suffix}_{h}.pkl")


class RoadNetwork:
    """Road network built from OSMnx, with shortest-path distance queries.

    Usage
    -----
    rn = RoadNetwork.from_bbox(40.50, 40.35, -79.85, -80.10)
    d = rn.get_dist_km(40.44, -79.99, 40.45, -80.00)   # road km between two lat/lon points
    D = rn.get_dist_matrix_km([(40.44, -79.99), (40.45, -80.00)])  # N×N matrix
    """

    def __init__(self, G):
        if not _OSMNX_AVAILABLE:
            raise ImportError("osmnx is required. Install with: pip install osmnx")
        self.G = ox.projection.project_graph(G, to_crs="EPSG:4326")
        # Pre-project to speed up nearest-node queries
        self._G_proj = ox.projection.project_graph(G)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_bbox(cls, north: float, south: float, east: float, west: float,
                  network_type: str = 'drive', cache_dir: str = 'cache') -> 'RoadNetwork':
        """Fetch drive network within a bounding box, with caching."""
        if not _OSMNX_AVAILABLE:
            raise ImportError("osmnx is required. Install with: pip install osmnx")
        os.makedirs(cache_dir, exist_ok=True)
        key = f"bbox:{north:.5f}:{south:.5f}:{east:.5f}:{west:.5f}:{network_type}"
        path = _cache_path(key, cache_dir, 'graph')
        if os.path.exists(path):
            logger.info(f"Loading cached road network from {path}")
            with open(path, 'rb') as f:
                G = pickle.load(f)
        else:
            logger.info(f"Fetching road network for bbox ({north},{south},{east},{west})")
            # osmnx 2.x: graph_from_bbox takes a tuple (left, bottom, right, top) = (west, south, east, north)
            try:
                G = ox.graph_from_bbox(bbox=(west, south, east, north), network_type=network_type)
            except TypeError:
                # osmnx 1.x fallback: keyword args (north, south, east, west)
                G = ox.graph_from_bbox(north=north, south=south, east=east, west=west,
                                       network_type=network_type)
            with open(path, 'wb') as f:
                pickle.dump(G, f)
            logger.info(f"Cached road network to {path}")
        return cls(G)

    @classmethod
    def from_place(cls, city_name: str, network_type: str = 'drive',
                   cache_dir: str = 'cache') -> 'RoadNetwork':
        """Fetch drive network for a place name (e.g., 'Pittsburgh, PA'), with caching."""
        if not _OSMNX_AVAILABLE:
            raise ImportError("osmnx is required. Install with: pip install osmnx")
        os.makedirs(cache_dir, exist_ok=True)
        key = f"place:{city_name}:{network_type}"
        path = _cache_path(key, cache_dir, 'graph')
        if os.path.exists(path):
            logger.info(f"Loading cached road network from {path}")
            with open(path, 'rb') as f:
                G = pickle.load(f)
        else:
            logger.info(f"Fetching road network for '{city_name}'")
            G = ox.graph_from_place(city_name, network_type=network_type)
            with open(path, 'wb') as f:
                pickle.dump(G, f)
            logger.info(f"Cached road network to {path}")
        return cls(G)

    # ------------------------------------------------------------------
    # Distance queries
    # ------------------------------------------------------------------

    def _nearest_node(self, lat: float, lon: float) -> int:
        return ox.distance.nearest_nodes(self.G, lon, lat)

    def get_dist_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Road-network shortest-path distance in km between two lat/lon points."""
        n1 = self._nearest_node(lat1, lon1)
        n2 = self._nearest_node(lat2, lon2)
        try:
            length_m = nx.shortest_path_length(self.G, n1, n2, weight='length')
            return length_m / 1000.0
        except nx.NetworkXNoPath:
            # Fallback: straight-line distance
            from math import radians, cos, sin, asin, sqrt
            R = 6371.0
            dlat = radians(lat2 - lat1)
            dlon = radians(lon2 - lon1)
            a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
            return R * 2 * asin(sqrt(a))

    def get_dist_matrix_km(self, latlon_list: List[Tuple[float, float]],
                           cache_dir: str = 'cache') -> np.ndarray:
        """N×N road-distance matrix (km) for a list of (lat, lon) points.

        Results are cached keyed on the sorted point list so repeated calls
        with the same set of centroids are free.
        """
        n = len(latlon_list)
        if n == 0:
            return np.zeros((0, 0))

        os.makedirs(cache_dir, exist_ok=True)
        key = "distmat:" + ":".join(f"{lat:.5f},{lon:.5f}" for lat, lon in latlon_list)
        path = _cache_path(key, cache_dir, 'distmat')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)

        nodes = [self._nearest_node(lat, lon) for lat, lon in latlon_list]
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    length_m = nx.shortest_path_length(self.G, nodes[i], nodes[j], weight='length')
                    D[i, j] = D[j, i] = length_m / 1000.0
                except nx.NetworkXNoPath:
                    # Straight-line fallback
                    D[i, j] = D[j, i] = self.get_dist_km(*latlon_list[i], *latlon_list[j])

        with open(path, 'wb') as f:
            pickle.dump(D, f)
        return D
