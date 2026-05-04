"""City-grid GeoData wrapper for cross-topology comparison.

Drops a synthetic 5x5 grid of "block" centroids inside a real-city bbox,
then exposes a GeoData-compatible interface so the regime sweep can run
each baseline through its normal pipeline. Distances between blocks are
either Euclidean (for the design layer) or road-network shortest-path
(when a RoadNetwork is supplied). Block "areas" are uniform.

Usage:
    cg = CityGeoData("manhattan", bbox=(40.78, 40.74, -73.96, -74.00),
                     grid_size=5)
    rn = cg.road_network  # OSMnx graph fetched and cached
    # Use cg as a GeoData drop-in: cg.short_geoid_list, cg.pos, ...
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import networkx as nx
from shapely.geometry import Polygon
import geopandas as gpd
from pyproj import Transformer

from lib.road_network import RoadNetwork


@dataclass
class CityConfig:
    name: str
    bbox: Tuple[float, float, float, float]   # (north, south, east, west)
    description: str = ""


# Cities chosen to span road topologies.
# bbox sized ~5x5 km so the 5x5 grid has ~1 km blocks.
CITY_CONFIGS = {
    "pittsburgh": CityConfig(
        "pittsburgh",
        (40.475, 40.425, -79.92, -79.99),
        "Hilly irregular network with rivers (PGH downtown + Oakland)",
    ),
    "chicago": CityConfig(
        "chicago",
        (41.91, 41.86, -87.61, -87.66),
        "Strict cardinal grid (Chicago Loop / River North)",
    ),
    "manhattan": CityConfig(
        "manhattan",
        (40.785, 40.745, -73.96, -74.00),
        "Avenue-and-street grid with diagonal Broadway (Midtown)",
    ),
    "la": CityConfig(
        "la",
        (34.075, 34.025, -118.22, -118.28),
        "Sprawled downtown grid with freeways (DTLA)",
    ),
}


class CityGeoData:
    """GeoData drop-in: 5x5 grid of synthetic blocks in a real city's bbox.

    Provides:
      short_geoid_list   - block IDs
      pos[bid]           - (x_m, y_m) in EPSG:2163 meters (consistent with GeoData)
      get_dist(b1, b2)   - Euclidean km between block centroids
      get_area(b)        - block area in km^2 (uniform)
      gdf                - GeoDataFrame of square block polygons (EPSG:2163)
      block_to_coord     - (row, col) index for each block
      G                  - 4-neighbor contiguity graph
      arc_list, block_graph - aliases for compatibility
    """

    def __init__(self, city: str, grid_size: int = 5, road_cache: str = "cache"):
        if city not in CITY_CONFIGS:
            raise ValueError(f"Unknown city {city!r}. Known: {list(CITY_CONFIGS)}")
        self.city = city
        self.cfg = CITY_CONFIGS[city]
        self.grid_size = int(grid_size)
        self.n_blocks = self.grid_size * self.grid_size
        self.level = "block_group"  # for GeoData duck-typing

        north, south, east, west = self.cfg.bbox
        self.bbox = self.cfg.bbox

        # Lay a regular grid of cell centroids in lat/lon
        lat_step = (north - south) / self.grid_size
        lon_step = (east - west) / self.grid_size
        self.short_geoid_list = []
        self.block_to_coord = {}
        cell_polys = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                bid = f"{city.upper()[:3]}{r:02d}{c:02d}"
                self.short_geoid_list.append(bid)
                self.block_to_coord[len(self.short_geoid_list) - 1] = (r, c)
                # Cell polygon (lat/lon)
                lat_lo = south + r * lat_step
                lat_hi = south + (r + 1) * lat_step
                lon_lo = west + c * lon_step
                lon_hi = west + (c + 1) * lon_step
                # NOTE: Polygon expects (lon, lat) for x,y
                cell_polys.append(
                    Polygon([(lon_lo, lat_lo), (lon_hi, lat_lo),
                             (lon_hi, lat_hi), (lon_lo, lat_hi)])
                )

        # Build GeoDataFrame, project to EPSG:2163 (matches Pittsburgh GeoData)
        self.gdf = gpd.GeoDataFrame(
            {"geoid": self.short_geoid_list},
            geometry=cell_polys,
            crs="EPSG:4326",
        ).to_crs("EPSG:2163")

        # pos[bid] in EPSG:2163 meters (centroid)
        self.pos = {}
        centroids = self.gdf.geometry.centroid
        for bid, c in zip(self.short_geoid_list, centroids):
            self.pos[bid] = (float(c.x), float(c.y))

        # Block area in km^2 (uniform)
        # Each cell is roughly (lat_step * 111km) x (lon_step * 111km * cos(lat))
        mid_lat = 0.5 * (north + south)
        cell_h_km = lat_step * 111.0
        cell_w_km = lon_step * 111.0 * math.cos(math.radians(mid_lat))
        self._area_km2 = float(cell_h_km * cell_w_km)

        # Distance cache (Euclidean km between centroids)
        self._dist = {}
        bids = self.short_geoid_list
        for i, b1 in enumerate(bids):
            x1, y1 = self.pos[b1]
            for j, b2 in enumerate(bids):
                x2, y2 = self.pos[b2]
                d_m = math.hypot(x1 - x2, y1 - y2)
                self._dist[(b1, b2)] = d_m / 1000.0

        # Contiguity graph (4-neighbor)
        self.G = nx.Graph()
        for bid in self.short_geoid_list:
            self.G.add_node(bid)
        for i in range(self.n_blocks):
            r1, c1 = self.block_to_coord[i]
            for dr, dc in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                rr, cc = r1 + dr, c1 + dc
                if 0 <= rr < self.grid_size and 0 <= cc < self.grid_size:
                    j = rr * self.grid_size + cc
                    self.G.add_edge(self.short_geoid_list[i],
                                    self.short_geoid_list[j])
        self.block_graph = self.G
        self.arc_list = []
        for e in self.G.edges():
            self.arc_list.append(e)
            self.arc_list.append((e[1], e[0]))

        # Road network (lazily initialised)
        self._road_network: Optional[RoadNetwork] = None
        self._road_cache_dir = road_cache

    # ------------------------------------------------------------------
    # GeoData duck-typing API
    # ------------------------------------------------------------------
    def get_dist(self, b1: str, b2: str) -> float:
        """Euclidean km between centroids."""
        return self._dist.get((b1, b2), float("inf"))

    def get_area(self, _bid: str) -> float:
        return self._area_km2

    def get_K(self, _bid: str) -> float:
        return 2.0  # arbitrary constant linehaul placeholder (not used in regime view)

    def get_F(self, _bid: str) -> float:
        return 0.0

    def get_arc_list(self):
        return self.arc_list

    # ------------------------------------------------------------------
    # Road network on demand
    # ------------------------------------------------------------------
    @property
    def road_network(self) -> RoadNetwork:
        if self._road_network is None:
            north, south, east, west = self.bbox
            pad = 0.005
            self._road_network = RoadNetwork.from_bbox(
                north + pad, south - pad, east + pad, west - pad,
                cache_dir=self._road_cache_dir,
            )
        return self._road_network
