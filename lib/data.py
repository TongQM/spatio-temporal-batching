import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd
from libpysal.weights import Queen
import polyline
import json
from shapely.geometry import LineString, Point
from shapely.ops import transform
from pyproj import Transformer
import logging
import pyproj


class GeoData:
    def __init__(self, filepath, geoid_list, level='tract'):
        """
        Initialize the geoData object.

        Parameters:
        - filepath: str, path to the shapefile
        - geoid_list: list, list of geoid values to filter
        - level: str, level of geography (default is 'tract')
        """
        self.filepath = filepath
        self.geoid_list = geoid_list
        self.level = level
        if level == 'tract':
            self.gdf = gpd.read_file(filepath)
            self.gdf["short_GEOID"] = self.gdf["GEOID"].str[-6:]
            self.gdf.set_index("short_GEOID", inplace=True)
            self.short_geoid_list = [geoid[-6:] for geoid in geoid_list]
            # Retrieve tracts with the specified GEOIDs
            self.gdf = self.gdf.loc[self.short_geoid_list]            
        
        elif level == 'block_group':
            self.gdf = gpd.read_file(filepath)
            # Pre-filter by FULL GEOID to avoid last-7-char collisions across
            # states/counties (e.g. NY has 16k BGs with only 10k unique 7-char
            # suffixes — tract+BG suffix repeats across counties).
            full_ids = [
                g[-12:] if g.startswith("1500000US") else g
                for g in geoid_list
            ]
            self.gdf = self.gdf[self.gdf["GEOID"].isin(full_ids)].copy()
            self.gdf = self.gdf.drop_duplicates(subset="GEOID")
            self.gdf["short_GEOID"] = self.gdf["GEOID"].str[-7:]
            self.gdf.set_index("short_GEOID", inplace=True)
            # Re-derive short_geoid_list from the filtered gdf to preserve
            # ordering and ensure consistency
            self.short_geoid_list = self.gdf.index.tolist()

        elif level == 'block':
            self.gdf = gpd.read_file(filepath)
            self.gdf["short_GEOID"] = self.gdf["GEOID20"].str[-10:]
            tract_short_geoid_list = [geoid[-6:] for geoid in geoid_list]
            self.gdf = self.gdf[self.gdf["short_GEOID"].str.startswith(tuple(tract_short_geoid_list))]
            self.gdf.set_index("short_GEOID", inplace=True)
            self.short_geoid_list = self.gdf.index.tolist()


        if self.gdf.crs.is_geographic:
            self.gdf = self.gdf.to_crs(epsg=2163)
        
        # Calculate the area for each geometry in square kilometers
        # Note: The area is calculated in the projected coordinate system (EPSG:2163)
        self.gdf['area'] = self.gdf.geometry.area / 1e6 # Convert to square kilometers

        # Compute contiguity weights
        w = Queen.from_dataframe(self.gdf, use_index=True)
        # Build graph using indices from gdf_subset
        self.G = nx.Graph()
        for node, neighbors in w.neighbors.items():
            for neighbor in neighbors:
                self.G.add_edge(node, neighbor)

        # Use centroids of each part for node positions
        self.pos = {short_geoid: (geom.centroid.x, geom.centroid.y) for short_geoid, geom in self.gdf.geometry.items()}
        # self.pos = {i: (geom.centroid.x, geom.centroid.y) for i, geom in enumerate(self.gdf.geometry)}

        # Compute Euclidean distances between centroids for each edge
        self.edge_labels = {}
        self.distance_dict = {}
        for (node1, node2) in self.G.edges():
            pt1 = self.pos[node1]
            pt2 = self.pos[node2]
            distance = ((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)**0.5 / 1000  # in kilometers
            self.G[node1][node2]['distance'] = distance  # Store distance in the graph
            self.edge_labels[(node1, node2)] = f"{distance:.2f} km"
            self.distance_dict[(node1, node2)] = distance

        # Road-distance matrix is populated lazily via attach_road_distances.
        # When set, get_dist() returns road km instead of Euclidean.
        self.road_dist_matrix = None
        self.road_dist_lookup = None
        self._short_to_idx = {b: i for i, b in enumerate(self.short_geoid_list)}

        # Compute shortest path lengths for all pairs using Dijkstra
        shortest_paths = dict(nx.all_pairs_dijkstra_path_length(self.G, weight='distance'))

        # Flatten the result into a dictionary with keys as (node1, node2)
        self.shortest_distance_dict = {}
        for src, paths in shortest_paths.items():
            for tgt, dist in paths.items():
                self.shortest_distance_dict[(src, tgt)] = dist
        
        # Build directed arc list for flow-based constraints
        self._build_arc_structures()

    def _build_arc_structures(self):
        """Build directed arc list and precompute in/out arc dictionaries for flow constraints"""
        # Build directed arc list from undirected graph
        self.arc_list = []
        for u, v in self.G.edges():
            self.arc_list.append((u, v))
            self.arc_list.append((v, u))
        self.arc_list = list(set(self.arc_list))  # Remove duplicates if any
        
        # Precompute out_arcs_dict and in_arcs_dict for all nodes
        self.out_arcs_dict = {
            node: [(node, neighbor) for neighbor in self.G.neighbors(node) if (node, neighbor) in self.arc_list] 
            for node in self.short_geoid_list
        }
        self.in_arcs_dict = {
            node: [(neighbor, node) for neighbor in self.G.neighbors(node) if (neighbor, node) in self.arc_list] 
            for node in self.short_geoid_list
        }
        
        logging.info(f"Built arc structures: {len(self.arc_list)} directed arcs for {len(self.short_geoid_list)} nodes")

    def get_arc_list(self):
        """Get the directed arc list"""
        return self.arc_list
    
    def get_in_arcs(self, node):
        """Get incoming arcs for a specific node"""
        return self.in_arcs_dict.get(node, [])
    
    def get_out_arcs(self, node):
        """Get outgoing arcs for a specific node"""
        return self.out_arcs_dict.get(node, [])

    def get_area(self, bg_geoid):
        return self.gdf['area'][bg_geoid]
    
    def get_dist(self, node1, node2):
        # Prefer real road-network shortest-path km when available.
        if self.road_dist_lookup is not None:
            d = self.road_dist_lookup.get((node1, node2))
            if d is not None:
                return d
        return self.shortest_distance_dict[(node1, node2)]

    def attach_road_distances(self, road_network) -> None:
        """Pre-compute road-network shortest-path distances between every pair
        of block-group centroids and cache as:

          - ``self.road_dist_matrix`` : N×N numpy matrix of road km
          - ``self.road_dist_lookup`` : ``(node1, node2) -> km`` dict
          - ``self.pos_road_km``      : ``node -> (x, y)`` MDS embedding such
            that Euclidean distance between two embedded positions
            approximates the road distance between the corresponding
            block groups

        After this call, ``get_dist`` returns road km; baselines that call
        ``geodata.get_dist`` automatically pick up the change. Baselines that
        use Euclidean ``np.linalg.norm`` over positions can opt in to road
        awareness by reading ``pos_road_km`` instead of ``pos``.
        """
        from pyproj import Transformer
        T = Transformer.from_crs("EPSG:2163", "EPSG:4326", always_xy=False)
        latlon = []
        for b in self.short_geoid_list:
            x, y = self.pos[b]
            lat, lon = T.transform(x, y)
            latlon.append((lat, lon))
        D = road_network.get_dist_matrix_km(latlon)
        # Symmetrise (road network may be slightly asymmetric due to one-ways)
        D = 0.5 * (D + D.T)
        np.fill_diagonal(D, 0.0)
        self.road_dist_matrix = D
        self.road_dist_lookup = {}
        for i, b1 in enumerate(self.short_geoid_list):
            for j, b2 in enumerate(self.short_geoid_list):
                self.road_dist_lookup[(b1, b2)] = float(D[i, j])

        # MDS embed road distances into 2D so Euclidean over embedded positions
        # approximates road km. Useful for baselines that use np.linalg.norm
        # internally (Multi-FR, TP-Lit, VCC).
        try:
            from sklearn.manifold import MDS
            mds = MDS(
                n_components=2,
                dissimilarity="precomputed",
                random_state=42,
                normalized_stress="auto",
                n_init=4,
                max_iter=300,
            )
            # MDS expects km-scale distances; output is also km-scale by default
            embedded = mds.fit_transform(D * 1000.0)  # in metres for stable fit
            embedded = embedded / 1000.0              # back to km
            self.pos_road_km = {
                b: (float(embedded[i, 0]), float(embedded[i, 1]))
                for i, b in enumerate(self.short_geoid_list)
            }
            self.mds_stress = float(getattr(mds, "stress_", -1.0))
        except Exception as e:
            print(f"[GeoData] MDS embedding failed: {e}")
            self.pos_road_km = None
            self.mds_stress = None

    def road_dist_km(self, b1: str, b2: str):
        """Explicit accessor returning road km if matrix is attached, else None."""
        if self.road_dist_lookup is None:
            return None
        return self.road_dist_lookup.get((b1, b2))

    def generate_demand_dist(self, commuting_df, population_df):
        '''
        Legacy code for generating demand distribution based on population and commuting data.
        This function is not used in the current implementation.
        It is recommended to construct the probaility distribution based on the demand data directly.
        '''
        # Population ratio manipulation
        selected_columns = ['Geography', 'Geographic Area Name', 'Estimate Total:']
        block_group_pop_data = population_df[selected_columns]
        new_column_names = ['GEOID', 'block_group_name', 'total_population']
        block_group_pop_data.columns = new_column_names
        tract_population = (
            block_group_pop_data
            .groupby('short_tract_GEOID', as_index=False)['total_population']
            .sum()
            .rename(columns={'total_population': 'tract_population'})
        )
        block_group_pop_data = block_group_pop_data.merge(tract_population, on='short_tract_GEOID', how='left')
        block_group_pop_data['population%'] = block_group_pop_data['total_population'] / block_group_pop_data['tract_population'] * 100

        # Commuting demand estimation proportional to population
        commuting_df = commuting_df.rename(columns={'GEOID': 'block_group_name'})
        commuting_df = commuting_df.merge(block_group_pop_data[['block_group_name', 'population%']], on='block_group_name', how='left')
        commuting_df['commuting_demand'] = commuting_df['population%'] * commuting_df['total_commuting']
        

    def plot_graph(self, savepath='./figures/'):
        # Plot the tracts and overlay the graph with distance labels
        fig, ax = plt.subplots(figsize=(30, 30))
        self.gdf.plot(ax=ax, color='lightgrey', edgecolor='black')
        if self.level == 'block':
            nx.draw(self.G, self.pos, ax=ax, node_color='red', edge_color='blue', with_labels=False, node_size=1)
        else:
            nx.draw(self.G, self.pos, ax=ax, node_color='red', edge_color='blue', with_labels=True, node_size=1)
            nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels=self.edge_labels, font_color='green', font_size=2)
        plt.savefig(f"{savepath}{self.level}_graph.png", dpi=300, bbox_inches='tight')
        plt.close(fig)


    def plot_partition(self, assignment):
        """
        Visualize the partition of block groups into districts.
        Parameters:
        assignment (np.ndarray): Binary array of shape (n_block_groups, n_centers),
                                where each row has exactly one 1.
        gdf (GeoDataFrame): GeoDataFrame of block groups; order must match assignment rows.
        Returns:
        centers (dict): Dictionary mapping each district (center index) to its center block group id.
        """
        # Convert binary assignment to a district label per block group.
        district_labels = np.argmax(assignment, axis=1)
        gdf = self.gdf.copy()  # avoid modifying the original GeoDataFrame
        gdf['district'] = district_labels

        # Try to map district indices to names if possible
        if 'district_name' in gdf.columns:
            district_names = gdf['district_name']
        else:
            district_names = gdf['district']

        # Custom color mapping
        custom_colors = {
            'East Pittsburgh': '#ADD8E6',  # light blue
            'Moronville': '#0000FF',      # blue
            'McKeesport': '#8B4513',      # brown
        }
        # Build color map for all districts
        unique_districts = np.unique(district_labels)
        color_map = {}
        cmap = plt.get_cmap('tab20', len(unique_districts))
        for i, district in enumerate(unique_districts):
            # Try to get the name from the most common value in the group
            name = gdf[gdf['district'] == district]['district'].mode().values[0]
            # If the name is in custom_colors, use it
            if name in custom_colors:
                color_map[district] = custom_colors[name]
            else:
                color_map[district] = cmap(i)

        fig, ax = plt.subplots(figsize=(15, 15))
        for district in unique_districts:
            subset = gdf[gdf['district'] == district]
            subset.plot(ax=ax, color=color_map[district], edgecolor='black')
        centers = {}
        for district in unique_districts:
            subset = gdf[gdf['district'] == district]
            centroids = subset.geometry.centroid
            avg_x = centroids.x.mean()
            avg_y = centroids.y.mean()
            distances = centroids.apply(lambda geom: ((geom.x - avg_x)**2 + (geom.y - avg_y)**2)**0.5)
            center_idx = distances.idxmin()
            centers[district] = center_idx
            ax.plot(avg_x, avg_y, marker='o', color='white', markersize=3)
        plt.show()
        return centers

    
    def get_K(self, block):
        if 'K' in self.gdf.columns:
            return self.gdf.loc[block, 'K']
        else:
            logging.warning(f"K not found for block {block}, returning default 0.0")
            return 0.0

    def get_F(self, block):
        if 'F' in self.gdf.columns:
            return self.gdf.loc[block, 'F']
        else:
            # logging.warning(f"F not found for block {block}, returning default 1.0")
            return 0.0

    def compute_K_for_all_blocks(self, depot_lat=40.38651, depot_lon=-79.82444):
        """
        Compute K_i for each block as the roundtrip distance from the depot (Walmart Garden Center)
        to the centroid of the block, using projected coordinates (meters, then convert to km).
        Store the result in self.gdf['K'] (in kilometers).
        """
        from shapely.geometry import Point
        import pyproj
        # Project depot to EPSG:2163
        wgs84 = pyproj.CRS('EPSG:4326')
        proj = self.gdf.crs
        transformer = pyproj.Transformer.from_crs(wgs84, proj, always_xy=True)
        depot_x, depot_y = transformer.transform(depot_lon, depot_lat)
        depot_pt = Point(depot_x, depot_y)
        K_list = []
        for idx, row in self.gdf.iterrows():
            centroid = row.geometry.centroid
            dist = depot_pt.distance(centroid) / 1000.0  # in km
            K_list.append(2 * dist)  # roundtrip
        self.gdf['K'] = K_list
        return self.gdf['K']
 

class RouteData:
    def __init__(self, data_path, geodata: GeoData):

        assert data_path.endswith('.json'), "Data path must be a JSON file."

        # Load the routes info from a JSON file
        with open(data_path, "r") as f:
            self.routes_info = json.load(f)

        self.geodata = geodata
        self.short_geoid_list = geodata.short_geoid_list
        self.level = geodata.level
        self.gdf = geodata.gdf

        # Create a transformer from EPSG:4326 (lat/lon) to EPSG:2163 (projected in meters)
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:2163", always_xy=True)

    def project_geometry(self, geometry):
        """Project a shapely geometry from EPSG:4326 to EPSG:2163."""
        return transform(self.transformer.transform, geometry)

    def visualize_routes_and_nodes(self):
        """
        Visualize routes overlaid on level nodes (projected in EPSG:2163),
        and return a dictionary mapping each route name to a list of stop info.
        
        Parameters:
        gdf (GeoDataFrame): Node polygons (CRS EPSG:2163).
        routes_info (list): List of route dictionaries, each with keys:
            - "Description": route name.
            - "EncodedPolyline": encoded polyline string.
            - "MapLineColor": color for plotting.
            - "Stops": list of stop dictionaries with keys "Latitude", "Longitude", "Description".
            
        Returns:
            dict: Keys are route names, and values are lists of stop info dictionaries.
        """
        stops_dict = {}
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot block groups as the base layer (already in EPSG:2163)
        self.gdf.plot(ax=ax, color='lightgrey', edgecolor='black')
        
        for route in self.routes_info:
            route_name = route.get("Description", "Unnamed Route")
            encoded_poly = route.get("EncodedPolyline")
            color = route.get("MapLineColor", "#000000")
            stops = route.get("Stops", [])
            
            # Decode the polyline (originally in lat/lon) and project to EPSG:2163.
            if encoded_poly:
                try:
                    # Decode returns (lat, lng) pairs. Swap to (lng, lat) for shapely.
                    coords = polyline.decode(encoded_poly)
                    coords_swapped = [(lng, lat) for lat, lng in coords]
                    route_line = LineString(coords_swapped)
                    projected_line = self.project_geometry(route_line)
                    ax.plot(*projected_line.xy, color=color, linewidth=2, label=route_name)
                except Exception as e:
                    print(f"Error decoding polyline for route '{route_name}': {e}")
            
            route_stops = []
            for stop in stops:
                lat = stop.get("Latitude")
                lng = stop.get("Longitude")
                desc = stop.get("Description", "")
                route_stops.append({"Latitude": lat, "Longitude": lng, "Description": desc})
                
                # Create a Point and project it
                pt = Point(lng, lat)
                projected_pt = self.project_geometry(pt)
                ax.scatter(projected_pt.x, projected_pt.y, color=color, s=50, edgecolor='k', zorder=5)
                ax.text(projected_pt.x, projected_pt.y, desc, fontsize=8, color=color)
            
            stops_dict[route_name] = route_stops

        ax.set_title("Routes and Block Groups (EPSG:2163 Projection)")
        ax.legend()
        plt.show()
        
        return stops_dict
    
    def find_nearest_stops(self):
        """
        For each node in self.gdf, find the nearest stop among all routes in self.routes_info.
        Returns:
            dict: node_idx -> {
                'route': <route Description string>,
                'stop': <original stop dict>,
                'distance': <distance in meters>
            }
        """
        nearest = {}
        # pre‐project all stops once
        all_stops = []
        for route in self.routes_info:
            route_name = route.get("Description", f"Route {route.get('RouteID')}")
            for stop in route.get("Stops", []):
                pt = Point(stop["Longitude"], stop["Latitude"])
                proj = self.project_geometry(pt)
                all_stops.append((route_name, stop, proj))

        # now loop over every node
        for idx, row in self.gdf.iterrows():
            cent = row.geometry.centroid
            best = None
            best_dist = float("inf")
            for route_name, stop_dict, stop_pt in all_stops:
                d = cent.distance(stop_pt)
                if d < best_dist:
                    best_dist = d
                    best = (route_name, stop_dict)
            nearest[idx] = {
                "route": best[0],
                "stop": best[1],
                "distance": best_dist
            }
        return nearest


    def _get_fixed_route_assignment(self, visualize=True):
        """
        Partition nodes into Districts based on the nearest stop, then plot with proper legend.
        """
        # Assign each node to nearest route
        nearest_stops = self.find_nearest_stops()
        district_assignment = {idx: nearest_stops[idx]['route'] for idx in self.gdf.index}
        self.gdf['district'] = self.gdf.index.map(district_assignment)
        center_nodes = {}
        for district, group in self.gdf.groupby('district'):
            centroids = group.geometry.centroid
            mean_point = Point(centroids.x.mean(), centroids.y.mean())
            best_idx, best_dist = None, float('inf')
            for idx, geom in group.geometry.items():
                d = geom.centroid.distance(mean_point)
                if d < best_dist:
                    best_dist, best_idx = d, idx
            center_nodes[district] = best_idx
        self.center_nodes = center_nodes
        if visualize:
            unique_districts = self.gdf['district'].unique()
            # Custom color mapping
            custom_colors = {
                'East Pittsburgh': '#ADD8E6',  # light blue
                'Moronville': '#0000FF',      # blue
                'McKeesport': '#8B4513',      # brown
            }
            cmap = plt.get_cmap('tab20', len(unique_districts))
            color_map = {}
            for i, district in enumerate(unique_districts):
                if district in custom_colors:
                    color_map[district] = custom_colors[district]
                else:
                    color_map[district] = cmap(i)
            fig, ax = plt.subplots(figsize=(10, 10))
            for district in unique_districts:
                subset = self.gdf[self.gdf['district'] == district]
                subset.plot(ax=ax, color=color_map[district], edgecolor='black')
            for route in self.routes_info:
                name = route.get("Description", f"Route {route.get('RouteID')}")
                poly = route.get("EncodedPolyline")
                color = route.get("MapLineColor", '#000000')
                if poly:
                    try:
                        coords = polyline.decode(poly)
                        line = LineString([(lng, lat) for lat, lng in coords])
                        proj_line = self.project_geometry(line)
                        ax.plot(*proj_line.xy, color=color, linewidth=2)
                    except Exception as e:
                        print(f"Error decoding polyline for {name}: {e}")
            for district, idx in center_nodes.items():
                geom = self.gdf.loc[idx].geometry
                boundary = getattr(geom, 'exterior', geom.boundary)
                ax.plot(*boundary.xy, color='red', linewidth=3)
                pt = geom.centroid
                ax.plot(pt.x, pt.y, 'o', color='red', markersize=10)
                ax.text(pt.x, pt.y, str(district), fontsize=12, fontweight='bold', color='red', ha='center', va='center')
            legend_handles = [
                mpatches.Patch(facecolor=color_map[d], edgecolor='black', label=str(d))
                for d in unique_districts
            ]
            ax.legend(handles=legend_handles, title='District', loc='upper right')
            # plt.title("Partition of Block Groups into Districts by Nearest Route Stop")
            plt.show()

    def build_assignment_matrix(self, visualize=True):
        """
        Construct a binary assignment matrix where entry (i, j) = 1 if the i-th block group
        (in self.short_geoid_list) is assigned to the center corresponding to the j-th center.
        Returns:
            assignment: np.ndarray of shape (n_groups, n_centers)
            center_list: list of center node identifiers (short GEOID index)
        """
        self._get_fixed_route_assignment(visualize=visualize)
        center_list = list(self.center_nodes.values())
        n_groups = len(self.short_geoid_list)
        n_centers = len(center_list)
        assignment = np.zeros((n_groups, n_centers), dtype=int)
        geo_to_row = {geoid: i for i, geoid in enumerate(self.short_geoid_list)}
        center_to_col = {center: j for j, center in enumerate(center_list)}
        for geoid in self.short_geoid_list:
            i = geo_to_row[geoid]
            district = self.gdf.loc[geoid, 'district']
            center = self.center_nodes[district]
            j = center_to_col[center]
            assignment[i, j] = 1
        return assignment, center_list


    def evaluate_partition(self, assignment, mode="fixed"):
        """
        Evaluate the partition of nodes into districts.
        
        Parameters:
        assignment (np.ndarray): Binary array of shape (n_block_groups, n_centers),
                                where each row has exactly one 1.
        mode (str): Mode of evaluation, either "fixed" or "tsp".
        
        Returns:
        centers (dict): Dictionary mapping each district (center index) to its center block group id.
        """
        assert mode in ["fixed", "tsp"], "Invalid mode. Choose 'fixed' or 'tsp'."

        # Convert binary assignment to a district label per block group.
        district_labels = np.argmax(assignment, axis=1)
        gdf = self.gdf.copy()



# class DemandData:
#     def __init__(self, meta_path, data_path, level='tract', datatype='commuting'):
#         """
#         Load data from a CSV file and rename columns based on metadata.
        
#         Parameters:
#         - meta_path: Path to the metadata CSV file.
#         - data_path: Path to the data CSV file.
        
#         Returns:
#         - DataFrame with renamed columns.
#         """
#         assert datatype in ['commuting', 'population'], "Invalid datatype. Choose 'commuting' or 'population'."
#         if datatype == 'commuting' and level != 'tract':
#             raise ValueError("Commuting data is only available at the tract level.")
        
#         self.level = level
#         meta = pd.read_csv(meta_path)
#         code_to_label = dict(zip(meta["Column Name"], meta["Label"]))
        
#         self.data = pd.read_csv(data_path)
#         self.data.rename(columns=code_to_label, inplace=True)
        
#         # Remove "!!" from column names
#         self.data.columns = self.data.columns.str.replace("!!", " ", regex=False)


def load_data(meta_path, data_path):
    """
    Load data from a CSV file and rename columns based on metadata.
    
    Parameters:
    - meta_path: Path to the metadata CSV file.
    - data_path: Path to the data CSV file.
    
    Returns:
    - DataFrame with renamed columns.
    """
    meta = pd.read_csv(meta_path)
    code_to_label = dict(zip(meta["Column Name"], meta["Label"]))
    
    data = pd.read_csv(data_path)
    data.rename(columns=code_to_label, inplace=True)
    
    # Remove "!!" from column names
    data.columns = data.columns.str.replace("!!", " ", regex=False)
    
    return data

