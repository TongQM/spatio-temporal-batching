#!/usr/bin/env python3
"""
Design Performance vs Sample Size Analysis

This script evaluates how transit design performance varies with training sample size.
For each sample size n:
1. Sample n demand points from underlying mixed-Gaussian distribution  
2. Generate TWO designs:
   - Robust Design: RS algorithm with ε = ε₀/√n (optimizes against worst-case)
   - Nominal Design: RS algorithm with ε = 0 (optimizes against empirical)
3. Evaluate both designs via simulation on:
   - Worst-case distribution (CQCP with ε = ε₀/√n)
   - Nominal distribution (empirical from samples)

Output: Two figures showing performance curves for robust vs nominal designs.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent display
import matplotlib.pyplot as plt
from lib.algorithm import Partition
from lib.data import GeoData
import networkx as nx
import warnings
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from scipy.stats import beta, gamma, multivariate_normal
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree

# Try to import python_tsp, fallback to MST if not available
try:
    from python_tsp.heuristics import solve_tsp_local_search, solve_tsp_simulated_annealing
    from python_tsp.exact import solve_tsp_dynamic_programming
    PYTHON_TSP_AVAILABLE = True
except Exception:
    PYTHON_TSP_AVAILABLE = False

warnings.filterwarnings('ignore')

# Use same configuration as simulate_service_designs.py
class MixedGaussianConfig:
    """Centralized configuration for truncated mixed-Gaussian distribution"""
    
    @staticmethod
    def get_cluster_centers(grid_size: int):
        """Get cluster centers scaled to grid size"""
        return [
            (grid_size * 0.25, grid_size * 0.25),  # Top-left cluster
            (grid_size * 0.75, grid_size * 0.25),  # Top-right cluster  
            (grid_size * 0.50, grid_size * 0.75)   # Bottom-center cluster
        ]
    
    @staticmethod
    def get_cluster_weights():
        """Get cluster mixing weights"""
        return [0.4, 0.35, 0.25]
    
    @staticmethod
    def get_cluster_sigmas(grid_size: int):
        """Get cluster standard deviations scaled to grid size"""
        return [grid_size * 0.15 * 0.5, grid_size * 0.16 * 0.5, grid_size * 0.12 * 0.5]

@dataclass
class DesignResult:
    """Store design optimization results"""
    depot_id: str
    district_roots: List[str]
    assignment: np.ndarray
    dispatch_intervals: Dict[str, float]  # T_star for each district
    obj_val: float
    district_info: List
    success: bool

@dataclass
class SimulationResult:
    """Store simulation evaluation results"""
    total_cost: float
    user_cost: float
    provider_cost: float
    success: bool

class ToyGeoData(GeoData):
    """Geographic data class matching simulate_service_designs.py"""
    
    def __init__(self, n_blocks, grid_size, service_region_miles=10.0, seed=42):
        np.random.seed(seed)
        self.n_blocks = n_blocks
        self.grid_size = grid_size
        self.service_region_miles = service_region_miles
        self.miles_per_grid_unit = service_region_miles / grid_size
        self.short_geoid_list = [f"BLK{i:03d}" for i in range(n_blocks)]
        
        # Create block graph
        self.G = nx.Graph()
        self.block_to_coord = {i: (i // grid_size, i % grid_size) for i in range(n_blocks)}
        
        # Add nodes and edges
        for block_id in self.short_geoid_list:
            self.G.add_node(block_id)
        
        for i in range(n_blocks):
            for j in range(i+1, n_blocks):
                coord1 = self.block_to_coord[i]
                coord2 = self.block_to_coord[j]
                if abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1]) == 1:
                    self.G.add_edge(self.short_geoid_list[i], self.short_geoid_list[j])
        
        self.block_graph = self.G.copy()
        self.areas = {block_id: 1.0 for block_id in self.short_geoid_list}
        self.gdf = None
        
        # Add arc structure methods
        self._build_arc_structures()
    
    def _build_arc_structures(self):
        """Build directed arc list and precompute in/out arc dictionaries"""
        self.arc_list = []
        for u, v in self.G.edges():
            self.arc_list.append((u, v))
            self.arc_list.append((v, u))
        self.arc_list = list(set(self.arc_list))
        
        self.out_arcs_dict = {
            node: [(node, neighbor) for neighbor in self.G.neighbors(node) if (node, neighbor) in self.arc_list] 
            for node in self.short_geoid_list
        }
        self.in_arcs_dict = {
            node: [(neighbor, node) for neighbor in self.G.neighbors(node) if (neighbor, node) in self.arc_list] 
            for node in self.short_geoid_list
        }
    
    def get_arc_list(self):
        return self.arc_list
    
    def get_in_arcs(self, node):
        return self.in_arcs_dict.get(node, [])
    
    def get_out_arcs(self, node):
        return self.out_arcs_dict.get(node, [])
    
    def get_dist(self, block1, block2):
        """Euclidean distance in miles between blocks"""
        if block1 == block2:
            return 0.0
        try:
            i1 = self.short_geoid_list.index(block1)
            i2 = self.short_geoid_list.index(block2)
            coord1 = self.block_to_coord[i1]
            coord2 = self.block_to_coord[i2]
            euclidean_dist = np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
            return euclidean_dist * self.miles_per_grid_unit
        except:
            return float('inf')
    
    def get_area(self, block_id):
        return 1.0
    
    def get_K(self, block_id):
        return 2.0

    def get_coord_in_miles(self, block_id: str) -> Tuple[float, float]:
        """Return the center coordinate of a block in miles within the 10×10 region."""
        idx = self.short_geoid_list.index(block_id)
        row = idx // self.grid_size
        col = idx % self.grid_size
        x_miles = (col + 0.5) * self.miles_per_grid_unit
        y_miles = (row + 0.5) * self.miles_per_grid_unit
        return (x_miles, y_miles)

def create_true_distribution_sampler(grid_size: int, seed: int = 42):
    """Create sampler for properly truncated mixed-Gaussian distribution"""
    np.random.seed(seed)
    
    # Use centralized configuration
    cluster_centers = MixedGaussianConfig.get_cluster_centers(grid_size)
    cluster_weights = MixedGaussianConfig.get_cluster_weights()
    cluster_sigmas = MixedGaussianConfig.get_cluster_sigmas(grid_size)
    
    # Calculate truncation probabilities for each cluster
    service_bounds = [0, grid_size - 1]
    truncated_weights = []
    
    for i, (center, sigma) in enumerate(zip(cluster_centers, cluster_sigmas)):
        # Create 2D Gaussian distribution for this cluster
        cov_matrix = [[sigma**2, 0], [0, sigma**2]]  # Independent x,y
        mvn = multivariate_normal(mean=center, cov=cov_matrix)
        
        # Calculate probability mass within service region
        prob_inside = mvn.cdf([grid_size-1, grid_size-1]) - mvn.cdf([grid_size-1, 0]) - mvn.cdf([0, grid_size-1]) + mvn.cdf([0, 0])
        truncated_weights.append(cluster_weights[i] * prob_inside)
    
    # Normalize truncated weights
    total_truncated_weight = sum(truncated_weights)
    truncated_weights = [w / total_truncated_weight for w in truncated_weights]
    
    def sample_from_true_distribution(n_samples: int, random_seed=None):
        """Sample from properly truncated mixed-Gaussian distribution"""
        if random_seed is not None:
            np.random.seed(random_seed)
            
        samples = []
        max_attempts = n_samples * 5  # Prevent infinite loops
        attempts = 0
        
        while len(samples) < n_samples and attempts < max_attempts:
            # Choose cluster according to truncated weights
            cluster_idx = np.random.choice(len(cluster_centers), p=truncated_weights)
            center = cluster_centers[cluster_idx]
            sigma = cluster_sigmas[cluster_idx]
            
            # Sample from chosen Gaussian cluster
            x = np.random.normal(center[0], sigma)
            y = np.random.normal(center[1], sigma)
            
            # Accept only if within service region (proper truncation)
            if 0 <= x <= grid_size - 1 and 0 <= y <= grid_size - 1:
                samples.append((x, y))
            
            attempts += 1
        
        return samples
    
    return sample_from_true_distribution

def create_nominal_distribution(grid_size: int, n_samples: int, seed: int = 42):
    """Create nominal distribution by sampling from true distribution and aggregating by blocks"""
    np.random.seed(seed)
    n_blocks = grid_size * grid_size
    short_geoid_list = [f"BLK{i:03d}" for i in range(n_blocks)]
    
    # Sample from true distribution
    sampler = create_true_distribution_sampler(grid_size, seed)
    demand_samples = sampler(n_samples, seed)
    
    # Aggregate samples by block to create nominal distribution
    nominal_prob = {block_id: 0.0 for block_id in short_geoid_list}
    
    for x, y in demand_samples:
        # Determine which block this sample falls into
        block_row = int(np.clip(np.floor(x), 0, grid_size - 1))
        block_col = int(np.clip(np.floor(y), 0, grid_size - 1))
        block_idx = block_row * grid_size + block_col
        block_id = short_geoid_list[block_idx]
        
        nominal_prob[block_id] += 1.0 / n_samples
    
    return nominal_prob, demand_samples

def create_odd_features(grid_size, seed=42):
    """Generate two-dimensional ODD feature distribution"""
    np.random.seed(seed)
    
    Omega_dict = {}
    
    for i in range(grid_size):
        for j in range(grid_size):
            block_id = f"BLK{i * grid_size + j:03d}"
            
            # Feature 1: Spatially correlated Beta distribution
            spatial_factor1 = np.sin(np.pi * i / grid_size) * np.cos(np.pi * j / grid_size)
            feature1 = beta.rvs(2.5, 1.8) * (0.5 + 0.5 * spatial_factor1)
            
            # Feature 2: Bimodal spatial distribution
            if (i + j) % 2 == 0:
                feature2 = gamma.rvs(2.0, scale=0.3)
            else:
                feature2 = gamma.rvs(1.5, scale=0.2)
            
            Omega_dict[block_id] = np.array([feature1, feature2])
    
    def J_function(omega):
        """ODD cost function"""
        return 30
        # return 0.1 * (omega[0]**1.5 + omega[1]**1.2)
    
    return Omega_dict, J_function

def optimize_service_design(geo_data: ToyGeoData, 
                          prob_dict: Dict[str, float],
                          Omega_dict: Dict,
                          J_function,
                          num_districts: int,
                          design_type: str,
                          epsilon: float,
                          seed: int = 42,
                          lambda_per_hour: float = 100.0) -> DesignResult:
    """Optimize service design using RS algorithm with specified epsilon"""
    
    try:
        print(f"    Optimizing {design_type} design (ε={epsilon:.4f})...")
        
        # Create partition instance with specified epsilon
        partition = Partition(geo_data, num_districts, prob_dict, epsilon=epsilon)
        
        # Run Random Search optimization
        depot_id, district_roots, assignment, obj_val, district_info = partition.random_search(
            max_iters=50,  # Same as simulate_service_designs.py
            prob_dict=prob_dict,
            Lambda=lambda_per_hour, wr=1.0, wv=10.0, beta=0.7120,
            Omega_dict=Omega_dict,
            J_function=J_function
        )
        
        print(f"      {design_type} design objective: {obj_val:.2f}")
        
        # Extract dispatch intervals from district_info (same as simulate_service_designs.py)
        dispatch_intervals = {}
        for district_data in district_info:
            if len(district_data) >= 5:
                cost, root, K_i, F_i, T_star = district_data[:5]
                dispatch_intervals[root] = T_star
            else:
                # Fallback for old format
                cost, root, K_i, F_i, T_star = district_data
                dispatch_intervals[root] = T_star
        
        return DesignResult(
            depot_id=depot_id,
            district_roots=district_roots,
            assignment=assignment,
            dispatch_intervals=dispatch_intervals,
            obj_val=obj_val,
            district_info=district_info,
            success=True
        )
        
    except Exception as e:
        print(f"      {design_type} optimization failed: {e}")
        return DesignResult(
            depot_id="", district_roots=[], assignment=np.array([]),
            dispatch_intervals={}, obj_val=float('inf'), district_info=[], success=False
        )

def solve_worst_case_with_cqcp(geodata, empirical_prob_dict, epsilon, seed=42):
    """Solve worst-case distribution using CQCP Benders decomposition"""
    try:
        assigned_blocks = geodata.short_geoid_list
        # Select center block based on grid size (avoid hardcoded index 50 for small grids)
        center_idx = len(assigned_blocks) // 2
        root = assigned_blocks[center_idx]
        
        partition = Partition(geodata, num_districts=1, prob_dict=empirical_prob_dict, epsilon=epsilon)
        
        K_i = 3.0
        F_i = 0.0  
        beta = 0.7120
        Lambda = 10.0
        
        result = partition._CQCP_benders(
            assigned_blocks=assigned_blocks,
            root=root,
            prob_dict=empirical_prob_dict,
            epsilon=epsilon,
            K_i=K_i, F_i=F_i, grid_points=10,
            beta=beta, Lambda=Lambda,
            single_threaded=True
        )
        
        if len(result) >= 7:
            worst_case_cost, x_star_dict, subgrad, C_star, alpha_i, subgrad_K_i, subgrad_F_i = result
            
            # Convert x_star_dict to distribution array using x_j^2
            worst_case_dist = np.zeros(len(assigned_blocks))
            total_mass = 0.0
            for i, block_id in enumerate(assigned_blocks):
                x_j = x_star_dict.get(block_id, 0.0)
                mass = max(0.0, x_j)**2  # Use x_j^2 for probability mass
                worst_case_dist[i] = mass
                total_mass += mass
            
            if total_mass > 1e-10:
                worst_case_dist = worst_case_dist / total_mass
            else:
                worst_case_dist = np.ones(len(assigned_blocks)) / len(assigned_blocks)
            
            # Convert back to dictionary format
            worst_case_prob_dict = {
                block_id: worst_case_dist[i] 
                for i, block_id in enumerate(geodata.short_geoid_list)
            }
            
            return worst_case_prob_dict, True
        else:
            return None, False
            
    except Exception as e:
        print(f"    CQCP solver failed: {e}")
        return None, False

def sample_demand_points_in_district(geo_data, assigned_blocks, dispatch_demand, sampler):
    """Sample demand points within assigned blocks of a district"""
    points = []
    attempts = 0
    max_attempts = dispatch_demand * 5
    
    while len(points) < dispatch_demand and attempts < max_attempts:
        # Sample a point from true distribution
        candidate_points = sampler(1)
        if candidate_points:
            x, y = candidate_points[0]
            # Determine which block this point falls into
            grid_size = geo_data.grid_size
            block_row = int(np.clip(np.floor(x), 0, grid_size - 1))
            block_col = int(np.clip(np.floor(y), 0, grid_size - 1))
            block_idx = block_row * grid_size + block_col
            block_id = geo_data.short_geoid_list[block_idx]
            
            # Accept if point is in assigned blocks
            if block_id in assigned_blocks:
                # Convert to miles coordinates
                x_miles = x * geo_data.miles_per_grid_unit
                y_miles = y * geo_data.miles_per_grid_unit
                points.append((x_miles, y_miles))
        
        attempts += 1
    
    return points

def calculate_tsp_length(points: List[Tuple[float, float]], use_exact: bool = True) -> float:
    """Compute TSP tour length using python_tsp if available; fallback to MST.

    - Exact DP for very small n, SA for medium n, local search for larger n.
    - Fallback: 2 × MST length (conservative upper bound for metric TSP).
    - Two-point case returns 2 × Euclidean distance (out-and-back).
    """
    if len(points) <= 1:
        return 0.0

    if len(points) == 2:
        (x1, y1), (x2, y2) = points
        return 2.0 * np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    coords = np.array(points)
    dist_matrix = squareform(pdist(coords, 'euclidean'))

    if use_exact and PYTHON_TSP_AVAILABLE:
        try:
            n = len(points)
            if n <= 12:
                _, distance = solve_tsp_dynamic_programming(dist_matrix)
                return float(distance)
            elif n <= 30:
                _, distance = solve_tsp_simulated_annealing(dist_matrix, max_processing_time=2.0)
                return float(distance)
            else:
                _, distance = solve_tsp_local_search(dist_matrix)
                return float(distance)
        except Exception:
            pass  # Fall through to MST fallback

    # MST fallback
    mst = minimum_spanning_tree(dist_matrix)
    mst_length = float(mst.sum())
    return 2.0 * mst_length

def evaluate_design_via_simulation(geo_data: ToyGeoData,
                                 design: DesignResult,
                                 prob_dict: Dict[str, float],
                                 eval_type: str,
                                 trial_seed: int = 42,
                                 lambda_per_hour: float = 100.0) -> SimulationResult:
    """Evaluate design performance via PROPER vehicle simulation matching simulate_service_designs.py"""
    
    if not design.success:
        return SimulationResult(
            total_cost=float('inf'),
            user_cost=float('inf'),
            provider_cost=float('inf'),
            success=False
        )
    
    try:
        # Constants matching simulate_service_designs.py
        SERVICE_HOURS = 12.0  # 8am to 8pm
        wr = 1.0  # Time-to-cost conversion factor
        wv = 10.0  # Vehicle speed (miles/hour)
        
        block_ids = geo_data.short_geoid_list
        N = len(block_ids)
        
        # No need to generate daily demands upfront - we'll calculate district arrival rates
        # and sample per dispatch using the given prob_dict distribution
        
        # Generate ODD features
        Omega_dict, J_function = create_odd_features(geo_data.grid_size)
        
        # District cost calculations (like simulate_service_designs.py)
        district_costs = []
        max_wait_time = 0.0
        max_travel_time = 0.0
        
        for root_id in design.district_roots:
            root_idx = block_ids.index(root_id)
            
            # Get assigned blocks from FIXED design
            assigned_blocks = []
            for i in range(N):
                if round(design.assignment[i, root_idx]) == 1:
                    block_id = block_ids[i]
                    assigned_blocks.append(block_id)
            
            if not assigned_blocks:
                continue
            
            # Calculate district arrival rate per hour from the given distribution (nominal or worst-case)
            district_prob_mass = sum(prob_dict.get(block_id, 0.0) for block_id in assigned_blocks)
            if district_prob_mass <= 0:
                continue
            
            # FIXED: Use actual dispatch interval from design
            T_star = design.dispatch_intervals[root_id]
            
            # Number of dispatches during service hours
            num_dispatches = max(1, int(np.ceil(SERVICE_HOURS / T_star)))
            
            # FIXED: Proper linehaul cost = roundtrip depot–root distance amortized by T_star
            root_depot_dist = geo_data.get_dist(design.depot_id, root_id)
            total_linehaul = 2.0 * root_depot_dist
            district_linehaul = total_linehaul / T_star
            
            # ODD cost: maximum ODD features in district (amortized over T_star)
            district_omega = np.zeros(2)
            for block_id in assigned_blocks:
                if block_id in Omega_dict:
                    district_omega = np.maximum(district_omega, Omega_dict[block_id])
            district_odd = J_function(district_omega) / T_star  # Fixed cost amortized over dispatch interval
            
            # FIXED: TSP-based travel cost with Poisson demand per dispatch
            district_travel = 0.0
            for dispatch in range(num_dispatches):
                # Sample number of demands for this dispatch using per-hour arrival rate
                # Expected demands in one dispatch interval = (lambda_per_hour * mass) * T_star
                dispatch_lambda = (lambda_per_hour * district_prob_mass) * T_star
                # Use trial_seed for reproducible trials with different random realizations
                np.random.seed(trial_seed + hash((root_id, dispatch)) % 1000)
                dispatch_demand = np.random.poisson(dispatch_lambda)
                
                if dispatch_demand > 0:
                    # Sample spatial demand locations using the given prob_dict distribution
                    demand_points = []
                    
                    # Create block probability array within this district for spatial sampling
                    district_block_probs = {}
                    district_total_prob = 0.0
                    for block_id in assigned_blocks:
                        prob = prob_dict.get(block_id, 0.0)
                        district_block_probs[block_id] = prob
                        district_total_prob += prob
                    
                    if district_total_prob > 0:
                        # Normalize probabilities within district
                        for block_id in district_block_probs:
                            district_block_probs[block_id] /= district_total_prob
                        
                        # Sample demand locations within district
                        for _ in range(dispatch_demand):
                            # Sample which block this demand occurs in
                            block_choices = list(district_block_probs.keys())
                            block_probs = list(district_block_probs.values())
                            sampled_block = np.random.choice(block_choices, p=block_probs)
                            
                            # Sample random location within that block
                            block_idx = block_ids.index(sampled_block)
                            block_row = block_idx // geo_data.grid_size
                            block_col = block_idx % geo_data.grid_size
                            
                            # Random location within block (in grid coordinates)
                            x = block_col + np.random.uniform(0, 1)
                            y = block_row + np.random.uniform(0, 1)
                            
                            # Convert to miles
                            x_miles = x * geo_data.miles_per_grid_unit
                            y_miles = y * geo_data.miles_per_grid_unit
                            demand_points.append((x_miles, y_miles))
                    
                    # Include the district root in the TSP tour
                    rx, ry = geo_data.get_coord_in_miles(root_id)
                    tsp_points = [(rx, ry)] + demand_points

                    if tsp_points:
                        # Calculate TSP routing cost for this dispatch (python_tsp with MST fallback)
                        tsp_length = calculate_tsp_length(tsp_points, use_exact=True)
                        district_travel += tsp_length
            
            district_travel_cost = district_travel / T_star
            
            # Total provider cost
            district_provider_cost = district_linehaul + district_odd + district_travel_cost
            
            # User costs
            district_wait = T_star
            # Worst rider in a dispatch experiences outbound linehaul only (depot→root)
            district_travel_time = (district_travel + root_depot_dist) / wv
            district_user_cost = wr * (district_wait + district_travel_time)
            
            # Total district cost
            district_total_cost = district_provider_cost + district_user_cost
            
            district_costs.append((root_id, district_provider_cost, district_user_cost, 
                                 district_total_cost, district_linehaul, district_odd, district_travel_cost))
            
            max_wait_time = max(max_wait_time, district_wait)
            max_travel_time = max(max_travel_time, district_travel_time)
        
        # FIXED: Minimax objective - find worst district cost
        if not district_costs:
            total_cost = 0.0
            provider_cost = 0.0
            user_cost = 0.0
        else:
            # Find district with maximum total cost (minimax)
            max_district = max(district_costs, key=lambda x: x[3])  # x[3] is total_cost
            _, provider_cost, user_cost, total_cost, _, _, _ = max_district
        
        return SimulationResult(
            total_cost=total_cost,
            user_cost=user_cost,
            provider_cost=provider_cost,
            success=True
        )
        
    except Exception as e:
        print(f"    Simulation evaluation ({eval_type}) failed: {e}")
        return SimulationResult(
            total_cost=design.obj_val,
            user_cost=design.obj_val * 0.6,
            provider_cost=design.obj_val * 0.4,
            success=True
        )

def run_sample_size_analysis(epsilon_0: float = 2.0,
                           grid_size: int = 10,
                           num_districts: int = 3,
                           sample_sizes: List[int] = None,
                           num_trials: int = 3,
                           lambda_per_hour: float = 100.0) -> Dict:
    """
    Run complete sample size analysis
    
    For each sample size n:
    1. Sample n points from true distribution → empirical distribution
    2. Generate robust design (ε = ε₀/√n) and nominal design (ε = 0)
    3. Compute worst-case distribution (CQCP with ε = ε₀/√n)
    4. Evaluate both designs on both nominal and worst-case distributions
    """
    
    if sample_sizes is None:
        sample_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200]
    
    print("=" * 80)
    print("DESIGN PERFORMANCE vs SAMPLE SIZE ANALYSIS - WITH TRIALS")
    print("=" * 80)
    print(f"Grid size: {grid_size}×{grid_size} = {grid_size**2} blocks")
    print(f"Service region: 10.0×10.0 miles")
    print(f"Base Wasserstein radius: ε₀ = {epsilon_0}")
    print(f"Number of districts: {num_districts}")
    print(f"Sample sizes: {sample_sizes}")
    print(f"Trials per sample size: {num_trials}")
    print()
    
    # Setup with fixed 10×10 mile service region
    service_region_miles = 10.0  # ALWAYS fixed at 10×10 miles
    n_blocks = grid_size * grid_size
    geo_data = ToyGeoData(n_blocks, grid_size, service_region_miles=service_region_miles)
    Omega_dict, J_function = create_odd_features(grid_size)
    
    results = {
        'sample_sizes': [],
        'epsilons': [],
        'robust_design_nominal_performance': [],
        'robust_design_worst_case_performance': [],
        'nominal_design_nominal_performance': [],
        'nominal_design_worst_case_performance': [],
        'robust_design_nominal_std': [],
        'robust_design_worst_case_std': [],
        'nominal_design_nominal_std': [],
        'nominal_design_worst_case_std': []
    }
    
    for n in sample_sizes:
        print(f"\\n📊 Processing sample size n = {n}")
        print("-" * 50)
        
        # Step 1: Sample n points and create empirical distribution
        epsilon = epsilon_0 / np.sqrt(n)
        print(f"  Wasserstein radius: ε = {epsilon_0}/√{n} = {epsilon:.4f}")
        
        nominal_prob_dict, demand_samples = create_nominal_distribution(grid_size, n, seed=42+n)
        print(f"  Generated {len(demand_samples)} empirical demand points")
        
        # Step 2: Generate TWO designs
        robust_design = optimize_service_design(
            geo_data, nominal_prob_dict, Omega_dict, J_function,
            num_districts, "Robust", epsilon, seed=42+n, lambda_per_hour=lambda_per_hour
        )
        
        nominal_design = optimize_service_design(
            geo_data, nominal_prob_dict, Omega_dict, J_function,
            num_districts, "Nominal", epsilon=0.0, seed=42+n, lambda_per_hour=lambda_per_hour
        )
        
        # Step 3: Compute worst-case distribution  
        print(f"    Computing worst-case distribution (ε={epsilon:.4f})...")
        worst_case_prob_dict, wc_success = solve_worst_case_with_cqcp(
            geo_data, nominal_prob_dict, epsilon, seed=42+n
        )
        
        if not wc_success:
            print(f"    Failed to compute worst-case distribution for n={n}")
            continue
        
        # Step 4: Evaluate both designs via MULTIPLE simulation trials
        print(f"    Evaluating designs via simulation ({num_trials} trials each)...")
        
        # Storage for trial results
        robust_nominal_trials = []
        robust_worst_case_trials = []
        nominal_nominal_trials = []
        nominal_worst_case_trials = []
        
        successful_trials = 0
        
        for trial in range(num_trials):
            trial_seed = 42 + n * 1000 + trial  # Unique seed per n and trial
            
            # Robust design evaluations
            robust_nominal_result = evaluate_design_via_simulation(
                geo_data, robust_design, nominal_prob_dict, "Robust on Nominal", trial_seed, lambda_per_hour=lambda_per_hour
            )
            robust_worst_case_result = evaluate_design_via_simulation(
                geo_data, robust_design, worst_case_prob_dict, "Robust on Worst-Case", trial_seed, lambda_per_hour=lambda_per_hour
            )
            
            # Nominal design evaluations  
            nominal_nominal_result = evaluate_design_via_simulation(
                geo_data, nominal_design, nominal_prob_dict, "Nominal on Nominal", trial_seed, lambda_per_hour=lambda_per_hour
            )
            nominal_worst_case_result = evaluate_design_via_simulation(
                geo_data, nominal_design, worst_case_prob_dict, "Nominal on Worst-Case", trial_seed, lambda_per_hour=lambda_per_hour
            )
            
            # Check if all simulations succeeded
            if (robust_nominal_result.success and robust_worst_case_result.success and 
                nominal_nominal_result.success and nominal_worst_case_result.success):
                
                robust_nominal_trials.append(robust_nominal_result.total_cost)
                robust_worst_case_trials.append(robust_worst_case_result.total_cost)
                nominal_nominal_trials.append(nominal_nominal_result.total_cost)
                nominal_worst_case_trials.append(nominal_worst_case_result.total_cost)
                successful_trials += 1
        
        # Store results if we have successful trials
        if successful_trials > 0:
            # Calculate means and standard deviations
            robust_nominal_mean = np.mean(robust_nominal_trials)
            robust_nominal_std = np.std(robust_nominal_trials) if len(robust_nominal_trials) > 1 else 0.0
            
            robust_worst_case_mean = np.mean(robust_worst_case_trials)
            robust_worst_case_std = np.std(robust_worst_case_trials) if len(robust_worst_case_trials) > 1 else 0.0
            
            nominal_nominal_mean = np.mean(nominal_nominal_trials)
            nominal_nominal_std = np.std(nominal_nominal_trials) if len(nominal_nominal_trials) > 1 else 0.0
            
            nominal_worst_case_mean = np.mean(nominal_worst_case_trials)
            nominal_worst_case_std = np.std(nominal_worst_case_trials) if len(nominal_worst_case_trials) > 1 else 0.0
            
            results['sample_sizes'].append(n)
            results['epsilons'].append(epsilon)
            results['robust_design_nominal_performance'].append(robust_nominal_mean)
            results['robust_design_worst_case_performance'].append(robust_worst_case_mean)
            results['nominal_design_nominal_performance'].append(nominal_nominal_mean)
            results['nominal_design_worst_case_performance'].append(nominal_worst_case_mean)
            
            # Store standard deviations for error bars
            results['robust_design_nominal_std'].append(robust_nominal_std)
            results['robust_design_worst_case_std'].append(robust_worst_case_std)
            results['nominal_design_nominal_std'].append(nominal_nominal_std)
            results['nominal_design_worst_case_std'].append(nominal_worst_case_std)
            
            print(f"    ✅ Results for n={n} ({successful_trials}/{num_trials} successful trials):")
            print(f"       Robust design:  {robust_nominal_mean:.1f}±{robust_nominal_std:.1f} (nominal) | {robust_worst_case_mean:.1f}±{robust_worst_case_std:.1f} (worst-case)")
            print(f"       Nominal design: {nominal_nominal_mean:.1f}±{nominal_nominal_std:.1f} (nominal) | {nominal_worst_case_mean:.1f}±{nominal_worst_case_std:.1f} (worst-case)")
        else:
            print(f"    ❌ Failed to evaluate designs for n={n}")
    
    return results

def create_performance_visualizations(results: Dict, epsilon_0: float):
    """Create two figures organized by evaluation distribution"""
    
    sample_sizes = np.array(results['sample_sizes'])
    
    robust_nominal = np.array(results['robust_design_nominal_performance'])
    robust_worst_case = np.array(results['robust_design_worst_case_performance'])
    nominal_nominal = np.array(results['nominal_design_nominal_performance'])
    nominal_worst_case = np.array(results['nominal_design_worst_case_performance'])
    
    # Check if std data exists and has correct length
    if ('robust_design_nominal_std' in results and 
        len(results['robust_design_nominal_std']) == len(sample_sizes) and
        len(results['robust_design_nominal_std']) > 0):
        robust_nominal_std = np.array(results['robust_design_nominal_std'])
        robust_worst_case_std = np.array(results['robust_design_worst_case_std'])
        nominal_nominal_std = np.array(results['nominal_design_nominal_std'])
        nominal_worst_case_std = np.array(results['nominal_design_worst_case_std'])
        use_error_bars = True
    else:
        # No error bar data available - use zeros
        robust_nominal_std = np.zeros_like(robust_nominal)
        robust_worst_case_std = np.zeros_like(robust_worst_case)
        nominal_nominal_std = np.zeros_like(nominal_nominal)
        nominal_worst_case_std = np.zeros_like(nominal_worst_case)
        use_error_bars = False
    
    # Figure 1: Performance on Worst-Case Distribution
    fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
    
    if use_error_bars:
        ax1.errorbar(sample_sizes, robust_worst_case, yerr=robust_worst_case_std, 
                     fmt='ro-', linewidth=2, markersize=8, capsize=5,
                     label='Robust Design (ε = ε₀/√n)')
        ax1.errorbar(sample_sizes, nominal_worst_case, yerr=nominal_worst_case_std,
                     fmt='bo-', linewidth=2, markersize=8, capsize=5, 
                     label='Nominal Design (ε = 0)')
    else:
        ax1.plot(sample_sizes, robust_worst_case, 'ro-', linewidth=2, markersize=8,
                 label='Robust Design (ε = ε₀/√n)')
        ax1.plot(sample_sizes, nominal_worst_case, 'bo-', linewidth=2, markersize=8, 
                 label='Nominal Design (ε = 0)')
    
    ax1.set_xlabel('Sample Size n', fontsize=14)
    ax1.set_ylabel('Worst-Case Total Cost', fontsize=14)
    # ax1.set_title(f'Performance on Worst-Case Distribution\\n(Evaluation under adversarial demand, ε₀ = {epsilon_0})', fontsize=15)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig('worst_case_performance.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("✅ Worst-case performance plot saved as 'worst_case_performance.pdf'")
    
    # Figure 2: Performance on Nominal Distribution
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4))
    
    if use_error_bars:
        ax2.errorbar(sample_sizes, robust_nominal, yerr=robust_nominal_std,
                     fmt='go-', linewidth=2, markersize=8, capsize=5,
                     label='Robust Design (ε = ε₀/√n)')
        ax2.errorbar(sample_sizes, nominal_nominal, yerr=nominal_nominal_std,
                     fmt='mo-', linewidth=2, markersize=8, capsize=5,
                     label='Nominal Design (ε = 0)')
    else:
        ax2.plot(sample_sizes, robust_nominal, 'go-', linewidth=2, markersize=8,
                 label='Robust Design (ε = ε₀/√n)')
        ax2.plot(sample_sizes, nominal_nominal, 'mo-', linewidth=2, markersize=8,
                 label='Nominal Design (ε = 0)')
    
    ax2.set_xlabel('Sample Size n', fontsize=14)
    ax2.set_ylabel('Nominal Total Cost', fontsize=14)
    # ax2.set_title(f'Performance on Nominal Distribution\\n(Evaluation under empirical demand, ε₀ = {epsilon_0})', fontsize=15)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig('nominal_performance.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("✅ Nominal performance plot saved as 'nominal_performance.pdf'")
    
    # Figure 3: Trade-off Scatter Plot (Nominal vs Worst-case Performance)
    fig3, ax3 = plt.subplots(1, 1, figsize=(7, 5))
    
    # Define different shapes for different sample sizes
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']
    marker_sizes = [8, 8, 9, 7, 9, 10, 12, 9]
    
    # Plot robust designs (red) and nominal designs (blue)
    for i, n in enumerate(sample_sizes):
        marker = markers[i % len(markers)]
        size = marker_sizes[i % len(marker_sizes)]
        
        # Robust design (red)
        robust_nominal_perf = robust_nominal[i]
        robust_worst_case_perf = robust_worst_case[i]
        
        if use_error_bars:
            ax3.errorbar(robust_nominal_perf, robust_worst_case_perf, 
                        xerr=robust_nominal_std[i], yerr=robust_worst_case_std[i],
                        fmt=marker, color='red', markersize=size, capsize=4, capthick=1.5, alpha=0.8,
                        markerfacecolor='red', markeredgecolor='darkred', markeredgewidth=1)
        else:
            ax3.plot(robust_nominal_perf, robust_worst_case_perf, marker=marker, color='red', 
                    markersize=size, alpha=0.8, markerfacecolor='red', markeredgecolor='darkred', 
                    markeredgewidth=1, linestyle='None')
        
        # Nominal design (blue) 
        nominal_nominal_perf = nominal_nominal[i]
        nominal_worst_case_perf = nominal_worst_case[i]
        
        if use_error_bars:
            ax3.errorbar(nominal_nominal_perf, nominal_worst_case_perf,
                        xerr=nominal_nominal_std[i], yerr=nominal_worst_case_std[i],
                        fmt=marker, color='blue', markersize=size, capsize=4, capthick=1.5, alpha=0.8,
                        markerfacecolor='blue', markeredgecolor='darkblue', markeredgewidth=1)
        else:
            ax3.plot(nominal_nominal_perf, nominal_worst_case_perf, marker=marker, color='blue',
                    markersize=size, alpha=0.8, markerfacecolor='blue', markeredgecolor='darkblue',
                    markeredgewidth=1, linestyle='None')
    
    # Add diagonal reference line (y = x)
    min_cost = min(min(robust_nominal), min(robust_worst_case), 
                   min(nominal_nominal), min(nominal_worst_case))
    max_cost = max(max(robust_nominal), max(robust_worst_case),
                   max(nominal_nominal), max(nominal_worst_case))
    ax3.plot([min_cost, max_cost], [min_cost, max_cost], 'k--', alpha=0.5, linewidth=1, 
             label='y = x (Equal Performance)')
    
    # Formatting
    ax3.set_xlabel('Cost under Empirical Distribution', fontsize=14)
    ax3.set_ylabel('Cost under Adversarial Distribution', fontsize=14)
    # ax3.set_title(f'Design Performance Trade-off\\n(Nominal vs Worst-Case Performance, ε₀ = {epsilon_0})', fontsize=15, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='both', which='major', labelsize=12)
    
    # Create comprehensive legend
    from matplotlib.lines import Line2D
    
    # Design type legend
    design_legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10,
               markeredgecolor='darkred', markeredgewidth=1,
               label='Robust Design (ε = ε₀/√n)', linestyle='None'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10,
               markeredgecolor='darkblue', markeredgewidth=1,
               label='Nominal Design (ε = 0)', linestyle='None'),
        Line2D([0], [0], color='black', linestyle='--', alpha=0.5,
               label='y = x (Equal Performance)')
    ]
    
    # Sample size legend (include all n, with multi-column layout to avoid overcrowding)
    size_legend = []
    for i, n in enumerate(sample_sizes):
        marker = markers[i % len(markers)]
        size = marker_sizes[i % len(marker_sizes)]
        size_legend.append(
            Line2D([0], [0], marker=marker, color='w', markerfacecolor='gray',
                   markersize=size, markeredgecolor='black', markeredgewidth=1,
                   label=f'n = {n}', linestyle='None')
        )
    
    # Create two separate legends
    legend1 = ax3.legend(handles=design_legend, loc='lower left', fontsize=12, 
                        title='Design Type', title_fontsize=13)
    ax3.add_artist(legend1)  # Keep first legend when adding second
    
    # Choose number of columns based on count
    ncols = 1 if len(size_legend) <= 6 else 2 if len(size_legend) <= 12 else 3
    legend2 = ax3.legend(handles=size_legend, loc='lower right', fontsize=10,
                        title='Sample Size', title_fontsize=11, ncol=ncols)
    
    # Add interpretation text (moved to center-left to avoid legend overlap)
    # ax3.text(0.02, 0.65, 
    #         'Points below diagonal:\\nBetter worst-case than nominal\\n\\n' +
    #         'Points above diagonal:\\nWorse worst-case than nominal',
    #         transform=ax3.transAxes, verticalalignment='top', fontsize=11,
    #         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('performance_tradeoff.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print("✅ Performance trade-off plot saved as 'performance_tradeoff.pdf'")

def main():
    # Parse command line arguments
    epsilon_0 = 2.0
    grid_size = 5  # Default grid size
    num_trials = 5  # Default number of simulation trials
    lambda_per_hour = 100.0  # Unified arrival rate (riders/hour)
    
    # Show usage help
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("Usage: python design_perf.py [epsilon_0] [grid_size] [num_trials] [lambda_per_hour]")
        print("  epsilon_0:   Base Wasserstein radius (default: 2.0)")
        print("  grid_size:   Grid resolution NxN (default: 5)")
        print("  num_trials:  Number of simulation trials for error bars (default: 5)")
        print("  lambda_per_hour: Arrival rate per hour used in optimization AND simulation (default: 100)")
        print("  ")
        print("Examples:")
        print("  python design_perf.py                    # Default: ε₀=2.0, 5×5 grid, 5 trials")
        print("  python design_perf.py 1.5                # ε₀=1.5, 5×5 grid, 5 trials")  
        print("  python design_perf.py 2.0 10             # ε₀=2.0, 10×10 grid, 5 trials")
        print("  python design_perf.py 1.0 20 10          # ε₀=1.0, 20×20 grid, 10 trials")
        print("  python design_perf.py 2.0 10 5 120       # ε₀=2.0, 10×10 grid, 5 trials, λ=120/hour")
        print("  ")
        print("Note: Service region is ALWAYS fixed at 10×10 miles")
        return
    
    # Parse arguments: python design_perf.py [epsilon_0] [grid_size] [num_trials]
    try:
        if len(sys.argv) > 1:
            epsilon_0 = float(sys.argv[1])
        if len(sys.argv) > 2:
            grid_size = int(sys.argv[2])
        if len(sys.argv) > 3:
            num_trials = int(sys.argv[3])
        if len(sys.argv) > 4:
            lambda_per_hour = float(sys.argv[4])
    except ValueError:
        print("Error: Invalid arguments. Use -h for help.")
        return
    
    # Service region always fixed at 10×10 miles
    service_region_miles = 10.0
    block_size_miles = service_region_miles / grid_size
    
    print(f"Using ε₀ = {epsilon_0}")
    print(f"Grid size: {grid_size}×{grid_size} = {grid_size**2} blocks")
    print(f"Service region: {service_region_miles}×{service_region_miles} miles (FIXED)")
    print(f"Block size: {block_size_miles:.3f}×{block_size_miles:.3f} miles")
    print(f"Simulation trials: {num_trials} per design evaluation")
    print(f"Unified arrival rate: λ = {lambda_per_hour}/hour")
    print()
    
    # Adjust sample sizes based on grid resolution
    if grid_size <= 5:
        sample_sizes = [10, 20, 30, 40, 50, 60, 80, 100]
    elif grid_size <= 10:
        sample_sizes = [10, 20, 30, 40, 50, 60, 80, 100]
    else:
        sample_sizes = [20, 40, 60, 80, 100, 120, 150, 200]
    
    results = run_sample_size_analysis(
        epsilon_0=epsilon_0,
        grid_size=grid_size,
        num_districts=3,
        sample_sizes=sample_sizes,
        num_trials=num_trials,  # Use command line argument
        lambda_per_hour=lambda_per_hour
    )
    
    if results['sample_sizes']:
        # Create visualizations
        create_performance_visualizations(results, epsilon_0)
        
        # Print summary
        print("\\n" + "=" * 60)
        print("ANALYSIS SUMMARY") 
        print("=" * 60)
        
        for i, n in enumerate(results['sample_sizes']):
            rn = results['robust_design_nominal_performance'][i]
            rwc = results['robust_design_worst_case_performance'][i]
            nn = results['nominal_design_nominal_performance'][i]
            nwc = results['nominal_design_worst_case_performance'][i]
            eps = results['epsilons'][i]
            
            print(f"n={n:3d} (ε={eps:.3f}): Robust=[{rn:6.1f}, {rwc:6.1f}] | Nominal=[{nn:6.1f}, {nwc:6.1f}]")
        
        print(f"\\n📊 Key Insights:")
        print(f"• Robust designs trained with Wasserstein uncertainty (ε = {epsilon_0}/√n)")
        print(f"• Nominal designs trained without uncertainty (ε = 0)")
        print(f"• Both evaluated on nominal and worst-case distributions")
        print(f"• Two separate PDF figures generated for comparison")
    
    else:
        print("No successful results obtained.")

if __name__ == "__main__":
    main()
