#!/usr/bin/env python3
"""
Pareto Frontier Analysis: User Cost vs Provider Cost Trade-offs
Analyzes the trade-off between user costs and provider costs by varying wr/wv ratios.
Uses 10×10 grid configuration for consistent comparison.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from lib.algorithm import Partition
from lib.data import GeoData
from lib.constants import TSP_TRAVEL_DISCOUNT
import networkx as nx
import time
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import json
from datetime import datetime

# Reuse configuration classes from simulate_service_designs.py
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

class ToyGeoData(GeoData):
    """Toy geographic data for simulation testing with fixed service region"""
    
    def __init__(self, n_blocks, grid_size, service_region_miles=10.0, seed=42):
        np.random.seed(seed)
        self.n_blocks = n_blocks
        self.grid_size = grid_size
        self.service_region_miles = service_region_miles  # Fixed service region size
        self.miles_per_grid_unit = service_region_miles / grid_size  # Resolution scaling
        self.short_geoid_list = [f"BLK{i:03d}" for i in range(n_blocks)]
        
        # Create block graph using block IDs as nodes
        self.G = nx.Graph()
        self.block_to_coord = {i: (i // grid_size, i % grid_size) for i in range(n_blocks)}
        
        # Add all block IDs as nodes
        for block_id in self.short_geoid_list:
            self.G.add_node(block_id)
        
        # Add edges between adjacent blocks
        for i in range(n_blocks):
            for j in range(i+1, n_blocks):
                coord1 = self.block_to_coord[i]
                coord2 = self.block_to_coord[j]
                if abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1]) == 1:
                    self.G.add_edge(self.short_geoid_list[i], self.short_geoid_list[j])
        
        # Create block graph for contiguity
        self.block_graph = nx.Graph()
        for i in range(n_blocks):
            self.block_graph.add_node(self.short_geoid_list[i])
        
        for i in range(n_blocks):
            for j in range(i+1, n_blocks):
                coord1 = self.block_to_coord[i]
                coord2 = self.block_to_coord[j]
                if abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1]) == 1:
                    self.block_graph.add_edge(self.short_geoid_list[i], self.short_geoid_list[j])
        
        # Set attributes
        self.areas = {block_id: 1.0 for block_id in self.short_geoid_list}
        self.gdf = None
        
        # Cache area data for faster access
        self.area_cache = {block_id: 1.0 for block_id in self.short_geoid_list}
    
    def get_area(self, block_id):
        """Get area for a block (required by algorithm)"""
        return self.area_cache.get(block_id, 1.0)
    
    def get_dist(self, block1, block2):
        """Euclidean distance between blocks in miles using resolution scaling"""
        if block1 == block2:
            return 0.0
        
        # Get block indices
        idx1 = self.short_geoid_list.index(block1) if block1 in self.short_geoid_list else None
        idx2 = self.short_geoid_list.index(block2) if block2 in self.short_geoid_list else None
        
        if idx1 is None or idx2 is None:
            return float('inf')
        
        # Get coordinates and compute distance
        coord1 = self.block_to_coord[idx1]
        coord2 = self.block_to_coord[idx2]
        
        # Euclidean distance in grid units
        dx = (coord1[0] - coord2[0])
        dy = (coord1[1] - coord2[1])
        grid_distance = (dx * dx + dy * dy) ** 0.5
        
        # Convert to miles using resolution scaling
        return grid_distance * self.miles_per_grid_unit

@dataclass
class ParetoPoint:
    """Single point on the Pareto frontier"""
    wr: float
    wv: float
    wr_wv_ratio: float
    user_cost: float
    user_cost_std: float
    provider_cost: float
    provider_cost_std: float
    total_cost: float
    total_cost_std: float
    depot_id: str
    num_districts: int
    objective_value: float
    computation_time: float

def create_nominal_distribution(grid_size: int, n_samples: int = 1000, seed: int = 42) -> Dict[str, float]:
    """Create nominal distribution by sampling from true mixed-Gaussian distribution"""
    np.random.seed(seed)
    n_blocks = grid_size * grid_size
    short_geoid_list = [f"BLK{i:03d}" for i in range(n_blocks)]
    
    # Generate samples from mixed-Gaussian distribution
    cluster_centers = MixedGaussianConfig.get_cluster_centers(grid_size)
    cluster_weights = MixedGaussianConfig.get_cluster_weights()
    cluster_sigmas = MixedGaussianConfig.get_cluster_sigmas(grid_size)
    
    samples = []
    attempts = 0
    max_attempts = n_samples * 10
    
    while len(samples) < n_samples and attempts < max_attempts:
        # Choose cluster based on weights
        cluster_idx = np.random.choice(len(cluster_centers), p=cluster_weights)
        center = cluster_centers[cluster_idx]
        sigma = cluster_sigmas[cluster_idx]
        
        # Sample from chosen Gaussian cluster
        x = np.random.normal(center[0], sigma)
        y = np.random.normal(center[1], sigma)
        
        # Accept only if within service region (proper truncation)
        if 0 <= x <= grid_size - 1 and 0 <= y <= grid_size - 1:
            samples.append((x, y))
        
        attempts += 1
    
    # Aggregate samples by block to create nominal distribution
    nominal_prob = {block_id: 0.0 for block_id in short_geoid_list}
    
    for x, y in samples:
        # Determine which block this sample falls into
        block_row = int(np.clip(np.floor(x), 0, grid_size - 1))
        block_col = int(np.clip(np.floor(y), 0, grid_size - 1))
        block_idx = block_row * grid_size + block_col
        block_id = short_geoid_list[block_idx]
        
        nominal_prob[block_id] += 1.0 / len(samples)
    
    return nominal_prob

def create_omega_distribution(grid_size: int, seed: int = 42) -> Tuple[Dict[str, np.ndarray], callable]:
    """Create parametrized ODD feature distribution"""
    np.random.seed(seed)
    n_blocks = grid_size * grid_size
    short_geoid_list = [f"BLK{i:03d}" for i in range(n_blocks)]
    
    Omega_dict = {}
    
    for i, block_id in enumerate(short_geoid_list):
        # Compute normalized coordinates
        row = i // grid_size
        col = i % grid_size
        x_norm = col / (grid_size - 1) if grid_size > 1 else 0.5
        y_norm = row / (grid_size - 1) if grid_size > 1 else 0.5
        
        # Feature 1: Spatially correlated Beta(2.5, 1.8) distribution
        spatial_base1 = 0.3 + 0.6 * (x_norm + y_norm) / 2.0
        omega1 = np.random.beta(2.5, 1.8) * spatial_base1 * 5.0
        omega1 = max(0.5, omega1)  # Minimum threshold
        
        # Feature 2: Bimodal spatial distribution with Gamma modes
        center_dist = np.sqrt((x_norm - 0.5)**2 + (y_norm - 0.5)**2)
        if center_dist < 0.3:
            # Central mode
            omega2 = np.random.gamma(2.0, 1.5) * 1.2
        else:
            # Peripheral mode
            omega2 = np.random.gamma(1.5, 0.8) * 2.0
        omega2 = max(0.5, omega2)
        
        Omega_dict[block_id] = np.array([omega1, omega2])
    
    def J_function(omega_vector):
        if hasattr(omega_vector, '__len__') and len(omega_vector) >= 2:
            return 0.6 * omega_vector[0] + 0.4 * omega_vector[1]
        return 0.0
    
    return Omega_dict, J_function

def compute_cost_breakdown(partition: Partition, assignment: np.ndarray, depot_id: str,
                          prob_dict: Dict[str, float], Omega_dict: Dict[str, np.ndarray],
                          J_function: callable, Lambda: float, wr: float, wv: float,
                          beta: float) -> Tuple[float, float, float]:
    """Compute provider and user costs using the real_case evaluator formulas.

    Provider per district (per hour):
      - linehaul = 2 * dr_min / T
      - travel   = mean_transit_distance_per_interval / T
        where mean_transit_distance_per_interval = beta * sqrt(Lambda * T) * alpha,
              alpha = sum_j sqrt(p_j) * sqrt(area_j) over assigned blocks
      - odd      = J(omega_district) / T with omega_district = elementwise max Omega over blocks

    User expected per district (aggregated by mass m):
      - wait    = T/2
      - transit = dr_min / wv + (mean_transit_distance_per_interval / (2 * wv))
      - cost    = wr * (wait + transit) * m
    """

    # Evaluate to obtain K_i (dr_min), F_i (ODD cost), T_star per district
    obj_val, district_info = partition.evaluate_partition_objective(
        assignment, depot_id, prob_dict, Lambda, wr, wv, beta, Omega_dict, J_function
    )

    block_ids = partition.short_geoid_list
    areas = {bid: float(partition.geodata.get_area(bid)) for bid in block_ids}
    travel_discount = getattr(partition.geodata, 'tsp_travel_discount', TSP_TRAVEL_DISCOUNT)

    total_provider = 0.0
    total_user = 0.0

    for info in district_info:
        if len(info) < 5:
            continue
        _, root, K_i, F_i, T = info[:5]
        # Gather assigned blocks for this root
        root_col = block_ids.index(root)
        assigned_blocks = [block_ids[i] for i in range(len(block_ids)) if assignment[i, root_col] > 0.5]
        if not assigned_blocks or T <= 0:
            continue

        # District mass and alpha (distribution-aware)
        m = sum(prob_dict.get(b, 0.0) for b in assigned_blocks)
        alpha = sum((prob_dict.get(b, 0.0) ** 0.5) * (areas.get(b, 1.0) ** 0.5) for b in assigned_blocks)
        mean_transit_dist = beta * float(np.sqrt(max(0.0, Lambda * T))) * alpha
        provider_transit_dist = mean_transit_dist / travel_discount

        # Provider components per hour
        linehaul_per_h = (2.0 * K_i) / T
        travel_per_h = provider_transit_dist / T
        odd_per_h = F_i / T

        total_provider += linehaul_per_h + travel_per_h + odd_per_h

        # User expected cost (mass-weighted)
        wait_h = T / 2.0
        transit_h = (K_i / wv) + (mean_transit_dist / (2.0 * wv))
        total_user += wr * (wait_h + transit_h) * m

    return total_provider, total_user, obj_val

def run_pareto_frontier_analysis(grid_size: int = 10, num_districts: int = 3, 
                                num_wr_wv_points: int = 15, max_iters: int = 100,
                                output_dir: str = "results", trials: int = 5,
                                seed: int = 42) -> List[ParetoPoint]:
    """
    Run Pareto frontier analysis by varying wr/wv ratios
    
    Args:
        grid_size: Grid dimension (default 10 for 10×10)
        num_districts: Number of districts
        num_wr_wv_points: Number of wr/wv ratios to test
        max_iters: Maximum iterations for optimization
        output_dir: Directory to save results
    
    Returns:
        List of ParetoPoint objects representing the frontier
    """
    
    print(f"🎯 Starting Pareto Frontier Analysis")
    print(f"   Grid: {grid_size}×{grid_size} ({grid_size**2} blocks)")
    print(f"   Districts: {num_districts}")
    print(f"   theta points: {num_wr_wv_points}")
    print(f"   Max iterations: {max_iters}")
    print(f"   Trials per ratio: {trials}")
    
    # Create test problem
    n_blocks = grid_size * grid_size
    geo_data = ToyGeoData(n_blocks, grid_size, seed=seed)
    nominal_prob = create_nominal_distribution(grid_size, seed=seed)
    Omega_dict, J_function = create_omega_distribution(grid_size, seed=seed)
    
    # Fix base speeds for evaluation (km/h or consistent units)
    wr_base = 5.04
    wv_base = 18.710765208297367
    geo_data.wr = wr_base
    geo_data.wv = wv_base

    # Define theta in (0,1]; lower theta emphasizes provider cost, higher theta emphasizes user cost
    # Avoid theta=0 to prevent division by zero when forming wr_eff = wr_base * ((1-theta)/theta)
    thetas = np.linspace(0.05, 1.0, num_wr_wv_points)
    
    # Fixed parameters
    Lambda = 25.0
    beta = 0.7120
    epsilon = 0.1  # Moderate robustness
    
    pareto_points = []
    
    for i, theta in enumerate(thetas):
        print(f"\n📊 Point {i+1}/{num_wr_wv_points}: theta = {theta:.3f}")
        # Effective wr passed into optimizer to realize provider + ((1-theta)/theta)*user
        wr_eff = wr_base * ((1.0 - theta) / theta)
        wv_eff = wv_base
        print(f"   wr_eff = {wr_eff:.3f}, wv_eff = {wv_eff:.3f} (evaluation uses wr={wr_base:.3f}, wv={wv_base:.3f})")
        
        user_list = []
        prov_list = []
        total_list = []
        obj_list = []
        time_list = []
        depot_last = None
        centers_last = None

        # Ensure the optimizer reads the effective weights
        geo_data.wr = wr_eff
        geo_data.wv = wv_eff

        for t in range(trials):
            start_time = time.time()
            # Fresh partition instance each trial (stochastic search)
            partition = Partition(geo_data, num_districts=num_districts, 
                                  prob_dict=nominal_prob, epsilon=epsilon)
            depot, centers, assignment, obj_val, district_info = partition.random_search(
                max_iters=max_iters,
                prob_dict=nominal_prob,
                Lambda=Lambda,
                wr=wr_eff,
                wv=wv_eff,
                beta=beta,
                Omega_dict=Omega_dict,
                J_function=J_function
            )
            computation_time = time.time() - start_time
            # Re-evaluate costs with base speeds for reporting and scalarization
            provider_cost, user_cost, _ = compute_cost_breakdown(
                partition, assignment, depot, nominal_prob, Omega_dict, J_function, Lambda, wr_base, wv_base, beta
            )
            total_cost = theta * provider_cost + (1.0 - theta) * user_cost
            user_list.append(user_cost)
            prov_list.append(provider_cost)
            total_list.append(total_cost)
            obj_list.append(obj_val)
            time_list.append(computation_time)
            depot_last, centers_last = depot, centers

        # Stats
        user_mean, user_std = float(np.mean(user_list)), float(np.std(user_list, ddof=1) if len(user_list)>1 else 0.0)
        prov_mean, prov_std = float(np.mean(prov_list)), float(np.std(prov_list, ddof=1) if len(prov_list)>1 else 0.0)
        total_mean, total_std = float(np.mean(total_list)), float(np.std(total_list, ddof=1) if len(total_list)>1 else 0.0)
        obj_mean = float(np.mean(obj_list))
        time_mean = float(np.mean(time_list))

        point = ParetoPoint(
            wr=wr_base,
            wv=wv_base,
            wr_wv_ratio=theta,
            user_cost=user_mean,
            user_cost_std=user_std,
            provider_cost=prov_mean,
            provider_cost_std=prov_std,
            total_cost=total_mean,
            total_cost_std=total_std,
            depot_id=depot_last,
            num_districts=len(centers_last) if centers_last is not None else num_districts,
            objective_value=obj_mean,
            computation_time=time_mean,
        )
        
        pareto_points.append(point)
        
        print(f"   ✅ Provider cost (eval): {prov_mean:.2f} ± {prov_std:.2f}")
        print(f"   ✅ User cost (eval): {user_mean:.2f} ± {user_std:.2f}")
        print(f"   ✅ Scalarized total θ·Prov+(1−θ)·User: {total_mean:.2f} ± {total_std:.2f}")
        print(f"   ⏱️  Time (mean over trials): {time_mean:.1f}s")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save detailed results as JSON
    results_data = {
        'metadata': {
            'grid_size': grid_size,
            'num_districts': num_districts,
            'num_points': num_wr_wv_points,
            'max_iters': max_iters,
            'trials': trials,
            'timestamp': timestamp
        },
        'pareto_points': [
            {
                'wr': p.wr,
                'wv': p.wv,
                'wr_wv_ratio': p.wr_wv_ratio,
                'user_cost': p.user_cost,
                'user_cost_std': p.user_cost_std,
                'provider_cost': p.provider_cost,
                'provider_cost_std': p.provider_cost_std,
                'total_cost': p.total_cost,
                'total_cost_std': p.total_cost_std,
                'depot_id': p.depot_id,
                'num_districts': p.num_districts,
                'objective_value': p.objective_value,
                'computation_time': p.computation_time
            }
            for p in pareto_points
        ]
    }
    
    json_path = os.path.join(output_dir, f'pareto_frontier_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Save CSV for easy analysis
    df = pd.DataFrame([
        {
            'wr_wv_ratio': p.wr_wv_ratio,
            'wr': p.wr,
            'wv': p.wv,
            'user_cost': p.user_cost,
            'user_cost_std': p.user_cost_std,
            'provider_cost': p.provider_cost,
            'provider_cost_std': p.provider_cost_std,
            'total_cost': p.total_cost,
            'total_cost_std': p.total_cost_std,
            'objective_value': p.objective_value,
            'computation_time': p.computation_time
        }
        for p in pareto_points
    ])
    
    csv_path = os.path.join(output_dir, f'pareto_frontier_{timestamp}.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"\n💾 Results saved:")
    print(f"   JSON: {json_path}")
    print(f"   CSV: {csv_path}")
    
    return pareto_points

def visualize_pareto_frontier(pareto_points: List[ParetoPoint], save_path: str = None):
    """Create comprehensive Pareto frontier visualization.

    If a save_path is provided, the figure is saved and the window is closed
    without calling plt.show() (which is blocking). If no save_path is given,
    the plot is shown interactively.
    """
    
    # Extract data
    user_costs = [p.user_cost for p in pareto_points]
    provider_costs = [p.provider_cost for p in pareto_points]
    user_err = [p.user_cost_std for p in pareto_points]
    prov_err = [p.provider_cost_std for p in pareto_points]
    ratios = [p.wr_wv_ratio for p in pareto_points]
    total_costs = [p.total_cost for p in pareto_points]
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Pareto Frontier Analysis: User vs Provider Costs', fontsize=16, fontweight='bold')
    
    # 1. Main Pareto frontier plot
    # Use log scale for the frontier panel; avoid zeros by flooring values
    eps = 1e-3
    prov_plot = [max(eps, v) for v in provider_costs]
    user_plot = [max(eps, v) for v in user_costs]
    prov_err_plot = [min(v*0.999, e) if v > eps else e for v, e in zip(prov_plot, prov_err)]
    user_err_plot = [min(v*0.999, e) if v > eps else e for v, e in zip(user_plot, user_err)]

    scatter = ax1.scatter(prov_plot, user_plot, c=ratios, s=100, 
                         cmap='viridis', alpha=0.8, edgecolors='black', linewidth=1)
    ax1.errorbar(prov_plot, user_plot, xerr=prov_err_plot, yerr=user_err_plot,
                 fmt='none', ecolor='gray', alpha=0.6, capsize=3)
    ax1.plot(prov_plot, user_plot, 'k--', alpha=0.5, linewidth=1)
    ax1.set_xlabel('Provider Cost', fontweight='bold')
    ax1.set_ylabel('User Cost', fontweight='bold')
    ax1.set_title('Pareto Frontier: User vs Provider Costs')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar for wr/wv ratio
    cbar1 = plt.colorbar(scatter, ax=ax1)
    # θ is the provider weight in the scalarization: θ·Provider + (1−θ)·User
    cbar1.set_label('θ (provider weight)', fontweight='bold')
    
    # 2. Cost vs wr/wv ratio
    ax2.errorbar(ratios, user_costs, yerr=user_err, fmt='o-', label='User Cost', color='red', linewidth=2, markersize=6, capsize=3)
    ax2.errorbar(ratios, provider_costs, yerr=prov_err, fmt='s-', label='Provider Cost', color='blue', linewidth=2, markersize=6, capsize=3)
    ax2.plot(ratios, total_costs, '^-', label='Total Cost', color='purple', linewidth=2, markersize=6)
    ax2.set_xlabel('θ (provider weight)', fontweight='bold')
    ax2.set_ylabel('Cost', fontweight='bold')
    ax2.set_title('Cost Components vs wr/wv Ratio')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Cost efficiency (total cost vs ratio)
    ax3.plot(ratios, total_costs, 'o-', color='purple', linewidth=2, markersize=8)
    min_total_idx = np.argmin(total_costs)
    ax3.plot(ratios[min_total_idx], total_costs[min_total_idx], 'r*', 
             markersize=15, label=f'Minimum Total Cost\n(ratio={ratios[min_total_idx]:.3f})')
    ax3.set_xlabel('θ (provider weight)', fontweight='bold')
    ax3.set_ylabel('Total Cost', fontweight='bold')
    ax3.set_title('Total Cost Efficiency')
    ax3.set_xscale('log')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Trade-off analysis (normalized costs)
    user_norm = np.array(user_costs) / np.max(user_costs)
    provider_norm = np.array(provider_costs) / np.max(provider_costs)
    ax4.plot(ratios, user_norm, 'o-', label='User Cost (norm)', color='red', linewidth=2)
    ax4.plot(ratios, provider_norm, 's-', label='Provider Cost (norm)', color='blue', linewidth=2)
    ax4.set_xlabel('wr/wv Ratio', fontweight='bold')
    ax4.set_ylabel('Normalized Cost', fontweight='bold')
    ax4.set_title('Normalized Cost Trade-offs')
    ax4.set_xscale('log')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved to: {save_path}")
        plt.close(fig)
    else:
        # Interactive session: this will block until the window is closed
        pass
        # plt.show()

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pareto Frontier Analysis for User vs Provider Costs')
    parser.add_argument('--grid-size', type=int, default=10, help='Grid size (default: 10)')
    parser.add_argument('--districts', type=int, default=3, help='Number of districts (default: 3)')
    parser.add_argument('--points', type=int, default=15, help='Number of wr/wv ratios to test (default: 15)')
    parser.add_argument('--iters', type=int, default=100, help='Max optimization iterations (default: 100)')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory (default: results)')
    parser.add_argument('--trials', type=int, default=5, help='Trials per wr/wv ratio (default: 5)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    parser.add_argument('--no-plot', action='store_true', help='Skip visualization')
    
    args = parser.parse_args()
    
    # Run analysis
    pareto_points = run_pareto_frontier_analysis(
        grid_size=args.grid_size,
        num_districts=args.districts,
        num_wr_wv_points=args.points,
        max_iters=args.iters,
        output_dir=args.output_dir,
        trials=args.trials,
        seed=args.seed
    )
    
    # Create visualization
    if not args.no_plot:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = os.path.join(args.output_dir, f'pareto_frontier_plot_{timestamp}.pdf')
        visualize_pareto_frontier(pareto_points, save_path=plot_path)
    
    # Print summary
    print(f"\n🎉 Pareto Frontier Analysis Complete!")
    print(f"   Generated {len(pareto_points)} points")
    print(f"   User cost range: {min(p.user_cost for p in pareto_points):.2f} - {max(p.user_cost for p in pareto_points):.2f}")
    print(f"   Provider cost range: {min(p.provider_cost for p in pareto_points):.2f} - {max(p.provider_cost for p in pareto_points):.2f}")
    
    # Find interesting points
    min_total_point = min(pareto_points, key=lambda p: p.total_cost)
    min_user_point = min(pareto_points, key=lambda p: p.user_cost)
    min_provider_point = min(pareto_points, key=lambda p: p.provider_cost)
    
    print(f"\n📈 Key Points:")
    print(f"   Minimum total cost: wr/wv={min_total_point.wr_wv_ratio:.3f}, total={min_total_point.total_cost:.2f}")
    print(f"   Minimum user cost: wr/wv={min_user_point.wr_wv_ratio:.3f}, user={min_user_point.user_cost:.2f}")
    print(f"   Minimum provider cost: wr/wv={min_provider_point.wr_wv_ratio:.3f}, provider={min_provider_point.provider_cost:.2f}")

if __name__ == '__main__':
    main()
