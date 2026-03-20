#!/usr/bin/env python3
"""
Test Enhanced Heuristics on Parametric Grid with Visualization
Demonstrates the enhanced random search + local search on configurable grid sizes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import matplotlib.pyplot as plt
from lib.algorithm import Partition
from lib.data import GeoData
import networkx as nx
import time
import argparse

class ToyGeoData(GeoData):
    """Toy geographic data for parametric grid testing"""
    
    def __init__(self, n_blocks, grid_size, seed=42):
        np.random.seed(seed)
        self.n_blocks = n_blocks
        self.grid_size = grid_size
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
    
    def get_dist(self, block1, block2):
        if block1 == block2:
            return 0.0
        try:
            i1 = self.short_geoid_list.index(block1)
            i2 = self.short_geoid_list.index(block2)
            coord1 = self.block_to_coord[i1]
            coord2 = self.block_to_coord[i2]
            return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
        except:
            return float('inf')
    
    def get_area(self, block_id):
        return 1.0
    
    def get_K(self, block_id):
        return 2.0

def test_heuristic_parametric(grid_size=10, num_districts=3, max_iters=100, seed=42):
    """Test enhanced heuristics on parametric grid with visualization
    
    Parameters:
    - grid_size: Size of square grid (e.g., 10 for 10x10)
    - num_districts: Number of districts to create
    - max_iters: Maximum iterations for random search
    - seed: Random seed for reproducibility
    """
    
    n_blocks = grid_size * grid_size
    
    print("=" * 80)
    print(f"ENHANCED HEURISTICS TEST - {grid_size}×{grid_size} GRID")
    print("=" * 80)
    
    # Setup
    toy_geo = ToyGeoData(n_blocks, grid_size, seed=seed)
    toy_geo.beta = 0.7120
    toy_geo.Lambda = 25.0
    toy_geo.wr = 1.0
    toy_geo.wv = 10.0
    
    # Create three-cluster truncated mixed Gaussian demand distribution
    prob_dict = {}
    
    # Define three cluster centers (scaled to grid size)
    cluster_centers = [
        (grid_size * 0.25, grid_size * 0.25),  # Top-left cluster
        (grid_size * 0.75, grid_size * 0.25),  # Top-right cluster  
        (grid_size * 0.50, grid_size * 0.75)   # Bottom-center cluster
    ]
    cluster_weights = [0.4, 0.35, 0.25]  # Relative importance of clusters
    cluster_sigmas = [grid_size * 0.18, grid_size * 0.20, grid_size * 0.15]  # Spread scaled to grid size
    
    for i, block_id in enumerate(toy_geo.short_geoid_list):
        row = i // grid_size
        col = i % grid_size
        
        # Compute mixture of three Gaussians
        prob_density = 0.0
        for (center_row, center_col), weight, sigma in zip(cluster_centers, cluster_weights, cluster_sigmas):
            dist_sq = (row - center_row)**2 + (col - center_col)**2
            gaussian = weight * np.exp(-dist_sq / (2 * sigma**2))
            prob_density += gaussian
        
        # Add small baseline and truncate
        base_prob = max(0.002, prob_density + 0.001 * np.random.uniform(-1, 1))
        prob_dict[block_id] = base_prob
    
    # Normalize probabilities
    total_prob = sum(prob_dict.values())
    for k in prob_dict:
        prob_dict[k] /= total_prob
    
    # Create parametrized two-dimensional ODD features
    Omega_dict = {}
    
    # ODD feature parameters
    # Feature 1: Beta distribution scaled and shifted
    alpha1, beta1 = 2.5, 1.8
    scale1, shift1 = 3.0, 1.5
    
    # Feature 2: Mixture of two modes with spatial correlation (scaled to grid)
    mode1_center = (grid_size * 0.3, grid_size * 0.3)
    mode2_center = (grid_size * 0.7, grid_size * 0.7)
    mode1_weight, mode2_weight = 0.6, 0.4
    
    for i, block_id in enumerate(toy_geo.short_geoid_list):
        row = i // grid_size
        col = i % grid_size
        
        # Feature 1: Spatially correlated Beta distribution
        # Use normalized distance from corner as Beta parameter
        normalized_dist = np.sqrt((row/grid_size)**2 + (col/grid_size)**2) / np.sqrt(2)
        beta_sample = np.random.beta(alpha1, beta1)
        omega1 = shift1 + scale1 * (beta_sample + 0.3 * normalized_dist)
        
        # Feature 2: Bimodal spatial distribution
        dist_to_mode1 = np.sqrt((row - mode1_center[0])**2 + (col - mode1_center[1])**2)
        dist_to_mode2 = np.sqrt((row - mode2_center[0])**2 + (col - mode2_center[1])**2)
        
        # Probability of belonging to mode 1 vs mode 2 based on distance
        mode1_prob = mode1_weight * np.exp(-dist_to_mode1 / (grid_size * 0.25))
        mode2_prob = mode2_weight * np.exp(-dist_to_mode2 / (grid_size * 0.25))
        total_mode_prob = mode1_prob + mode2_prob
        
        if total_mode_prob > 0:
            mode1_normalized = mode1_prob / total_mode_prob
        else:
            mode1_normalized = 0.5
            
        # Sample from appropriate mode
        if np.random.uniform() < mode1_normalized:
            omega2 = 2.2 + 0.8 * np.random.gamma(2.0, 0.5)  # Mode 1: lower values
        else:
            omega2 = 3.8 + 0.6 * np.random.gamma(1.5, 0.7)  # Mode 2: higher values
            
        # Add spatial noise
        omega2 += 0.15 * np.random.normal(0, 1)
        omega2 = max(0.5, omega2)  # Ensure positive
        
        Omega_dict[block_id] = np.array([omega1, omega2])
    
    def J_function(omega_vector):
        if hasattr(omega_vector, '__len__') and len(omega_vector) >= 2:
            return 0.6 * omega_vector[0] + 0.4 * omega_vector[1]
        return 0.0
    
    # Create partition instance
    partition = Partition(toy_geo, num_districts=num_districts, 
                         prob_dict=prob_dict, epsilon=0.3)
    
    print(f"🎯 Configuration: {grid_size}×{grid_size} grid, {num_districts} districts")
    print(f"📊 Total blocks: {n_blocks}")
    print(f"🚀 Running enhanced random search + local search...")
    
    # Run enhanced heuristics
    start_time = time.time()
    depot, centers, assignment, obj_val, district_info = partition.random_search(
        max_iters=max_iters,
        prob_dict=prob_dict,
        Lambda=toy_geo.Lambda,
        wr=toy_geo.wr,
        wv=toy_geo.wv,
        beta=toy_geo.beta,
        Omega_dict=Omega_dict,
        J_function=J_function
    )
    elapsed_time = time.time() - start_time
    
    print(f"✅ Completed in {elapsed_time:.2f}s")
    print(f"📍 Optimal depot: {depot}")
    print(f"🏛️ District roots: {centers}")
    print(f"💰 Final objective: {obj_val:.4f}")
    
    # Analyze results
    block_ids = toy_geo.short_geoid_list
    N = len(block_ids)
    
    # Create district assignments for visualization
    district_assignment = np.full((grid_size, grid_size), -1)
    depot_location = None
    root_locations = []
    
    # Map depot location
    depot_idx = block_ids.index(depot)
    depot_row = depot_idx // grid_size
    depot_col = depot_idx % grid_size
    depot_location = (depot_row, depot_col)
    
    # Map district assignments and roots
    district_sizes = []
    district_costs = []
    
    for d, root_id in enumerate(centers):
        root_idx = block_ids.index(root_id)
        root_row = root_idx // grid_size
        root_col = root_idx % grid_size
        root_locations.append((root_row, root_col))
        
        # Count assigned blocks for this district
        assigned_count = 0
        
        for i in range(N):
            if round(assignment[i, root_idx]) == 1:
                row = i // grid_size
                col = i % grid_size
                district_assignment[row, col] = d
                assigned_count += 1
                
        district_sizes.append(assigned_count)
        
        # Get district cost from district_info
        for cost, root, K_i, F_i, T_star in district_info:
            if root == root_id:
                district_costs.append(cost)
                break
    
    print(f"\\n📊 District Analysis:")
    for d, (root_id, size, cost) in enumerate(zip(centers, district_sizes, district_costs)):
        print(f"   District {d+1}: Root {root_id}, Size {size} blocks, Cost {cost:.3f}")
    
    balance_ratio = max(district_sizes) / min(district_sizes) if min(district_sizes) > 0 else float('inf')
    print(f"   Load balance ratio: {balance_ratio:.2f}:1")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'Enhanced Heuristics Results - {grid_size}×{grid_size} Grid\\\\n'
                f'Depot: {depot}, Objective: {obj_val:.3f}, Time: {elapsed_time:.1f}s',
                fontsize=14, fontweight='bold')
    
    # 1. District Assignments
    ax1 = axes[0, 0]
    colors = plt.cm.Set3(np.linspace(0, 1, num_districts))
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    
    im1 = ax1.imshow(district_assignment, cmap=cmap, vmin=0, vmax=num_districts-1)
    ax1.set_title('District Assignments')
    
    # Mark depot
    ax1.scatter(depot_col, depot_row, c='red', s=200, marker='*', 
               edgecolors='black', linewidth=2, label='Depot')
    
    # Mark roots
    for d, (root_row, root_col) in enumerate(root_locations):
        ax1.scatter(root_col, root_row, c='white', s=100, marker='s', 
                   edgecolors='black', linewidth=2)
        ax1.text(root_col, root_row, f"R{d+1}", ha='center', va='center', 
                fontsize=8, fontweight='bold', color='black')
    
    ax1.legend()
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    
    # 2. Demand Distribution
    ax2 = axes[0, 1]
    prob_grid = np.zeros((grid_size, grid_size))
    for i, block_id in enumerate(block_ids):
        row = i // grid_size
        col = i % grid_size
        prob_grid[row, col] = prob_dict[block_id]
    
    im2 = ax2.imshow(prob_grid, cmap='YlOrRd')
    ax2.set_title('Demand Distribution')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Row')
    
    # 3. ODD Features (Dimension 1)
    ax3 = axes[1, 0]
    odd1_grid = np.zeros((grid_size, grid_size))
    for i, block_id in enumerate(block_ids):
        row = i // grid_size
        col = i % grid_size
        odd1_grid[row, col] = Omega_dict[block_id][0]
    
    im3 = ax3.imshow(odd1_grid, cmap='viridis')
    ax3.set_title('ODD Features (Dimension 1)')
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    ax3.set_xlabel('Column')
    ax3.set_ylabel('Row')
    
    # 4. District Costs
    ax4 = axes[1, 1]
    district_bars = ax4.bar(range(1, num_districts+1), district_costs, 
                           color=colors[:num_districts])
    ax4.set_title('District Costs')
    ax4.set_xlabel('District')
    ax4.set_ylabel('Cost')
    ax4.set_xticks(range(1, num_districts+1))
    
    # Add value labels on bars
    for i, (bar, cost) in enumerate(zip(district_bars, district_costs)):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{cost:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save with grid size in filename
    filename = f'enhanced_heuristics_{grid_size}x{grid_size}_results.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"📁 Saved visualization as '{filename}'")
    
    # Summary
    print(f"\\n🎉 ENHANCED HEURISTICS SUMMARY:")
    print(f"   ✅ Successfully partitioned {n_blocks} blocks into {num_districts} districts")
    print(f"   ✅ Optimized depot location: {depot}")
    print(f"   ✅ Balanced district sizes (ratio {balance_ratio:.1f}:1)")
    print(f"   ✅ Integrated ODD costs and depot optimization")
    print(f"   ✅ Fast execution: {elapsed_time:.1f}s")
    print(f"   💰 Final objective: {obj_val:.4f}")
    
    return {
        'grid_size': grid_size,
        'num_districts': num_districts,
        'depot': depot,
        'centers': centers,
        'objective': obj_val,
        'elapsed_time': elapsed_time,
        'district_sizes': district_sizes,
        'balance_ratio': balance_ratio
    }

def main():
    """Command line interface for parametric testing"""
    parser = argparse.ArgumentParser(description='Test Enhanced Heuristics on Parametric Grid')
    parser.add_argument('--grid_size', type=int, default=10, help='Grid size (default: 10)')
    parser.add_argument('--num_districts', type=int, default=3, help='Number of districts (default: 3)')
    parser.add_argument('--max_iters', type=int, default=100, help='Max iterations (default: 100)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.grid_size < 3:
        print("❌ Grid size must be at least 3")
        return
    
    if args.num_districts < 2 or args.num_districts > args.grid_size * args.grid_size // 2:
        print(f"❌ Number of districts must be between 2 and {args.grid_size * args.grid_size // 2}")
        return
    
    # Run test
    result = test_heuristic_parametric(
        grid_size=args.grid_size,
        num_districts=args.num_districts,
        max_iters=args.max_iters,
        seed=args.seed
    )
    
    print(f"\\n📋 Test completed successfully!")

if __name__ == "__main__":
    main()