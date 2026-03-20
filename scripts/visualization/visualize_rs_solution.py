#!/usr/bin/env python3
"""
Visualize Random Search Heuristic Solution
Creates a figure similar to Figure 6 showing optimal service design and demand distributions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from lib.algorithm import Partition
from lib.data import GeoData
import networkx as nx
from scipy.stats import multivariate_normal, beta, gamma

# ===== CENTRALIZED MIXED-GAUSSIAN PARAMETERS =====
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
        return [grid_size * 0.15 *0.5, grid_size * 0.16 *0.5, grid_size * 0.12 *0.5]

class ToyGeoData(GeoData):
    """Toy geographic data for visualization"""
    
    def __init__(self, n_blocks, grid_size, service_region_miles=10.0, seed=42):
        np.random.seed(seed)
        self.n_blocks = n_blocks
        self.grid_size = grid_size
        self.service_region_miles = service_region_miles
        self.miles_per_grid_unit = service_region_miles / grid_size
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
            grid_dist = np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
            return grid_dist * self.miles_per_grid_unit
        except:
            return float('inf')
    
    def get_area(self, block_id):
        # Area per block = total service region area / number of blocks
        # Service region: 10 miles × 10 miles = 100 square miles
        return 100.0 / self.n_blocks
    
    def get_K(self, block_id):
        return 2.0

def create_true_distribution_sampler(grid_size: int, seed: int = 42):
    """Create sampler for properly truncated mixed-Gaussian distribution"""
    from scipy.stats import multivariate_normal
    
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
        # Use numerical integration over the rectangle [0, grid_size-1] x [0, grid_size-1]
        prob_inside = mvn.cdf([grid_size-1, grid_size-1]) - mvn.cdf([grid_size-1, 0]) - mvn.cdf([0, grid_size-1]) + mvn.cdf([0, 0])
        
        # Renormalized weight = original_weight * prob_inside / sum(all prob_inside)
        truncated_weights.append(cluster_weights[i] * prob_inside)
    
    # Normalize truncated weights
    total_truncated_weight = sum(truncated_weights)
    truncated_weights = [w / total_truncated_weight for w in truncated_weights]
    
    print(f"Original weights: {cluster_weights}")
    print(f"Truncated weights: {[f'{w:.3f}' for w in truncated_weights]}")
    
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
        
        if len(samples) < n_samples:
            print(f"Warning: Only generated {len(samples)} out of {n_samples} samples")
        
        return samples
    
    return sample_from_true_distribution

def create_nominal_distribution(grid_size: int, n_samples: int = 1000, seed: int = 42):
    """
    Create nominal distribution by sampling from true distribution and aggregating by blocks
    
    Returns:
    - nominal_prob: Block-based nominal distribution (proper probabilities that sum to 1)
    """
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
    
    return nominal_prob

def generate_odd_features(grid_size, seed=42):
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
    
    return Omega_dict

def J_function(omega):
    """ODD cost function"""
    return 50 * (omega[0]**1.5 + omega[1]**1.2)

def create_continuous_demand_field(grid_size, service_region_miles=10.0, resolution=100):
    """Create continuous TRUE truncated mixed-Gaussian demand distribution for visualization"""
    from scipy.stats import multivariate_normal
    
    # Use same parameters as the true distribution sampler
    miles_per_grid_unit = service_region_miles / grid_size
    
    # Use centralized configuration
    cluster_centers_grid = MixedGaussianConfig.get_cluster_centers(grid_size)
    cluster_weights = MixedGaussianConfig.get_cluster_weights()
    cluster_sigmas = MixedGaussianConfig.get_cluster_sigmas(grid_size)
    
    # Convert grid coordinates to miles
    cluster_centers = [(cx * miles_per_grid_unit, cy * miles_per_grid_unit) 
                      for cx, cy in cluster_centers_grid]
    cluster_sigmas_miles = [sigma * miles_per_grid_unit for sigma in cluster_sigmas]
    
    # Calculate truncation probabilities for proper normalization
    truncated_weights = []
    for i, (center_grid, sigma_grid) in enumerate(zip(cluster_centers_grid, cluster_sigmas)):
        # Create 2D Gaussian distribution for this cluster
        cov_matrix = [[sigma_grid**2, 0], [0, sigma_grid**2]]
        mvn = multivariate_normal(mean=center_grid, cov=cov_matrix)
        
        # Calculate probability mass within service region
        prob_inside = mvn.cdf([grid_size-1, grid_size-1]) - mvn.cdf([grid_size-1, 0]) - mvn.cdf([0, grid_size-1]) + mvn.cdf([0, 0])
        truncated_weights.append(cluster_weights[i] * prob_inside)
    
    # Normalize truncated weights
    total_weight = sum(truncated_weights)
    truncated_weights = [w / total_weight for w in truncated_weights]
    
    # Create high-resolution grid
    x = np.linspace(0, service_region_miles, resolution)
    y = np.linspace(0, service_region_miles, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Compute truncated continuous demand field
    demand_field = np.zeros((resolution, resolution))
    for k, (center_miles, sigma_miles) in enumerate(zip(cluster_centers, cluster_sigmas_miles)):
        cov_miles = [[sigma_miles**2, 0], [0, sigma_miles**2]]
        rv = multivariate_normal(center_miles, cov_miles)
        
        for i in range(resolution):
            for j in range(resolution):
                # Only add density if within service region
                if 0 <= X[i, j] <= service_region_miles and 0 <= Y[i, j] <= service_region_miles:
                    density = rv.pdf([X[i, j], Y[i, j]])
                    demand_field[i, j] += truncated_weights[k] * density
    
    return demand_field * 2000  # Scale for visibility

def create_visualization(geodata, depot_id, district_roots, assignment, dispatch_intervals, 
                        prob_dict, Omega_dict, district_info=None):
    """Create the dual-panel visualization"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    grid_size = geodata.grid_size
    service_region_miles = geodata.service_region_miles
    
    # === LEFT PANEL: Optimal Service Design ===
    
    # 1. Continuous demand distribution background
    continuous_demand = create_continuous_demand_field(grid_size, service_region_miles)
    im1 = ax1.imshow(continuous_demand, cmap='Blues', alpha=0.6, origin='lower', 
                     extent=[0, service_region_miles, 0, service_region_miles])
    
    # 2. District boundaries and assignments
    colors = ['lightcoral', 'lightgreen', 'lightyellow', 'lightcyan', 'lightpink']
    block_to_district = {}
    
    # Find which district each block belongs to
    for i, block_id in enumerate(geodata.short_geoid_list):
        for district_idx, root in enumerate(district_roots):
            root_idx = int(root[3:]) if root.startswith('BLK') else int(root)
            if assignment[i, root_idx] == 1:
                block_to_district[i] = district_idx
                break
    
    # Draw district assignments
    miles_per_block = service_region_miles / grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            block_idx = i * grid_size + j
            if block_idx in block_to_district:
                district_idx = block_to_district[block_idx]
                x_start = j * miles_per_block
                y_start = i * miles_per_block
                rect = Rectangle((x_start, y_start), miles_per_block, miles_per_block, 
                               linewidth=2, edgecolor=colors[district_idx % len(colors)], 
                               facecolor=colors[district_idx % len(colors)], alpha=0.3)
                ax1.add_patch(rect)
    
    # 3. ODD feature bars for each block
    max_omega = max(np.max(omega) for omega in Omega_dict.values())
    
    for i in range(grid_size):
        for j in range(grid_size):
            block_id = f"BLK{i * grid_size + j:03d}"
            omega = Omega_dict[block_id]
            
            # Convert to miles coordinates
            x_block = j * miles_per_block
            y_block = i * miles_per_block
            bar_width = miles_per_block * 0.12
            
            # Feature 1 bar (left bar, closer to center)
            bar_height1 = (omega[0] / max_omega) * miles_per_block * 0.8
            rect1 = Rectangle((x_block + miles_per_block * 0.35, y_block + miles_per_block * 0.1), 
                            bar_width, bar_height1, 
                            facecolor='gray', edgecolor='black', linewidth=0.5)
            ax1.add_patch(rect1)
            
            # Feature 2 bar (right bar, closer to center)  
            bar_height2 = (omega[1] / max_omega) * miles_per_block * 0.8
            rect2 = Rectangle((x_block + miles_per_block * 0.53, y_block + miles_per_block * 0.1), 
                            bar_width, bar_height2,
                            facecolor='darkgray', edgecolor='black', linewidth=0.5)
            ax1.add_patch(rect2)
    
    # 4. Depot location (red square)
    depot_idx = int(depot_id[3:]) if depot_id.startswith('BLK') else int(depot_id)
    depot_i, depot_j = depot_idx // grid_size, depot_idx % grid_size
    depot_x = depot_j * miles_per_block + miles_per_block * 0.3
    depot_y = depot_i * miles_per_block + miles_per_block * 0.3
    depot_size = miles_per_block * 0.4
    depot_rect = Rectangle((depot_x, depot_y), depot_size, depot_size, 
                          facecolor='red', edgecolor='darkred', linewidth=2)
    ax1.add_patch(depot_rect)
    
    # 5. Dispatch intervals as text
    for district_idx, root in enumerate(district_roots):
        root_idx = int(root[3:]) if root.startswith('BLK') else int(root)
        root_i, root_j = root_idx // grid_size, root_idx % grid_size
        interval = dispatch_intervals.get(root, 0)
        root_x = root_j * miles_per_block + miles_per_block * 0.5
        root_y = root_i * miles_per_block + miles_per_block * 0.5
        ax1.text(root_x, root_y, f'{interval:.1f}', 
                ha='center', va='center', fontsize=10, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    ax1.set_xlim(0, service_region_miles)
    ax1.set_ylim(0, service_region_miles)
    ax1.set_xlabel('x (miles)')
    ax1.set_ylabel('y (miles)')
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar for demand
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label('Demand density')
    
    # Add legend for left panel
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Depot'),
        Patch(facecolor='gray', label='ODD feature 1'),
        Patch(facecolor='darkgray', label='ODD feature 2'),
        Patch(facecolor='black', alpha=0.7, label='Dispatch intervals')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # === RIGHT PANEL: Demand Distributions ===
    
    # True demand points (blue dots) - sample from underlying mixed-Gaussian distribution
    np.random.seed(42)
    n_samples = 100  # Increased for better visualization
    
    # Create true distribution sampler
    true_sampler = create_true_distribution_sampler(grid_size, seed=42)
    
    # Sample from true mixed-Gaussian distribution
    empirical_points_grid = true_sampler(n_samples)
    
    # Convert grid coordinates to miles coordinates
    empirical_points = []
    for x_grid, y_grid in empirical_points_grid:
        x_miles = x_grid * miles_per_block
        y_miles = y_grid * miles_per_block
        empirical_points.append((x_miles, y_miles))
    
    empirical_x, empirical_y = zip(*empirical_points)
    ax2.scatter(empirical_x, empirical_y, c='blue', s=20, alpha=0.6, label='True demand samples')
    
    # Extract worst-case distribution from CQCP solutions if available
    worst_case_dist = np.zeros((grid_size, grid_size))
    
    if district_info is not None:
        # Get worst-case distribution from CQCP subproblem solutions
        # district_info contains tuples: (cost, root, K_i, F_i, T_star, x_star_dict)
        # Aggregate worst-case intensities across all districts using MAX
        worst_case_intensities = {}
        
        for district_data in district_info:
            if len(district_data) >= 6:  # Has x_star_dict
                x_star_dict = district_data[5] if isinstance(district_data[5], dict) else {}
                print(f"District {district_data[1]} x_star sample: {list(x_star_dict.items())[:5]}")
                for block_id, intensity in x_star_dict.items():
                    if block_id not in worst_case_intensities:
                        worst_case_intensities[block_id] = 0.0
                    worst_case_intensities[block_id] = max(worst_case_intensities[block_id], intensity)
        
        # Fill grid with worst-case intensities
        for i in range(grid_size):
            for j in range(grid_size):
                block_id = f"BLK{i * grid_size + j:03d}"
                if block_id in worst_case_intensities:
                    worst_case_dist[i, j] = worst_case_intensities[block_id]
                else:
                    worst_case_dist[i, j] = 0.0
    else:
        # Fallback: show nominal distribution 
        for i in range(grid_size):
            for j in range(grid_size):
                block_id = f"BLK{i * grid_size + j:03d}"
                worst_case_dist[i, j] = prob_dict[block_id]
    
    im2 = ax2.imshow(worst_case_dist, cmap='Oranges', alpha=0.8, origin='lower', 
                     extent=[0, service_region_miles, 0, service_region_miles])
    
    ax2.set_xlim(0, service_region_miles)
    ax2.set_ylim(0, service_region_miles)
    ax2.set_xlabel('x (miles)')
    ax2.set_ylabel('y (miles)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add title for right panel
    # ax2.set_title('True Demand Samples vs Worst-Case Distribution')
    
    # Add colorbar for worst-case distribution
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label('Worst-case intensity (CQCP solution)')
    
    # Add title for left panel  
    # ax1.set_title('Optimal Service Design')
    
    plt.tight_layout()
    
    # Add overall figure title
    # fig.suptitle(f'Random Search Optimization Results ({grid_size}×{grid_size} Grid)\nTrue Mixed-Gaussian Demand vs Distributionally Robust Design', 
    #             fontsize=14, fontweight='bold', y=0.98)
    
    return fig

def main():
    # Setup (reduced grid size for faster execution)
    grid_size = 10
    n_blocks = grid_size * grid_size
    num_districts = 3
    
    print("Setting up toy geographic data...")
    geodata = ToyGeoData(n_blocks, grid_size, service_region_miles=10.0, seed=42)
    
    print("Generating true demand samples and constructing nominal distribution...")
    
    # Step 1: Sample from true mixed-Gaussian distribution
    true_sampler = create_true_distribution_sampler(grid_size, seed=42)
    true_demand_samples = true_sampler(1000)  # Sample 1000 points
    
    # Step 2: Construct nominal distribution from true samples
    prob_dict = {}
    short_geoid_list = [f"BLK{i:03d}" for i in range(n_blocks)]
    
    # Initialize all blocks with zero probability
    for block_id in short_geoid_list:
        prob_dict[block_id] = 0.0
    
    # Count samples in each block
    for x, y in true_demand_samples:
        # Determine which block this sample falls into
        block_row = int(np.clip(np.floor(x), 0, grid_size - 1))
        block_col = int(np.clip(np.floor(y), 0, grid_size - 1))
        block_idx = block_row * grid_size + block_col
        block_id = short_geoid_list[block_idx]
        prob_dict[block_id] += 1.0 / len(true_demand_samples)
    
    print(f"Constructed nominal distribution from {len(true_demand_samples)} true samples")
    
    # Generate ODD features
    Omega_dict = generate_odd_features(grid_size, seed=42)
    
    # Verify probabilities sum to 1
    total_prob = sum(prob_dict.values())
    print(f"Total probability: {total_prob:.6f}")
    
    # Show some sample probabilities
    sample_probs = list(prob_dict.items())[:5]
    print(f"Sample probabilities: {sample_probs}")
    
    print("Creating partition instance and running Random Search...")
    partition = Partition(geodata, num_districts, prob_dict, epsilon=1)
    
    # Run Random Search heuristic (reduced iterations for faster execution)
    depot_id, district_roots, assignment, obj_val, district_info = partition.random_search(
        max_iters=20,
        prob_dict=prob_dict,
        Lambda=100, wr=1.0, wv=10.0, beta=0.7120,
        Omega_dict=Omega_dict,
        J_function=J_function
    )
    
    # Extract dispatch intervals from district_info
    dispatch_intervals = {}
    for i, root in enumerate(district_roots):
        if district_info and i < len(district_info):
            # district_info contains tuples: (cost, root, K_i, F_i, T_star)
            district_data = district_info[i]
            if len(district_data) >= 5:
                dispatch_intervals[root] = district_data[4]  # T_star
            else:
                dispatch_intervals[root] = 0.0
        else:
            dispatch_intervals[root] = 0.0
    
    print(f"Random Search completed:")
    print(f"  Objective value: {obj_val:.2f}")
    print(f"  Depot location: {depot_id}")
    print(f"  District roots: {district_roots}")
    print(f"  Dispatch intervals: {dispatch_intervals}")
    
    print("Creating visualization...")
    fig = create_visualization(geodata, depot_id, district_roots, assignment, 
                             dispatch_intervals, prob_dict, Omega_dict, district_info)
    
    # Save figure as PDF
    plt.savefig('rs_optimal_design.pdf', dpi=300, bbox_inches='tight')
    print("Visualization saved as 'rs_optimal_design.pdf'")
    
    plt.show()

if __name__ == "__main__":
    main()