#!/usr/bin/env python3
"""
Visualize Worst-Case Distribution over 10×10 Mile Service Region

This script generates and visualizes:
1. Truncated mixed-Gaussian underlying demand distribution
2. Empirical samples from the distribution  
3. Worst-case distribution computed via CQCP Benders decomposition
4. Comparison between empirical and worst-case distributions

Service region: 10×10 miles divided into 10×10 grid (100 blocks)
Distance metric: Euclidean distance
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'analysis'))

import numpy as np
import matplotlib.pyplot as plt
from lib.algorithm import Partition
from lib.data import GeoData
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# Import classes and functions from validation script
from validate_sample_cutoff import ToyGeoData, create_mixed_gaussian_samples

def create_truncated_mixed_gaussian_sampler(grid_size=10, service_region_miles=10.0, seed=42):
    """Create truncated mixed-Gaussian demand distribution sampler"""
    
    np.random.seed(seed)
    
    # Mixed-Gaussian parameters (in miles)
    cluster_centers = [
        (2.5, 2.5),   # Bottom-left cluster
        (7.5, 2.5),   # Bottom-right cluster  
        (5.0, 7.5)    # Top-center cluster
    ]
    cluster_weights = [0.4, 0.35, 0.25]
    cluster_sigmas = [1.5, 1.6, 1.2]  # Standard deviations in miles
    
    def sampler(n_samples):
        """Generate n_samples from truncated mixed-Gaussian"""
        samples = []
        max_attempts = n_samples * 10
        attempts = 0
        
        while len(samples) < n_samples and attempts < max_attempts:
            # Choose cluster
            cluster_idx = np.random.choice(len(cluster_centers), p=cluster_weights)
            center = cluster_centers[cluster_idx]
            sigma = cluster_sigmas[cluster_idx]
            
            # Generate sample from Gaussian
            x = np.random.normal(center[0], sigma)
            y = np.random.normal(center[1], sigma)
            
            # Accept if within service region bounds
            if 0 <= x <= service_region_miles and 0 <= y <= service_region_miles:
                samples.append((x, y))
            
            attempts += 1
        
        return samples
    
    return sampler

def samples_to_block_distribution(samples, geodata, grid_size):
    """Convert continuous samples to block-based probability distribution"""
    n_blocks = grid_size * grid_size
    block_counts = np.zeros(n_blocks)
    
    miles_per_block = geodata.service_region_miles / grid_size
    
    for x_miles, y_miles in samples:
        # Determine which block this sample falls into
        block_col = int(np.clip(np.floor(x_miles / miles_per_block), 0, grid_size - 1))
        block_row = int(np.clip(np.floor(y_miles / miles_per_block), 0, grid_size - 1))
        block_idx = block_row * grid_size + block_col
        block_counts[block_idx] += 1
    
    # Convert to probability distribution
    prob_dist = block_counts / len(samples)
    return prob_dist

def solve_worst_case_cqcp(geodata, empirical_prob_dict, epsilon):
    """Solve worst-case distribution using CQCP Benders decomposition"""
    try:
        # Use entire service region as one district
        assigned_blocks = geodata.short_geoid_list  # All 100 blocks
        root = assigned_blocks[50]  # Center block as root
        
        # Create partition instance
        partition = Partition(geodata, num_districts=1, prob_dict=empirical_prob_dict, epsilon=epsilon)
        
        # CQCP subproblem parameters
        K_i = 3.0     # Fixed linehaul cost
        F_i = 0.0     # No ODD cost
        beta = 0.7120 # Intensity parameter
        Lambda = 10.0 # Cost scaling parameter
        
        # Solve CQCP subproblem
        result = partition._CQCP_benders(
            assigned_blocks=assigned_blocks,
            root=root,
            prob_dict=empirical_prob_dict,
            epsilon=epsilon,
            K_i=K_i,
            F_i=F_i,
            grid_points=10,
            beta=beta,
            Lambda=Lambda,
            single_threaded=True
        )
        
        if len(result) >= 7:
            worst_case_cost, x_star_dict, subgrad, C_star, alpha_i, subgrad_K_i, subgrad_F_i = result
            
            # Extract worst-case distribution
            worst_case_dist = np.zeros(len(assigned_blocks))
            for i, block_id in enumerate(assigned_blocks):
                x_j = x_star_dict.get(block_id, 0.0)
                # Use x_j^2 for probability mass (ensure non-negative)
                worst_case_dist[i] = max(0.0, x_j)**2
            
            # Normalize to proper probability distribution
            total_mass = np.sum(worst_case_dist)
            if total_mass > 1e-10:
                worst_case_dist = worst_case_dist / total_mass
            else:
                worst_case_dist = np.ones(len(assigned_blocks)) / len(assigned_blocks)
            
            return worst_case_dist, True
        else:
            print("CQCP solver returned incomplete result")
            return None, False
            
    except Exception as e:
        print(f"CQCP solver failed: {e}")
        return None, False

def measure_uniformity(distribution):
    """Measure uniformity using entropy-based metric"""
    dist_safe = distribution + 1e-12  # Avoid log(0)
    entropy = -np.sum(dist_safe * np.log(dist_safe))
    max_entropy = np.log(len(distribution))  # Uniform distribution entropy
    uniformity = entropy / max_entropy if max_entropy > 0 else 0.0
    return max(0.0, min(1.0, uniformity))

def create_visualization(geodata, samples, empirical_dist, worst_case_dist, epsilon, n_samples):
    """Create comprehensive visualization"""
    
    grid_size = geodata.grid_size
    service_region_miles = geodata.service_region_miles
    
    # Calculate uniformity metrics
    empirical_uniformity = measure_uniformity(empirical_dist)
    worst_case_uniformity = measure_uniformity(worst_case_dist)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Empirical sample points only
    if samples:
        sample_x, sample_y = zip(*samples)
        ax1.scatter(sample_x, sample_y, c='blue', s=30, alpha=0.7, edgecolors='navy', linewidth=0.5)
    
    ax1.set_xlim([0, service_region_miles])
    ax1.set_ylim([0, service_region_miles])
    ax1.set_xlabel('x coordinate (miles)', fontsize=12)
    ax1.set_ylabel('y coordinate (miles)', fontsize=12)
    ax1.set_title(f'Empirical Demand Samples (n={n_samples})', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Add grid lines to show block boundaries
    for i in range(grid_size + 1):
        coord = i * (service_region_miles / grid_size)
        ax1.axhline(coord, color='gray', alpha=0.3, linewidth=0.5)
        ax1.axvline(coord, color='gray', alpha=0.3, linewidth=0.5)
    
    # Plot 2: Empirical distribution with samples
    empirical_grid = empirical_dist.reshape(grid_size, grid_size)
    vmax_empirical = max(np.max(empirical_grid), 0.02)
    
    im2 = ax2.imshow(empirical_grid, cmap='Greens', origin='lower', 
                    vmin=0, vmax=vmax_empirical, 
                    extent=[0, service_region_miles, 0, service_region_miles])
    
    if samples:
        ax2.scatter(sample_x, sample_y, c='blue', s=20, alpha=0.6, edgecolors='navy', linewidth=0.3)
    
    ax2.set_xlabel('x coordinate (miles)', fontsize=12)
    ax2.set_ylabel('y coordinate (miles)', fontsize=12)
    ax2.set_title(f'Empirical Distribution\nUniformity = {empirical_uniformity:.3f}', fontsize=14, fontweight='bold')
    plt.colorbar(im2, ax=ax2, shrink=0.8, label='Probability density')
    
    # Plot 3: Worst-case distribution with samples
    worst_case_grid = worst_case_dist.reshape(grid_size, grid_size)
    vmax_shared = max(np.max(empirical_grid), np.max(worst_case_grid), 0.02)
    
    im3 = ax3.imshow(worst_case_grid, cmap='YlOrRd', origin='lower', 
                    vmin=0, vmax=vmax_shared, 
                    extent=[0, service_region_miles, 0, service_region_miles])
    
    if samples:
        ax3.scatter(sample_x, sample_y, c='blue', s=20, alpha=0.6, edgecolors='navy', linewidth=0.3)
    
    ax3.set_xlabel('x coordinate (miles)', fontsize=12)
    ax3.set_ylabel('y coordinate (miles)', fontsize=12)
    ax3.set_title(f'Worst-Case Distribution\nε={epsilon:.3f}, Uniformity = {worst_case_uniformity:.3f}', 
                 fontsize=14, fontweight='bold')
    plt.colorbar(im3, ax=ax3, shrink=0.8, label='Probability density')
    
    # Plot 4: Side-by-side comparison (bar chart)
    block_indices = np.arange(len(empirical_dist))
    width = 0.35
    
    ax4.bar(block_indices - width/2, empirical_dist, width, 
           label='Empirical', alpha=0.7, color='green')
    ax4.bar(block_indices + width/2, worst_case_dist, width, 
           label='Worst-case', alpha=0.7, color='red')
    
    ax4.set_xlabel('Block Index', fontsize=12)
    ax4.set_ylabel('Probability Mass', fontsize=12)
    ax4.set_title('Distribution Comparison', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add summary statistics
    max_empirical = np.max(empirical_dist)
    max_worst_case = np.max(worst_case_dist)
    ax4.text(0.02, 0.98, f'Max empirical: {max_empirical:.4f}\nMax worst-case: {max_worst_case:.4f}', 
            transform=ax4.transAxes, verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig

def main():
    """Main visualization function"""
    
    # Parse command line arguments
    n_samples = 50      # Number of demand samples
    epsilon = 1.0       # Wasserstein radius
    grid_size = 50      # Grid size (20×20 for finer resolution)
    seed = 42           # Random seed
    
    # Parse arguments from command line
    for arg in sys.argv[1:]:
        try:
            if '=' in arg:
                key, value = arg.split('=')
                if key == 'n_samples':
                    n_samples = int(value)
                elif key == 'epsilon':
                    epsilon = float(value)
                elif key == 'grid_size':
                    grid_size = int(value)
                elif key == 'seed':
                    seed = int(value)
        except ValueError:
            print(f"Invalid argument: {arg}")
    
    service_region_miles = 10.0  # Fixed at 10×10 miles
    block_size_miles = service_region_miles / grid_size
    
    print("=" * 80)
    print("WORST-CASE DISTRIBUTION VISUALIZATION")
    print("=" * 80)
    print(f"Service region: {service_region_miles}×{service_region_miles} miles")
    print(f"Grid resolution: {grid_size}×{grid_size} blocks ({grid_size**2} total)")
    print(f"Block size: {block_size_miles:.2f}×{block_size_miles:.2f} miles")
    print(f"Number of demand samples: {n_samples}")
    print(f"Wasserstein radius: ε = {epsilon}")
    print(f"Distance metric: Euclidean")
    print(f"Underlying distribution: Truncated mixed-Gaussian (3 clusters)")
    print()
    
    # Create geodata
    print("Creating service region geodata...")
    geodata = ToyGeoData(grid_size=grid_size, service_region_miles=service_region_miles, seed=seed)
    
    # Create demand sampler
    print("Setting up truncated mixed-Gaussian demand distribution...")
    demand_sampler = create_truncated_mixed_gaussian_sampler(
        grid_size=grid_size, service_region_miles=service_region_miles, seed=seed
    )
    
    # Generate empirical samples
    print(f"Generating {n_samples} demand samples...")
    samples = demand_sampler(n_samples)
    print(f"Successfully generated {len(samples)} samples")
    
    # Convert to block distribution
    print("Converting samples to block-based empirical distribution...")
    empirical_dist = samples_to_block_distribution(samples, geodata, grid_size)
    empirical_prob_dict = {
        block_id: empirical_dist[i] 
        for i, block_id in enumerate(geodata.short_geoid_list)
    }
    
    empirical_uniformity = measure_uniformity(empirical_dist)
    print(f"Empirical distribution uniformity: {empirical_uniformity:.4f}")
    
    # Solve worst-case distribution
    print(f"Solving worst-case distribution with ε = {epsilon}...")
    worst_case_dist, success = solve_worst_case_cqcp(geodata, empirical_prob_dict, epsilon)
    
    if success and worst_case_dist is not None:
        worst_case_uniformity = measure_uniformity(worst_case_dist)
        print(f"Worst-case distribution uniformity: {worst_case_uniformity:.4f}")
        print()
        
        # Create visualization
        print("Creating visualization...")
        fig = create_visualization(
            geodata, samples, empirical_dist, worst_case_dist, epsilon, n_samples
        )
        
        # Save figure
        filename = f'worst_case_visualization_n{n_samples}_eps{epsilon:.1f}.pdf'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Visualization saved as '{filename}'")
        
        # Show key results
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"Number of samples: {n_samples}")
        print(f"Wasserstein radius: ε = {epsilon}")
        print(f"Empirical uniformity: {empirical_uniformity:.4f}")
        print(f"Worst-case uniformity: {worst_case_uniformity:.4f}")
        
        # Determine regime
        if worst_case_uniformity >= 0.95:
            regime = "Uniform Regime"
        elif worst_case_uniformity >= 0.85:
            regime = "Transition Regime"
        else:
            regime = "Data-Driven Regime"
        print(f"Regime: {regime}")
        
        max_empirical = np.max(empirical_dist)
        max_worst_case = np.max(worst_case_dist)
        print(f"Max empirical mass: {max_empirical:.4f}")
        print(f"Max worst-case mass: {max_worst_case:.4f}")
        print(f"Concentration ratio: {max_worst_case/max_empirical:.2f}×")
        
        plt.show()
        
    else:
        print("❌ Failed to solve worst-case distribution")
        
        # Still create empirical visualization
        print("Creating empirical-only visualization...")
        uniform_dist = np.ones(grid_size**2) / (grid_size**2)  # Fallback uniform
        fig = create_visualization(
            geodata, samples, empirical_dist, uniform_dist, epsilon, n_samples
        )
        filename = f'empirical_only_n{n_samples}.pdf'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✅ Empirical visualization saved as '{filename}'")
        plt.show()

if __name__ == "__main__":
    main()