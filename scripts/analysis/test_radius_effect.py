#!/usr/bin/env python3
"""
Direct Test: Wasserstein Radius Effect on Worst-Case Distribution

This script tests the same empirical sample with different Wasserstein radii
to visualize the phase transition from uniform to data-driven worst-case distributions.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from validate_sample_cutoff import ToyGeoData, create_mixed_gaussian_samples, samples_to_empirical_distribution, measure_uniformity
from lib.algorithm import Partition
import warnings
warnings.filterwarnings('ignore')

def test_radius_effect_on_single_sample(n_samples=100, grid_size=10, seed=42):
    """
    Test how different Wasserstein radii affect the worst-case distribution
    for a single fixed empirical sample.
    """
    
    print("=" * 80)
    print("WASSERSTEIN RADIUS EFFECT ON WORST-CASE DISTRIBUTION")
    print("=" * 80)
    print(f"Fixed sample size: n = {n_samples}")
    print(f"Grid size: {grid_size}×{grid_size} = {grid_size**2} blocks")
    print(f"Testing different Wasserstein radii from large to small")
    print()
    
    # Create geodata
    geodata = ToyGeoData(grid_size, service_region_miles=10.0, seed=seed)
    
    # Generate ONE fixed empirical sample
    samples = create_mixed_gaussian_samples(grid_size, n_samples, seed=seed)
    empirical_dist = samples_to_empirical_distribution(samples, grid_size)
    empirical_prob_dict = {
        block_id: empirical_dist[i] 
        for i, block_id in enumerate(geodata.short_geoid_list)
    }
    
    print(f"Generated empirical distribution from {n_samples} mixed-Gaussian samples")
    print(f"Empirical distribution entropy: {measure_uniformity(empirical_dist):.4f}")
    print()
    
    # Test different Wasserstein radii
    # Start large (uniform regime) → small (data-driven regime)
    epsilon_values = [5.0, 2.0, 1.0, 0.5, 0.3, 0.2, 0.1, 0.05]
    
    results = {
        'epsilons': [],
        'uniformities': [],
        'worst_case_distributions': [],
        'max_masses': [],
        'success': []
    }
    
    for epsilon in epsilon_values:
        print(f"Testing ε = {epsilon:.3f}...", end=' ')
        
        try:
            # Create partition instance
            partition = Partition(geodata, num_districts=1, prob_dict=empirical_prob_dict, epsilon=epsilon)
            
            # Use entire service region as one district
            assigned_blocks = geodata.short_geoid_list  # All blocks
            root = assigned_blocks[50]  # Center block as root
            
            # CQCP subproblem parameters
            K_i = 3.0  # Fixed linehaul cost
            F_i = 0.0  # No ODD cost
            beta = 0.7120
            Lambda = 10.0
            
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
                # Note: x_star_dict contains x_j values, but probability mass is x_j^2
                worst_case_dist = np.zeros(len(assigned_blocks))
                for i, block_id in enumerate(assigned_blocks):
                    x_j = x_star_dict.get(block_id, 0.0)
                    # Ensure non-negative and use x_j^2 for probability mass
                    worst_case_dist[i] = max(0.0, x_j)**2
                
                # Normalize to ensure proper probability distribution
                total_mass = np.sum(worst_case_dist)
                if total_mass > 1e-10:
                    worst_case_dist = worst_case_dist / total_mass
                else:
                    worst_case_dist = np.ones(len(assigned_blocks)) / len(assigned_blocks)
                
                # Measure uniformity
                uniformity = measure_uniformity(worst_case_dist)
                max_mass = np.max(worst_case_dist)
                
                # Store results
                results['epsilons'].append(epsilon)
                results['uniformities'].append(uniformity)
                results['worst_case_distributions'].append(worst_case_dist)
                results['max_masses'].append(max_mass)
                results['success'].append(True)
                
                print(f"✓ Uniformity = {uniformity:.4f}, Max mass = {max_mass:.4f}")
            else:
                print("✗ Incomplete result")
                results['success'].append(False)
                
        except Exception as e:
            print(f"✗ Error: {e}")
            results['success'].append(False)
    
    return results, empirical_dist

def create_radius_effect_visualization(results, empirical_dist, grid_size):
    """Create visualization showing how worst-case distribution changes with radius"""
    
    successful_results = [i for i, success in enumerate(results['success']) if success]
    if len(successful_results) < 4:
        print("Insufficient successful results for visualization")
        return
    
    n_plots = len(successful_results)
    n_cols = 4
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Plot empirical distribution first
    empirical_grid = empirical_dist.reshape(grid_size, grid_size)
    ax_emp = fig.add_subplot(n_rows, n_cols, 1)
    im_emp = ax_emp.imshow(empirical_grid, cmap='Greens', origin='lower')
    ax_emp.set_title('Empirical Distribution\n(Fixed Sample)', fontsize=12, fontweight='bold')
    ax_emp.set_xlabel('x coordinate')
    ax_emp.set_ylabel('y coordinate')
    plt.colorbar(im_emp, ax=ax_emp, shrink=0.8)
    
    # Plot worst-case distributions
    plot_idx = 2  # Start from second subplot
    for i in successful_results:
        if plot_idx > n_rows * n_cols:
            break
            
        row = (plot_idx - 1) // n_cols
        col = (plot_idx - 1) % n_cols
        ax = axes[row, col]
        
        epsilon = results['epsilons'][i]
        uniformity = results['uniformities'][i]
        worst_case = results['worst_case_distributions'][i]
        max_mass = results['max_masses'][i]
        
        worst_case_grid = worst_case.reshape(grid_size, grid_size)
        
        # Use different colormaps to distinguish regime
        if uniformity > 0.95:
            cmap = 'Blues'  # Uniform regime
            regime = "Uniform"
        elif uniformity > 0.85:
            cmap = 'Oranges'  # Transition
            regime = "Transition"
        else:
            cmap = 'Reds'  # Data-driven regime
            regime = "Data-Driven"
        
        im = ax.imshow(worst_case_grid, cmap=cmap, origin='lower')
        ax.set_title(f'{regime} Regime\nε={epsilon:.3f}, U={uniformity:.3f}', fontsize=11)
        ax.set_xlabel('x coordinate')
        ax.set_ylabel('y coordinate')
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        plot_idx += 1
    
    # Hide unused subplots
    for plot_idx in range(plot_idx, n_rows * n_cols + 1):
        if plot_idx <= n_rows * n_cols:
            row = (plot_idx - 1) // n_cols
            col = (plot_idx - 1) % n_cols
            if row < n_rows and col < n_cols:
                axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('radius_effect_visualization.pdf', dpi=300, bbox_inches='tight')
    print("✅ Radius effect plots saved as 'radius_effect_visualization.pdf'")
    
    return fig

def create_phase_transition_plot(results):
    """Create plot showing uniformity vs epsilon to identify phase transition"""
    
    successful_results = [i for i, success in enumerate(results['success']) if success]
    if len(successful_results) < 3:
        print("Insufficient results for phase transition plot")
        return
    
    epsilons = [results['epsilons'][i] for i in successful_results]
    uniformities = [results['uniformities'][i] for i in successful_results]
    max_masses = [results['max_masses'][i] for i in successful_results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Uniformity vs Epsilon
    ax1.semilogx(epsilons, uniformities, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Wasserstein Radius ε', fontsize=14)
    ax1.set_ylabel('Worst-Case Distribution Uniformity', fontsize=14)
    ax1.set_title('Phase Transition: Uniformity vs Wasserstein Radius', fontsize=15)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.5, 1.0])
    
    # Add regime annotations
    uniform_threshold = 0.95
    ax1.axhline(y=uniform_threshold, color='red', linestyle='--', alpha=0.7, 
                label=f'Uniform threshold = {uniform_threshold}')
    ax1.legend()
    
    # Plot 2: Maximum Mass vs Epsilon
    ax2.semilogx(epsilons, max_masses, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Wasserstein Radius ε', fontsize=14)
    ax2.set_ylabel('Maximum Probability Mass', fontsize=14)
    ax2.set_title('Concentration: Max Mass vs Wasserstein Radius', fontsize=15)
    ax2.grid(True, alpha=0.3)
    
    # Theoretical uniform mass line
    uniform_mass = 1.0 / (10 * 10)  # 1/N for N blocks
    ax2.axhline(y=uniform_mass, color='blue', linestyle=':', alpha=0.7,
                label=f'Uniform mass = {uniform_mass:.3f}')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('phase_transition_analysis.pdf', dpi=300, bbox_inches='tight')
    print("✅ Phase transition analysis saved as 'phase_transition_analysis.pdf'")
    
    return fig

def main():
    import sys
    
    # Parse arguments
    n_samples = 100
    grid_size = 10
    seed = 42
    
    for arg in sys.argv[1:]:
        try:
            if '=' in arg:
                key, value = arg.split('=')
                if key == 'n_samples':
                    n_samples = int(value)
                elif key == 'grid_size':
                    grid_size = int(value)
                elif key == 'seed':
                    seed = int(value)
        except ValueError:
            print(f"Invalid argument: {arg}")
    
    print(f"Testing with n_samples={n_samples}, grid_size={grid_size}, seed={seed}")
    print()
    
    # Run radius effect test
    results, empirical_dist = test_radius_effect_on_single_sample(n_samples, grid_size, seed)
    
    # Create visualizations
    if any(results['success']):
        create_radius_effect_visualization(results, empirical_dist, grid_size)
        create_phase_transition_plot(results)
        
        # Print summary
        successful_results = [i for i, success in enumerate(results['success']) if success]
        if successful_results:
            print("\n" + "=" * 60)
            print("RADIUS EFFECT SUMMARY")
            print("=" * 60)
            print(f"{'Epsilon':<10} {'Uniformity':<12} {'Max Mass':<12} {'Regime'}")
            print("-" * 50)
            
            for i in successful_results:
                epsilon = results['epsilons'][i]
                uniformity = results['uniformities'][i]
                max_mass = results['max_masses'][i]
                
                if uniformity > 0.95:
                    regime = "Uniform"
                elif uniformity > 0.85:
                    regime = "Transition"
                else:
                    regime = "Data-Driven"
                
                print(f"{epsilon:<10.3f} {uniformity:<12.4f} {max_mass:<12.4f} {regime}")
            
            # Find phase transition point
            uniformities = [results['uniformities'][i] for i in successful_results]
            epsilons = [results['epsilons'][i] for i in successful_results]
            
            transition_indices = [i for i, u in enumerate(uniformities) if u < 0.95]
            if transition_indices:
                critical_idx = transition_indices[0]  # First point below threshold
                critical_epsilon = epsilons[critical_idx]
                print(f"\n🎯 Phase transition occurs around ε ≈ {critical_epsilon:.3f}")
                print(f"   Above this: Uniform regime (U ≥ 0.95)")
                print(f"   Below this: Data-driven regime (U < 0.95)")
            else:
                print(f"\n⚠️  No clear phase transition found in tested range")
                print(f"   All distributions remain uniform (U ≥ 0.95)")
                print(f"   Try smaller epsilon values to find the transition")
        
    else:
        print("No successful results obtained")
    
    plt.show()

if __name__ == "__main__":
    main()