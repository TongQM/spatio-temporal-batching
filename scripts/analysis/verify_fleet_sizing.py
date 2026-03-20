#!/usr/bin/env python3
"""
Realistic Fleet Sizing Verification for Propositions 5 and 6

Uses:
- 10×10 mile service region with depot at center (5,5)
- python_tsp for actual TSP route calculations
- Truncated mixed-Gaussian continuous demand distribution
- Realistic parameters requiring multiple vehicles

Verifies:
1. Proposition 5: Fleet size bound for achieving target service level
2. Proposition 6: Monotonicity and upper envelope properties
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import multivariate_normal
import numpy.random as rng
try:
    from python_tsp.exact import solve_tsp_dynamic_programming
    from python_tsp.heuristics import solve_tsp_simulated_annealing
    TSP_AVAILABLE = True
except ImportError:
    print("Warning: python_tsp not available, using MST approximation")
    TSP_AVAILABLE = False

def create_truncated_mixed_gaussian_sampler(service_region_size=10.0, seed=42):
    """
    Create sampler for truncated mixed-Gaussian demand distribution
    in 10×10 mile service region
    """
    np.random.seed(seed)
    
    # Mixed-Gaussian parameters (same as previous work)
    cluster_centers = [
        (service_region_size * 0.25, service_region_size * 0.25),  # Top-left
        (service_region_size * 0.75, service_region_size * 0.25),  # Top-right  
        (service_region_size * 0.50, service_region_size * 0.75)   # Bottom-center
    ]
    cluster_weights = [0.4, 0.35, 0.25]
    cluster_sigmas = [service_region_size * 0.15 * 0.5, service_region_size * 0.16 * 0.5, service_region_size * 0.12 * 0.5]
    
    # Calculate truncation weights
    truncated_weights = []
    for i, (center, sigma) in enumerate(zip(cluster_centers, cluster_sigmas)):
        cov_matrix = [[sigma**2, 0], [0, sigma**2]]
        mvn = multivariate_normal(mean=center, cov=cov_matrix)
        
        # Probability mass within [0, service_region_size] × [0, service_region_size]
        prob_inside = (mvn.cdf([service_region_size, service_region_size]) - 
                      mvn.cdf([service_region_size, 0]) - 
                      mvn.cdf([0, service_region_size]) + 
                      mvn.cdf([0, 0]))
        
        truncated_weights.append(cluster_weights[i] * prob_inside)
    
    # Normalize
    total_weight = sum(truncated_weights)
    truncated_weights = [w / total_weight for w in truncated_weights]
    
    def sample_demands(n_samples, random_seed=None):
        """Sample demand locations from truncated mixed-Gaussian"""
        if random_seed is not None:
            np.random.seed(random_seed)
        
        samples = []
        max_attempts = n_samples * 5
        attempts = 0
        
        while len(samples) < n_samples and attempts < max_attempts:
            # Choose cluster
            cluster_idx = np.random.choice(len(cluster_centers), p=truncated_weights)
            center = cluster_centers[cluster_idx]
            sigma = cluster_sigmas[cluster_idx]
            
            # Sample from cluster
            x = np.random.normal(center[0], sigma)
            y = np.random.normal(center[1], sigma)
            
            # Accept if within service region
            if 0 <= x <= service_region_size and 0 <= y <= service_region_size:
                samples.append((x, y))
            
            attempts += 1
        
        return samples[:n_samples]
    
    return sample_demands

def calculate_tsp_distance(depot_location, demand_locations, use_bhh=False, beta=0.7120, K_i=3.0):
    """
    Calculate tour distance using TSP or BHH approximation
    
    Args:
        depot_location: (x, y) coordinates of depot
        demand_locations: List of (x, y) coordinates of demands
        use_bhh: If True, use BHH approximation; if False, use TSP
        beta: BHH coefficient for approximation
        K_i: Fixed linehaul cost for BHH approximation
    
    Returns:
        Total tour distance (depot -> demands -> depot)
    """
    if len(demand_locations) == 0:
        return 0.0
    
    if len(demand_locations) == 1:
        # Round trip to single customer
        dx = demand_locations[0][0] - depot_location[0]
        dy = demand_locations[0][1] - depot_location[1]
        return 2 * np.sqrt(dx**2 + dy**2)
    
    # Option 1: BHH Approximation
    if use_bhh:
        n_customers = len(demand_locations)
        return K_i + beta * np.sqrt(n_customers)  # Simplified BHH formula
    
    # Create distance matrix (depot + customers)
    all_locations = [depot_location] + demand_locations
    n = len(all_locations)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = all_locations[i][0] - all_locations[j][0]
                dy = all_locations[i][1] - all_locations[j][1]
                distance_matrix[i][j] = np.sqrt(dx**2 + dy**2)
    
    if TSP_AVAILABLE and n <= 8:  # Only use TSP for very small instances
        try:
            permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
            
            # Ensure tour starts and ends at depot (index 0)
            if permutation[0] != 0:
                depot_idx = permutation.index(0)
                permutation = permutation[depot_idx:] + permutation[:depot_idx]
            
            return distance
        
        except Exception as e:
            pass  # Fall through to MST approximation
    
    # Fallback: MST approximation (2 * MST)
    from scipy.sparse.csgraph import minimum_spanning_tree
    mst = minimum_spanning_tree(distance_matrix).toarray()
    mst_cost = mst.sum()
    return 2 * mst_cost  # Classical 2-approximation

def poisson_quantile_left(lam, zeta):
    """Compute left ζ-quantile of Poisson(λ) distribution"""
    if lam <= 0:
        return 0
    return int(stats.poisson.ppf(zeta, lam))

def compute_fleet_bound(K_i, beta, S_i, m_star_i, w_v, Ci):
    """
    Compute fleet size lower bound from Proposition 5
    Vi ≥ (K_i + β S_i √m_i*) / (w_v Ci)
    """
    numerator = K_i + beta * S_i * np.sqrt(m_star_i)
    theoretical_bound = numerator / (w_v * Ci)
    return max(1.0, theoretical_bound)

def compute_upper_envelope(K_i, beta, S_i, w_v, Ci, Lambda, zeta=0.95):
    """
    Compute upper envelope V̄_i(Ci) from Proposition 6
    Uses the complex formula with y_ζ(λ_i) where λ_i = Λ Ci ∫_P_i f
    """
    # Compute λ_i = Λ Ci ∫_P_i f (assuming unit area ∫_P_i f = 1)
    lambda_i = Lambda * Ci
    
    # Compute t_ζ and y_ζ(λ_i) from Proposition 6
    if lambda_i > 0:
        # t_ζ formula from the paper
        log_term = np.log(1/(1-zeta))
        t_zeta = (1/np.sqrt(lambda_i)) * log_term + np.sqrt((1/lambda_i) * log_term**2 + 2*log_term)
        y_zeta = lambda_i + t_zeta * np.sqrt(lambda_i)
    else:
        y_zeta = 0
    
    # Compute upper envelope: V̄_i(Ci) = (K_i + β S_i √y_ζ) / (w_v Ci)
    numerator = K_i + beta * S_i * np.sqrt(y_zeta)
    theoretical_envelope = numerator / (w_v * Ci)
    return max(1.0, theoretical_envelope)

def simulate_service_level_realistic(Vi, Ci, depot_location, demand_sampler, 
                                   Lambda, w_v, use_bhh=False, beta=0.7120, K_i=3.0, 
                                   n_simulations=30, seed=42):
    """
    Realistic service level simulation using actual TSP routing
    
    Args:
        Vi: Number of vehicles (integer)
        Ci: Dispatch Subinterval  
        depot_location: (x, y) depot coordinates
        demand_sampler: Function to sample demand locations
        Lambda: Demand intensity (arrivals per unit time)
        w_v: Vehicle speed
        n_simulations: Number of Monte Carlo simulations
        
    Returns:
        service_level: Fraction of dispatches completing within T_i = Ci * Vi
        travel_times: Array of simulated travel times
    """
    np.random.seed(seed)
    
    T_i = Ci * Vi  # Available time budget
    lambda_arrival = Lambda * Ci  # Expected arrivals in dispatch Subinterval
    
    travel_times = []
    successful_dispatches = 0
    
    for sim in range(n_simulations):
        # Generate Poisson arrivals
        n_arrivals = np.random.poisson(lambda_arrival)
        
        if n_arrivals == 0:
            # No customers, just depot round-trip time
            travel_time = 0.0
        else:
            # Sample customer locations
            demand_locations = demand_sampler(n_arrivals, random_seed=None)
            
            # Calculate tour distance (TSP or BHH)
            tour_distance = calculate_tsp_distance(depot_location, demand_locations, 
                                                  use_bhh=use_bhh, beta=beta, K_i=K_i)
            
            # Travel time = distance / speed
            travel_time = tour_distance / w_v
        
        travel_times.append(travel_time)
        
        # Check if vehicle completes tour within time budget
        if travel_time <= T_i:
            successful_dispatches += 1
    
    service_level = successful_dispatches / n_simulations
    return service_level, np.array(travel_times)

def verify_propositions_realistic(use_bhh=False, n_simulations=30, n_error_runs=5):
    """Main verification with realistic simulation
    
    Args:
        use_bhh: If True, use BHH approximation; if False, use TSP routing
        n_simulations: Number of Monte Carlo simulations per test
        n_error_runs: Number of independent runs for error bar calculation
    """
    
    # Realistic parameters
    service_region_size = 10.0  # 10×10 mile region
    depot_location = (5.0, 5.0)  # Center of region
    K_i = 3.0           # Fixed linehaul distance (depot round-trip)
    beta = 0.7120       # BHH coefficient  
    S_i = 5.0           # Spatial scaling factor
    w_v = 30.0          # Vehicle speed (mph)
    Lambda = 17.0        # Demand intensity (arrivals per hour)
    zeta = 0.95         # Target service level (95%)
    
    # Range of dispatch Subintervals (hours)
    C_values = np.logspace(-1, 1, 100)  # From 0.1 to 10 hours (increased resolution)
    
    print("=" * 80)
    print("REALISTIC FLEET SIZING VERIFICATION (Propositions 5 & 6)")
    print("=" * 80)
    print(f"Service region: {service_region_size}×{service_region_size} miles")
    print(f"Depot location: {depot_location}")
    print(f"Routing method: {'BHH Approximation' if use_bhh else 'TSP (with MST fallback)'}")
    print(f"Parameters: K_i = {K_i}, β = {beta}, S_i = {S_i}")
    print(f"Vehicle speed: {w_v} mph, Demand intensity: {Lambda} arrivals/hour")
    print(f"Target service level: {zeta}")
    print()
    
    # Create demand sampler
    demand_sampler = create_truncated_mixed_gaussian_sampler(service_region_size, seed=42)
    
    # Storage for results
    fleet_bounds = []
    upper_envelopes = []
    actual_min_fleets = []
    actual_min_fleets_std = []  # Standard deviations for error bars
    service_levels_at_bound = []
    service_levels_std = []  # Standard deviations for service levels
    theoretical_bounds_raw = []  # Before max(1, .) constraint
    
    for i, Ci in enumerate(C_values):
        print(f"Testing Ci = {Ci:.2f} hours ({i+1}/{len(C_values)})")
        
        # Compute theoretical fleet bound
        lambda_i = Lambda * Ci
        m_star_i = poisson_quantile_left(lambda_i, zeta)
        
        # Raw theoretical bound (before integer constraint)
        numerator = K_i + beta * S_i * np.sqrt(m_star_i)
        V_bound_raw = numerator / (w_v * Ci)
        theoretical_bounds_raw.append(V_bound_raw)
        
        # Actual bound (with integer constraint)
        V_bound = compute_fleet_bound(K_i, beta, S_i, m_star_i, w_v, Ci)
        fleet_bounds.append(V_bound)
        
        # Upper envelope from Proposition 6
        V_envelope = compute_upper_envelope(K_i, beta, S_i, w_v, Ci, Lambda, zeta)
        upper_envelopes.append(V_envelope)
        
        # Find actual minimum fleet size via integer search (multiple runs for error bars)
        V_actual_min_runs = []
        sl_at_bound_runs = []
        
        for run in range(n_error_runs):
            run_seed = 42 + run * 1000  # Different seed for each run
            
            # Find minimum fleet for this run
            V_actual_min = None
            for V_test in range(1, 21):  # Test 1 to 20 vehicles
                sl, _ = simulate_service_level_realistic(
                    V_test, Ci, depot_location, demand_sampler, Lambda, w_v, 
                    use_bhh=use_bhh, beta=beta, K_i=K_i, n_simulations=n_simulations, seed=run_seed
                )
                
                if sl >= zeta:
                    V_actual_min = float(V_test)
                    break
            
            if V_actual_min is None:
                V_actual_min = 20.0  # Cap at 20 if still not sufficient
            
            V_actual_min_runs.append(V_actual_min)
            
            # Service level at theoretical bound for this run
            sl_at_bound, _ = simulate_service_level_realistic(
                int(V_bound), Ci, depot_location, demand_sampler, Lambda, w_v,
                use_bhh=use_bhh, beta=beta, K_i=K_i, n_simulations=n_simulations, seed=run_seed
            )
            sl_at_bound_runs.append(sl_at_bound)
        
        # Calculate means and standard deviations
        actual_min_fleets.append(np.mean(V_actual_min_runs))
        actual_min_fleets_std.append(np.std(V_actual_min_runs))
        service_levels_at_bound.append(np.mean(sl_at_bound_runs))
        service_levels_std.append(np.std(sl_at_bound_runs))
        
        print(f"  Expected arrivals: {lambda_i:.1f}")
        print(f"  95% quantile: m* = {m_star_i}")
        print(f"  Theoretical bound: Vi ≥ {V_bound:.1f}")
        print(f"  Upper envelope: V̄_i = {V_envelope:.1f}")
        print(f"  Actual minimum: Vi = {np.mean(V_actual_min_runs):.1f} ± {np.std(V_actual_min_runs):.1f}")
        print(f"  Service level at bound: {np.mean(sl_at_bound_runs):.3f} ± {np.std(sl_at_bound_runs):.3f}")
        print()
    
    # Convert to numpy arrays
    fleet_bounds = np.array(fleet_bounds)
    upper_envelopes = np.array(upper_envelopes)
    actual_min_fleets = np.array(actual_min_fleets)
    actual_min_fleets_std = np.array(actual_min_fleets_std)
    service_levels_at_bound = np.array(service_levels_at_bound)
    service_levels_std = np.array(service_levels_std)
    theoretical_bounds_raw = np.array(theoretical_bounds_raw)
    
    # Create visualization with error bars
    create_realistic_fleet_plots(C_values, fleet_bounds, upper_envelopes, actual_min_fleets, 
                                service_levels_at_bound, theoretical_bounds_raw, zeta,
                                actual_min_fleets_std, service_levels_std)
    
    # Summary
    print("=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    # Proposition 5 verification
    bound_violations = np.sum(service_levels_at_bound < zeta)
    print(f"Proposition 5 - Fleet Bound Verification:")
    print(f"  Target service level: {zeta:.3f}")
    print(f"  Violations of bound: {bound_violations}/{len(C_values)}")
    print(f"  Success rate: {(len(C_values) - bound_violations)/len(C_values)*100:.1f}%")
    
    # Proposition 6 verification (check if decreasing)
    fleet_decreasing = np.all(np.diff(fleet_bounds) <= 0.1)  # Small tolerance for numerical issues
    envelope_decreasing = np.all(np.diff(upper_envelopes) <= 0.1)
    envelope_bounds = np.all(actual_min_fleets <= upper_envelopes + 0.1)  # Vi ≤ V̄_i
    
    print(f"\nProposition 6 - Upper Envelope Properties:")
    print(f"  Is Vi(Ci) decreasing? {'✅ Yes' if fleet_decreasing else '❌ No'}")
    print(f"  Is V̄_i(Ci) decreasing? {'✅ Yes' if envelope_decreasing else '❌ No'}")
    print(f"  Vi ≤ V̄_i for all Ci? {'✅ Yes' if envelope_bounds else '❌ No'}")
    
    return {
        'C_values': C_values,
        'fleet_bounds': fleet_bounds,
        'actual_min_fleets': actual_min_fleets,
        'service_levels': service_levels_at_bound,
        'theoretical_bounds_raw': theoretical_bounds_raw
    }

def create_realistic_fleet_plots(C_values, fleet_bounds, upper_envelopes, actual_min_fleets, 
                               service_levels, theoretical_bounds_raw, zeta,
                               actual_min_fleets_std=None, service_levels_std=None):
    """Create comprehensive plots for realistic fleet sizing verification with error bars"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Fleet size vs dispatch Subinterval
    ax1.loglog(C_values, fleet_bounds, 'b-', linewidth=3, 
               label='Fleet Bound Vi (Prop 5)')
    ax1.loglog(C_values, upper_envelopes, 'g--', linewidth=3, 
               label='Upper Envelope V̄_i (Prop 6)')
    
    if actual_min_fleets_std is not None:
        ax1.errorbar(C_values, actual_min_fleets, yerr=actual_min_fleets_std, 
                     fmt='ro-', linewidth=2, markersize=6, capsize=3, capthick=1,
                     label='Simulated Minimum Vi ± σ', alpha=0.8)
    else:
        ax1.loglog(C_values, actual_min_fleets, 'ro-', linewidth=2, markersize=6, 
                   label='Simulated Minimum Vi', alpha=0.8)
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    ax1.set_xlabel('Dispatch Subinterval Ci (hours)', fontsize=14)
    ax1.set_ylabel('Fleet Size Vi (vehicles)', fontsize=14)
    # ax1.set_title('Fleet Size Requirements vs Dispatch Subinterval', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # Plot 2: Service level verification with theoretical bound reference
    if service_levels_std is not None:
        ax2.errorbar(C_values, service_levels, yerr=service_levels_std,
                     fmt='go-', linewidth=2, markersize=6, capsize=3, capthick=1,
                     label='Simulated SL at Theoretical Bound ± σ')
    else:
        ax2.semilogx(C_values, service_levels, 'go-', linewidth=2, markersize=6, 
                     label='Simulated SL at Theoretical Bound')
    
    ax2.axhline(y=zeta, color='red', linestyle='--', linewidth=3, 
                label=f'Target Service Level ζ = {zeta}')
    ax2.axhline(y=1.0, color='black', linestyle='-', alpha=0.5, 
                label='Perfect Service Level = 1.0')
    ax2.set_xscale('log')
    
    ax2.set_xlabel('Dispatch Subinterval Ci (hours)', fontsize=14)
    ax2.set_ylabel('Service Level SL_i(Vi)', fontsize=14)
    # ax2.set_title('Service Level Verification (Proposition 5)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.set_ylim([0.85, 1.02])
    
    # Plot 3: Fleet size ratio and tightness
    ratio = actual_min_fleets / fleet_bounds
    if actual_min_fleets_std is not None:
        # Error propagation for ratio: σ(A/B) ≈ (A/B) * √((σA/A)² + (σB/B)²)
        # Since fleet_bounds has no error, σ(ratio) = σ(actual)/fleet_bounds
        ratio_std = actual_min_fleets_std / fleet_bounds
        ax3.errorbar(C_values, ratio, yerr=ratio_std, fmt='mo-', 
                     linewidth=2, markersize=4, capsize=3, capthick=1)
    else:
        ax3.semilogx(C_values, ratio, 'mo-', linewidth=2, markersize=4)
    
    ax3.axhline(y=1.0, color='black', linestyle='-', alpha=0.5, 
                label='Perfect Bound (ratio = 1)')
    ax3.set_xscale('log')
    ax3.set_xlabel('Dispatch Subinterval Ci (hours)', fontsize=14)
    ax3.set_ylabel('V_actual / V_bound', fontsize=14)
    # ax3.set_title('Tightness of Fleet Size Bound', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=12)
    ax3.tick_params(axis='both', which='major', labelsize=12)
    
    # Plot 4: Upper envelope gap analysis
    gap = upper_envelopes - actual_min_fleets
    if actual_min_fleets_std is not None:
        # Gap error is just the error in actual_min_fleets (upper_envelopes is deterministic)
        ax4.errorbar(C_values, gap, yerr=actual_min_fleets_std, fmt='co-', 
                     linewidth=2, markersize=4, capsize=3, capthick=1,
                     label='V̄_i - V_actual ± σ')
    else:
        ax4.semilogx(C_values, gap, 'co-', linewidth=2, markersize=4, 
                     label='V̄_i - V_actual')
    
    ax4.axhline(y=0.0, color='black', linestyle='-', alpha=0.5, 
                label='Perfect envelope (gap = 0)')
    ax4.set_xscale('log')
    
    ax4.set_xlabel('Dispatch Subinterval Ci (hours)', fontsize=14)
    ax4.set_ylabel('Upper Envelope Gap (vehicles)', fontsize=14)
    # ax4.set_title('Upper Envelope Overestimate (Proposition 6)', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=12)
    ax4.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig('fleet_sizing_verification.pdf', dpi=300, bbox_inches='tight')
    print("✅ Fleet sizing verification plots saved as 'fleet_sizing_verification.pdf'")
    
    return fig

if __name__ == "__main__":
    import sys
    
    # Command line arguments
    use_bhh = False
    n_simulations = 30  # Default value
    n_error_runs = 5  # Default value
    
    # Parse command line arguments
    numeric_args = []
    for arg in sys.argv[1:]:
        if arg.lower() in ['bhh', 'true', '1']:
            use_bhh = True
        elif arg.isdigit():
            numeric_args.append(int(arg))
    
    # Assign numeric arguments in order: n_simulations, n_error_runs
    if len(numeric_args) >= 1:
        n_simulations = numeric_args[0]
    if len(numeric_args) >= 2:
        n_error_runs = numeric_args[1]
    
    print(f"Running verification with {'BHH approximation' if use_bhh else 'TSP routing'}")
    print(f"Number of simulations per test: {n_simulations}")
    print("Usage: python verify_fleet_sizing.py [bhh] [n_simulations] [n_error_runs]")
    print("Examples:")
    print("  python verify_fleet_sizing.py          # TSP routing, 30 simulations, 5 error runs")
    print("  python verify_fleet_sizing.py bhh      # BHH approximation, 30 simulations, 5 error runs")
    print("  python verify_fleet_sizing.py 100      # TSP routing, 100 simulations, 5 error runs")
    print("  python verify_fleet_sizing.py bhh 50 10 # BHH approximation, 50 simulations, 10 error runs")
    print()
    
    print(f"Number of error bar runs: {n_error_runs}")
    
    results = verify_propositions_realistic(use_bhh=use_bhh, n_simulations=n_simulations, n_error_runs=n_error_runs)
    plt.show()