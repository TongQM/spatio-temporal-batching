#!/usr/bin/env python3
"""
Analytical Verification of Proposition 4
Tests the theoretical monotonicity relationships directly using the Newton method optimization
"""

import numpy as np
import matplotlib.pyplot as plt

def newton_optimize_ci(Ki, Fi, alpha_i, wr=1.0, wv=10.0, max_iter=20):
    """
    Find optimal c_i using Newton's method for given Ki, Fi, alpha_i
    
    Objective: g_bar = (Ki + Fi) * ci^-2 + wr/(2*wv) * Ki + wr * ci^2 + (ci^-1 + wr/wv * ci) * alpha_i
    """
    
    def g_bar_derivative(ci):
        """First derivative dg_bar/dci"""
        return (-2*(Ki + Fi) * ci**-3 + 2*wr*ci - alpha_i * ci**-2 + (wr/wv) * alpha_i)
    
    def g_bar_second_derivative(ci):
        """Second derivative d²g_bar/dci²"""
        return (6*(Ki + Fi) * ci**-4 + 2*wr + 2*alpha_i * ci**-3)
    
    # Initial guess
    ci_star = np.sqrt(max(1.0, min(10.0, wv/wr)))
    
    # Newton iterations
    for _ in range(max_iter):
        grad = g_bar_derivative(ci_star)
        hess = g_bar_second_derivative(ci_star)
        
        if abs(grad) < 1e-10:  # Convergence
            break
            
        ci_new = ci_star - grad / hess
        ci_new = max(0.01, min(ci_new, 20.0))  # Bounds
        
        if abs(ci_new - ci_star) < 1e-12:
            break
            
        ci_star = ci_new
    
    return ci_star

def verify_proposition4_analytical():
    """Create analytical verification of Proposition 4"""
    
    wv, wr = 10.0, 1.0
    threshold = wv**2 / wr  # = 100
    beta = 0.7120
    
    print("=" * 70)
    print("ANALYTICAL VERIFICATION OF PROPOSITION 4")
    print("=" * 70)
    print(f"Parameters: wv = {wv}, wr = {wr}, β = {beta}")
    print(f"Critical threshold: wv²/wr = {threshold}")
    print()
    
    # Define test cases with varying alpha_i (via different Lambda values)
    lambda_values = np.logspace(0, 8, 200)  # From 1 to 100000 for better convergence
    
    # Simulate alpha_i = beta * sqrt(Lambda) * raw_alpha 
    # Assume raw_alpha ≈ 1.0 for simplicity (can be adjusted)
    raw_alpha = 1.0
    alpha_values = beta * np.sqrt(lambda_values) * raw_alpha
    
    # Test cases
    test_cases = [
        {"name": "Case 1: Ki + Fi < wv²/wr", "Ki": 20, "Fi": 30, "color": "blue"},   # 50 < 100
        {"name": "Case 2: Ki + Fi > wv²/wr", "Ki": 80, "Fi": 50, "color": "red"},   # 130 > 100
        {"name": "Case 3: Ki + Fi = wv²/wr", "Ki": 60, "Fi": 40, "color": "green"}  # 100 = 100
    ]
    
    # Create single compressed plot
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    # fig.suptitle('Proposition 4 Verification: Monotonicity of c_i* vs α(P_i)', 
    #             fontsize=16, fontweight='bold')
    
    results = []
    
    for i, case in enumerate(test_cases):
        Ki, Fi = case['Ki'], case['Fi']
        
        print(f"Testing {case['name']}: Ki + Fi = {Ki + Fi}")
        
        # Calculate c_i* for each alpha_i
        ci_star_values = []
        for alpha_i in alpha_values:
            ci_star = newton_optimize_ci(Ki, Fi, alpha_i, wr, wv)
            ci_star_values.append(ci_star)
        
        # Plot on main plot with markers every 10th point for clarity
        marker_indices = range(0, len(alpha_values), 10)
        ax.plot(alpha_values, ci_star_values, '-', 
                color=case['color'], label=f"{case['name']} (K+F={Ki+Fi})", 
                linewidth=3, alpha=0.8)
        ax.scatter([alpha_values[i] for i in marker_indices], 
                  [ci_star_values[i] for i in marker_indices],
                  color=case['color'], s=50, alpha=0.9, zorder=5)
        
        # Analyze monotonicity
        if len(ci_star_values) > 10:
            # Calculate overall trend using linear regression
            slope = np.polyfit(alpha_values, ci_star_values, 1)[0]
            
            if Ki + Fi < threshold:
                expected = "increasing"
                verified = slope > 0.0005
            elif Ki + Fi > threshold:
                expected = "decreasing" 
                verified = slope < -0.0005
            else:
                expected = "constant"
                verified = abs(slope) < 0.0005
            
            status = "✓" if verified else "✗"
            print(f"  Slope: {slope:.6f}, Expected: {expected}, Verified: {verified}")
            
            # Add verification status to legend
            case['status'] = status
            case['slope'] = slope
        
        results.append((case, alpha_values, ci_star_values))
    
    # Format main plot
    ax.set_xlabel('α(Pi)', fontsize=16)
    ax.set_ylabel('ci*', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    # ax.set_title('Monotonicity of Optimal Dispatch Interval vs Demand Intensity', fontsize=14, pad=20)
    ax.grid(True, alpha=0.3)
    
    # Add theoretical limit to main plot
    limit_ci = np.sqrt(wv/wr)
    ax.axhline(y=limit_ci, color='black', linestyle='--', alpha=0.8, linewidth=2,
               label=f'Asymptotic limit: √(wv/wr) = {limit_ci:.2f}')
    
    # Set log scale for x-axis to better show convergence
    ax.set_xscale('log')
    
    # Create enhanced legend with verification status
    legend_labels = []
    for case in test_cases:
        if hasattr(case, 'status'):
            legend_labels.append(f"{case['name']} {case['status']}")
        else:
            legend_labels.append(case['name'])
    
    ax.legend(fontsize=14, loc='center left')
    
    # Add threshold annotation with improved positioning
    ax.text(0.98, 0.98, f'Critical threshold: Ki + Fi = {threshold}=wv²/wr\nΛ range: 1 to 100,000', 
            transform=ax.transAxes, 
            horizontalalignment='right', verticalalignment='top',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9),
            fontsize=13)
        
    plt.tight_layout()
    plt.savefig('proposition4_verification.pdf', dpi=300, bbox_inches='tight')
    print(f"\n✅ Compressed verification saved as 'proposition4_compressed_verification.pdf'")
    
    # Summary
    print("\n" + "=" * 70)
    print("PROPOSITION 4 VERIFICATION SUMMARY")
    print("=" * 70)
    
    for case, alpha_vals, ci_vals in results:
        if len(ci_vals) > 10:
            slope = np.polyfit(alpha_vals, ci_vals, 1)[0]
            Ki, Fi = case['Ki'], case['Fi']
            
            if Ki + Fi < threshold:
                expected = "increasing"
                verified = slope > 0.0005
            elif Ki + Fi > threshold:
                expected = "decreasing"
                verified = slope < -0.0005
            else:
                expected = "constant"
                verified = abs(slope) < 0.0005
            
            status = "✅ VERIFIED" if verified else "❌ FAILED"
            
            print(f"{case['name']}: {status}")
            print(f"  Ki + Fi = {Ki + Fi:3.0f} vs threshold = {threshold:3.0f}")
            print(f"  Observed slope: {slope:.6f}")
            print(f"  Expected: {expected}")
            print(f"  Range: c_i* ∈ [{min(ci_vals):.3f}, {max(ci_vals):.3f}]")
            print()
    
    return results

if __name__ == "__main__":
    results = verify_proposition4_analytical()
    plt.show()