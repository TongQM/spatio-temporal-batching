#!/usr/bin/env python3
"""
Focused comparison of Wasserstein distances:
Normal sampling (appearance_prob=1.0) vs. sampling with appearance filter (appearance_prob<1.0)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from test_wasserstein_distance import WassersteinTester
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compare_appearance_effects():
    """Compare Wasserstein distances for different appearance probabilities."""
    
    # Test parameters
    n_points = 50
    dimension = 2
    n_samples_list = [100, 200, 500, 1000]
    appearance_probs = [0.3, 0.5, 0.7, 0.9, 1.0]  # 1.0 = normal sampling
    n_repetitions = 10
    
    # Create tester
    tester = WassersteinTester(n_points=n_points, dimension=dimension, seed=42)
    
    # Run experiment
    logger.info("Running focused comparison experiment...")
    results = tester.run_experiment(n_samples_list, appearance_probs, n_repetitions)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Mean Wasserstein distance by appearance probability
    mean_by_app_prob = df.groupby('appearance_prob')['wasserstein_distance'].agg(['mean', 'std']).reset_index()
    axes[0, 0].errorbar(mean_by_app_prob['appearance_prob'], mean_by_app_prob['mean'], 
                       yerr=mean_by_app_prob['std'], marker='o', capsize=5, linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Appearance Probability', fontsize=12)
    axes[0, 0].set_ylabel('Mean Wasserstein Distance', fontsize=12)
    axes[0, 0].set_title('Effect of Appearance Probability on Wasserstein Distance', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Normal Sampling')
    axes[0, 0].legend()
    
    # Plot 2: Box plot by appearance probability
    sns.boxplot(data=df, x='appearance_prob', y='wasserstein_distance', ax=axes[0, 1])
    axes[0, 1].set_xlabel('Appearance Probability', fontsize=12)
    axes[0, 1].set_ylabel('Wasserstein Distance', fontsize=12)
    axes[0, 1].set_title('Distribution of Wasserstein Distances', fontsize=14)
    axes[0, 1].axvline(x=4, color='red', linestyle='--', alpha=0.7, label='Normal Sampling')  # x=4 corresponds to 1.0
    axes[0, 1].legend()
    
    # Plot 3: Mean distance by sample size, colored by appearance prob
    for app_prob in appearance_probs:
        subset = df[df['appearance_prob'] == app_prob]
        mean_by_samples = subset.groupby('n_samples')['wasserstein_distance'].mean()
        label = f'App. Prob = {app_prob}' + (' (Normal)' if app_prob == 1.0 else '')
        linestyle = '-' if app_prob == 1.0 else '--'
        linewidth = 3 if app_prob == 1.0 else 2
        axes[1, 0].plot(mean_by_samples.index, mean_by_samples.values, 
                       marker='o', label=label, linestyle=linestyle, linewidth=linewidth)
    
    axes[1, 0].set_xlabel('Number of Samples', fontsize=12)
    axes[1, 0].set_ylabel('Mean Wasserstein Distance', fontsize=12)
    axes[1, 0].set_title('Convergence Rate by Appearance Probability', fontsize=14)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Ratio to normal sampling (appearance_prob=1.0)
    normal_sampling = df[df['appearance_prob'] == 1.0].groupby(['n_samples'])['wasserstein_distance'].mean()
    
    ratios_data = []
    for app_prob in [0.3, 0.5, 0.7, 0.9]:
        subset = df[df['appearance_prob'] == app_prob]
        mean_by_samples = subset.groupby('n_samples')['wasserstein_distance'].mean()
        ratio = mean_by_samples / normal_sampling
        for n_samples, ratio_val in ratio.items():
            ratios_data.append({'n_samples': n_samples, 'appearance_prob': app_prob, 'ratio': ratio_val})
    
    ratio_df = pd.DataFrame(ratios_data)
    
    for app_prob in [0.3, 0.5, 0.7, 0.9]:
        subset = ratio_df[ratio_df['appearance_prob'] == app_prob]
        axes[1, 1].plot(subset['n_samples'], subset['ratio'], marker='o', 
                       label=f'App. Prob = {app_prob}', linewidth=2)
    
    axes[1, 1].axhline(y=1.0, color='red', linestyle='-', alpha=0.7, linewidth=3, label='Normal Sampling')
    axes[1, 1].set_xlabel('Number of Samples', fontsize=12)
    axes[1, 1].set_ylabel('Ratio to Normal Sampling', fontsize=12)
    axes[1, 1].set_title('Relative Performance vs. Normal Sampling', fontsize=14)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('./figures/appearance_probability_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed comparison
    print("\n" + "="*70)
    print("DETAILED COMPARISON: NORMAL vs APPEARANCE-FILTERED SAMPLING")
    print("="*70)
    
    # Normal sampling results
    normal_results = df[df['appearance_prob'] == 1.0]['wasserstein_distance']
    print(f"Normal Sampling (appearance_prob = 1.0):")
    print(f"  Mean: {normal_results.mean():.4f}")
    print(f"  Std:  {normal_results.std():.4f}")
    print(f"  Min:  {normal_results.min():.4f}")
    print(f"  Max:  {normal_results.max():.4f}")
    
    # Comparison for each appearance probability
    print(f"\nComparison with appearance-filtered sampling:")
    print(f"{'App. Prob':<10} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8} {'Ratio to Normal':<15}")
    print("-" * 70)
    
    for app_prob in [0.3, 0.5, 0.7, 0.9]:
        subset_results = df[df['appearance_prob'] == app_prob]['wasserstein_distance']
        ratio = subset_results.mean() / normal_results.mean()
        print(f"{app_prob:<10.1f} {subset_results.mean():<8.4f} {subset_results.std():<8.4f} "
              f"{subset_results.min():<8.4f} {subset_results.max():<8.4f} {ratio:<15.2f}")
    
    # Statistical significance test
    from scipy.stats import ttest_ind
    print(f"\nStatistical significance (t-test vs normal sampling):")
    print(f"{'App. Prob':<10} {'t-statistic':<12} {'p-value':<10} {'Significant':<12}")
    print("-" * 50)
    
    for app_prob in [0.3, 0.5, 0.7, 0.9]:
        subset_results = df[df['appearance_prob'] == app_prob]['wasserstein_distance']
        t_stat, p_value = ttest_ind(subset_results, normal_results)
        significant = "Yes" if p_value < 0.05 else "No"
        print(f"{app_prob:<10.1f} {t_stat:<12.2f} {p_value:<10.4f} {significant:<12}")
    
    return df

if __name__ == "__main__":
    results_df = compare_appearance_effects()