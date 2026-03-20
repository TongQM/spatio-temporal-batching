#!/usr/bin/env python3
"""
Automated LBBD (Logic-Based Benders Decomposition) execution script
Based on the workflow from data_process.ipynb
"""

import os
import sys
import time
import json
import pickle
import logging
import argparse
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from lib import GeoData, load_data
from lib.algorithm import Partition
from config import DATA_PATHS, BG_GEOID_LIST
from lbbd_config import LBBD_CONFIGS, OUTPUT_SETTINGS, COMPUTATION_SETTINGS, get_config_by_name

def setup_logging():
    """Setup logging configuration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'lbbd_run_{timestamp}.log'
    
    logging.basicConfig(
        level=getattr(logging, OUTPUT_SETTINGS['log_level']),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def load_bg_demand_data():
    """Load block group demand data from commuting files"""
    logging.info("Loading block group demand data...")
    
    bg_demand_dict = {}
    
    try:
        # Try to load actual commuting data
        commuting_file = DATA_PATHS['commuting_data']
        if os.path.exists(commuting_file):
            df_commuting = pd.read_csv(commuting_file)
            logging.info(f"Loaded commuting data with {len(df_commuting)} records")
            # TODO: Add actual data processing logic here
        else:
            logging.warning(f"Commuting file not found: {commuting_file}")
            
        # For now, generate sample demand data (replace with actual data loading logic)
        np.random.seed(COMPUTATION_SETTINGS['random_seed'])
        for bg_id in BG_GEOID_LIST:
            # Use short GEOID as key to match bg_geodata.short_geoid_list
            short_geoid = bg_id[-7:]
            bg_demand_dict[short_geoid] = {
                'total_commuting': max(10, np.random.poisson(50)),  # Random commuting demand
                'population': max(50, np.random.poisson(200))       # Random population
            }
            
        logging.info(f"Created demand data for {len(bg_demand_dict)} block groups")
        
    except Exception as e:
        logging.error(f"Error loading demand data: {e}")
        raise
    
    return bg_demand_dict

def setup_geodata():
    """Initialize GeoData with block group shapefile"""
    logging.info("Setting up GeoData...")
    
    bgfile_path = DATA_PATHS['shapefile']
    
    if not os.path.exists(bgfile_path):
        logging.error(f"Shapefile not found: {bgfile_path}")
        raise FileNotFoundError(f"Shapefile not found: {bgfile_path}")
    
    try:
        bg_geodata = GeoData(bgfile_path, BG_GEOID_LIST, level='block_group')
        logging.info(f"GeoData initialized with {len(bg_geodata.short_geoid_list)} block groups")
        return bg_geodata
    except Exception as e:
        logging.error(f"Error initializing GeoData: {e}")
        raise

def create_probability_dict(bg_geodata, bg_demand_dict):
    """Create probability dictionary from demand data"""
    logging.info("Creating probability dictionary...")
    
    # Calculate total commuting
    total_commuting = sum(bg_demand_dict[bg]['total_commuting'] for bg in bg_geodata.short_geoid_list)
    
    # Create probability dictionary
    probability_dict = {
        bg: bg_demand_dict[bg]['total_commuting'] / total_commuting 
        for bg in bg_geodata.short_geoid_list
    }
    
    logging.info(f"Created probability dict with total probability: {sum(probability_dict.values()):.4f}")
    return probability_dict

def run_lbbd_experiment(bg_geodata, probability_dict, config):
    """Run LBBD experiment with given configuration"""
    logging.info(f"Starting LBBD experiment: {config['name']}")
    logging.info(f"Config: {config['description']}")
    
    # Initialize partition problem
    partition = Partition(
        bg_geodata, 
        num_districts=config['num_districts'], 
        prob_dict=probability_dict, 
        epsilon=config['epsilon']
    )
    
    # Run Benders decomposition
    start_time = time.time()
    
    try:
        best_partition, best_cost, history = partition.benders_decomposition(
            max_iterations=config['max_iterations'],
            tolerance=config['tolerance'],
            max_cuts=config['max_cuts'],
            verbose=config['verbose']
        )
        
        end_time = time.time()
        runtime = end_time - start_time
        
        logging.info(f"LBBD completed in {runtime:.2f} seconds")
        logging.info(f"Best cost: {best_cost:.4f}")
        logging.info(f"Converged in {len(history)} iterations")
        
        return {
            'best_partition': best_partition,
            'best_cost': best_cost,
            'history': history,
            'runtime': runtime,
            'config': config,
            'converged': len(history) < config['max_iterations']
        }
        
    except Exception as e:
        logging.error(f"Error during LBBD execution: {e}")
        raise

def save_results(results, output_dir=None):
    """Save LBBD results to files"""
    if output_dir is None:
        output_dir = OUTPUT_SETTINGS['results_dir']
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = results['config']['name']
    
    # Save results as pickle
    pickle_file = os.path.join(output_dir, f"lbbd_{config_name}_{timestamp}.pkl")
    with open(pickle_file, 'wb') as f:
        pickle.dump(results, f)
    
    # Save summary as JSON
    summary = {
        'best_cost': results['best_cost'],
        'runtime': results['runtime'],
        'converged': results['converged'],
        'iterations': len(results['history']),
        'config': results['config'],
        'timestamp': timestamp
    }
    
    json_file = os.path.join(output_dir, f"lbbd_{config_name}_{timestamp}.json")
    with open(json_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logging.info(f"Results saved to {pickle_file} and {json_file}")
    return pickle_file, json_file

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run LBBD experiments')
    parser.add_argument('--config', '-c', type=str, 
                       help='Configuration name to run (see lbbd_config.py)')
    parser.add_argument('--list-configs', action='store_true',
                       help='List available configurations')
    parser.add_argument('--all', action='store_true',
                       help='Run all configurations')
    parser.add_argument('--output-dir', '-o', type=str,
                       help='Output directory for results')
    return parser.parse_args()

def main():
    """Main execution function"""
    args = parse_arguments()
    
    # List configurations if requested
    if args.list_configs:
        print("Available LBBD configurations:")
        print("-" * 60)
        for config in LBBD_CONFIGS:
            print(f"Name: {config['name']}")
            print(f"Description: {config['description']}")
            print(f"Districts: {config['num_districts']}, Epsilon: {config['epsilon']}")
            print(f"Max iterations: {config['max_iterations']}, Tolerance: {config['tolerance']}")
            print("-" * 60)
        return
    
    # Setup logging
    log_file = setup_logging()
    logging.info("Starting automated LBBD execution")
    logging.info(f"Log file: {log_file}")
    
    # Determine which configurations to run
    if args.config:
        try:
            configs = [get_config_by_name(args.config)]
        except ValueError as e:
            logging.error(str(e))
            sys.exit(1)
    elif args.all:
        configs = LBBD_CONFIGS
    else:
        # Default: run quick_test and small_scale
        configs = [get_config_by_name('quick_test'), get_config_by_name('small_scale')]
    
    try:
        # Load data
        logging.info("Loading data...")
        bg_demand_dict = load_bg_demand_data()
        bg_geodata = setup_geodata()
        probability_dict = create_probability_dict(bg_geodata, bg_demand_dict)
        
        # Run experiments
        all_results = []
        for i, config in enumerate(configs):
            logging.info(f"\n{'='*60}")
            logging.info(f"Running experiment {i+1}/{len(configs)}: {config['name']}")
            logging.info(f"{'='*60}")
            
            try:
                results = run_lbbd_experiment(bg_geodata, probability_dict, config)
                all_results.append(results)
                
                # Save individual results
                pickle_file, json_file = save_results(results, args.output_dir)
                
                # Print summary
                logging.info(f"✓ Experiment {config['name']} completed successfully")
                logging.info(f"  Best cost: {results['best_cost']:.4f}")
                logging.info(f"  Runtime: {results['runtime']:.2f}s")
                logging.info(f"  Iterations: {len(results['history'])}")
                logging.info(f"  Converged: {results['converged']}")
                
            except Exception as e:
                logging.error(f"✗ Experiment {config['name']} failed: {e}")
                continue
        
        # Save combined results
        if all_results:
            output_dir = args.output_dir or OUTPUT_SETTINGS['results_dir']
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            combined_file = os.path.join(output_dir, f"lbbd_combined_{timestamp}.pkl")
            with open(combined_file, 'wb') as f:
                pickle.dump(all_results, f)
            logging.info(f"Combined results saved to {combined_file}")
            
            # Print final summary
            logging.info(f"\n{'='*60}")
            logging.info("FINAL SUMMARY")
            logging.info(f"{'='*60}")
            for result in all_results:
                config_name = result['config']['name']
                logging.info(f"{config_name:15} | Cost: {result['best_cost']:8.4f} | "
                           f"Time: {result['runtime']:6.2f}s | "
                           f"Iter: {len(result['history']):3d} | "
                           f"Conv: {'Yes' if result['converged'] else 'No'}")
        
        logging.info("Automated LBBD execution completed successfully")
        
    except Exception as e:
        logging.error(f"Fatal error in main execution: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 