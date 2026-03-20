"""
LBBD experiment configurations: run profiles, output settings, and computational settings.
"""

# Experiment configurations
LBBD_CONFIGS = [
    {
        'name': 'tiny_test',
        'description': 'Tiny test with just 6 blocks for debugging',
        'num_districts': 2,
        'epsilon': 0.1,
        'max_iterations': 5,
        'tolerance': 1e-2,
        'max_cuts': 10,
        'verbose': True
    },
    {
        'name': 'quick_test',
        'description': 'Quick test with minimal iterations',
        'num_districts': 3,
        'epsilon': 0.1,
        'max_iterations': 10,
        'tolerance': 1e-2,
        'max_cuts': 20,
        'verbose': True
    },
    {
        'name': 'small_scale',
        'description': 'Small scale experiment',
        'num_districts': 3,
        'epsilon': 0.1,
        'max_iterations': 20,
        'tolerance': 1e-3,
        'max_cuts': 50,
        'verbose': True
    },
    {
        'name': 'standard',
        'description': 'Standard LBBD run',
        'num_districts': 3,
        'epsilon': 0.05,
        'max_iterations': 50,
        'tolerance': 1e-4,
        'max_cuts': 100,
        'verbose': True
    },
    {
        'name': 'high_precision',
        'description': 'High precision with tight tolerance',
        'num_districts': 3,
        'epsilon': 0.01,
        'max_iterations': 100,
        'tolerance': 1e-5,
        'max_cuts': 200,
        'verbose': True
    },
    {
        'name': 'five_districts',
        'description': 'Test with 5 districts',
        'num_districts': 5,
        'epsilon': 0.05,
        'max_iterations': 50,
        'tolerance': 1e-4,
        'max_cuts': 100,
        'verbose': True
    }
]

# Output settings
OUTPUT_SETTINGS = {
    'results_dir': 'results/lbbd',
    'log_level': 'INFO',
    'save_plots': True,
    'save_detailed_history': True
}

# Computational settings
COMPUTATION_SETTINGS = {
    'random_seed': 42,
    'parallel_processing': False,
    'memory_limit_gb': 8,
    'time_limit_hours': 2
}

def get_config_by_name(name):
    """Get configuration by name"""
    for config in LBBD_CONFIGS:
        if config['name'] == name:
            return config
    raise ValueError(f"Configuration '{name}' not found")

def list_available_configs():
    """List all available configurations"""
    print("Available LBBD configurations:")
    print("-" * 50)
    for config in LBBD_CONFIGS:
        print(f"Name: {config['name']}")
        print(f"Description: {config['description']}")
        print(f"Districts: {config['num_districts']}, Epsilon: {config['epsilon']}")
        print(f"Max iterations: {config['max_iterations']}, Tolerance: {config['tolerance']}")
        print("-" * 50)

if __name__ == "__main__":
    list_available_configs()
