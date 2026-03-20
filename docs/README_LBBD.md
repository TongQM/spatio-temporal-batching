# LBBD Automation Scripts

This directory contains automated scripts for running Logic-Based Benders Decomposition (LBBD) experiments based on the workflow from `data_process.ipynb`.

## Files

- `run_lbbd.py` - Main automation script
- `lbbd_config.py` - Configuration file with experiment parameters
- `README_LBBD.md` - This documentation file

## Quick Start

### 1. List Available Configurations
```bash
python run_lbbd.py --list-configs
```

### 2. Run a Specific Configuration
```bash
python run_lbbd.py --config quick_test
```

### 3. Run Default Experiments (quick_test + small_scale)
```bash
python run_lbbd.py
```

### 4. Run All Configurations
```bash
python run_lbbd.py --all
```

### 5. Specify Output Directory
```bash
python run_lbbd.py --config standard --output-dir my_results/
```

## Available Configurations

The following pre-defined configurations are available in `lbbd_config.py`:

| Name | Districts | Epsilon | Max Iter | Tolerance | Description |
|------|-----------|---------|----------|-----------|-------------|
| `quick_test` | 3 | 0.1 | 10 | 1e-2 | Quick test with minimal iterations |
| `small_scale` | 3 | 0.1 | 20 | 1e-3 | Small scale experiment |
| `standard` | 3 | 0.05 | 50 | 1e-4 | Standard LBBD run |
| `high_precision` | 3 | 0.01 | 100 | 1e-5 | High precision with tight tolerance |
| `five_districts` | 5 | 0.05 | 50 | 1e-4 | Test with 5 districts |

## Customizing Experiments

Edit `lbbd_config.py` to:

1. **Add new configurations** to the `LBBD_CONFIGS` list
2. **Modify data paths** in the `DATA_PATHS` dictionary
3. **Update block group list** in `BG_GEOID_LIST`
4. **Change output settings** in `OUTPUT_SETTINGS`

Example of adding a new configuration:
```python
{
    'name': 'my_experiment',
    'description': 'Custom experiment description',
    'num_districts': 4,
    'epsilon': 0.02,
    'max_iterations': 30,
    'tolerance': 1e-3,
    'max_cuts': 75,
    'verbose': True
}
```

## Output Files

The script generates several output files:

### Log Files
- `lbbd_run_YYYYMMDD_HHMMSS.log` - Detailed execution log

### Result Files (per experiment)
- `lbbd_{config_name}_YYYYMMDD_HHMMSS.pkl` - Complete results (pickle format)
- `lbbd_{config_name}_YYYYMMDD_HHMMSS.json` - Summary results (JSON format)

### Combined Results
- `lbbd_combined_YYYYMMDD_HHMMSS.pkl` - All experiments combined

## Result Structure

Each result file contains:
```python
{
    'best_partition': np.array,  # Best partition matrix
    'best_cost': float,          # Best objective value
    'history': list,             # Iteration history
    'runtime': float,            # Execution time in seconds
    'config': dict,              # Configuration used
    'converged': bool            # Whether algorithm converged
}
```

## Data Requirements

The script expects the following data files:
- `2023_shape_files/census_shape_block_group/tl_2023_42_bg.shp` - Block group shapefile
- `2023_target_tract_commuting/ACSDT5Y2023.B08303-Data.csv` - Commuting data (optional)
- `2023_target_blockgroup_population/ACSDT5Y2023.B01003-Data.csv` - Population data (optional)

**Note**: If actual data files are not available, the script will generate synthetic demand data for testing purposes.

## Command Line Arguments

```
usage: run_lbbd.py [-h] [--config CONFIG] [--list-configs] [--all] [--output-dir OUTPUT_DIR]

Run LBBD experiments

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG, -c CONFIG
                        Configuration name to run (see lbbd_config.py)
  --list-configs        List available configurations
  --all                 Run all configurations
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        Output directory for results
```

## Example Usage Scenarios

### Development Testing
```bash
# Quick test during development
python run_lbbd.py --config quick_test
```

### Production Run
```bash
# Run standard configuration with custom output
python run_lbbd.py --config standard --output-dir production_results/
```

### Batch Processing
```bash
# Run all configurations for comprehensive analysis
python run_lbbd.py --all --output-dir batch_results/
```

### Parameter Sensitivity Analysis
```bash
# Run multiple configurations to analyze parameter sensitivity
python run_lbbd.py --config small_scale
python run_lbbd.py --config standard  
python run_lbbd.py --config high_precision
```

## Troubleshooting

### Common Issues

1. **Missing shapefile**: Ensure the shapefile path in `DATA_PATHS` is correct
2. **Gurobi license**: Make sure Gurobi is properly licensed
3. **Memory issues**: Reduce `max_cuts` or `max_iterations` for large problems
4. **Convergence issues**: Adjust `tolerance` or increase `max_iterations`

### Debug Mode
To enable more detailed logging, modify `OUTPUT_SETTINGS['log_level']` to `'DEBUG'` in `lbbd_config.py`.

### Performance Tips
- Use `quick_test` for initial debugging
- Start with smaller `num_districts` for faster convergence
- Increase `max_cuts` for better bounds but slower iterations
- Use appropriate `epsilon` values (0.01-0.1 typically work well)

## Integration with Jupyter Notebooks

Results can be loaded back into Jupyter notebooks for analysis:

```python
import pickle
import pandas as pd

# Load results
with open('lbbd_results/lbbd_standard_20241201_143022.pkl', 'rb') as f:
    results = pickle.load(f)

# Analyze convergence
history_df = pd.DataFrame(results['history'])
print(f"Best cost: {results['best_cost']:.4f}")
print(f"Runtime: {results['runtime']:.2f} seconds")
``` 