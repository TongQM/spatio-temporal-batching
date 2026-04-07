"""
Shared constants used across the transit partitioning project.
"""

# Use the same per-km provider travel scaling across modes.
# No autonomous-vehicle mileage discount is applied in the current experiments.
TSP_TRAVEL_DISCOUNT = 1.0

# Default provider-side operating cost per active vehicle-equivalent.
DEFAULT_FLEET_COST_RATE = 0.0
