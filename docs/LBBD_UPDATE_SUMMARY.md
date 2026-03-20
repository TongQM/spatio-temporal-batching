# LBBD Implementation Update Summary

## Overview
Updated the Logic-Based Benders Decomposition (LBBD) implementation to match the modified mathematical formulation (7) with depot location decisions and partition-dependent costs.

## Key Changes

### 1. **Master Problem Decision Variables**

#### **New Variables Added:**
- **`δ_j`** (delta): Binary variables for depot location selection
  - `δ_j = 1` if depot is located at block j, 0 otherwise
  - Constraint: `Σ_j δ_j = 1` (exactly one depot)

- **`ω_i`** (omega): Continuous variables for ODD summary per district
  - `ω_i ≥ Σ_j Ω_j z_ji` (ODD constraint from formulation 7i)

- **`K_i`**: Continuous variables for linehaul costs per district
  - `K_i = 2d(κ_i, Σ_j κ_j δ_j)` (distance from district root to depot, formulation 7g)

- **`F_i`**: Continuous variables for ODD costs per district
  - `F_i = J(ω_i)` (ODD cost function, formulation 7h)

#### **Existing Variables:**
- **`z[j,i]`**: Assignment variables (unchanged)
- **`o`**: Objective variable (unchanged)
- **`f[root_id, arc]`**: Flow variables for contiguity (unchanged)

### 2. **Master Problem Constraints**

#### **New Constraints:**
- **(7c)** One depot: `Σ_j δ_j = 1`
- **(7g)** Linehaul cost: `K_i = 2 * Σ_j d(root_i, block_j) * δ_j`
- **(7h)** ODD cost: `F_i = J(ω_i)` (implemented with piecewise linear approximation)
- **(7i)** ODD constraint: `ω_i ≥ Σ_j Ω_j z_ji`

#### **Existing Constraints (unchanged):**
- **(7d)** Assignment: `Σ_i z_ji = 1`
- **(7e)** Number of districts: `Σ_i z_ii = k`
- **(7f)** Symmetry breaking: `z_ji ≤ z_ii`
- **(7j-7m)** Flow-based contiguity constraints

### 3. **Subproblem Updates**

#### **New Method: `_SDP_benders_updated`**
- Accepts **partition-dependent costs** `K_i` and `F_i` as parameters
- Updated objective function:
  ```python
  g_bar = (K_i + F_i) * ci^(-2) + alpha_i * ci^(-1) + wr/(2*wv) * K_i + wr/wv * alpha_i * ci + wr * ci^2
  ```
- Computes subgradients w.r.t. master variables:
  - `∂g/∂α_i = ci^(-1) + wr/wv * ci`
  - `∂g/∂K_i = ci^(-2) + wr/(2*wv)`
  - `∂g/∂F_i = ci^(-2)`

### 4. **ODD Cost Function Handling**

#### **Piecewise Linear Approximation:**
- Uses SOS2 (Special Ordered Sets of type 2) variables for exact representation
- Creates breakpoints: `ω ∈ [0, max_omega]`
- Evaluates `J(ω)` at breakpoints
- Constraints:
  ```python
  ω_i = Σ_k ω_k * λ_k
  F_i = Σ_k J(ω_k) * λ_k
  Σ_k λ_k = 1
  ```

### 5. **Solution Extraction**

#### **Enhanced Information:**
- **Depot location**: `depot_block` from `δ_j` solution
- **Partition-dependent costs**: `K_sol`, `F_sol`, `omega_sol`
- **Comprehensive results dictionary**:
  ```python
  {
      'best_partition': z_sol,
      'best_cost': float,
      'best_depot': block_id,
      'best_K': [K_i values],
      'best_F': [F_i values], 
      'best_omega': [ω_i values],
      'history': iteration_history,
      'converged': boolean,
      'iterations': int,
      'final_gap': float
  }
  ```

### 6. **API Changes**

#### **New Parameters:**
- **`Omega_dict`**: Dictionary mapping `block_id → Ω_j` (ODD summary vector)
- **`J_function`**: Function `J(ω) → F` (ODD cost function)

#### **Usage Example:**
```python
# Define ODD parameters
Omega_dict = {block_id: odd_value for block_id in blocks}
J_function = lambda omega: 0.5 + 0.1*omega + 0.05*omega**2

# Run updated LBBD
result = partition.benders_decomposition(
    max_iterations=50,
    tolerance=1e-3,
    Omega_dict=Omega_dict,
    J_function=J_function
)

# Access results
depot = result['best_depot']
K_costs = result['best_K']
F_costs = result['best_F']
```

## Mathematical Correspondence

### **Formulation (7) Implementation:**
- **(7a)** Objective: `min o + Σ_j φ(κ_j)δ_j` → Handled in master objective
- **(7b-7f)** Binary/assignment constraints → Implemented as Gurobi constraints
- **(7g)** Linehaul cost → Bilinear constraint `K_i = 2d(...)`
- **(7h)** ODD cost → Piecewise linear `F_i = J(ω_i)`
- **(7i)** ODD constraint → Linear constraint `ω_i ≥ Σ_j Ω_j z_ji`
- **(7j-7m)** Contiguity → Flow conservation constraints
- **(7n)** Benders cuts → `o ≥ cut_constant + Σ cut_coeff * z_ji`

## Testing

Created `test_updated_lbbd.py` to verify:
- ✅ Depot location decisions
- ✅ Partition-dependent cost calculation
- ✅ ODD constraint handling
- ✅ Piecewise linear J(ω) approximation
- ✅ Enhanced result reporting

## Backward Compatibility

- **Old LBBD method**: Still available as `_SDP_benders` (original implementation)
- **New LBBD method**: Uses `_SDP_benders_updated` with partition-dependent costs
- **Default parameters**: Provided for `Omega_dict` and `J_function` if not specified

