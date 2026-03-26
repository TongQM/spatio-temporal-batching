# Claude Code Guidance

## Purpose

This repository studies **service-design optimization** and **service evaluation** for multiple service modes under uncertain demand.

The project has two distinct layers:

1. **Optimization / design generation**
   - Produce a service design using an optimization model, heuristic, or paper-inspired adaptation.
   - This layer may use approximations, relaxations, discretizations, surrogate costs, or simplified recourse models.

2. **Evaluation / simulation**
   - Evaluate a finished design by simulating service under sampled demand.
   - This is the common comparison layer across methods.

These layers are intentionally modular and must not be conflated.

---

## Core Principle 1: Optimization and Evaluation Are Modular

### Design / optimization layer
In optimization, methods may use approximations such as:
- BHH approximations
- block-wise demand assignment
- geometric partition surrogates
- LP relaxations
- restricted line libraries
- discretized time or load states
- other tractability-driven simplifications

Examples:
- the project’s own frameworks use discretization and approximation
- Carlsson-style methods use continuous approximations in optimization
- MiND-style or other paper-inspired baselines may use adapted or relaxed design-time models

These approximations are acceptable in optimization if they are:
- explicit
- documented
- methodologically consistent

### Evaluation / simulation layer
Evaluation should be treated as the higher-fidelity comparison layer.

In evaluation, we generally do **not** use:
- BHH approximations for realized service
- blockwise surrogate travel approximations as the final metric
- optimization-only relaxations as if they were true operational outcomes

Instead, evaluation should simulate service by:
- sampling demand realizations
- running the service design
- computing realized user/provider metrics using the mode’s actual service logic

If you change optimization logic, do not silently change evaluation semantics unless that is the explicit goal.

---

## Core Principle 2: Cost Structure Depends On Service Mode

Different service modes have different cost components. Do not force all modes into the same implicit cost model.

Examples:
- **Partition-based modes** may have **linehaul cost**
  - vehicles travel from the depot to a district before in-district service
- **Non-partition modes** such as:
  - fully on-demand
  - fixed-route
  - fixed-route with detours
  may have **no linehaul cost** in the same sense
- **Walk time** only exists for relevant modes such as:
  - fixed-route
  - fixed-route with detours
  - VCC-like coordination modes
- Some modes incur:
  - in-district routing cost
  - dispatch-interval wait cost
  - ODD cost
  - coordination or transfer effects
  while others do not

These examples are not exhaustive.

### Rule
When implementing or modifying a mode:
- include only the cost components that operationally belong to that mode
- do not inherit costs from another mode just because the evaluation table has a common schema
- map mode-specific costs into the shared output structure carefully and honestly

The common output table is for comparison, not proof that all methods share the same internal cost model.

---

## Core Principle 3: Examples Are Not Exhaustive

Always ask:
- what costs exist for this mode?
- what operational steps actually occur?
- what approximations are used only during optimization?
- what should be simulated during evaluation?
- what is absent for this mode and should remain zero or omitted?

Do not assume that because one baseline has a component, another should too.

---

## Environment

Use the `optimization` conda environment.

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate optimization
```

Common dependencies:
- `numpy`
- `scipy`
- `pandas`
- `matplotlib`
- `gurobipy`
- `networkx`
- `geopandas`
- `shapely`
- `pyproj`

---

## Important Files

### Shared interfaces and evaluation
- `lib/baselines/base.py`
  - `ServiceDesign`
  - `EvaluationResult`
  - shared evaluation helpers
  - common simulation/evaluation logic

### Baseline implementations
- `lib/baselines/*.py`

Important examples:
- `lib/baselines/spatial_only.py`
- `lib/baselines/temporal_only.py`
- `lib/baselines/spatio_temporal.py`
- `lib/baselines/carlsson.py`
- `lib/baselines/fr_detour.py`
- `lib/baselines/vcc.py`
- `lib/baselines/fully_od.py`

### Supporting optimization / data code
- `lib/algorithm.py`
- `lib/evaluate.py`
- `lib/data.py`

### Main experiment entrypoint
- `scripts/experiments/baseline_comparison.py`

Outputs:
- `results/baseline_comparison/.../results.json`
- `results/baseline_comparison/.../summary.csv`

---

## Working Rules

### If changing optimization logic
- Preserve the separation between optimization and evaluation.
- Document approximations clearly in the method docstring.
- Do not describe an optimization approximation as if it were the final evaluation model.

### If changing evaluation logic
- Be explicit about whether the change affects:
  - one mode only
  - all modes
  - comparability across baselines
- Do not accidentally import optimization-time approximations into evaluation.

### If adding or modifying a baseline
Before coding, identify:
1. what the mode operationally does
2. what cost components it should have
3. what approximations are used only for tractable design optimization
4. how it should be simulated in evaluation

### If implementing a paper-inspired method
State explicitly:
- what is faithful to the paper
- what is adapted to this repo
- what is only an optimization-time surrogate
- how evaluation differs from the paper model, if applicable

Do not overclaim faithfulness.

---

## Shared Output Structure

`EvaluationResult` provides a common comparison schema, including:
- user:
  - wait
  - in-vehicle
  - walk
  - user cost
- provider:
  - linehaul
  - in-district
  - total travel
  - fleet
  - ODD cost
  - provider cost
- aggregate:
  - total cost

This shared schema does **not** imply that every mode should populate every component in the same way.

Instead:
- populate components according to the mode’s operational logic
- keep unused or inapplicable components zero only when that is correct

---

## Testing Expectations

After meaningful changes to baselines or shared evaluation code, run:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate optimization
python scripts/experiments/baseline_comparison.py --mode synthetic --grid_size 5 --districts 3
```

Check:
- the script completes
- all intended baselines appear
- outputs are written
- metrics are finite
- cost components make sense for each mode
- no mode is accidentally charged for inapplicable costs
- no optimization-time approximation leaked into evaluation unexpectedly

Additional tests may still be useful for specific algorithm changes, but the baseline-comparison workflow is the main integration check.

---

## Project-Specific Pitfalls

- `fr_detour.py` is sensitive to overclaiming paper faithfulness.
- Synthetic and real geodata expose positions differently; use `_get_pos(...)`.
- Evaluation comparability is easy to break if a baseline bypasses shared metrics.
- Road-network mode and Euclidean/BHH mode are different evaluation regimes; do not mix them accidentally.
- The common result table is shared, but the underlying operational cost structure is mode-specific.

---

## What Claude Should Optimize For

When making changes, optimize for:
1. correct separation of optimization vs evaluation
2. correct mode-specific cost structure
3. honest documentation of approximations
4. comparability across service modes
5. minimal disruption to experiment workflows

Prefer a smaller, honest, mode-aware change over a broad but muddled rewrite.
