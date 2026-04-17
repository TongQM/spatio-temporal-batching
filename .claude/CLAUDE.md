# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Research code for **service-design optimization and evaluation under uncertain demand**, applied to Pittsburgh HCT and synthetic grids. It compares several last-mile/transit service modes (partition-based spatio-temporal batching, fully on-demand, fixed-route, fixed-route with detours, VCC, Carlsson-style continuous approximations) against a shared evaluation harness. The partition optimizer uses Logic-Based Benders Decomposition (LBBD) with Wasserstein-DRO subproblems; heuristics (random search + local search) are also provided.

## Environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate optimization
```

Key deps: `numpy`, `scipy`, `pandas`, `gurobipy` (license required), `networkx`, `geopandas`, `shapely`, `pyproj`, `libpysal`, `simpy`, `polyline`. All scripts run from the project root.

## Common commands

Integration check after touching baselines or shared evaluation:
```bash
python scripts/experiments/baseline_comparison.py --mode synthetic --grid_size 5 --districts 3
# outputs: results/baseline_comparison/.../{results.json,summary.csv}
```

Targeted tests (pytest, or run as scripts where applicable):
```bash
python -m pytest tests/test_fr_detour.py -k <name>
python tests/test_lbbd.py            # LBBD on a 4x4 synthetic grid
python tests/test_rs_heuristic.py
python tests/test_vcc.py
```

LBBD on real Pittsburgh data:
```bash
python -m real_case.experiment --method LBBD --districts 3
python scripts/experiments/run_lbbd.py --config standard   # edit lbbd_config.py for profiles
```

Other experiment entrypoints: [scripts/experiments/baseline_cloud_sweep.py](scripts/experiments/baseline_cloud_sweep.py), [scripts/experiments/design_perf.py](scripts/experiments/design_perf.py), [scripts/experiments/simulate_service_designs.py](scripts/experiments/simulate_service_designs.py), [scripts/experiments/pareto_frontier_analysis.py](scripts/experiments/pareto_frontier_analysis.py).

## Architecture: two layers, kept strictly separate

The codebase has two layers that **must not be conflated**:

1. **Design / optimization layer** — each baseline produces a `ServiceDesign` using whatever approximations make optimization tractable: BHH, block-wise demand assignment, geometric partition surrogates, LP relaxations, restricted line libraries, discretized time/load states, etc. These approximations are *acceptable in optimization* if explicit and documented.

2. **Evaluation / simulation layer** — `EvaluationResult` is computed by sampling demand, running the design, and measuring realized user/provider metrics using the mode's actual service logic. Evaluation should **not** use BHH or block-wise surrogate travel as the final metric, and must not silently inherit optimization-time relaxations.

If you change optimization logic, do not change evaluation semantics unless that is the explicit goal.

### Key files
- [lib/baselines/base.py](lib/baselines/base.py) — `ServiceDesign`, `EvaluationResult`, shared evaluation/simulation helpers. This is the comparability boundary.
- [lib/baselines/](lib/baselines/) — one file per mode: `spatial_only`, `temporal_only`, `spatio_temporal`, `carlsson`, `fixed_route`, `fr_detour`, `vcc`, `fully_od`.
- [lib/algorithm.py](lib/algorithm.py) — LBBD (master = Hess assignment + depot selection; subproblems = per-district CQCP under Wasserstein DRO with multi-cut acceleration), random search, local search.
- [lib/evaluate.py](lib/evaluate.py), [lib/data.py](lib/data.py), [lib/road_network.py](lib/road_network.py), [lib/synthetic.py](lib/synthetic.py), [lib/tsp_solver.py](lib/tsp_solver.py) — supporting infrastructure.
- [scripts/experiments/baseline_comparison.py](scripts/experiments/baseline_comparison.py) — main integration entrypoint, drives every baseline through the shared evaluator.
- [real_case/](real_case/) — Pittsburgh HCT case study (data loaders + experiment driver).

## Cost structure is mode-specific

The shared `EvaluationResult` schema (user: wait / in-vehicle / walk; provider: linehaul / in-district / fleet / ODD) is for **comparison**, not a claim that every mode populates every component the same way. When implementing or modifying a baseline:

- **Partition-based modes** typically have linehaul cost (depot → district).
- **Non-partition modes** (fully on-demand, fixed-route, fr-detour) generally do **not** have linehaul cost in that sense.
- **Walk time** only applies to fixed-route, fr-detour, VCC-like modes.
- Other components (in-district routing, dispatch wait, ODD, coordination/transfer effects) are present or absent depending on the mode.

Rule: include only what operationally belongs to the mode. Do not inherit costs from another baseline just because the schema has a slot. Leave inapplicable components zero only when zero is the *correct* value.

## Working rules

- **Adding/modifying a baseline**: identify (1) what the mode operationally does, (2) which cost components belong to it, (3) which approximations are optimization-only, (4) how it should be simulated. Map mode-specific costs into `EvaluationResult` honestly.
- **Paper-inspired methods**: state explicitly what is faithful to the paper, what is adapted, what is only an optimization-time surrogate, and how evaluation differs. Do not overclaim faithfulness — [lib/baselines/fr_detour.py](lib/baselines/fr_detour.py) is particularly sensitive to this.
- **Changing evaluation logic**: be explicit whether the change is local to one mode, affects all modes, or breaks comparability. Do not import optimization-time approximations into evaluation.
- Prefer a smaller, honest, mode-aware change over a broad but muddled rewrite.

### FR-Detour specific guidance

- Treat FR-Detour optimization as a **service-point assignment** problem:
  - checkpoint candidates define route geometry,
  - stop points are assigned to selected lines/trips,
  - demand blocks are primarily an evaluation-side sampling device.
- Use fixed-common-origin / stochastic-destination / last-mile wording consistently.
  Earlier reverse first-mile wording may be mathematically symmetric in the
  simplified model, but it is not the intended semantic description for this repo.
- Unselected checkpoint candidates still matter:
  - before optimization they are candidate checkpoint anchors,
  - after optimization they are ordinary demand/service locations unless actually selected.
- FR-Detour evaluation currently uses a deliberate repo-specific innovation:
  - sampled passengers inside a demanded block choose the **nearest realized service point globally**
  - realized service points are selected checkpoints and detour-served stop points
  - if no detour stop serves that location, passengers fall back to the nearest selected checkpoint
- Do not silently reintroduce block-to-route partitioning in FR-Detour evaluation. Optimization may assign stop points uniquely to lines, but evaluation is allowed to use global nearest-realized-service-point access.

## Project-specific pitfalls

- Synthetic and real geodata expose positions differently — use the `_get_pos(...)` helper rather than indexing directly.
- Road-network mode and Euclidean/BHH mode are different evaluation regimes; do not mix them.
- Evaluation comparability is easy to break if a baseline bypasses the shared metrics in `lib/baselines/base.py`.
- Gurobi is required for LBBD master/subproblems; tests that skip silently without a license can hide regressions.
