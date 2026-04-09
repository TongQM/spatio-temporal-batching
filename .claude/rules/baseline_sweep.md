# Baseline comparison sweep — methodology reference

This is the local source of truth for how baseline performance clouds are
produced and interpreted in this repo. The sweep script is
[scripts/experiments/baseline_cloud_sweep.py](scripts/experiments/baseline_cloud_sweep.py).
It supports two views. Keep the two views separate — they answer different
questions and the artifacts are stored in different output subfolders.

## Two views

### Regime view (`--view regime`, default, main comparison)

Sweep shared **system regime** inputs `(Λ, wr)` on a small grid. Every
baseline sees the same grid and re-optimizes at its own library defaults.
Each cloud point means:

> "method M's best design at system regime `(Λ, wr)`"

All clouds live in the same plane, use the same x-/y-axis semantics, and are
directly comparable.

Default grid: `Λ ∈ {50, 150, 300}` pax/hour, `wr ∈ {0.3, 1.0, 3.0}` →
9 points per baseline. Change with `--regime_lambdas` and `--regime_wrs`.

### Hyper view (`--view hyper`, diagnostic)

Sweep per-baseline hyperparameters (fleet for FR, `T_fixed` for SP-Lit,
`num_districts` for Joint, walk-tolerance for VCC, …). Useful to see each
method's own Pareto shape and to understand how its hyperparameters bend
the frontier. **Not apples-to-apples across baselines** — different methods
sweep different knobs, span different axes of the cost plane, and live in
different absolute ranges. Use this view for single-method diagnostics, not
for cross-method claims.

## Shared-parameter definitions

These semantics are assumed by both views and are the authoritative contract
between baselines and the sweep harness. If a future change touches any of
these, update this file.

### `Λ` — total arrival rate `[passengers / hour]`

- Poisson arrival rate summed over the whole service area.
- Spatial distribution is fixed by `prob_dict`, so the per-block rate is
  `Λ · p_b`.
- **System input, not a design choice.** Every baseline receives the same
  `Λ` via `design(..., Lambda=Λ)` and `evaluate(..., Lambda=Λ)` and must
  design a service that handles it. No baseline rescales `Λ` internally.

### `wr` — rider time cost weight `[cost / rider-hour]`

- Weight on total user time (`wait + in-vehicle + walk`) in the design
  objective and in the reported `EvaluationResult.user_cost =
  wr · total_user_time_per_rider · expected demand`.
- Used by `_newton_T_star` ([lib/baselines/base.py:157](lib/baselines/base.py))
  for every partition baseline that picks `T*` automatically.

### `wv` — vehicle operating cost weight `[cost / vehicle-km]`

- **Pinned to `1.0` in regime view.**
- Intended meaning: cost of one vehicle-km of provider travel
  (linehaul + in-district + detour).
- Do not interpret `wv` as vehicle speed. Physical speed is a separate model
  constant.
- In `FRDetour`, rider wait and in-vehicle time should both be weighted by
  `wr`; `wv` is only a provider-side cost scale there.
- **Known asymmetry**: the optimization side uses `wv` explicitly
  (`_newton_T_star` at [lib/baselines/base.py:174](lib/baselines/base.py);
  the FR-Detour master at [lib/baselines/fr_detour.py:894](lib/baselines/fr_detour.py)),
  but the evaluation side builds
  `provider_cost = linehaul_km + in_district_km + odd + fleet_cost_rate · raw_fleet`
  **without multiplying by `wv`** ([lib/baselines/base.py:645,651](lib/baselines/base.py)).
  The evaluation implicitly assumes `wv = 1 [cost / km]`, while the
  optimizer uses whatever `wv` you pass. Fixing `wv = 1` in regime view
  keeps both sides on the same scale. Do not sweep `wv` unless the
  `evaluate_design` / `simulate_design` helpers are fixed first.

## Regime-responsiveness at design time

Not every baseline re-optimizes its design when `(Λ, wr)` changes. This
cheat sheet tells you what to expect:

| Baseline | Design uses `Λ` | Design uses `wr`/`wv` | Notes |
|---|---|---|---|
| Multi-FR / Multi-FR-Detour | yes | yes | Benders master; `T` over internal `T_values` grid |
| SP-Lit (Carlsson) | no | no | Fixed `T_fixed` → **regime-insensitive design** |
| TP-Lit (Liu) | yes | partial | Uses `Λ·T` for fleet sizing |
| Joint-Nom / Joint-DRO | yes | yes | Newton `T*` uses `wr/wv` |
| VCC | yes | no | `Λ` drives planning horizon only |
| OD (FullyOD) | no | no | Fixed `T_min` → **regime-insensitive design** |

For regime-insensitive designs the cloud answers "how does this fixed
design behave when the world around it changes" — a legitimate data point
and explicitly called out in the regime-view `varying_note` for that
baseline. Do **not** describe them as if they were re-optimizing.

### Fleet-size parity across baselines

Regime view is set up so baselines with **scheduled** vehicles all operate
at roughly the same fleet scale, tied to `--districts`:

- Partition methods (SP-Lit, Joint-Nom, Joint-DRO) dispatch one tour per
  district, so their effective fleet is ≈ `K = --districts` (maybe +1 per
  district if utilization exceeds one vehicle-period).
- FR / FR-Detour: `max_fleet` is deliberately left unset in
  `_build_regime_specs`, so the `FRDetour` constructor's fall-back
  (`fleet_budget = num_districts`, see
  [lib/baselines/fr_detour.py:2351](lib/baselines/fr_detour.py)) gives FR
  exactly `K` vehicles. This matches SP-Lit's fleet footprint and makes
  FR vs partition a routing-structure comparison, not a fleet-scale one.

Baselines with **unscheduled** vehicles do NOT fit this parity rule and
are on a different fleet axis by design:

- OD: fully on-demand; every request can trigger its own vehicle. Its
  effective fleet is whatever the simulation draws, typically ≫ K.
- VCC: currently hardcoded `fleet_size = 30` in `_build_regime_specs`.
  This is NOT comparable to partition/FR on the fleet axis — VCC at `K=3`
  vehicles would be broken (the DAR-VCC model assumes enough vehicles to
  absorb request overlap with walk-tolerance slack). Treat VCC cloud
  position as "VCC at its own scale" and not a direct fleet-for-fleet
  comparison with SP-Lit or FR.

When you add a new scheduled baseline, wire it through the same
`num_districts` → fleet-budget default so the fleet-parity rule holds.

## Cloud interpretation

### Regime view
- Each baseline produces `|Λ| × |wr|` points (default 9).
- Marker **size ∝ Λ** (small = smallest Λ, large = largest Λ).
- Marker **color/shape = baseline** (fixed per-method styling).
- The convex hull of points is drawn lightly to show each method's spread
  across the regime grid.
- Comparing two clouds directly is valid.

### Hyper view
- Each baseline produces up to `--settings_per_baseline` points (default 20).
- Marker size is uniform; shape/color identifies the baseline.
- The hull shape depends entirely on which hyperparameter axis each method
  sweeps — NOT a Pareto frontier across methods. Treat the hulls as
  per-method shapes, not as comparable regions.
- Do **not** sweep Joint-DRO's `epsilon` in the cloud — it is a robustness
  knob, not a provider-vs-user Pareto knob. The current script drops it
  from the cloud automatically; put it in a dedicated robustness plot
  instead.

## Output layout

`results/baseline_clouds/<view>_<mode>_<descriptor>_K<K>_L<Λ>_wr<wr>_wv<wv>_cloud_<timestamp>/`

- `experiment.json` — sweep metadata, including `view`, `regime_lambdas`,
  `regime_wrs`, the pinned `wv`, and each baseline's `varying_note`.
- `aggregate_results.json` — all rows + metadata.
- `aggregate_summary.csv` — flat CSV; `Lambda` and `wr` columns are always
  populated so rows are self-describing regardless of view.
- `scatter_user_provider_cloud.png` — the main plot.
- `<baseline_slug>/`
  - `settings.json` — the sweep settings for this baseline.
  - `summary.csv`
  - `setting_01.json … setting_NN.json` — per-setting cache. `--resume`
    reuses these without recomputing, so the same output folder can be
    iterated on.

## Adding a new baseline

To make a new baseline work cleanly under the regime view, its `design(...)`
method must:

1. **Accept `Lambda`, `wr`, `wv` via kwargs**. The base abstract method is
   keyword-only for these; the concrete baselines already follow this
   pattern (e.g. [lib/baselines/fr_detour.py:2297](lib/baselines/fr_detour.py)).
2. **Use them internally** when choosing its design (through
   `_newton_T_star`, its own MIP objective, or whatever mechanism is
   appropriate). A baseline that ignores `Λ`, `wr`, `wv` will produce an
   identical design across the whole regime grid — still plottable, but
   must be marked `regime-insensitive design` in the cheat-sheet above and
   in the baseline's `varying_note` string in `_build_regime_specs`.
3. **Populate the mode-appropriate slots** of `EvaluationResult`. Follow the
   mode-specific cost-structure rule from
   [.claude/CLAUDE.md](.claude/CLAUDE.md) — do not inherit cost components
   that do not operationally belong to the mode.

## Commands

Activate the env first:

```bash
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh && conda activate optimization
```

### Smoke tests

```bash
# Regime view (main) — fast baselines only, small grid
python scripts/experiments/baseline_cloud_sweep.py \
  --view regime \
  --mode synthetic --grid_size 4 --districts 2 \
  --services sp_lit joint_nom od \
  --regime_lambdas 100 300 900 \
  --regime_wrs 0.3 1.0 3.0 \
  --jobs 1

# Hyper view (diagnostic) — small grid, single baseline
python scripts/experiments/baseline_cloud_sweep.py \
  --view hyper \
  --mode synthetic --grid_size 4 --districts 2 \
  --services joint_dro \
  --settings_per_baseline 5 --jobs 1
```

### Production shape

```bash
python scripts/experiments/baseline_cloud_sweep.py \
  --view regime \
  --mode synthetic --grid_size 5 --districts 3 \
  --services multi_fr multi_fr_detour sp_lit tp_lit joint_nom joint_dro vcc od \
  --regime_lambdas 100 300 900 \
  --regime_wrs 0.3 1.0 3.0 \
  --jobs 4
```

## What NOT to do

- Do not sweep `wv` unless `evaluate_design` / `simulate_design` are fixed
  to multiply `provider_cost` by `wv`. The asymmetry above is a known
  limitation; pin `wv = 1` until it is actually fixed.
- Do not put Joint-DRO's `epsilon` in the main cloud. Separate robustness
  plot.
- Do not compare hull areas across hyper-view baselines and call it a
  comparison — that is exactly what the regime view was added to fix.
- Do not silently add a new regime knob (e.g. `K`, `fleet_cost_rate`) to
  the grid. If you want to study `K` sensitivity, run separate sweeps per
  `K` and compare as a small multiple, not by blowing up the single cloud
  plot.
