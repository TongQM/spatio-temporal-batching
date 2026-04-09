## FR-Detour Summary

### Model interpretation

- The intended repo semantics for `FRDetour` are:
  - fixed common origin / depot
  - stochastic destinations
  - last-mile service
- Earlier first-mile wording is mathematically close in the simplified Euclidean setting, but it is not the intended semantic description for this repo.

### Optimization semantics

- Route geometry is built from **checkpoint candidates**.
- **Stop points** are assigned to selected lines / trips as service locations.
- Checkpoint candidates and stop points are intended to be globally disjoint:
  checkpoint-candidate locations should not reappear as detour stop points.
- **Demand blocks** are mainly an evaluation-side sampling device, not the primary first-stage assignment object.
- Unselected checkpoint candidates still matter:
  - before optimization they are candidate geometry anchors
  - after optimization they behave like ordinary demand/service locations unless actually selected

### Evaluation semantics

- Detour decisions are **second-stage** decisions made after scenario demand is realized.
- For each realized scenario, evaluation now follows a four-step flow:
  - sample rider destinations continuously within demanded blocks
  - greedily project each rider to the nearest candidate service location
  - solve route-by-route second-stage detours on the projected stop demand
  - reassign riders projected to unserved stop points to the nearest realized service point
- Candidate service locations for the greedy projection are only:
  - selected checkpoints
  - assigned stop points
- Unselected checkpoint candidates are not second-stage detour candidates unless they are selected into the backbone.
- Realized service points in a scenario are:
  - selected checkpoints
  - detour-served stop points with positive projected demand
- Demand is discrete at support locations, but passenger positions are sampled continuously within each demanded block for walking.

### Figure semantics

- The FR comparison plot is a **one-scenario diagnostic** overlaid on top of average evaluation metrics.
- Current map semantics:
  - blue square = selected checkpoint
  - filled colored circle = detour-served stop point with positive projected demand
  - hollow colored circle = assigned corridor stop point not served in that scenario
  - small gray dots = sampled destination points for realized riders
  - thin gray dashed lines = connection from each sampled destination to the realized service point that serves it
- The title keeps aggregate demand counts:
  - `C` = demand units served at selected checkpoints
  - `D` = demand units served at detour stop points
  - `W` = demand units that are reassigned after their projected stop point is not served
- The map no longer prints `C#`, `D#`, `W#` labels at individual locations.

### Important interpretation rules

- A positive `delta` is intended as a **reachability relaxation**:
  - FR remains a feasible special case of FR-Detour
  - if detours are not worthwhile, the model should be free to choose no detours
- The top-row “optimized separately” figure is **not** a pure monotonicity test on one fixed design.
- The bottom-row same-design diagnostic is the cleaner view for isolating the effect of allowing detours.

### Current calibration decisions

- In `FRDetour`, rider time is now weighted by `wr`:
  - wait time uses `wr`
  - in-vehicle time also uses `wr`
  - walk time uses `walk_time_weight * wr`
- `wv` should be read as a deprecated alias for a provider-side
  `vehicle_cost_weight`; for FR-style comparisons it should typically be
  pinned at `1.0`.
- Current intended relative weights are roughly:
  - vehicle travel: `1x`
  - in-vehicle time: `1x` via `wr`
  - wait time: `1x` via `wr`
  - walk time: `5x` via `walk_time_weight * wr`
- Fleet remains constrained in design; explicit fleet pricing depends on `fleet_cost_rate`.

### Current implementation notes

- Plot semantics were recently corrected so `C` / `D` counts correspond to final realized service-point loads, not demand-block labels.
- Scenario evaluation is now driven by projected rider demand at candidate service locations before second-stage detour selection.
- Rider-level assignment records continue to drive the dashed scenario visualization, but now reflect:
  - projected candidate service locations
  - final realized service locations after reassignment
- `.claude/rules/fr_detour_mind_origin_fixed.md` remains the more detailed source of truth.
