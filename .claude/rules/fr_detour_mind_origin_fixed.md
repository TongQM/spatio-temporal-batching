# FR-Detour: Origin-Fixed MiND-VRP Rule

This note is the local source of truth for how `FRDetour` should be understood, implemented, tested, and compared in this repo.

## What FR-Detour is supposed to represent

`FRDetour` is the repo's implementation target for the **MiND-VRP** model from:

- *A Double Decomposition Algorithm for Network Planning and Operations in Deviated Fixed-route Microtransit*

The target problem variant here is:

- **fixed origin**
- **stochastic destinations**
- **deviated fixed-route microtransit**

Wording note:

- Some earlier discussion in this repo used the reverse first-mile wording:
  stochastic origins traveling to a common destination.
- For the simplified Euclidean service model used here, the two views are
  largely symmetric mathematically.
- The intended semantic interpretation for this repo is nevertheless:
  riders start from the common origin / depot and travel to realized
  destination/service locations.
- Code comments, captions, and docs should therefore use last-mile /
  fixed-origin / stochastic-destination wording consistently.

That means:

- first stage chooses a scheduled backbone of reference trips,
- second stage reacts to realized destination demand,
- all riders conceptually start from the same origin / depot side of the service,
- second-stage routing decides where to deviate and which realized destinations are served.

## Paper-faithful model target

When working on `lib/baselines/fr_detour.py`, the intended model should be treated as:

### First stage

- candidate **reference trips** are defined by:
  - a reference line,
  - a departure time / headway choice,
  - scheduled checkpoint arrival times.
- route geometry is built from **checkpoint candidates** only.
- non-checkpoint **stop points** are service locations, not route-geometry anchors.
- checkpoint candidates and stop points must be treated as two globally
  disjoint sets.
- unselected checkpoint candidates remain ordinary demand/service locations after optimization.
- binary trip-selection variable `x_(l,t)`
- scenario-dependent assignment variables linking realized **service locations** to trips
- each stop point is assigned to at most one selected line / trip
- fleet budget constraint
- lower and upper load-factor constraints

In this repo, candidate-line generation may remain preprocessing, but once candidate trips exist, the optimization should behave like the paper's first stage rather than an ad hoc route-compatibility heuristic.

### Second stage

- use a **load-expanded checkpoint network**
- nodes represent:
  - checkpoint,
  - remaining onboard load
- arcs represent **subpaths between checkpoints**
- time feasibility must come from **scheduled checkpoint times**, not a free geometric surrogate
- linking should be **assignment-to-service** in the paper sense, not a loose route-compatibility score

For this repo's target variant, second-stage load should follow **fixed-origin delivery semantics**:

- vehicle starts with assigned riders onboard,
- load decreases as riders are dropped off,
- detours serve destination stops between checkpoints.


There is one subtle point in the second-stage optimization when we combine
continuously sampled rider destinations with the paper's stop-level demand
framework: the paper assumes realized demand already occurs directly at stop
points or checkpoints, so the second stage sees those realized location-level
demands immediately. In our repo, riders are first sampled in continuous space,
so the second stage does not directly know which stop points should carry
realized demand. This creates a chicken-egg problem.

The intended resolution is:

1. after rider destinations are realized, greedily preassign each rider by
   Euclidean distance to the nearest **candidate service location**
2. the candidate service locations for this preassignment are only:
   - selected checkpoints
   - assigned stop points
   - assigned stop points must come only from the non-checkpoint stop-point
     set; a checkpoint candidate must not reappear as another route's stop
     point
3. aggregate those rider preassignments into realized stop/checkpoint demand
4. selected-checkpoint demand is treated as fixed realized checkpoint service
5. run second-stage detour optimization only on assigned stop points with
   positive projected demand
6. after the second-stage solve:
   - riders projected to stop points that are actually served stay there
   - riders projected to stop points that are not served are reassigned to the
     nearest realized service point among served stop points and selected
     checkpoints
   - walk distance is recomputed to that final realized service point

Important correction after revisiting the paper semantics:

- if a checkpoint is **not selected** into the route backbone, it is not part
  of that route's second-stage service network and cannot be visited
- therefore unselected checkpoint candidates are **not** second-stage detour
  candidates unless they are actually selected into the route

## Double decomposition target

The intended algorithmic structure is:

- outer **multi-cut Benders decomposition**
- inner **column generation**
- pricing by **label-setting / subpath generation**

Interpretation:

- pricing adds subpath arcs,
- restricted routing LP combines them into a route for one trip and one scenario,
- Benders master updates first-stage trip selection and assignments.

## Exactness policy

Do not overclaim.

There are two distinct targets:

1. **DD relaxed core**
   - mixed-integer first stage
   - continuous second stage
   - this is the acceptable Milestone 1 target

2. **Exact paper extensions**
   - `DD&ILS`
   - `UB&DD`
   - these are Milestone 2 targets and should not be claimed implemented unless they actually exist in code

If only the relaxed DD core is present, metadata and docs should say so explicitly.

## Repo-facing API constraints

Even when internals change, keep the public baseline API stable:

- `FRDetour.design(...)`
- `FRDetour.evaluate(...)`
- `FRDetour.custom_simulate(...)`

Return type should remain `ServiceDesign`.

Use `service_metadata["fr_detour"]` to expose paper-relevant internal structure needed by diagnostics. At minimum it should contain stable keys for:

- selected trip identities
- checkpoint sequence by active route
- selected checkpoints by route
- checkpoint schedule by active route
- assigned stop points by route
- reachable stop points by route
- checkpoint-candidate pool
- demand-support locations
- assignment / compatibility summaries
- scenario assigned loads
- planned fleet size
- solve mode:
  - `dd_relaxed`
  - `dd_ils`
  - `ub_dd`

## What should NOT happen

- Do not describe stop-level compatibility as paper-faithful passenger assignment.
- Do not replace scheduled checkpoint timing with a purely geometric surrogate and still call it faithful MiND.
- Do not use block-level partition logic as the primary first-stage assignment semantics.
- Do not claim `DD&ILS` or `UB&DD` exists unless integer L-shaped or UB-and-branch logic is actually implemented.
- Do not let evaluation silently use different routing semantics from optimization unless the difference is deliberate and documented.

## Evaluation rules

Optimization and evaluation should use the same operational meaning of:

- scheduled checkpoints
- detour feasibility
- skip limit `K`
- fixed-origin rider flow

Evaluation is intentionally more realistic than the paper's first-stage assignment:

- demand support is discrete over demand/service locations
- passenger positions are sampled continuously inside demanded blocks
- before second-stage routing, riders are greedily projected to the nearest
  candidate service location:
  - selected checkpoint, or
  - assigned stop point
- second-stage detour optimization is then solved on the projected stop demand
- final rider access uses the realized service set:
  - selected checkpoints
  - detour-served stop points
- riders projected to a stop point that is not actually served are reassigned
  to the nearest realized service point and their walk is recomputed

Cost-weight interpretation for `FRDetour`:

- `wr` is the rider-time weight and should weight:
  - wait time
  - in-vehicle time
  - walk time through `walk_time_weight * wr`
- `wv` should be treated as a deprecated alias for a provider-side
  vehicle-travel cost scale, not as vehicle speed and not as a rider-time
  weight

Do not revert evaluation to:

- a route-partition rule based on nearest selected checkpoint, or
- a semantics where stop points are marked served before rider demand is first
  projected onto candidate service locations

## Testing rules

The primary gate for FR / FR-Detour is a **small deterministic acceptance suite**, not only large synthetic regressions.

Required test themes:

- `delta = 0` only serves checkpoint-feasible demand
- `delta > 0` can serve a corridor-feasible off-route destination that FR cannot
- outside-corridor or schedule-infeasible demand remains unserved
- at-most-one assignment is respected
- lower and upper load-factor logic is exercised
- `K = 0` vs `K = 1` skip behavior is visible
- optimization metadata and evaluation semantics are consistent
- unselected checkpoint candidates fall back to selected checkpoints in evaluation unless detour-served
- sampled passengers can choose the nearest realized service point globally, even across route boundaries
- stop points with zero projected demand are not marked as served
- riders projected to unserved stop points are reassigned to the nearest
  realized service point

Large synthetic comparisons are secondary checks, not the primary source of truth.

## Comparison and plotting rules

`scripts/plots/fr_detour_comparison.py` should support two views:

1. **Primary**: FR and FR-Detour optimize separately and are compared end-to-end
2. **Secondary diagnostic**: both are evaluated on the same selected design to isolate second-stage detour effects

Plots should separate:

- optimization differences:
  - selected routes
  - schedules
  - checkpoint structure
- evaluation differences:
  - checkpoint-served demand
  - detour-stop-served demand
  - reassigned demand after unserved projected stop points
  - user metrics
  - provider metrics


## Current implementation status convention

When describing the code, use these labels precisely:

- `paper-faithful`: directly matches the MiND-VRP intent
- `adapted`: same architectural role, but simplified or repo-specific
- `missing`: paper component not implemented

This file should be updated whenever `FRDetour` meaningfully changes.
