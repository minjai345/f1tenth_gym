# Repository knowledge base

This is the developer-facing map of `f1tenth_gym`. Use it to orient feature
work, then read the named source and tests before changing a subsystem. The
published Sphinx pages are the user-facing source of truth for behavior and
examples; this file focuses on architecture, invariants, and change seams.

Baseline reviewed: Phase 2 after merge `a9cc351` on 2026-08-28. Treat counts
and the known-rough-edges section as a snapshot and re-check them against
`HEAD`.

## What this repository is

`f1tenth_gym` is a Python 3.12+ Gymnasium environment for simulating one or
more 1/10-scale race cars. It combines:

- CoG-referenced kinematic and single-track vehicle models; the transitional
  MB enum is rejected as unsupported;
- Euler or RK4 integration with optional integration substeps;
- raster or exact segment-based 2D LiDAR;
- wall and car-to-car collision detection, plus JAX impulse-based contact;
- occupancy-grid tracks, centerlines, racelines, and Frenet coordinates;
- multi-agent reset strategies, rewards, termination, rendering, and wrappers;
- deterministic seeding, actuator/sensor noise, and domain randomization.

The installed package is `f1tenth_gym` version `1.0.0dev`. Importing the
package registers `f1tenth-v0`; normal use is the namespaced Gymnasium id
`f1tenth_gym:f1tenth-v0`. There is no ROS runtime in this repository despite
the history and branch names.

The active code line has diverged substantially from the old upstream package.
In particular, there is no per-agent `RaceCar` object and no dictionary/YAML
environment config. Agents are rows in shared arrays, and `EnvConfig` is the
only accepted configuration surface.

## Minimal contract

```python
import gymnasium as gym
import numpy as np

from f1tenth_gym.envs.env_config import EnvConfig

env = gym.make(
    "f1tenth_gym:f1tenth-v0",
    config=EnvConfig(render_enabled=False),
)
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(
    np.array([[0.0, 2.0]], dtype=np.float32)
)
env.close()
```

Key API facts:

- Native actions have shape `(num_agents, 2)` and columns
  `[steering, longitudinal]`. Steering is always column zero.
- The default control interprets those columns as target steering angle in
  radians and target speed in m/s. Controllers turn setpoints into steering
  rate and acceleration; commands are not direct state writes.
- `step()` validates shape but does not clip values to `action_space`.
- The default observation is
  `dict[agent_id -> dict[field -> float32 ndarray]]`.
- Default fields are `scan`, `std_state`, `state`, `collision`, `lap_time`,
  `lap_count`, `sim_time`, and `frenet_pose`, subject to LiDAR/Frenet gating.
- `reset()` returns `(obs, info)`; `step()` returns Gymnasium's five-tuple.
- Reset info has lap/time keys and copied per-agent `terminated_agents` status.
  Step info additionally has copied per-agent `collisions` and `progress`
  arrays.
- `terminated` represents collision or lap completion. `truncated` represents
  `max_episode_steps` only.
- The default scalar reward is the simulation timestep (`0.01`), not progress.

Imports are intentionally shallow at package roots. Deep-import configuration,
models, observations, wrappers, and tracks from their defining modules.

## Default behavior at a glance

| Concern | Default |
|---|---|
| Track | `Spielberg`, scale `1.0` |
| Agents | one, with `agent_0` as ego |
| Dynamics | single-track (`ST`), RK4, `0.01 s` step/integrator step |
| Controls | steering-angle and target-speed setpoints |
| LiDAR | enabled, 1080 beams, 270 degrees, 30 m, exact `SEGMENT` backend |
| Collisions | `SEGMENT_CONTACT`, JAX contact on CPU |
| Observations | per-agent `DEFAULT` dictionary |
| Reset | static grid near the start of the raceline |
| Reward | survival reward equal to timestep |
| Episode end | ego collision or one lap; configurable EGO/ANY/ALL reduction; no default step limit |
| Rendering | enabled in config, but no renderer is built until a render mode is selected |

Named tracks are downloaded from `https://api.f1tenth.org/<name>.tar.xz` into
the repository's ignored `maps/` directory when absent. A clean checkout
therefore needs network access for the default environment. Passing a path or
a prebuilt `Track` avoids name resolution; sharing one `Track` between envs
also shares expensive cached geometry.

## Architecture

```text
import f1tenth_gym
  -> gym.register("f1tenth-v0")

gym.make(..., config=EnvConfig)
  -> F110Env                         Gymnasium lifecycle and episode semantics
       -> Track                      map, reference lines, Frenet transforms
       -> F110Simulator              dynamics, state, sensing, contact
            -> SimulationState       struct-of-arrays runtime buffers
            -> action adapters       setpoint/pass-through control conversion
            -> dynamics + integrator per-agent state transition
            -> contact adapter       host/JAX wall and body contact seam
            -> LiDAR backend         raster or segment scan
       -> Observation provider       observation values and spaces
       -> ResetFn                    seeded initial-pose strategy
       -> RenderClock/renderer       optional pacing and pixels
```

### Environment layer

`f1tenth_gym/envs/f110_env.py` owns the public Gymnasium contract:

- accepting and flattening `EnvConfig`;
- resolving the track and constructing all components;
- action/observation spaces;
- reset selection and reset-seed semantics;
- lap counting, progress, reward, termination, and truncation;
- the public `info` dictionaries;
- render pacing, render callbacks, and resource cleanup;
- full reconfiguration through `configure()`.

It must not become a second physics implementation. Feature calculations that
are part of the state transition belong in the simulator or a focused
subsystem; episode semantics and the Gymnasium surface belong here.

Current `F110Env.step()` order is load-bearing:

```text
sim.step(action)
-> increment episode step count and render clock
-> lap/termination check
-> per-agent Frenet progress
-> observation
-> update env.sim_time from sim.state.sim_time
-> truncation
-> copied info
-> reward (CUSTOM rewards therefore see final info)
```

Because observation happens before `env.sim_time` is refreshed, the
observation field `sim_time` is deliberately/currently one step behind
`info["sim_time"]`. This behavior is documented and tested indirectly; change
it only as an explicit API correction.

### Simulator layer

`f1tenth_gym/envs/simulator.py` owns one shared simulation update. Its step
order is:

```text
validate `(N, 2)` action
-> add command noise
-> apply throttle and steering FIFO delays
-> for each agent in Python:
     setpoint conversion -> integration substeps -> yaw wrap -> state mirrors
-> resolve SEGMENT_CONTACT walls and body pairs, if selected
-> update each agent's Frenet state from its corrected CoG pose
-> compute the wall scan and opponent occlusion, then observed LiDAR
   noise/dropout/bias
-> increment simulation time
```

The dynamics loop is per-agent Python, not vectorized. Contact and exact scan
kernels are JAX-backed, but this is still a Gymnasium/NumPy environment, not a
fully jitted environment.

### State and coordinate frames

`SimulationState` is a struct of arrays. Important buffers are:

| Buffer | Shape | Meaning |
|---|---:|---|
| `state` | `(N, state_dim)` | native model state |
| `standard_state` | `(N, 7)` | `[X,Y,delta,speed,yaw,yaw_rate,beta]` |
| `control_input` | `(N, 2)` | delayed commands reaching the model |
| `scans` | `(N, beams)` | observed ranges; internal size is 1 when LiDAR is off |
| `poses` | `(N, 3)` | native `[x,y,yaw]` mirror used by geometry |
| `frenet` | `(N, 3)` | centerline `[s,ey,ephi]` |
| `collisions` | `(N,)` | float32 contact flags |

Buffers are mutated in place, so observations and `info` arrays must be
copies. Never return a live view that can corrupt replay buffers or stored
logs.

Supported model-native poses share one frame:

- `KS` native `x/y` is CoG referenced and has state dimension 5.
- `ST` native `x/y` is CoG referenced and has state dimension 7.
- `standard_state`, derived observations, and Frenet coordinates are also CoG
  referenced. The rear-axle KS equation remains a test oracle only.

`DynamicModel.get_initial_state()` currently attaches `state_dim` and
`control_dim` to the enum member as a side effect. The simulator calls it
before reading `model.control_dim`; do not reorder that construction without
first removing the process-global side effect.

Continuous runtime buffers are float32; lap counters and delay heads use
integer arrays. Dynamics may compute at wider precision, but simulator state
writes cast back to float32 each step. Compare floating time with tolerances
rather than equality.

## Subsystem map

| Path | Responsibility |
|---|---|
| `f1tenth_gym/__init__.py` | Gymnasium registration only |
| `envs/env_config.py` | frozen config tree, rewards, termination, DR validation |
| `envs/f110_env.py` | environment API and episode bookkeeping |
| `envs/simulator.py` | state transition, LiDAR, collisions, contact wiring |
| `envs/state.py` | array allocation/reset and steering-delay ring buffer |
| `envs/action.py` | action enums, setpoint conversion, action spaces |
| `envs/integrators.py` | plain-Python Euler and RK4 |
| `envs/dynamic_models/` | KS/ST kernels, standardizers, vehicle parameters; legacy MB data/code pending removal |
| `envs/observation/` | field vocabulary, providers, spaces, presets |
| `envs/lidar/` | LiDAR config plus mutable raster/exact-segment adapters |
| `envs/collision_models.py` | collision enum and collision-body vertices |
| `envs/contact/` | JAX manifolds/solvers and the NumPy/JAX adapter |
| `f1tenth_gym/jax/` | pure dynamics, state, track/reset tables, clean sensing and wall/pair contact |
| `envs/track/track.py` | map/reference-line loading and Frenet transforms |
| `envs/track/walls.py` | oriented wall extraction from occupancy maps |
| `envs/track/budget.py` | exact allocation budgets and guards |
| `envs/track/accel.py` | contact tile index and cache |
| `envs/track/ray_tiles.py` | scan tile index and cache |
| `envs/reset/` | reset strategy registry and samplers |
| `envs/rendering/` | PyQt6/OpenGL renderer, objects, callbacks |
| `envs/wrappers.py` | single-agent and observation-delay wrappers |
| `examples/` | waypoint following, video, telemetry, synthetic tracks |
| `tests/` | 568 collected tests across 42 `test_*.py` files |
| `docs/` | Sphinx user documentation plus behavioral measurements |

## Configuration model

All config sections are frozen dataclasses. Use `with_updates()`/dataclass
replacement instead of mutation. Validation lives in each section's
`__post_init__`; cross-section constraints live in `EnvConfig.__post_init__`.

The tree is:

- `EnvConfig`
  - `ControlConfig`
  - `SimulationConfig`
  - `ObservationConfig`
  - `ResetConfig`
  - `LiDARConfig`
  - `ContactConfig`
  - `RenderConfig`
  - `TerminationConfig`
  - `RewardConfig`
  - `DomainRandomizationConfig`

`configure(new_config)` reconstructs track, simulator, spaces, reset strategy,
render clock, and renderer. It also re-arms `EnvConfig.seed` for the next
unseeded reset. `update_params()` is the narrower shared-vehicle update; an
agent-specific index is unsupported.

Domain randomization draws one shared `VehicleParameters` object per episode,
not one per agent. Gymnasium spaces remain fixed and are built from the widest
configured parameter extrema.

## Vehicle-parameter ABI

`VehicleParameters` has a positional float32 wire format consumed by numba
dynamics kernels. `PARAMETER_ORDER` is exactly dataclass declaration order and
currently has 88 entries. KS/ST hard-code indices 0-17; the retained legacy MB
kernel reads later slots but is not selectable by `EnvConfig`.

This is an ABI:

- append-only changes are possible when kernels do not need the new field;
- inserting, deleting, or reordering a field silently rewires positional
  kernel reads;
- update kernel indices, presets, docs, and the literal ABI test together when
  intentionally changing it;
- never add a per-model slice unless the whole API is deliberately redesigned.

`tests/test_vehicle_params_abi.py` is the gate. Small-scale presets leave the
MB-only block as NaN; `FULLSCALE_VEHICLE_PARAMETERS` preserves the measured
values as data but does not make MB selectable.

## LiDAR and collision/contact architecture

LiDAR backends implement a small shared adapter surface: construction with
beam/range geometry, `set_map(track, scale)`, `scan(pose, rng)`, and
`get_increment()`.

- `RASTER` sphere-traces a distance transform and has cell-center range bias.
- `SEGMENT` intersects oriented wall segments analytically through a JAX
  kernel and is the default.
- Gaussian noise, systematic per-beam bias, dropout, and range clipping affect
  observed scans only. LiDAR is not a collision response mechanism.
- Other vehicles shorten scans through opponent body ray casting after the
  wall scan.

Collision modes are behaviorally different:

| Mode | Walls | Other cars | Response |
|---|---|---|---|
| `NONE` | off | off | no flags |
| `SEGMENT_CONTACT` | wall segments | SAT/JAX pair manifold | impulse and position correction |

`SEGMENT_CONTACT` is the sole production collision response; `NONE` explicitly
disables detection and response. GJK exists only as a test oracle for SAT.

The established portable JAX geometry seam is intentionally narrow:

- `envs/contact/kernels.py`
- `envs/contact/solver.py`
- `f1tenth_gym/jax/lidar_kernels.py`

The functional seam under `f1tenth_gym/jax/` currently owns traced vehicle and
episode parameters, KS/ST dynamics, controllers, actuator noise/FIFOs and
free-flight `lax.scan` rollouts. Host preprocessing produces fixed-shape spline,
wall, tile and RL-reset tables; exact-shape buckets are the default for
heterogeneous maps, with shared `Track` objects stored once and referenced by
index. Clean exact scans use masked ray-tile candidates, the current LiDAR
mounting calculation and simultaneous all-edge opponent occlusion. Runtime
``range_max`` must not exceed the ray table's preprocessed reach. Functional
wall contact vmaps the existing pure manifolds/Jacobi solver across agents and
converts KS/ST native state to/from rigid-body velocity without host marshalling.
It preserves the current host oracle's CoG tile lookup and its discarded
speculative-only clamp; change either only as an explicit behavior correction.
Fixed-capacity pair tables enumerate canonical unordered body pairs with masked
padding. Pair manifolds share one state snapshot, and every global Jacobi sweep
scatter-adds equal-and-opposite impulses/corrections before one native-state
write per body. ``resolve_contacts`` preserves wall-then-pair ordering and unions
fresh per-step events without freezing or latching agents.
`DynamicsConfig` contains structural choices; `EpisodeParams` contains values
that vary under environment-level `vmap` without recompilation. The functional
step is still free-flight-only and does not yet integrate sensing/contact or
episode semantics.

Keep these pure JAX/array math, fixed-shape, jittable/vmappable, free of Gym
and package-local imports, and free of NumPy marshalling. Host conversion,
device selection, map preprocessing, and caches belong in adapters. The
contract is enforced by `tests/test_migration_seam.py`.

JAX initializes multithreaded runtime state. Async Gymnasium vector workers
must use a spawn context with the segment backend; forking after JAX
initialization can deadlock.

## Tracks, resets, and laps

A `Track` combines a ROS-style binary occupancy map with two closed
`Raceline` objects:

- `centerline` defines the simulator's Frenet frame, progress, and default lap
  arclength;
- `raceline` carries the optimized driving line and velocity profile;
- default RL resets spawn on the raceline, so reset `ey` is normally non-zero;
- `ResetConfig(reference_line=ReferenceLine.CENTERLINE)` gives centerline
  spawns instead.

Map image rows represent world Y and columns represent world X. Images are
flipped vertically on load. In `occupancy_map`, `0` is occupied and `255` is
free. Wall extraction retains the unthresholded grayscale image when possible
to recover subpixel contours and outward normals.

Track geometry is cached on `Track` instances. Cache keys must include every
input that changes geometry, scale, body reach, tolerance, or allocation cap.
Both contact and scan indexes have explicit projected-allocation guards; do
not bypass them with an unbudgeted table.

Frenet projection uses a local search centered on each agent's previous `s`
during stepping. Teleport/reset uses a global search. A one-step translation
larger than the search window can select the wrong nearby part of a track.

Lap modes are Frenet crossing and winding angle. The default
`count_partial_first_lap=True` counts the first finish-line crossing; setting
it false treats the spawn-to-line segment as an out lap.

## Observations, rewards, and wrappers

`envs/observation/fields.py` is the single field vocabulary. Adding a packaged
field generally requires all of:

1. add the name to `BASE_FIELDS` or `DERIVED_FIELDS`;
2. define a finite Gymnasium space in `full.field_space()`;
3. produce a copied float32 value in `FullObservation.observe()`;
4. decide whether `RawObservation` needs a batched key;
5. update presets only when the new field belongs in a stable preset;
6. test value, shape, dtype, bounds, gating, and aliasing;
7. update `docs/observations.rst` and relevant RL examples.

`DIRECT` is the exceptional observation: a flat dictionary of agent-batched
arrays rather than per-agent dictionaries. `FRENET_DYNAMIC_STATE` is a legacy
name and does not contain `frenet_pose`.

Progress is wrap-corrected centerline arclength per agent. The built-in scalar
reward always uses the ego agent. A custom reward is called after final step
info is assembled. Keep reward signals in `info` even when built-in reward
modes do not consume them.

`SingleAgentWrapper` requires exactly one agent, unwraps `agent_0`, and changes
actions from `(1, 2)` to `(2,)`. Pair it with Gymnasium's
`FlattenObservation` for conventional RL libraries. `ObservationDelayWrapper`
delays only observations; reward, termination, truncation, and info stay
current.

## Rendering

The only concrete renderer is PyQt6 + pyqtgraph/OpenGL. Pixel modes require an
X display; use `xvfb-run` in headless environments. The renderer consumes a
copy of DEFAULT-vocabulary observations even when the policy uses a different
observation preset.

Rendering has two separate clocks:

- `RenderConfig.render_fps` controls human redraws or distinct RGB frames;
- `metadata["render_fps"]` is `round(1 / timestep)` for Gymnasium
  `RecordVideo`, which captures once per step.

Human modes may pace to a real-time factor. RGB-array mode never sleeps and
caches frames between configured render instants.

## Where to extend a feature

| Change | Primary seams | Minimum focused tests/docs |
|---|---|---|
| Config knob | owning config dataclass, cross-validation, `_apply_env_config`, consumer | `test_env_config.py`, subsystem test, `configuration.rst` |
| Action/control mode | action enum/adapters/space, simulator dispatch | `test_action.py`, `test_actuation.py`, `actions.rst` |
| Observation field/preset | field vocabulary, space, provider, optional raw key | `test_observation.py`, `observations.rst` |
| Dynamics model | model enum/frame/state init/standardizer, kernel, params | `test_dynamics.py`, ABI tests, `dynamics.rst` |
| Integrator behavior | `integrators.py`, simulator substep validation | `test_integrators.py`, `test_env_config.py` |
| LiDAR backend | config enum, `_make_scan_simulator`, backend adapter | scan tests, migration seam, observations/config docs |
| Collision mode | collision enum, `_build_contact()`, contact step dispatch | collision/contact behavior tests, config docs |
| Contact physics | pure kernels/solver, adapter, indexes/budgets | kernel, solver, body/mode, migration tests |
| Track format/geometry | loader, `TrackSpec`, walls/index caches | track/wall/index tests, `tracks.rst` |
| Reset strategy | enum, `_RESET_BUILDERS`, sampler, config applicability | `test_reset.py`, `test_env_config.py` |
| Reward/termination | config plus `F110Env` episode bookkeeping | reward/termination/lap tests, `rl.rst` |
| Renderer/callback | render config, factory/backend, render observations | `test_renderer.py`, `rendering.rst` |
| Wrapper/API adapter | `wrappers.py` without changing native env | wrapper/env tests, `rl.rst` |

Prefer explicit enum dispatch with a final error over implicit fallthrough.
Validate incompatible combinations once at `EnvConfig` construction when the
constraint spans subsystems.

## Tests and validation

The current tree collects 568 tests across 42 `test_*.py` files. Most tests use
`unittest.TestCase` but run through pytest. Tests cover public behavior and
low-level numerical contracts, including JIT/vmap/gradient properties,
allocation guards, cache invalidation, observation aliasing, vector envs,
rendering, and the vehicle ABI.

Use the existing environment without syncing when dependencies are present:

```bash
env -u PYTHONPATH UV_CACHE_DIR=/tmp/f1tenth-gym-uv-cache \
  uv run --no-sync pytest
```

Unset `PYTHONPATH` because ROS installations can inject incompatible pytest
plugins. Set a writable `UV_CACHE_DIR` in restricted environments. For the
full rendering path on a headless machine:

```bash
xvfb-run -a env -u PYTHONPATH UV_CACHE_DIR=/tmp/f1tenth-gym-uv-cache \
  uv run --no-sync pytest
```

Focused tests are preferred during iteration, followed by the full suite when
behavior or shared infrastructure changes.

Documentation CI has three meaningful gates:

```bash
UV_CACHE_DIR=/tmp/f1tenth-gym-uv-cache uv run --no-sync \
  python -m sphinx -W -b html docs docs/_build/html
UV_CACHE_DIR=/tmp/f1tenth-gym-uv-cache uv run --no-sync \
  python -m sphinx -b doctest docs docs/_build/doctest
UV_CACHE_DIR=/tmp/f1tenth-gym-uv-cache uv run --no-sync \
  python docs/style_lint.py --strict
```

`flake8` is advisory in CI. Scope it to live source; a bare `flake8 .` can
walk the large local virtualenv:

```bash
uv run --no-sync flake8 f1tenth_gym tests examples --statistics
```

`uv sync` includes the `dev`, `examples`, and `gpu` default groups. On a CPU
machine use `uv sync --no-group gpu`. `uv.lock` is intentionally ignored in
the reviewed baseline, so installs are resolved rather than lock-replayed.

## Documentation map

- `docs/installation.rst`: editable install, Docker, display and ROS-path traps
- `docs/quickstart.rst`: complete episode lifecycle
- `docs/configuration.rst`: exhaustive config field reference
- `docs/actions.rst`: action ordering, limits, setpoint controllers
- `docs/observations.rst`: observation types, fields, shapes, bounds
- `docs/dynamics.rst`: models, state layouts, frame conversion, integration
- `docs/tracks.rst`: maps, reference lines, Frenet semantics, synthetic tracks
- `docs/rl.rst`: wrappers, flattening, reward modes, vector environments
- `docs/sim2real.rst`: delays, noise, LiDAR offsets, domain randomization
- `docs/reproducibility.rst`: reset seed tree and stream-shifting choices
- `docs/rendering.rst`: render modes, clocks, video, callbacks
- `docs/examples.rst`: shipped examples and optional dependencies
- `docs/api/index.rst`: autodoc surface
- `docs/contact_measurements.md`: contact design and benchmark record
- `docs/scan_measurements.md`: segment-vs-raster scan measurements

The Sphinx prose has protected canonical facts and a strict style linter in
`docs/_style/`. When changing a documented contract, update the canonical
page rather than duplicating the same full explanation across pages.

## Known rough edges at the reviewed baseline

These are current-code observations, not permission to fix them as part of an
unrelated feature:

- Per-agent vehicle parameter updates are not supported; all agents share one
  `VehicleParameters` instance.
- `MB` is unsupported. KS contact cannot represent diagonal sliding faithfully
  because KS has no slip angle or yaw-rate state.
- `MAP_RANDOM_STATIC` samples `np.where(mask)` as `(x, y)`, although NumPy
  returns `(row_y, col_x)`. It is unreliable on non-square maps and samples
  general map free space rather than track-valid poses.
- `Track.from_raceline_file()` computes a synthetic-map origin and then
  overwrites its `TrackSpec` with origin `(0, 0, 0)`, so map/world alignment is
  suspect for racelines away from the origin.
- `Track.save_centerline()` writes a leading UUID comment that its current
  centerline loader does not accept as the header row.
- Reset pose/state shape checks in `F110Env.reset()` use `assert`, which is
  stripped by optimized Python.
- Frenet `Track.s_guess` is stored on the shared `Track`; simulator stepping
  passes each agent's own explicit previous `s`, but direct callers using the
  implicit guess share the last projection.
- Named-map download writes into the package source layout and has no checked
  checksum. Consider path/`Track` injection for packaged or offline use.
- Rendering requires X/GL, and async vectorization with the default JAX scan
  backend requires process spawning.

Reproduce and add a regression test before turning any item into a fix.

## Workspace and change hygiene

- Preserve existing user changes. Inspect `git status` before editing.
- `maps/`, `.venv/`, `.jax_cache/`, Sphinx builds, and local agent planning
  files are ignored generated/local state, not product source.
- Do not use ignored `CLAUDE.md`, `ISSUES.md`, or planning files as canonical;
  they may contain valuable analysis but can lag `HEAD`. Verify claims in code,
  tests, and published docs.
- Keep feature work focused. Do not opportunistically repair a known rough edge
  unless it is required by the requested feature or explicitly authorized.
- New behavior needs tests at the narrowest numerical layer and at the public
  environment seam when users can observe it.
- Update user docs whenever config, shapes, defaults, timing, frames, or public
  exceptions change.
