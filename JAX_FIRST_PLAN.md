# JAX-First f1tenth_gym

A framework-neutral, pure-JAX simulation core that can run thousands of
environments in one compiled accelerator program. The existing Gymnasium API
remains the compatibility surface for SBX/SB3 and current users; the shipped
device-batched API does not make a training framework the simulation contract.
An official-JaxMARL adapter remains a possible future consumer, not a v1
dependency.

Working branch `dev-features` · merge source `jax/main` exactly once ·
`jax-backend` is read-only research material · supported dynamics are KS-CoG
and ST · `SEGMENT_CONTACT` is the sole production collision response.

---

## 00 · Goal and performance hypothesis

The end state is:

```text
host-side configuration and track preprocessing
        -> fixed-shape JAX arrays
        -> pure reset/step functions
             -> Gymnasium adapter       (SBX/SB3/current users)
             -> device-batched adapter  (vmap + scan training)
             -> optional JaxMARL adapter (deferred beyond v1)
```

The pure core owns state transition, sensing, contact, progress and per-agent
events. It does **not** download maps, call NumPy/SciPy, render, auto-reset, or
inherit from a training framework. Adapters own API-specific observations,
termination tuples, auto-reset and host/device conversion.

`docs/contact_measurements.md` correctly finds that GPU contact loses to CPU
contact for one environment and crosses over at roughly 52 bodies in one wide
launch. Environment batching may provide that width: 1024 two-agent
environments expose 2048 independent bodies to XLA. This is a hypothesis about
the final lowering, not proof that the existing benchmark transfers unchanged.

Consequences:

- Keep the NumPy/Gymnasium contact path on CPU during migration.
- Do not change a public contact-device default from the crossover estimate.
- Re-measure the final `jit(vmap(scan(...)))` program with outputs consumed,
  warm-up excluded, `block_until_ready()`, compile time and peak memory reported.
- Treat the environment batch as the primary throughput axis; agents remain a
  smaller fixed axis inside each environment.

---

## 01 · Attribution and source policy

Hongrui **"Billy"** Zheng (`billyzheng.bz@gmail.com`) authored the JAX fork.
The original signed history is reachable from `jax/main`; it must enter
`dev-features` through one real two-parent merge.

At the reviewed baseline (`dev-features` at `f3ff709`):

- `jax/main` is `1b4eb3f`;
- the merge base of `dev-features` and `jax/main` is `5a301bd`;
- `.mailmap` has landed as `282d8f8`;
- the rewritten JAX parent inside `jax-backend`, `6780536`, has the same tree as
  `jax/main`, but different unsigned commit objects;
- the **tip** of `jax-backend` is not content-identical to `jax/main`.

Before implementation, fetch and re-run those checks rather than assuming the
snapshot is still current:

```bash
git fetch jax
git merge-base dev-features jax/main
git rev-parse jax/main
git diff --stat 6780536 jax/main
git rev-list --left-right --count jax/main...jax-backend
```

Merge policy:

```bash
git merge --no-ff jax/main
```

- Merge `jax/main` exactly once. Never merge both JAX histories.
- Never squash, rebase or replace the merge with a file copy. If this lands by
  pull request, the repository must not squash the PR.
- Resolve the merge against current `dev-features`; do not assume the old
  conflict count or merge base in earlier notes.
- Preserve the original commits even where modern integration later rewrites
  or deletes their files. The DAG preserves authorship; live `git blame` is not
  promised for code that is substantively reimplemented.

### `jax-backend` is inspiration only

Do not merge, cherry-pick, rebase, replay or treat commits from `jax-backend` as
implementation inputs. That branch targeted an older gym and many of its
decisions are stale.

It may be inspected for:

- previously discovered failure modes and benchmark methodology;
- possible algorithms, such as traced vehicle parameters;
- tests or invariants worth reproducing against current `HEAD`.

Every adopted idea is re-derived in current source, implemented afresh and
gated by current tests. In particular, commit `2f36b04` is a useful example of
the static/dynamic parameter split, not code to transplant.

---

## 02 · Architecture and public contracts

### Functional core

The internal API is explicit and has no automatic reset:

```text
reset_core(key, static, episode_params) -> (observation_data, state)

step_core(key, state, actions, static, episode_params)
    -> (observation_data, next_state, rewards, events, metrics)
```

The concrete names may change, but these properties do not:

- inputs and outputs are JAX pytrees;
- every leaf has a static shape under `jit`, `vmap` and `lax.scan`;
- arrays stay on device throughout a compiled rollout;
- the core never auto-resets or converts to NumPy;
- randomness comes only from explicit keys;
- state is immutable from the caller's perspective;
- `events` contains per-agent facts, not framework-specific done semantics.

`observation_data` is a canonical internal vocabulary. Adapters may package it
as the current per-agent Gymnasium dictionaries, flat policy arrays, or JaxMARL
agent dictionaries without recomputing physics.

### Static configuration versus episode parameters

Fields that change shapes, dispatch or compiled topology are static, including:

- number of agents and beams;
- dynamics and integrator choice;
- enabled observation fields;
- contact solver iteration counts;
- padded track-table dimensions.

Per-episode physical values are traced leaves, including vehicle parameters,
noise samples and domain-randomization draws. They live in an explicit pytree
or state, not in static `self` and not behind `static_argnums`.

Required properties:

- `vmap` can run different vehicle parameters in each environment;
- a new DR draw does not compile a new executable;
- gradients with respect to supported physical parameters are finite in
  free-flight dynamics;
- incompatible structural changes fail before compilation.

### Gymnasium and SBX/SB3

SBX follows the SB3 API and consumes Gymnasium or SB3 `VecEnv` environments. It
does not consume a JaxMARL `MultiAgentEnv` directly.

The existing `F110Env` remains the native Gymnasium contract and the NumPy
correctness reference during migration. A JAX-backed Gymnasium adapter must:

- implement the same five-tuple and seeding rules;
- expose finite Gymnasium spaces;
- pass Gymnasium and SB3 environment checks for the supported observation mode;
- work with `SingleAgentWrapper`/flattening for conventional single-agent SBX;
- preserve `terminated` versus `truncated` and the documented `info` values.

It also preserves native action shape/order, controller semantics and the fact
that `step()` validates but does not clip an out-of-space action. Gym-facing
normalized `[-1, 1]` actions belong in a wrapper; device-native policies use a
pure scaling helper inside their compiled rollout.

A normal Gymnasium adapter introduces Python dispatch and host/device transfer.
It proves ecosystem compatibility, not maximum simulator throughput. A separate
device-batched adapter or native JAX training loop is required to keep thousands
of environments inside one compiled rollout. If an SB3 `VecEnv` adapter is
added, its performance must be measured rather than assumed.

SBX/SB3 are single-policy APIs. Supported multi-car training modes must be
explicit:

- ego policy with externally controlled opponents;
- centralized/joint policy through a flattened joint action/observation;
- genuine multi-policy MARL through a separate adapter.

### Optional JaxMARL adapter

JaxMARL is an optional consumer of the functional core, not its base class.

- Depend on the official package in an optional extra if the adapter ships.
- Do not retain copied `MultiAgentEnv` or spaces as a silent vendored fork.
- The adapter implements key-first reset/step, agent dictionaries and its own
  auto-reset behavior.
- JaxNav may inform continuous-control conventions, but it is not an
  inheritance target.
- JaxMARL-specific shape constraints are adapter tests, not core architecture.

This boundary also keeps third-party Apache-2.0 code and its attribution out of
the MIT package unless the official dependency is deliberately enabled.

**v1 decision:** no JaxMARL adapter is shipped. The official API currently
translates arrays to agent-keyed dictionaries and auto-resets its public
``step`` on ``done["__all__"]``. The dense native batch already supplies the
current multi-agent rollout contract, preserves terminal observations
separately from selective reset observations and passes a complete PPO gate.
Adding the external environment/algorithm dependency stack without a concrete
multi-policy consumer would not improve that core. Revisit this decision for a
specific IPPO/MAPPO integration, depend on the official package and run the
continuous multi-agent smoke gate described below; do not vendor its base
classes or spaces.

### Collision response and episode termination

Collision detection/response and episode termination are independent:

```text
contact detected
-> solve impulse and position correction
-> update all vehicle states
-> record per-agent collision events/status
-> reduce terminal status with EGO / ANY / ALL
-> either continue the whole simulation or terminate the whole simulation
```

There is no permanently frozen collided-agent state. When collision is not a
stopping condition, every agent continues under the corrected physics and keeps
receiving actions. When collision is a stopping condition and the configured
reduction fires, the entire environment terminates on that transition.

The core maintains per-agent event/status arrays. A single episode-end policy
reduces relevant status with:

- `EGO`: stop when the ego agent satisfies the terminal condition;
- `ANY`: stop when any agent satisfies it;
- `ALL`: stop once every agent has satisfied it.

Events such as contact are per-step facts. Terminal status may latch the fact
that an agent has already satisfied a condition, which makes `ALL` meaningful
without freezing that vehicle or requiring all agents to collide/cross a line
on the same step.

Gymnasium returns one global `terminated`. The JaxMARL adapter initially sets
every `done[agent]` and `done["__all__"]` to the same global result; per-agent
collision/lap facts remain available in state/info. Partial agent removal and
absorbing-agent semantics require a separate future design.

Timeout remains truncation in Gymnasium. APIs without a distinct truncation
slot carry the distinction in metrics/info.

---

## 03 · What comes from each code line

`jax/main` is a complete historical JAX environment and training example, not
the behavior specification for the modern gym. Use it for provenance and as a
starting implementation only where current tests confirm behavior.

| Capability | Current `dev-features` | `jax/main` | JAX-first action |
|---|---|---|---|
| KS / ST dynamics | NumPy/numba reference | JAX variants | revalidate and modernize |
| Integrators | Python Euler/RK4 + substeps | JAX Euler/RK4 | revalidate and adopt |
| State/API | mutable Gymnasium | functional JAX/JaxMARL-like | build framework-neutral core |
| Vehicle parameters | 88-slot NumPy ABI | static flat `Param` | traced episode pytree |
| LiDAR | exact segment default | raster ray march + `jax-pf` | port current segment path |
| Contact | segment impulse response | SAT flags only | port current response, redesign pairs |
| Tracks | current map/Frenet semantics | older track stack | preprocess current Track |
| Resets/seeding | strategy registry/seed tree | one sampler | port supported current contract |
| Observations | presets + finite bounds | one flat layout | canonical data + adapters |
| Rewards/termination/info | current configurable contract | older reduced contract | port deliberately |
| DR/delays/noise | supported | incomplete | traced, fixed-shape implementation |
| Training examples | conventional Gym examples | PPO/MPPI + weights | reuse patterns, retrain policies |
| MB | present but outside JAX/contact support | absent | retire from supported v1 surface |

Checked-in policies from `jax/main` are smoke-test artifacts only. Observation,
control, reward and physics changes can make their shapes or learned behavior
invalid; they are not acceptance oracles.

---

## 04 · Implementation phases

Every phase lands with the applicable source tests, documentation gates and one
repository-wide `pytest` command green. A raw broken merge is never an
intermediate deliverable.

### Phase 0 — contract and compatibility preparation

- `.mailmap` remains its own landed commit (`282d8f8`).
- Re-check `jax/main`, merge base, signatures and repository merge settings.
- Record public names and import paths:
  - `F110Env`: existing Gymnasium/NumPy reference;
  - `f1tenth_gym.jax`: functional core and JAX adapters;
  - `JaxF110Env`: Gymnasium adapter if a distinct public class is needed.
- Spike imported dynamics and required pytree dependencies on the current JAX
  floor (`>=0.11.1` at this snapshot). Do not downgrade the modern kernels to
  the old `<0.8` pin merely to make the merge compile.
- Decide whether Flax/Chex are necessary or whether JAX-native registered
  dataclasses can keep the core dependency surface smaller.
- Add current-HEAD regression tests for:
  - KS rear-axle versus CoG transform equivalence;
  - raw KS state-frame migration;
  - ST reverse motion and finite zero-speed gradients;
  - collision response independent from episode termination;
  - `EGO`/`ANY`/`ALL` whole-environment termination.

*Done when:* the design tests exist, current suite is green, refs are recorded
and the JAX dependency direction is decided.

#### Phase 0 implementation record — 2026-08-28

- Preparation started from `dev-features` at
  `f3ff709546f605dea53298efc5b2d1dc5de9a522`.
- `jax/main` resolves to `1b4eb3f5161756bb925987753b965b549097742f`;
  its merge base with the preparation head is
  `5a301bd0ae1ceaf7dec653e7549c8d099db58a6b`.
- The branches contain 374 current-line-only and 170 JAX-line-only commits at
  this point. `merge.ff` and `merge.conflictstyle` are unset; local
  `commit.gpgsign` is true.
- The five sampled `jax/main` tips report Git signature status `N`. Preserve
  the incoming commit objects through a real merge, but do not claim that those
  sampled commits carry verifiable signatures.
- The installed JAX is 0.11.1. A local spike verified
  `jax.tree_util.register_dataclass`, `jit`, per-environment parameter `vmap`,
  reverse velocity and a finite zero-speed gradient together. The core will use
  JAX-native registered dataclasses; Flax and Chex are not required base
  dependencies.
- Current-HEAD regression coverage now pins the KS rear-axle/CoG derivative
  transform, the existing raw-state frame, ST reverse and zero-speed behavior,
  collision/termination separation, and latched `EGO`/`ANY`/`ALL` reduction.

### Phase 1 — merge `jax/main` exactly once

Create one real two-parent merge. Resolve shared project files in favor of the
current build, docs and public behavior unless the plan explicitly adopts the
JAX side. The merge resolution must leave one installable project and a green
suite; temporary source namespaces may coexist until Phase 2.

Do not import commits from `jax-backend`. Similar-looking solutions are
reimplemented against current code and tests.

*Done when:* both parents are in the DAG, original signatures remain reachable,
the project builds on the chosen JAX version, and all enabled tests pass.

#### Phase 1 implementation record — 2026-08-28

- Merge commit `a9cc351c0f25b4dcaf73aa92c57732c67f7a3d12` has parents
  `d73b2379f81c94357f630a9b8d717b93b8885e5d` and the original
  `jax/main` tip `1b4eb3f5161756bb925987753b965b549097742f`.
- No `jax-backend` commit was merged, cherry-picked or replayed.
- Current packaging, dependencies, docs and shared runtime files won merge
  resolution. The standalone JAX tree was retained only long enough to perform
  history-preserving Phase 2 moves.

### Phase 2 — one package and one supported surface

- Move useful JAX implementation into `f1tenth_gym.jax` using history-preserving
  moves where the source remains applicable.
- Remove the standalone `f1tenth_gym_jax` package after imports/tests/examples
  have migrated.
- Remove vendored JaxMARL base/spaces; adapters use either internal types or an
  explicit official optional dependency.
- Retire `LoopCounterMode.TOGGLE`.
- Make `SEGMENT_CONTACT` the only production collision response; retain `NONE`
  as the explicit no-collision option.
- Remove `BOUNDING_BOX` and freeze-on-collision `LIDAR_SCAN` production modes.
  Keep a small GJK implementation only in tests as an independent SAT oracle.
- Retire `DynamicModel.MB` from the supported v1 surface and remove its dynamics
  dispatch/package/tests/docs. If release policy requires a transition, the
  temporary enum/config path raises a named unsupported-model error before its
  final removal.
- Preserve hard-won full-scale parameter values separately from active dynamics.
  Do not expose MB-only DR knobs as silent no-ops.
- Promote the CoG-referenced KS to production and remove `PoseReference`/
  `_cog_offset` only after state, reset, observations, collision, LiDAR,
  rendering and docs all agree on CoG. Keep the rear-axle CommonRoad kernel as
  a test-only oracle.
- Keep RASTER temporarily because it is the differential oracle for the scan
  inherited from `jax/main`; remove both together in Phase 4.

*Done when:* there is one package, supported models are KS-CoG/ST, production
collision is SEGMENT_CONTACT/NONE, and all public frame changes are documented.

#### Phase 2 implementation record — 2026-08-28

- Applicable dynamics and integration history moved into `f1tenth_gym.jax` and
  was rewritten for JAX 0.11.1. The old package, vendored JaxMARL surface,
  incoming examples and stale policy artifacts were removed.
- The functional seam now provides traced `DynamicsParams`, KS-CoG, ST, Euler,
  RK4 and fixed integration substeps. Tests cover NumPy parity, `jit`, `vmap`,
  parameter gradients, reverse motion and finite zero-speed derivatives.
- Production collision modes are now exactly `NONE` and `SEGMENT_CONTACT`.
  GJK is a test-only SAT oracle; LiDAR no longer adjudicates or freezes contact.
- KS raw and standardized states are CoG-referenced. `PoseReference`, CoG-offset
  conversion and collision rewind/halt scaffolding were removed.
- `DynamicModel.MB` remains only as a transitional enum and retained research
  implementation; `EnvConfig` rejects it unconditionally, and rejects DR over
  its inactive parameter block. `LoopCounterMode.TOGGLE` was removed.

### Phase 3 — functional dynamics, state, parameters and tracks

- Implement the framework-neutral reset/step core.
- Port KS-CoG, ST, Euler, RK4 and integration substeps against current numerical
  references.
- Split static configuration from traced episode parameters.
- Port action adapters/controllers, delays and command noise without static
  per-episode values.
- Preprocess the current `Track` on the host into fixed-shape device tables:
  spline coefficients, reference-line data, walls, tile indexes and masks.
- Implement reset strategies as pure key-driven gathers/sampling over those
  tables; map loading/download/extraction remains outside JIT.
- Define shared-map batching: unique maps are stored once and environments carry
  indexes. Compare global padding against bucketing batches by map shape before
  committing to a single padded maximum.

*Done when:* dynamics and rollouts match current NumPy behavior to declared
tolerances; `jit`, `vmap`, `lax.scan`, `jax.eval_shape` and DR-over-parameters
tests pass without recompilation per episode.

#### Phase 3 implementation record — 2026-08-28

- `DynamicsConfig` separates hashable structural choices from traced
  `EpisodeParams`; immutable `DynamicsState` carries native model state,
  delayed commands, fixed-shape FIFO buffers/heads and simulation time.
- Pure action adapters match the current target-speed controller, steering P
  controller/relay hatch, direct modes and `[steering, longitudinal]` ordering.
- `step_dynamics` applies explicit-key command noise, steering/throttle delays,
  controller conversion and vmapped integration. `rollout_dynamics` composes it
  through `lax.scan` without auto-reset or host conversion.
- Differential tests cover KS/ST host-simulator rollouts, FIFO order, key
  replay, `jit`, per-environment-parameter `vmap`, `eval_shape` and gradients.
- Host preprocessing now emits fixed-shape spline, wall, contact-tile, ray-tile
  and reset-candidate pytrees with explicit masks. Pure global Frenet transforms
  and key-driven grid/all-track reset sampling run under `jit`/`vmap`.
- Reused `Track` objects are stored once and referenced by map index. Different
  shapes default to exact-shape buckets; a layout report quantifies global
  padding overhead. `MAP_RANDOM_STATIC` remains explicitly outside the device
  surface pending a decision on its known host row/column bug.

### Phase 4 — exact LiDAR and contact

- Replace `jax-pf`/raster ray marching with the current segment kernels and
  fixed-shape ray-tile candidates.
- Port LiDAR mounting transforms and opponent occlusion. Scans remain sensors;
  segment wall/body geometry drives collision independently, while
  noise/bias/dropout affect observed ranges only.
- Remove RASTER, EDT and the theta LUT only after differential tests pass.
  There is no live `jax-pf` dependency at the start of this phase; keep it out
  of the built artifacts.
- Port wall contact manifolds and Jacobi impulse/position solver without host
  marshalling in the compiled path.
- Redesign car-to-car contact as a simultaneous fixed-pair Jacobi update:
  precomputed pair table, masks, accumulated per-body impulses/corrections and
  no pair-order-dependent state mutation.
- Implement per-step collision events without a persistent freeze latch.

*Done when:* scan/contact scenarios match current behavior to tolerance,
opponent occlusion is covered, pair order does not change the result, momentum
tests pass and the compiled batched path contains no NumPy conversion.

#### Phase 4a implementation record — 2026-08-28

- The owning ray/segment kernel now lives under ``f1tenth_gym.jax``; the old
  deep environment module is a compatibility import for the mutable adapter.
- Clean scans gather masked fixed-shape ray-tile candidates, preserve the
  current CoG-pose-to-LiDAR mounting calculation, and intersect every beam with
  all opponent body edges without branch-heavy angular culling.
- Collision-body geometry is traced separately from dynamics and preserves the
  configured rear-axle-to-body-centre conversion in the common CoG frame.
- Host construction rejects a scan range beyond the ray table's reach. Tests
  cover wall and opponent parity, empty/padded geometry, one-beam sweeps,
  pair-order invariance, ``jit``, environment ``vmap``, ``eval_shape`` and
  finite pose gradients.
- Noise, fixed episode bias and dropout remain in Phase 5. The host RASTER
  backend was retained through the scan/contact differential gate and removed
  in Phase 4d.

#### Phase 4b implementation record — 2026-08-28

- Functional wall contact gathers masked contact-tile candidates, vmaps the
  established pure manifold/gap kernels and runs the fixed-sweep Jacobi impulse
  and position solver without NumPy conversion or device pinning.
- Shared traced body geometry and KS/ST rigid-body conversions preserve the
  collision-body offset, torque about CoG, exact ST velocity writeback and the
  documented KS course projection.
- Collision outputs are fresh boolean per-step events. A prior event never
  freezes an agent or latches into the next call.
- Differential tests cover complete KS/ST dynamics-plus-contact steps against
  the live simulator, empty/padded candidates, ``jit``, environment ``vmap``,
  ``eval_shape`` and finite free-space gradients.
- Two oracle quirks are explicit and tested: broad-phase lookup remains at CoG,
  and a speculative clamp without a penetrating manifold is discarded. Pair
  contact remains for Phase 4c's simultaneous global Jacobi redesign.

#### Phase 4c implementation record — 2026-08-28

- Fixed-capacity ``PairTable`` arrays enumerate every canonical unordered
  vehicle pair and use masks for padding. Host preprocessing validates safe
  indexes for every slot plus complete live topology, canonical ordering and
  uniqueness before compilation.
- Every pair manifold is built from one shared body-state snapshot. Each global
  Jacobi sweep computes all pair proposals from the same velocities, then
  scatter-adds equal-and-opposite linear/angular impulses and applies one update
  per body. A common per-pair relaxation based on the larger endpoint degree
  preserves momentum while stabilising bodies participating in several pairs.
- Pair position corrections are accumulated by body and native KS/ST response
  is written once per agent. ``resolve_contacts`` deliberately preserves the
  current macro order of wall response followed by simultaneous pair response,
  and returns the union of fresh per-step wall/pair events without latching.
- Tests cover isolated-pair and two-car live-simulator parity, disjoint pairs,
  pair-row order and agent-label invariance, symmetric multi-car contact,
  momentum, non-increasing zero-restitution energy, masks, event clearing,
  ``jit``, environment ``vmap``, ``lax.scan`` and finite free-space gradients.
- Sensing and contact are now complete standalone functional layers. They are
  not yet wired into the free-flight ``step_dynamics`` state transition; that
  integration remains part of the complete core/adapters work.

#### Phase 4d implementation record — 2026-08-29

- The raster sphere tracer, SciPy EDT construction, discretized-angle sine/
  cosine tables and ``ScanBackend`` selector were removed. Exact segment
  intersection is now the only mutable-environment LiDAR implementation;
  ``scan_device`` remains the explicit CPU/GPU placement choice.
- Opponent-body occlusion was separated into a focused Numba geometry module,
  preserving the incumbent in-place ray-cast contract and its randomized
  brute-force oracle rather than deleting it with the raster scanner.
- The weak recorded legacy-scan fixture and generator were deleted after the
  exact scanner passed wall, opponent, environment, noise and JAX parity gates.
  Historical accuracy/performance measurements remain as the rationale for the
  migration, with retired behavior labelled accordingly.

### Phase 5 — observations, episode semantics and adapters

Port or deliberately exclude every current behavior rather than using a short
"feature parity" label:

- observation fields/presets, shapes, dtypes, bounds and copy semantics;
- finish-line laps, wrap-corrected progress and lap times;
- reward modes and current information signals;
- deterministic seed streams and domain randomization;
- LiDAR noise, dropout, fixed episode bias and actuator delays;
- collision/lap/timeout status and `EGO`/`ANY`/`ALL` reduction;
- Gymnasium terminated/truncated distinction;
- reset options and supported wrappers;
- native action ordering, bounds validation and setpoint-controller behavior;
- `configure()` rebuild semantics and shape-preserving `update_params()`;
- the documented step ordering, including current observation/info timing,
  unless a separately approved API correction changes it.

Arbitrary Python `CUSTOM` reward callbacks cannot execute inside a compiled
device rollout. The Gymnasium adapter may preserve them after host conversion;
the device-batched API accepts only a pure JAX reward callable or built-in mode.
Both paths consume the same final metrics, and the limitation is documented.

Then add:

- the Gymnasium JAX adapter and SB3 environment-check smoke test;
- a single-agent SBX training smoke test using a finite flat or supported Dict
  observation;
- the device-batched adapter used by native JAX rollouts;
- the optional official-JaxMARL adapter and its reset/step/auto-reset tests if
  multi-policy training is in scope for the release.

*Done when:* the explicit capability checklist has no accidental gaps, both
Gymnasium backends agree on shared scenarios, and adapter-specific contracts
pass independently.

#### Phase 5a implementation record — 2026-08-29

- ``ScanParams`` now carries traced range/noise/dropout magnitudes in addition
  to clean sensor geometry. ``ScanState`` owns the fixed-shape per-agent,
  per-beam bias sampled from an explicit reset key; a new draw therefore does
  not alter compiled topology or mutate structural configuration.
- ``observed_scan`` preserves the mutable environment's sensor order exactly:
  clean range plus Gaussian noise plus episode bias, clipping to
  ``[range_min, range_max]``, then dropout to exactly ``range_max``. Clean
  geometry remains a separately callable, immutable input.
- Seed replay, reset-dependent bias, zero/one dropout, arithmetic order,
  validation, ``jit``, environment ``vmap`` and ``lax.scan`` are executable
  gates. NumPy and JAX random streams are intentionally not byte-paired.
- This is the first Phase 5 slice. Sensing is not yet composed with dynamics,
  contact, Frenet state or episode semantics in ``step_core``.

#### Phase 5b implementation record — 2026-08-29

- Stepping Frenet projection now uses a fixed-shape local candidate mask
  centered on each agent's explicit previous ``s``. It matches the mutable
  simulator's 10 m window across displaced and wrap-adjacent poses; the global
  projector remains the reset/teleport oracle.
- ``EpisodeState`` seeds separate progress and lap references from the reset
  pose, tracks wrap-corrected cumulative arclength, finish splits, lap counts,
  elapsed steps and latched per-agent terminal status without touching vehicle
  state. Finish timing preserves the current pre-refresh clock guard and
  partial-first-lap/out-lap behavior.
- ``BookkeepingParams`` keeps collision/lap/step enables and limits plus reward
  weights as traced leaves. Built-in SURVIVAL/PROGRESS rewards are per-agent for
  device/MARL consumers; the future Gymnasium adapter selects its ego scalar,
  while Python ``CUSTOM`` remains an adapter-only callback.
- EGO/ANY/ALL termination, fresh collision/lap events, mixed terminal causes,
  newly terminated status and independent timeout truncation pass ``jit``,
  heterogeneous environment ``vmap`` and ``lax.scan`` gates. A scripted
  out-lap/reward/status sequence agrees with the mutable environment.
- These functions remain standalone layers. ``step_core`` still needs to
  compose dynamics, wall/pair contact, local Frenet projection, observed scans
  and episode bookkeeping in the documented order.

#### Phase 5c implementation record — 2026-08-30

- ``CoreConfig`` now carries hashable dynamics/reset/sensor/contact/Frenet/
  episode topology, while ``CoreTables`` and ``CoreParams`` keep fixed device
  arrays and per-episode traced values separate. Cross-topology agent/state
  mismatches fail before compilation.
- ``reset_core`` uses fixed named key children to sample RL poses, initialize
  zero-velocity native state, globally project each CoG pose, reset episode
  state and sensor bias, and return a real observed first scan. Reset does not
  adjudicate contact, matching the mutable simulator.
- ``step_core`` composes command noise/FIFOs and dynamics, optional wall-then-
  pair response, per-agent local Frenet projection from the corrected pose,
  clean and observed scans, then reward/lap/termination/truncation bookkeeping.
  It returns the planned five-part observation/state/reward/event/metric tree
  and never auto-resets or freezes a terminated vehicle.
- ``CoreObservation`` is one fixed agent-batched vocabulary: observed scans,
  native and seven-channel standardized states, fresh collisions, Frenet pose,
  lap values and the deliberately lagged observation clock. ``CoreMetrics``
  retains post-transition time and distinct termination/truncation status.
- Collision ``NONE`` and disabled sensing are static branches that do not trace
  unused geometry; the latter uses the mutable simulator's one-beam internal
  placeholder. With Frenet disabled, progress stays zero and lap termination is
  explicitly off; PROGRESS reward is rejected structurally.
- Aggregate reset/step replay, ``jit``, ``eval_shape``, shared-table parameter
  ``vmap``, multi-step ``lax.scan``, contact termination without state freezing
  and a six-step mutable-environment parity rollout are executable gates. The
  remaining Phase 5 work is host configuration/parameter construction, reset
  overrides, observation packaging and Gymnasium/device-batched adapters.

#### Phase 5d implementation record — 2026-08-31

- The deep-import host builder translates the supported ``EnvConfig`` surface
  and one already-resolved ``Track`` into ``CoreConfig``, ``CoreTables`` and
  ``CoreParams``. ``build_core`` selects the single device requested by active
  contact/sensing, places every array tree there and returns one ``CoreBundle``
  ready for ``reset_core``/``step_core``; mixed active devices fail before map
  preprocessing.
- Static model/integrator/controller/substep/delay/reset/contact/sensor/reward/
  termination choices and traced physical, noise, solver, limit and reward
  values now come from the current frozen config tree. Production continuous
  parameters are explicitly float32, counters int32 and flags boolean even
  when Python inputs are integral or process-wide JAX x64 is enabled.
- Unsupported host behavior fails explicitly rather than falling through:
  winding-angle laps, ``MAP_RANDOM_STATIC``, Python ``CUSTOM`` rewards and
  non-default contact wall extraction tolerance are not part of this core
  builder. Reset overrides and custom callback execution remain adapter work.
- Varying domain-randomization bounds require an explicit already-sampled
  shared ``VehicleParameters`` until key-driven device sampling lands. The
  draw is checked across every supported dynamics/body field, while contact
  tables are sized from the full bound envelope. Constant bounds need no draw.
- Structurally disabled contact and LiDAR no longer build unused acceleration
  indexes. Collision ``NONE`` receives constant-size masked contact/pair
  placeholders, disabled LiDAR receives a masked ray placeholder, and wall
  extraction is skipped when neither subsystem consumes it. Contact broad-
  phase reach now includes the collision body's CoG-relative offset as well as
  its half extents, including independent domain-randomization endpoints.
- Mapping, rejection, device placement, dtype, disabled-allocation, reset/
  jitted-step smoke and CoG-relative contact-budget tests are executable gates.
  At this checkpoint Phase 5 remained open for reset overrides, observation
  packaging, key-driven parameter draws and Gymnasium/device-batched framework
  adapters.

#### Phase 5e implementation record — 2026-08-31

- Explicit pose and full-native-state reset functions now share the sampled
  core's complete initializer. Pose overrides zero every motion channel; state
  overrides preserve KS/ST values after production-dtype conversion. Both
  globally seed Frenet/lap/progress references, clear controls/FIFOs/contact
  events, draw sensor bias, return a real first scan and never solve contact at
  spawn.
- Every reset path reserves the same named pose/bias/scan key children. An
  override discards only the sampling child, so choosing a reset mode does not
  shift functional sensor streams. This is intentionally different from the
  mutable environment's sequential NumPy generator and remains deterministic
  under ``jit`` and environment ``vmap``.
- Shared component-based observation bound/space helpers let host adapters
  preserve the mutable provider's finite spaces without constructing a fake
  simulator. The deep-import ``GymObservationAdapter`` resolves all six current
  observation types, gates scan/Frenet fields, transfers only selected canonical
  dependencies, computes established derived fields, and returns independent
  float32 NumPy leaves with the documented agent dictionaries or ``DIRECT``
  batched keys.
- The internal disabled-LiDAR beam and disabled-Frenet values never enter a
  public layout. Widest domain-randomization limits, integrator-step overshoot,
  state dimension, track extents and configured sensor geometry feed the same
  space construction as the mutable environment. Observation time comes only
  from ``CoreObservation`` so the deliberate metrics/info clock split survives.
- Observation layout construction consumes the paired ``CoreBundle`` rather
  than independently supplied topology. The bundle retains its source
  ``EnvConfig`` and resolved host ``Track``; an optional ``ObservationConfig``
  changes only the public view. The adapter also validates the traced scan
  range, preventing stale pieces from declaring bounds the core can violate.
- This slice provides reusable reset and packaging boundaries, not a new
  Gymnasium environment. Seed continuation, reset-option parsing, actions,
  scalar ego rewards, final info, custom callbacks, configure/update behavior,
  rendering, device-batched policy layouts and key-driven parameter draws remain
  Phase 5 adapter work.

#### Phase 5f implementation record — 2026-08-31

- The deep-import ``JaxF110Env`` is the first complete Gymnasium lifecycle over
  the functional core. It is deliberately not registered and does not replace
  ``f1tenth_gym:f1tenth-v0``. Construction resolves the configured host track,
  builds one paired ``CoreBundle``/``GymObservationAdapter`` and jits the
  reset and transition entry points for that topology.
- The adapter shape-validates native ``(num_agents, 2)`` actions and converts
  them to production float32 without clipping. It supports sampled,
  pose-override and full-native-state resets, packages independent NumPy
  observations, selects the ego scalar from device rewards and returns Python
  termination/truncation flags plus the established copied reset/step
  information dictionaries. Observation time remains one transition behind
  step ``info`` by design.
- Gymnasium seeding owns a host NumPy generator and derives explicit per-reset
  JAX keys from it. Same-class seeded replay is deterministic, including sensor
  streams, while NumPy and JAX backends are not byte-paired. Functional reset
  modes reserve the same pose/bias/scan key children, so an explicit pose/state
  override does not shift the functional sensor children.
- Varying domain-randomization bounds are sampled once per episode on the host
  into the same shared ``VehicleParameters`` model used by the mutable gym, then
  passed through the builder's bound validation into traced core parameters.
  Python ``CUSTOM`` reward callbacks also stay at the Gym boundary and run only
  after final packaged observations and ``info`` exist. Neither behavior enters
  a compiled device rollout.
- ``configure()``, ``update_map()`` and shared ``update_params()`` rebuild the
  bundle, spaces, compiled entry points and renderer state as required. The
  existing PyQt/OpenGL renderer, render clock, callbacks, DEFAULT renderer view,
  human modes and cached RGB frames are reused rather than reimplemented in the
  transition.
- The current ``SingleAgentWrapper`` and Gymnasium ``FlattenObservation`` turn a
  one-agent ``KINEMATIC_STATE`` adapter into finite float32 ``Box(5,)`` and
  ``Box(2,)`` interfaces suitable for SB3/SBX APIs. That compatibility path
  transfers selected observations from device to NumPy every Gym step; it is
  not the device-batched adapter, no trainer dependency is added, and no
  throughput comparison is claimed. Native batched training remains Phase 6
  work.
- Raw and wrapped Gymnasium checker runs, copied public values, stochastic seed
  replay, reset overrides, invalid actions, custom rewards, shared DR,
  truncation without freezing, config/map/parameter rebuilds and a mocked RGB
  renderer lifecycle are executable adapter gates. Optional-dependency gates
  also pass the Stable-Baselines3 checker and complete one eight-step SBX PPO
  update. Display-backed video and sustained training remain separate Phase 6
  validation.

### Phase 6 — batched training and performance decisions

- [x] Run a complete device-native PPO training job; imported training code is a
  starting pattern, not a frozen implementation.
- [x] Keep the SBX smoke/compatibility job separate from the native throughput
  job.
- [x] Evaluate JaxMARL separately. It does not ship in v1; if a later release
  adds it, run at least one continuous multi-agent IPPO/MAPPO smoke job.
- [x] Benchmark batch sizes, agents, maps and beam counts with anti-DCE outputs,
  synchronization, compile time, steady-state throughput and peak memory.
- [x] Re-measure contact CPU/GPU behavior on the final batched transition before
  changing any default.
- [x] Commit `uv.lock` and make CI/images use the locked environment once the
  runtime JAX and optional dependency decision is final.

*Done when:* training completes, numerical gates pass at scale, memory and
throughput are published, and defaults follow the measured final program.

#### Phase 6a implementation record — 2026-08-31

- ``ActiveVehicleParams`` is the finite 20-field supported prefix of the host
  vehicle ABI. Key-driven device randomization samples stable field-index
  substreams and replaces dynamics plus collision-body geometry from the same
  draw, preserving the ``lr``/CoG body-offset correlation. Fixed endpoints are
  exact, disabled or constant bounds return the nominal vehicle, and the draw
  remains one scalar parameter set shared by every agent in an environment.
- ``CoreBundle`` now carries device-placed randomization metadata. Varying
  bounds no longer require a host sample to build the core, while optional
  explicit host episode parameters remain bounds-validated. Active values and
  independent cross-field limit intervals receive finite/kernel-safety checks,
  and contact preprocessing still covers the full configured body envelope.
  An authoritative ``target_device`` can place a state-only training bundle on
  an accelerator instead of inheriting the CPU fallback.
- ``reset_batch`` and its override variants vmap the unchanged core over one
  shared ``CoreTables`` and sample independent episode parameters from each
  reset key. ``step_batch`` preserves raw terminal behavior without reset or
  freezing. ``step_batch_autoreset`` receives separate transition/reset keys,
  resets only whole environment rows selected by terminated-or-truncated, and
  preserves the terminal transition observation separately from the reset
  observation used as the next carry.
- Ordered policy layouts produce decentralized
  ``(batch, agents, features)`` device arrays; centralized flattening and ego
  reward selection are explicit. Device ``CUSTOM`` rewards are pure JAX
  callables vmapped per environment, receive actions/events/metrics/active
  parameters and return one reward per agent. The disabled LiDAR/Frenet
  placeholders cannot be selected into a policy layout.
- Scalar parity, key replay/isolation, device DR, reset overrides, raw done
  continuation, mixed termination/timeout selective auto-reset, terminal-value
  preservation, custom rewards, policy channel order, transfer guards,
  ``jit(lax.scan(vmap(step)))``-equivalent composition and import purity are
  executable gates. The first adapter intentionally supports one shared map;
  complete-table indexed maps must bucket reset and track shapes together.
- This checkpoint establishes the native rollout contract, not completion of
  Phase 6. A sustained PPO training job, measured CPU/GPU throughput and memory,
  final contact device measurements, dependency locking, indexed map routing
  and any optional official-JaxMARL adapter remain open. JaxMARL should wait
  until the native training/update contract and intended multi-policy mode are
  proven.

#### Phase 6b implementation record — 2026-08-31

- ``batch_action_bounds`` derives physical ``(batch, agents, 2)`` limits from
  each environment's active traced vehicle parameters and the compiled
  controller modes. ``scale_normalized_actions`` maps bounded policy output to
  those limits without clipping, so domain-randomized rows do not silently use
  nominal action scales.
- The repository-only ``validation/jax_native_ppo.py`` job is a fresh JAX-only
  PPO implementation, not a packaged trainer and not copied from either old JAX
  branch. It keeps pre-tanh latent actions and joint log probabilities, stops
  rollout gradients, clips policy/value updates and the global gradient norm,
  and uses the terminal transition observation for timeout bootstrapping.
  Natural termination zero-bootstraps; both termination and truncation stop GAE
  recurrence across an auto-reset boundary.
- The fixed CPU gate completed 98,304 environment steps reproducibly. Its
  deterministic fixed-key evaluation improved the bounded speed objective from
  ``0.0`` to ``0.924272`` and mean speed from ``0.0`` to ``2.964633 m/s`` for a
  ``3 m/s`` target. The first compiled rollout/update took ``1.62 s`` and the
  remaining updates sustained about ``49,229 environment-steps/s`` on the
  reviewed CPU. These are validation-workload figures, not simulator-only
  throughput.
- ``IndexedCoreTables`` stacks complete equal-shape reset, track and pair tables.
  Indexed reset, override, raw-step and selective-auto-reset functions choose a
  table row per environment without changing ``BatchState`` or the core. The
  host ``build_indexed_core`` deduplicates repeated ``Track`` identities,
  buckets every complete table leaf by exact shape/dtype, supplies bucket-local
  map indexes and source-row routing, and never pads a small map to a larger
  bucket.
- Action, PPO, indexed scalar-parity, JIT, reset and host-routing gates pass as
  one focused 67-test slice. Commit ``8697e24`` records the training and indexed
  runtime seams; the standalone measurement harness follows separately.
- Phase 6 still requires published final-program measurements and a locked
  dependency snapshot. The optional JaxMARL decision remains separate from the
  now-proven native training contract.

#### Phase 6c completion record — 2026-08-31

- ``benchmarks/phase6_rollout.py`` records schema-validated JSON for equivalent
  functional and mutable scenarios. It separates construction/reset and first
  compilation from steady timing, synchronizes every call, consumes every
  transition leaf to prevent dead-code elimination and reports environment and
  agent rates, resident input/table bytes and real device allocator statistics
  when the backend exposes them. Commit ``cc3477f`` records the base harness;
  ``b58521a`` replaces its empty sensor/contact map with a 154-segment annular
  road, and ``e573dfd`` requires sustained contact on every agent-step. Empty or
  partly active contact measurements cannot pass the final JSON schema.
- On the reviewed i9-11980HK CPU, state-only KS reached 800,971
  environment-steps/s at batch 16, 1080-beam LiDAR reached 10,715 at batch 64,
  persistent two-agent ST contact reached 38,564 at batch 96 and the full
  ST/contact/1080-beam transition reached 4,817 at batch 8. Exact-shape four-map
  indexing reached 607,768 at batch 64 and occupied exactly four 30,924-byte
  tables.
- The RTX 3080 Laptop GPU was initially hidden by the execution sandbox, not
  absent from the host. With device nodes visible, JAX reported the GPU and all
  PPO/state/LiDAR/contact/full/indexed gates completed without OOM. State-only
  KS reached 2.03 million environment-steps/s at batch 1024; 1080-beam LiDAR
  reached 67,296 at batch 64; the widest measured full transition reached
  34,302 environment-steps/s at batch 64. The largest isolated simulator peak
  was 148.3 MiB on the 16 GiB device.
- Sustained two-agent ST contact kept CPU ahead through batch 48 and favored GPU
  from batch 64. GPU/CPU rates were 0.83x at batch 48, 1.08x at batch 64, 1.55x
  at batch 96 and 9.44x at batch 256. The mutable/single-environment default
  therefore remains CPU. Native batched programs choose an explicit device from
  the complete rollout shape; no universal crossover threshold is encoded.
- The official JaxMARL surface was reevaluated after the native contract was
  proven. Its agent-dictionary and public auto-reset boundary would be an
  adapter translation, not a better core. A separate ephemeral gate passed the
  Stable-Baselines3 2.9.0 environment checker and one SBX 0.28.0 eight-step PPO
  update against JAX 0.11.1. Because that requested Gymnasium path and the
  device-native PPO path are both covered, v1 deliberately ships no JaxMARL
  dependency or vendored compatibility layer.
- ``uv.lock`` is now source controlled. CI and the CPU container use
  ``uv sync --frozen`` (and omit the default GPU group where appropriate), so
  resolution drift can no longer change those gates. Flax, Chex, Optax,
  JaxMARL and trainer packages remain intentionally absent: the validation PPO
  is repository-only and no external training adapter ships. The measured
  methodology, command shapes, CPU/GPU tables and placement guidance are
  published in ``docs/jax_performance.rst``. Phase 6 is complete.

#### Phase 6d responsibility-first cleanup record — 2026-09-01

- The Phase 1 through Phase 6c records above preserve the names and paths that
  existed when each slice landed. They are historical, not the current import
  surface. The separate ``f1tenth_gym.jax`` namespace was subsequently removed
  and its implementation redistributed beside the owning environment,
  dynamics, action, integration, contact, LiDAR, reset, track, episode,
  observation and batching subsystems under ``f1tenth_gym.envs``.
- ``JaxSimulator(config, track, device=...)`` now owns host configuration,
  preprocessing, validation, placement and compiled single-environment reset/
  step entry points. ``IndexedJaxSimulator`` owns exact-shape indexed-map
  construction, and ``GymObservationAdapter.from_simulator()`` derives a host
  observation view from that configured simulator. These surfaces supersede
  the staged ``build_core``/``build_indexed_core`` functions, ``CoreBundle``
  facade and ``from_bundle()`` adapter construction described in the earlier
  implementation records; the necessary ``CoreConfig``/``CoreTables``/
  ``CoreParams`` split remains internal and explicit for JAX tracing.
- The current parameter names describe their ownership directly:
  ``DynamicsRuntimeParams(vehicle=...)`` holds traced dynamics values,
  ``EpisodeParams`` holds episode bookkeeping, and ``CoreParams`` exposes them
  as ``dynamics`` and ``episode``.
- The repository-only native PPO CPU gate and full-rollout CPU/RTX 3080 gates
  were rerun successfully immediately before removal. The
  ``benchmarks`` and ``validation`` directories and their 18 harness-only tests
  were then deleted. ``docs/jax_performance.rst`` retains the protocol, measured
  tables and placement conclusions as historical migration evidence without
  presenting deleted commands as a supported interface.

---

## 05 · Numerical and architectural traps

### ST low-speed behavior

The current NumPy ST switches below `0.5 m/s`. Billy's JAX switching model uses
`1.5 m/s` and clips velocity with `V = clip(V, min=0.001)`, which prevents
reverse motion. Do not adopt either behavior accidentally.

The JAX implementation must retain the signed raw velocity for kinematics,
control constraints and branch selection. Only the denominators of the dynamic
branch receive a safe nonzero velocity. Select the final branch using the raw
velocity and the current `0.5 m/s` threshold.

The dynamic branch divides by velocity several times. A plain outer `where`
produces the right forward value at zero but can return NaN gradients because
the inactive branch still contains undefined adjoints. Use the double-where
guard: sanitize the velocity inside the partially defined branch, then select
the result using the original predicate.

Under `vmap`, `lax.cond` over a batched predicate lowers to select-like behavior,
so it is not a substitute for sanitizing the inactive branch. Test forward
values, reverse motion and gradients.

The switch remains discontinuous at `0.5 m/s`. Finite gradients away from the
boundary are required; smooth contact/dynamics across the boundary is not part
of this migration.

### Pair contact

The current agent-pair loop recomputes vertices after each pair, making results
order-dependent. A vmap around that loop does not fix the algorithm. Pair
generation, manifolds and solver updates must be fixed-shape and simultaneous,
with impulses/corrections accumulated by body before applying a Jacobi sweep.

### Hard physics is not globally differentiable

Pure JAX means transformable and accelerator-compatible, not differentiable at
every event. Segment minima, contact activation, impact response, termination,
lap crossings and reset choices have genuine discontinuities. Validation asks
for useful finite gradients in free flight and away from geometric switches; it
does not promise end-to-end gradients through impacts or resets.

### Precision

Numba dynamics may compute in float64 before state is recast to float32. JAX
defaults to float32, moving that narrowing inside RK4 stages. Production uses an
explicit dtype policy and tolerance budget.

`jax_enable_x64` is process-global, not an environment config field. Run x64
oracle checks and float32 production checks in separate processes/CI jobs. Do
not mix implicit float64 constants into an otherwise float32 rollout.

### Fixed shapes and maps

All map-dependent device tables need masks and allocation budgets. A shared
stack avoids duplicating identical maps, but padding every map to the largest
wall/tile/spline shape can transfer the worst map's memory and compute cost to
the whole batch. Measure global padding against shape-bucketed batches.

### Auto-reset

The core never auto-resets. Gymnasium resets when its caller asks; the batched
training adapter may fuse reset with step; JaxMARL may calculate reset candidates
on every step. Each adapter must keep reset/step leaves shape-compatible, but
that rule must not force framework behavior into the physics core.

---

## 06 · Validation gates

### Differential oracle

Use the current NumPy implementation as a live oracle. Each JAX numerical layer
gets randomized differential tests, followed by controlled end-to-end rollouts.
Do not use `jax-backend`'s `validation/reference/numpy_reference.npz`; it predates
current occupancy semantics, CoG normalization, controllers, lap counting and
contact defaults.

Required numerical gates include:

- KS rear-axle reference versus production CoG transform;
- KS/ST derivatives and Euler/RK4 rollouts, including reverse and zero speed;
- action controller and integration-substep behavior;
- Frenet projection and wrap-corrected progress over the full track;
- segment scan values, LiDAR transform and opponent occlusion;
- wall and car-to-car contact, momentum and pair-order invariance;
- seeded reset, noise, delay and DR streams;
- free-flight gradients versus finite differences where the model is smooth.

### Public contracts

- Gymnasium and SB3 environment checkers;
- action/observation spaces, shapes, dtypes and finite bounds;
- `reset` and five-tuple `step` behavior;
- info arrays are copies at host API boundaries;
- global termination reduction for `EGO`, `ANY` and `ALL`;
- collision response continues when termination is disabled;
- timeout remains distinguishable from termination;
- optional JaxMARL key-first/reset/step/auto-reset contract, only if that
  adapter is shipped;
- device-batched `jit(vmap(lax.scan))` smoke tests.

### Performance gates

- prevent dead-code elimination by returning/consuming scans and contact state;
- synchronize before timing;
- separate compile latency from steady-state throughput;
- report environment-steps and agent-steps separately;
- report accelerator, JAX version, dtype, batch shape and peak memory;
- compare against the NumPy environment on the same behavior configuration.

---

## 07 · Supported-surface cleanup

These changes are intentional API corrections for the JAX-first v1 surface,
not opportunistic cleanup:

- [x] Remove `LoopCounterMode.TOGGLE`.
- [x] Move GJK into a test-only reference and remove `BOUNDING_BOX` production
      dispatch without renumbering existing serialized enum values.
- [x] Remove freeze-on-collision `LIDAR_SCAN`; keep `NONE` as explicit disablement.
- [x] Make `SEGMENT_CONTACT` the sole production response and separate response
      from termination policy.
- [x] Remove/retire `DynamicModel.MB` and its production dispatch, tests and docs.
- [x] Preserve full-scale parameter data needed by KS/ST or possible future MB
      research, but reject MB-only runtime/DR knobs rather than silently ignoring them.
- [x] Promote KS-CoG, migrate the raw state contract, then delete frame-conversion
      scaffolding once no supported model uses a rear-axle native state.
- [x] Remove RASTER after the segment scan gate passes; keep absent `jax-pf`
      out of package dependencies and built artifacts.
- [x] Correct the stale `collision_check` documentation.
- [x] Reconcile the documented 20,000 SAT/GJK cases with the 4,000-case executable
      regression before moving GJK out of production.

Unrelated dead helpers, rendering knobs and generated `__pycache__` directories
are not migration prerequisites. Handle them in focused cleanup changes only if
they materially reduce the implementation surface.

---

## 08 · Explicit decisions

1. Merge original `jax/main` once; preserve its DAG and signatures.
2. Use `jax-backend` only for ideas and warnings; import no commits or fixtures.
3. Build a framework-neutral functional JAX core with no automatic reset.
4. Keep Gymnasium as the SBX/SB3 compatibility API and add a device-batched
   API around the core. Defer the optional official-JaxMARL adapter beyond v1.
5. Keep the NumPy environment until JAX parity and training gates pass; it is the
   live correctness oracle, not the long-term high-throughput path.
6. Support KS-CoG and ST. MB is not supported by the JAX-first v1 surface.
7. Use SEGMENT_CONTACT as the only production collision response; `NONE` disables
   collision. Do not freeze individual collided agents.
8. Reduce per-agent terminal status to one whole-environment result using
   configurable `EGO`, `ANY` or `ALL` semantics.
9. Keep the current discontinuous ST switch at `0.5 m/s`, preserve reverse motion
   and sanitize its inactive dynamic branch for finite gradients.
10. Separate static topology from traced episode parameters so DR and parameter
    sweeps vmap without recompilation.
11. Treat x64 as a process-wide validation mode; production dtype is measured on
    the final implementation.
12. Store unique preprocessed maps once and bucket complete tables by exact
    shape; measurements reject global worst-map padding.
13. Keep `uv.lock` committed and use frozen resolution in CI/images.

---

## 09 · Out of scope

- JAX multi-body dynamics.
- Partial removal or absorbing states for individually completed agents.
- A promise of differentiability through hard contact, lap crossings,
  termination or reset.
- Device-native Qt rendering; rendering remains a host-side consumer.
- An official-JaxMARL adapter in the v1 release; add one only for a concrete
  multi-policy consumer and keep it outside the core.
- Reusing old trained weights as a behavior oracle.
- Deleting the NumPy implementation before the JAX core and adapters pass their
  numerical, API and training gates.
