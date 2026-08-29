# JAX-First f1tenth_gym

A framework-neutral, pure-JAX simulation core that can run thousands of
environments in one compiled accelerator program. The existing Gymnasium API
remains the compatibility surface for SBX/SB3 and current users; optional
adapters expose device-batched and JaxMARL-style APIs without making either
framework the simulation contract.

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
             -> optional JaxMARL adapter
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
that `step()` validates but does not clip an out-of-space action. Normalized
`[-1, 1]` actions, when useful to an RL library, belong in a wrapper.

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

### Phase 6 — batched training and performance decisions

- Run a complete device-native PPO training job; imported training code is a
  starting pattern, not a frozen implementation.
- Run an SBX smoke/compatibility job separately from the native throughput job.
- If JaxMARL ships, run at least one continuous multi-agent IPPO/MAPPO smoke job.
- Benchmark batch sizes, agents, maps and beam counts with anti-DCE outputs,
  synchronization, compile time, steady-state throughput and peak memory.
- Re-measure contact CPU/GPU behavior on the final batched transition before
  changing any default.
- Commit `uv.lock` and make CI/images use the locked environment once the JAX,
  Flax/Chex and optional training dependency set is final.

*Done when:* training completes, numerical gates pass at scale, memory and
throughput are published, and defaults follow the measured final program.

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
- optional JaxMARL key-first/reset/step/auto-reset contract;
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
4. Keep Gymnasium as the SBX/SB3 compatibility API; add device-batched and
   optional official-JaxMARL adapters around the core.
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
12. Store unique preprocessed maps once, but choose padding versus shape bucketing
    from measured memory and throughput.
13. Commit `uv.lock` after dependency reconciliation and use locked CI/images.

---

## 09 · Out of scope

- JAX multi-body dynamics.
- Partial removal or absorbing states for individually completed agents.
- A promise of differentiability through hard contact, lap crossings,
  termination or reset.
- Device-native Qt rendering; rendering remains a host-side consumer.
- Reusing old trained weights as a behavior oracle.
- Deleting the NumPy implementation before the JAX core and adapters pass their
  numerical, API and training gates.
