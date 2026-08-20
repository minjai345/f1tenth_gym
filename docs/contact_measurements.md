# Contact system — measured numbers

Every number here was produced on this tree by the command shown. Numbers in
`IMPROVED CONTACT AND SCANS PLAN/CONTACT_IMPLEMENTATION_PLAN.md` §7 predate the
sub-pixel extraction and are superseded by this file.

Machine: RTX 3080 Laptop, py3.13, jax 0.11.1, numpy 2.4, scikit-image 0.26.

---

## Phase 1 — wall extraction

```bash
env -u PYTHONPATH DISPLAY= uv run pytest tests/test_walls.py -q
```

Spielberg, Austin and Monza carry anti-aliased edges (234 distinct intermediate
grey levels), so extraction traces the sub-pixel level set. A ROS `map_saver` map
has one intermediate value (205 = unknown) and falls back to the binarised grid;
`has_subpixel_edges` separates the two.

| | binarised grid | sub-pixel level set |
|---|---|---|
| Spielberg segments | 9,501 | **833** |
| median turn between adjacent segments | 45.00° | **3.73°** |
| turns > 60° | 0.0% | 0.2% |
| extraction time | 0.49 s | **0.11 s** |
| per-segment normal error, synthetic wall (length-weighted) | 19.19° | **0.004°** |

Agreement with the LiDAR's EDT, 3,000 free points on Spielberg, `(EDT − segment
distance) / cell`: median **+0.146** (binarised: +0.480), range [−0.141, +0.998].
The offset does not grow with range.

Systematic biases, both one-sided: marching squares bevels convex corners inward by
at most √2/4 ≈ 0.354 cells, so obstacles are never over-reported; an isolated single
occupied cell becomes a diamond of half its area.

**float32 limit.** Endpoints are float32 at world magnitudes, so a 4 cm segment 77 m
from the origin loses up to 2e-4 rad when you subtract them. Read the stored `n`;
do not recompute it from `a` and `b`. A UTM-scale origin collapses endpoints outright
and `extract_walls` raises rather than emitting zero-length segments.

### Contact normal accuracy

Depth-weighted over the segments an F1TENTH collision box actually touches, 560
contacts per angle, binarised grid:

| approach | median | p95 | max |
|---|---|---|---|
| 0° | 0.00° | 0.00° | 0.00° |
| 15° | 5.80° | 10.94° | 14.25° |
| 30° | 8.55° | 13.98° | 17.32° |
| 45° | 0.00° | 16.39° | 30.51° |
| 90° | 0.00° | 13.10° | 28.64° |

45° is not a special case — it has the best median, tied with the grid-aligned
angles. Adjacency blending at λ = body half-width was measured against this and moves
the error by less than 0.5°: it smooths the normal *along* the wall as a car slides,
it does not remove staircase bias. The sub-pixel path removes the bias at source.

**Open question (plan §8.3), answered on a single-sided wall fixture.** The
length-weighted aggregate normal over a whole face is accurate to ≤0.15° at every
angle. Over a 0.58 m contact patch the worst case is 2–6°, rising to 26.6° at exactly
45° where a short patch can catch an unbalanced mix of the two 45°-family normals.

---

## Phase 2 — tile budget

```bash
env -u PYTHONPATH DISPLAY= uv run pytest tests/test_budget.py -q
```

`k_tile` is the exact maximum candidate count over every tile, by difference-array
accumulation — **not** sampled. Verified against brute force over all 68,400 Monza
tiles at three tile sizes. F1TENTH body, `QH = hypot(0.58, 0.31)/2 = 0.32882 m`,
tile 0.5 m, margin 1.25:

| map | segments | KTILE | safe | tile grid | table | extract |
|---|---|---|---|---|---|---|
| Spielberg | 833 | 15 | 19 | 133×207 | 2.09 MB | 0.11 s |
| Austin | 1022 | 13 | 17 | 168×286 | 3.27 MB | 0.11 s |
| Monza | 650 | 10 | 13 | 342×200 | 3.56 MB | 0.09 s |
| Spielberg_blank | 0 | 0 | 0 | 0×0 | 0 | — |

The plan's §7 figures (KTILE 66, 7.48 MB) were computed over 9,501 binarised
segments and no longer apply.

Tile-size sweep, Spielberg — larger tiles trade a slightly higher `K` for a much
smaller table:

| tile | KTILE | tile grid | table |
|---|---|---|---|
| 0.25 m | 13 | 265×413 | 7.44 MB |
| 0.5 m | 15 | 133×207 | 2.09 MB |
| 1.0 m | 16 | 67×104 | 0.56 MB |
| 2.0 m | 20 | 34×52 | 0.18 MB |
| 4.0 m | 43 | 17×26 | 0.10 MB |

**Domain randomization.** `QH` is a function of body length and width, which DR can
grow, so budgets are sized from the DR range endpoints. Note `widest_params()` is the
wrong helper: it widens *limit* fields for the observation bounds and leaves the body
alone.

---

## Phase 3 — tile index

```bash
env -u PYTHONPATH DISPLAY= uv run pytest tests/test_accel.py -q
```

`gather()` must return a **superset** of the segments whose grown bounding box
contains the query centre; a miss is a silently wrong contact. Verified against
brute force over 4,000 real free poses per map, synthetic grids at four tile sizes,
and points sitting exactly on tile seams.

| map | segments | K | table | build | misses | extra candidates |
|---|---|---|---|---|---|---|
| Spielberg | 833 | 19 | 2.09 MB | 0.11 s | **0** | 0.35 |
| Monza | 650 | 13 | 3.56 MB | 0.09 s | **0** | 0.19 |

**Memory guard.** A tile grid costs `rows x cols` per entry, so halving `tile_size`
quadruples it and `map_scale=10` multiplies it by 100. Monza at scale 10 with a
0.25 m tile projects to a 6799x3972 grid — 0.65 GB for the budget accumulator alone
and 1.4 GB for the table. Both `tile_budget` and `build_tile_index` now compute the
projection and raise `MemoryError` **before allocating**, in 92 ms, naming the tile
size and the cap. Default ceiling 512 MB, raisable via `max_bytes`. Scale 10 at
0.5 m (135 MB) still builds.

---

## Phase 4 — narrow phase

```bash
env -u PYTHONPATH DISPLAY= uv run pytest tests/test_contact_kernels.py -q
```

`segment_contact` in `envs/contact/kernels.py`: pure JAX, fixed shape, `vmap`-ready.
Each §3 invariant measured against a Cyrus-Beck oracle that clips the segment against
the body's four half-planes.

| invariant | measured |
|---|---|
| §3.2 far face of a 2-px strip | near face 2 points, far face **0** (without the straddle test it reports 0.271 m) |
| §3.3 body face axes required | full kernel **0** false positives; dropping the two axes gives **10.6%** |
| §3.4 strict separation | flat face contacts at 1e-2, 1e-4 and 1e-6 m penetration |
| §3.5 manifold | flat → **2** points, corner-on → **1**; every point within the circumradius |

**Precondition.** `normal` must be perpendicular to `seg_b - seg_a`. `WallSegments`
guarantees it; a hand-written normal that is not perpendicular makes `seg_a @ normal`
cease to be a plane, and the gates then admit contacts up to 0.66 m deep. Cost me an
afternoon — it looked exactly like a kernel defect.

**The reference-face clip declines some real overlaps on an isolated segment** — its
crossing can fall outside the incident face. On a continuous wall the neighbours cover
it: **0 misses over 1,950 poses** whose body genuinely overlaps a Monza wall cell, at
4.3 contact points per pose. Isolated-stub false negatives are therefore not a defect
of the chain, but a solver fed hand-made single segments should know.

**`blended_normal` is not implemented, deliberately.** The plan's §5.4 blending exists
to smooth a staircase. Sub-pixel extraction removed the staircase at source (median
turn 3.73° against 45°), and adjacency blending at λ = body half-width was measured to
move the contact-normal error by **less than 0.5°**. Revisit only if a solver shows
step-to-step normal jitter.

---

## Phase 5 — solver

```bash
env -u PYTHONPATH DISPLAY= uv run pytest tests/test_contact_solver.py -q
```

**Jacobi, not Gauss-Seidel.** The plan assumed sequential impulses. Measured on the
3080 with a flat two-point manifold, incoming 5 m/s, `e = 0`, residual is
`max |v_c . n|` after solving:

| sweeps | sequential residual | Jacobi residual | sequential cost | Jacobi cost |
|---|---|---|---|---|
| 16 | 0.032 | 0.536 | 3.55 ms | 0.51 ms |
| 64 | 0.000 | **0.00066** | ~14 ms | **0.82 ms** |

Sequential converges far better per sweep, and loses anyway: a `lax.scan` over
contact slots is launch-bound, so each sweep costs ~0.25 ms of latency for a handful
of flops. Jacobi makes a sweep one vectorised op, so sweeps are nearly free and you
can afford 64. Same mechanism as the plan's `while_loop` anti-goal. Default is 64
sweeps, 1.26 ms/solve over a realistic 38 slots.

Restitution lands exactly: `e = 0.0 / 0.3 / 0.8` from a 5 m/s impact gives separating
speeds of `0.00 / +1.50 / +4.00` m/s. A 2,000-step rest under gravity settles to
exactly `slop` penetration with zero yaw drift and flat kinetic energy.

**Two sign bugs the physics tests caught**, both invisible to the geometry tests:

- Restitution bias was `e * -approach`, which drives `v_n` toward *minus* the bounce —
  the body kept closing at 1.5 m/s for `e = 0.3`. It must be `e * approach`.
- The speculative clamp fired while already penetrating. A negative gap makes
  `-gap/dt` positive, so instead of clamping it flung the body out at `gap/dt`,
  turning a 1 cm/s resting contact into 0.1 m/s and multiplying its energy by 100.
  It is now gated on `gap > 0`.

---

## Phase 6 — gym integration

```bash
env -u PYTHONPATH DISPLAY= uv run pytest tests/test_segment_contact_mode.py -q
```

`CollisionCheckMode.SEGMENT_CONTACT = 3`, `ContactConfig`, and a `resolve_contacts`
hook after the integration loop and before the Frenet block, so the corrected pose is
what the Frenet frame, the scan and the observation all see with nothing re-derived.

**The result the project exists for.** Coasting into a wall in `ACCL` mode, friction 0,
measuring the first contact step:

| approach | halt keeps | segment contact keeps | tangential fraction |
|---|---|---|---|
| 30° | 0.0% | **50.0%** | 50.0% |
| 60° | 0.0% | **93.3%** | 86.6% |
| 75° | 0.0% | **98.9%** | 96.6% |

**The LiDAR-off trap is fixed for the new mode.** Driven into a wall with
`lidar_config.enabled=False`: the halt flags 0/120 steps, segment contact 106/120.

**Cost.** 3.98 ms/step against the halt's 0.231 ms, of which ~2.2 ms is the adapter
call including the numpy-JAX boundary — the plan's unmeasured open question §8.5.
That is 40% of a 10 ms budget for one agent; multi-agent vmap is unmeasured.

**Four wiring bugs, all silent.** Three were flag ownership: the scan path's `else`
cleared the flag the solver had just set; the LiDAR-off branch did the same
unconditionally; and both were invisible because contact still *resolved* correctly
(speed fell 4.0 to 0.002) while `info["collisions"]` read 0. The fourth was mine and
worse: replacing an 8-space-indented line matched it as a *substring* of a
12-space-indented one inside `reset()`, silently de-indenting the Frenet block out of
its per-agent loop. `tests/test_frenet_multiagent.py` caught it.

**Config guards.** `collision_check` is now coerced through `CollisionCheckMode(...)`,
closing the raw-int footgun where `0` did not disable collisions and `1` did not
select `LIDAR_SCAN`. `SEGMENT_CONTACT` + `DynamicModel.MB` raises at config build, one
guard, before any map download or JIT.

---

## Phase 7 — car-to-car

```bash
env -u PYTHONPATH DISPLAY= uv run pytest tests/test_body_contact.py -q
```

`body_contact` is a separating-axis test over the two bodies' four unique face
normals; the minimum-overlap axis is the MTV, and the manifold comes from the same
reference-face clip the wall path uses. `resolve_pair` is the two-body solver, where
the effective mass carries both translational and both rotational terms.

**The plan's claim reproduces exactly: SAT and the existing GJK agree on
20,000/20,000 random pose pairs.** So this is a strict extension, not a replacement.

Two cars closing head-on at 3 m/s each, in the gym:

| mode | restitution | car 0 after | car 1 after |
|---|---|---|---|
| `BOUNDING_BOX` (GJK boolean) | — | 3.00 m/s | 3.00 m/s — **passes through** |
| `SEGMENT_CONTACT` | 0.0 | 0.00 m/s | 0.00 m/s |
| `SEGMENT_CONTACT` | 0.5 | −1.50 m/s | +1.50 m/s |

Momentum is conserved because the impulses are internal: over 174 random overlapping
pairs with friction 0.5, worst linear drift **1.1e-5 kg m/s** and worst angular drift
**1.9e-6 kg m^2/s**, both at float32 noise level.

Pairs are resolved alongside the walls, before the Frenet block, so a corrected pose
never leaves the Frenet frame or the scan stale. A numpy bounding-box test runs first,
so the common case of nobody touching costs no JAX dispatch at all.

---

## Broad-phase back end (plan §8.1)

Measured with `bench_broadphase.py` on the 3080, µs per query, all four methods
agreeing with brute force to 0.0:

| method | 256 envs | 1024 | 4096 |
|---|---|---|---|
| bvh + `while_loop` | 50.09 | 20.60 | 17.75 |
| bvh + `fori` | 12.38 | 2.96 | 0.732 |
| **tile gather** | **0.381** | **0.221** | **0.137** |

Tile gather wins by 32× / 13× / 5.3×, not the "within 4%" the CPU measurement
suggested. The BVH is not shipping.

---

## Incumbent baseline (what SEGMENT_CONTACT replaces)

`_halt_on_collision` zeroes `model.velocity_indices()` with no reference to the
wall's direction. Coasting into a wall in `ACCL` mode with a zero command so the
speed controller cannot interfere, sweeping approach angle, 21 impacts:

| approach | entry | into wall | along wall | speed after | kept |
|---|---|---|---|---|---|
| 5° | 8.0 | 7.970 | 0.697 | 0.000000 | 0.0% |
| 30° | 8.0 | 6.928 | 4.000 | 0.000000 | 0.0% |
| 60° | 8.0 | 4.000 | 6.928 | 0.000000 | 0.0% |
| 75° | 8.0 | 2.071 | **7.727** | 0.000000 | 0.0% |

Best retained speed across all 21 impacts: `0.000000 m/s`.

**The speed controller.** Under the default `SPEED` mode `pid_accl` re-commands
`kp = 4.755` per m/s of deficit, saturating at `m·a_max = 35.57 N` against a
`m·g = 36.69 N` car. Contact is therefore a driven equilibrium, and a pinned car
under the halt sits in a bit-identical step-rate limit cycle: 1200/1200 steps
flagged, one distinct velocity, 21.34 N discarded per step. Resting-contact energy
tests must command zero speed or use `ACCL`, or the controller reads as a leak.

## Step cost (RTX 3080 Laptop, Spielberg, 833 segments, k=19, 64 sweeps)

The contact call is one launch of a ~40-point solve per agent per step, so latency
dominates and the arithmetic does not register.

| | before | after |
|---|---|---|
| `SEGMENT_CONTACT` step, N=1 | 2.346 ms | **0.502 ms** |
| `_resolve_contacts` | 1.862 ms | **0.147 ms** |
| `LIDAR_SCAN` step, N=1 (reference) | 0.258 ms | 0.245 ms |

Two causes, measured separately:

| | GPU | CPU |
|---|---|---|
| wall kernel, synced | 1.531 ms | **0.084 ms** |
| adapter path incl. marshalling | 1.812 ms | **0.372 ms** |

| argument marshalling | cost |
|---|---|
| `jnp.asarray` x6 then call | 0.216 + 0.081 ms |
| `np.asarray` x6 then call | **0.002 + 0.099 ms** |
| output conversion (2 arrays, float, bool) | 0.008 ms — already cheap |

Sweeps are not the lever on CPU: 1/8/64/128 sweeps cost 0.099/0.118/0.143/0.169 ms,
so the default 64 costs 0.044 ms over a single sweep. On GPU they are the lever, in
the wrong direction — see the cost model below.

`jax.jit(..., out_shardings=SingleDeviceSharding(cpu))` is what pins execution;
`device_put` on the closure constants alone does not (output still lands on CUDA).

Physics is unchanged: over a 900-step wall-scrape both devices flag 807 contact
steps, max state divergence 6.3e-6 (float32 reassociation).

### Scaling

| N | `LIDAR_SCAN` | `SEGMENT_CONTACT` | contact |
|---|---|---|---|
| 1 | 0.254 ms | 0.497 ms | +0.242 |
| 2 | 1.410 ms | 1.875 ms | +0.465 |
| 4 | 6.377 ms | 7.315 ms | +0.938 |

Contact is linear in N at ~0.24 ms/agent (one dispatch each). At N=4 the LiDAR
`ray_cast` quadratic is 87% of the step, so batching agents into one `vmap` call
would recover ~10% and is not the next lever.

### Why a broad-phase skip was not added

Skipping the solve when no wall is near sounds free, but F1TENTH tracks are tight:
the racing-line clearance to the nearest wall is a median 0.401 m (Spielberg),
0.298 m (Monza), 0.275 m (Austin) against a body half-diagonal of 0.329 m. A
circle test would skip only 62/46/41% of steps, cost 21 us of numpy each time, and
save at most 0.147 ms. Segment AABBs are looser still — a tile-occupancy test skips
just 0.6% of steps, because one long diagonal wall has a huge axis-aligned box.

## Device choice (verified against three adversarial attacks)

The GPU cost is **not** a flat launch fee. It is roughly

    0.35 ms host dispatch  +  0.015 ms x solver_iterations

because `resolve` is a `lax.fori_loop` of `solver_iterations` sweeps and
`speculative_clamp` a `lax.scan` over contact slots, so one call is a chain of >=64
tiny sequential kernels: ~21.5 us per sweep on GPU against ~2 us on CPU. Flat in the
number of bodies, rising in configuration.

Consequence: every dial a user can turn to add work adds *depth*, which makes the
GPU relatively worse, never better. Measured at one body per launch:

| config | cpu | gpu | ratio |
|---|---|---|---|
| `solver_iterations=64` (default) | 0.113 ms | 1.378 ms | 10.5x |
| `solver_iterations=256` | 0.189 ms | 3.493 ms | 18.4x |
| `solver_iterations=2048` | 0.978 ms | 24.43 ms | 25.0x |
| `tile_size=4.0` (k=54) | 0.155 ms | 1.420 ms | 9.2x |
| `tile_size=8.0` (k=83) | 0.188 ms | 2.235 ms | 11.9x |

The only axis that amortises the sweep chain is **width** — bodies per launch — and
nothing in the shipped code populates it. Crossover for a single vmapped launch at
default settings is **~52 bodies** (cpu 1.140 ms at 48, 2.759 ms at 56; gpu flat
1.39-1.48 ms). The knee is a cache/threading cliff on the CPU side, not a slope change.

`_resolve_contacts` resolves one body per launch, so no `num_agents` reaches it:

| N | cpu ms/step | gpu ms/step | ratio |
|---|---|---|---|
| 1 | 0.131 | 1.378 | 10.5x |
| 8 | 1.020 | 9.681 | 9.5x |
| 32 | 3.538 | 38.27 | 10.8x |
| 64 | 6.991 | 76.34 | 10.9x |

Deferring the host sync to pipeline launches does not rescue it: with device-resident
args and one `block_until_ready` at the end, per-call GPU cost is still ~1.0 ms flat,
and 77% of a call is *host* dispatch, which serializes in a Python loop exactly as
device time does. Per call the GPU path also burns ~10x more host CPU than the CPU path.

### Batching agents (not yet implemented)

One vmapped launch over an env's agents against N sequential launches, on CPU:

| N | sequential | batched | speedup |
|---|---|---|---|
| 2 | 0.229 ms | 0.126 ms | 1.8x |
| 8 | 0.844 ms | 0.249 ms | 3.4x |
| 16 | 1.608 ms | 0.433 ms | 3.7x |
| 32 | 3.267 ms | 0.792 ms | 4.1x |

Worth doing as a *CPU* optimisation. It does not enable the GPU: even a perfectly
batched single env needs ~52 bodies, and nobody races 52 F1TENTH cars. Note
`_resolve_agent_contacts` cannot be vmapped as written — it recomputes each body's
vertices after every resolved pair, so pair resolution is order-dependent and its
launch count stays O(N^2).

With the LiDAR on, contact is 42% of an N=1 step but only 7.5% at N=8, because
`ray_cast` is quadratic and owns 91% of an N=8 step. With the LiDAR off, contact is
63-76% of the step at every N.

### Env count is not a batching axis

`gym.spec("f1tenth-v0").vector_entry_point` is None, so `gym.make_vec` can only build
sync/async loops of independent `F110Env`s. Passing a shared `Track` dedupes the numpy
side (same segment arrays, same tile table) but each `F110Simulator` still builds its
own `WallContact` with the table baked in as a traced constant: `same jitted callable:
False`. Sync sub-envs step in a Python loop; async sub-envs are separate processes.

That last point is also why a GPU default would be hazardous: `AsyncVectorEnv` forks K
worker processes, and JAX preallocates a large fraction of VRAM *per process*.

## Phase 8 — gradient surrogate

`deepest_depth` is the differentiable companion to `segment_contact`: the penetration
of the single deepest body vertex, with an optional softmin over vertices.

Measured over 417 poses on Spielberg (one per sampled wall segment, 5 mm penetration,
yaw jittered +/-0.6 rad), against central differences at h = 1e-4. Median relative
error, and the count of poses where finite differences return no signal on any axis:

| loss | d/dx | d/dy | d/dpsi | FD dead |
|---|---|---|---|---|
| manifold sum, float32 | 9.7% | 8.4% | 6.8% | 12 / 417 |
| deepest point, float32 | 1.4% | 1.0% | 1.8% | **0 / 417** |
| manifold sum, float64 | 0.0% | 0.0% | 0.0% | 0 / 417 |
| deepest point, float64 | 0.0% | 0.0% | 0.0% | 0 / 417 |

Softened (3 mm) and swept by penetration depth, float64, median relative d/dpsi error
is **0.03% at every depth from 2 mm to 200 mm** — the solver's slop is 2 mm, so the
whole realistic range is covered.

### Two plan claims that do not reproduce

Plan §3.9 says summed manifold depth is discontinuous in yaw, with central differences
spiking to `|d/dpsi| = 10503` against 2.3 for the deepest point. **On this tree both
sit at 1.0-1.5**, and in float64 both match autodiff to 0.0% on every component. The
surrogate is still worth having, but for the float32 reasons in the table above, not
for a discontinuity that is not there.

`PLAN_REVISION.md` §Phase 8 says the float32 deepest-point signal is "exactly 0.0 at
every pose" at ~100 m world coordinates. **Not reproduced**: 0 of 417 poses are dead
in either frame. Body-frame recentring is a real but modest precision gain, not a
correctness fix — same maths in world axes, same probes:

| projection frame | d/dx | d/dpsi | max depth difference |
|---|---|---|---|
| body-centred (shipped) | **0.86%** | **1.72%** | — |
| world axes | 3.25% | 2.43% | 1.55e-5 m |

Script: `scratchpad/frame_ab.py`.

### The residual discontinuities are real, and small

Two remain, both inherent to summing one-sided per-segment depths:

**Flush contact ties two vertices.** With the body exactly parallel to a wall the hard
minimum breaks the tie arbitrarily and reports a median `|d/dpsi|` of **0.29** where the
true two-sided derivative is near zero. 0.5 mm of softening cuts that to **0.0004**, and
3 mm to 0.0001. Use a nonzero `softness` for anything that learns from yaw.

**A segment's tangential gate can flip while it is still deep.** At 1 pose in 417 a body
straddling a seven-segment corner rotates 1e-4 rad and one segment's contribution drops
0.3146 -> 0.0000, giving `|FD d/dpsi| = 1572`. Verified not to be the broad phase: the
candidate set is identical either side. Rare and geometry-driven, not depth-driven.

## Phase 9 — migration seam

The seam is a property, not a deliverable: `kernels.py` and `solver.py` must stay
pure JAX so the eventual rewrite deletes `adapter.py` rather than rewriting them.
`tests/test_migration_seam.py` makes that mechanical rather than a matter of care:

- An AST walk over both modules rejects any import outside
  `jax / numpy / typing / math / dataclasses / functools`, **including imports nested
  inside functions**, which a top-of-file grep would miss.
- Both modules are loaded straight off disk with `importlib`, outside the package, to
  prove there is no gym coupling at all.
- Neither may import numpy: that is the adapter's job, and physics drifting into the
  adapter is what would stop the seam being a deletion.
- Every public kernel is `jit(vmap(...))`-ed over a batch of 8, and the surrogate
  additionally through `jax.grad`, pinning fixed shapes and no data-dependent control flow.
- `WallSegments` is checked to carry a unit outward normal per segment, and
  `segment_contact` to require one. A wall array built for a scan backend would drop
  the normals -- segment-segment intersection returns a boolean with no side, which
  contact cannot use.

Verified to fail when it should: adding `from ..state import SimulationState` at the
top of `kernels.py` fails two of the tests, and hiding
`from ..collision_models import get_vertices` inside a function body still fails one.

## Opponent occlusion — the field-of-view wrap

`get_blocked_view_indices` took the min and max of the four corner beam indices. That
is only the body's angular extent when the body does not straddle the ends of the
scan. An opponent directly behind has corners either side of +/-pi, both outside the
270 degree field of view, so they clamped to opposite ends and the bounds became the
whole scan — every beam tested against a body no beam can reach.

| opponent, 1080 beams / 270 deg | beams swept | beams actually hit | per call |
|---|---|---|---|
| ahead 3 m | 28 | 26 | 30.1 us |
| ahead-left 3 m | 49 | 49 | 45.9 us |
| behind-right 3 m | 26 | 25 | 28.9 us |
| **directly behind 3 m** | **1080** | **0** | **871.3 us** |
| **behind 0.6 m, touching** | **1080** | **0** | **869.1 us** |

Replaced by recovering the arc as the complement of the widest gap between corner
bearings — well defined for any convex body the scanner sits outside — then
intersecting that arc with the scan's own span. A body across the ends yields two
tails, kept separate: unioning them is what produced the full sweep.

Rear cases now cost **1.3 us**. Visible cases keep byte-identical index ranges.

### It is a correctness fix too

Verified against a brute-force all-beams sweep, 500 random poses per configuration.
The culled sweep never misses a beam the brute force shortens, and the full-sweep
pathology is gone everywhere:

| configuration | full sweeps before | after | beams swept before | after |
|---|---|---|---|---|
| 1080 beams, 270 deg | 83 | **0** | 203,852 | 113,721 |
| same, close quarters | 297 | **0** | 771,668 | 450,419 |
| 108 beams, 270 deg | 106 | **0** | 26,092 | 13,901 |
| 1080 beams, 360 deg | 0 | **0** | 197,279 | 113,061 |
| 720 beams, 180 deg | 103 | **0** | 152,224 | 76,286 |

The old rule also **missed** occlusions: 83 of 4,000 poses at 360 degrees and 6 at
close quarters, where a partially wrapped body left one tail outside the swept range.
There is no case where the new rule misses one the old caught.

### Step time: the quadratic is gone

Same session, same machine, `LIDAR_SCAN`, Spielberg, grid spawn:

| N | before | after | speedup | after, real-time factor at dt=0.01 |
|---|---|---|---|---|
| 1 | 0.243 ms | 0.244 ms | 1.0x | 41x |
| 2 | 1.373 ms | 0.550 ms | 2.5x | 18x |
| 4 | 6.340 ms | 1.073 ms | **5.9x** | 9.3x |
| 8 | 26.489 ms | 2.286 ms | **11.6x** | 4.4x |
| 12 | 59.634 ms | 3.729 ms | **16.0x** | 2.7x |

Scaling is now close to linear in N rather than quadratic — doubling the field roughly
doubles the step. An 8-car race went from 0.38x real time to 4.4x.
