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
