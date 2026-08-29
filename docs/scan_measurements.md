# LiDAR scanner migration — measured numbers

The raster/EDT backend measured here was retired on 2026-08-29. Exact segment
intersection is now the sole scanner. The former results remain as the
reproducible rationale for that removal, not as a list of selectable backends.

Every number here was produced on this tree. Machine: RTX 3080 Laptop, py3.13,
jax 0.11.1, numpy 2.4. Ground truth throughout is exact ray-segment intersection
against `wall_segments(track)`, itself validated against a synthetic half-plane whose
wall position is known in closed form: **0.000 um max error over 721 beams**.

## Why the former raster backend was removed

`get_dt = resolution * distance_transform_edt(bitmap)` measures the distance to the
nearest occupied cell **centre**. The wall face is half a cell nearer, so every range
overshoots. Sphere tracing converges on the zero level of that field, so the error is
definitional, not a tracing bug.

Per-beam range error, EDT minus exact, over ~730,000 beams (150 raceline stations x 3
lateral offsets x yaw jitter, poses within 0.45 m of a wall rejected):

| track | resolution | median signed | % reading long | median abs | p99 | in pixels |
|---|---|---|---|---|---|---|
| Spielberg | 0.05796 | **+0.0297 m** | 83.0 | 0.0328 | 0.211 | **+0.513** |
| Monza | 0.09585 | **+0.0472 m** | 81.6 | 0.0532 | 0.336 | **+0.493** |
| Austin | 0.08089 | **+0.0400 m** | 83.0 | 0.0445 | 0.210 | **+0.495** |

Half a pixel on three tracks at three resolutions. It is a bias, not scatter, and it is
**range-independent in absolute terms** (~+3 cm everywhere on Spielberg), so relative
error falls from 4.07% at 0-1 m to 0.46% at 20-30 m.

Grazing incidence hurts twice — the error grows 9x and **the sign flips**:

| incidence from the normal | median signed | median abs |
|---|---|---|
| 0-15 deg | +0.0304 | 0.0304 |
| 60-70 deg | +0.0291 | 0.0367 |
| 80-85 deg | **-0.0058** | 0.0949 |
| 85-90 deg | **-0.1546** | 0.2721 |

## What the fuzzy point cloud is made of

Ranked by contribution to the band, perpendicular to the wall (Spielberg, 25,740 beams).
The terms telescope exactly.

| # | cause | perp. mean | perp. rms |
|---|---|---|---|
| 1 | EDT sphere-trace terminal overshoot | **+2.35 cm** | 2.87 cm |
| 2 | raster grid vs the sub-pixel contour | -0.27 cm | 1.49 cm |
| 3 | `noise_std = 0.01` | +0.00 cm | 0.79 cm |
| 4 | beam-angle LUT quantisation | -0.00 cm | 0.49 cm |
| 5 | sphere-trace `eps` | **0.000 cm** | **0.000** |

Cumulative ablation, distance from the drawn point to the true wall:

| layer | mean | rms | p95 |
|---|---|---|---|
| observed | 2.70 cm | 3.31 cm | 6.30 cm |
| noise off | 2.64 | 3.22 | 6.04 |
| + exact beam angles | 2.60 | 3.18 | 6.02 |
| + eps = 1e-9 | 2.60 | 3.18 | 6.02 |
| exact cast vs the raster | 1.22 | 1.49 | 2.78 |
| **exact cast vs sub-pixel** | **0.00** | **0.00** | **0.00** |

Two findings worth keeping:

**`eps` is inert.** `distance_transform` is a bare nearest-cell lookup with no
interpolation, so the smallest non-zero EDT value is exactly one cell (5.796 cm on
Spielberg). Any `eps` below that terminates on the identical cell — the default `1e-4`
is 580x too small to change anything, and `eps = 0.058` shifts ranges by up to 21 m.
It is a step function, not a tolerance.

**Rank 1 leads on bias, not spread.** On zero-mean spread alone it is 1.65 cm against
rank 2's 1.49 cm, essentially tied; its dominance comes from the systematic +2.35 cm.
Ranks 3 and 4 swap depending on whether grazing flyaways are counted. Both leading
terms are the raster path, so the conclusion is unaffected.

## Cost

Per 1080-beam scan on Spielberg, 833 segments, k = 584 candidates per tile:

| backend | N=1 | N=4 | N=8 | N=12 |
|---|---|---|---|---|
| EDT raster (per agent) | 0.133 ms | 0.133 | 0.133 | 0.133 |
| segment, `scan_device="cpu"` | 0.158 | 0.094 | 0.089 | 0.155 |
| segment, `scan_device="gpu"` | 0.146 | 0.058 | **0.030** | **0.019** |

The GPU total is nearly flat — 0.146 ms at N=1 against 0.225 ms at N=12 — so the
per-agent cost collapses as the field grows: **7.09x faster than the EDT at 12 agents**.
This is the opposite of the contact kernels, which are launch-bound and lose on GPU by
10x; a scan is 1080 x 584 intersections, four orders of magnitude more work than a
40-point contact solve, so the launch cost amortises.

Those batched figures need `_update_scans` to make one vmapped call. The mutable
environment still loops per agent, so its path is the N=1 column.

Brute force over all segments is not viable: 3.7 ms (float32) to 9.3 ms (float64) per
scan, 27-117x the EDT.

Ray tile tables, built once and cached on the `Track`:

| track | segments | grid | k | table | build |
|---|---|---|---|---|---|
| Spielberg | 833 | 14x21 | 584 | 0.69 MB | 127 ms |
| Monza | 650 | 34x20 | 216 | 0.59 MB | 121 ms |
| Austin | 1022 | 17x29 | 656 | 1.29 MB | 148 ms |

Note k/S is 70% on Spielberg and 64% on Austin: at `range_max = 30 m` on a ~100 m
circuit, most of the track is reachable from every tile, so the tile gather culls far
less than it would on a large map. It still beats the 32 MB EDT it replaces.

## Removal gate

A recorded `legacy_scan.npz` fixture compared scans at `mse < 2.0` before removal.
Exact segment casting scored **0.0891 / 0.1029 / 0.0205** on Spielberg / Monza /
Austin against the former EDT's 0.0015 / 0.0052 / 0.0049 — 20-100x inside the
threshold. Because `mse < 2.0` against a ~1 m median range is roughly a 140% RMS
tolerance, the weak fixture and its stale generator were deleted with the backend;
exact geometry, host/JAX parity and environment tests are the surviving gates.

The #91 no-false-collision pin holds: min(scan - side_distances) over ~109 centerline
poses is 0.9150 m under the EDT and 0.9025 m exact.

The sensor never entered occupied geometry in 400 steps of a wall-scrape crash.
LiDAR is now independent of collision adjudication; geometric contact handles the
stopping response, while `CollisionCheckMode.NONE` deliberately disables it.

Douglas-Peucker at `wall_tolerance_px = 0.25` leaves 0.008% (Spielberg) / 0.025% (Monza)
of beams more than 1 m from the un-simplified contour, against the EDT's 0.101% / 0.201%
— a ~10x reduction, not elimination.
