import math
import unittest

import numpy as np

from f1tenth_gym.envs.dynamic_models import F1TENTH_VEHICLE_PARAMETERS as F1TENTH
from f1tenth_gym.envs.env_config import DomainRandomizationConfig
from f1tenth_gym.envs.track import Track
from f1tenth_gym.envs.track.budget import (
    query_half_extent,
    tile_budget,
    track_budget,
    widest_query_half_extent,
)
from f1tenth_gym.envs.track.walls import extract_walls, wall_segments

QH_TEST = query_half_extent(0.58, 0.31)


def brute_force_k(walls, qh, tile_size, origin, tile_shape):
    """Largest candidate count over every tile, counted one tile at a time."""
    a, b = walls.a.astype(np.float64), walls.b.astype(np.float64)
    lo, hi = np.minimum(a, b) - qh, np.maximum(a, b) + qh
    rows, cols = tile_shape
    worst = 0
    for r in range(rows):
        y0, y1 = origin[1] + r * tile_size, origin[1] + (r + 1) * tile_size
        band = (hi[:, 1] >= y0) & (lo[:, 1] < y1)
        if not band.any():
            continue
        lo_b, hi_b = lo[band], hi[band]
        for c in range(cols):
            x0, x1 = origin[0] + c * tile_size, origin[0] + (c + 1) * tile_size
            worst = max(worst, int(((hi_b[:, 0] >= x0) & (lo_b[:, 0] < x1)).sum()))
    return worst


def blobs_grid(size=120):
    grid = np.full((size, size), 255.0)
    grid[20:30, 15:70] = 0.0
    grid[60:64, 30:100] = 0.0
    grid[80:100, 80:86] = 0.0
    return grid


class TestExactness(unittest.TestCase):
    """The whole point of the module: a maximum, not an estimate."""

    def test_matches_brute_force_over_every_tile(self):
        walls = extract_walls(blobs_grid(), 0.05)
        qh = query_half_extent(0.58, 0.31)
        for tile_size in (0.25, 0.5, 1.0):
            budget = tile_budget(walls, qh, tile_size=tile_size)
            self.assertEqual(
                budget.k_tile,
                brute_force_k(walls, qh, tile_size, budget.origin, budget.tile_shape),
                f"tile_size={tile_size}",
            )

    def test_no_real_query_pose_exceeds_the_budget(self):
        track = Track.from_track_name("Monza", 1.0)
        walls = wall_segments(track)
        qh = query_half_extent(F1TENTH.length, F1TENTH.width)
        budget = tile_budget(walls, qh)
        a, b = walls.a.astype(np.float64), walls.b.astype(np.float64)
        lo, hi = np.minimum(a, b) - qh, np.maximum(a, b) + qh

        rng = np.random.default_rng(11)
        free = np.argwhere(track.occupancy_map != 0.0)
        pick = free[rng.choice(len(free), 1500, replace=False)]
        res = float(track.spec.resolution)
        centres = np.stack(
            [
                track.spec.origin[0] + (pick[:, 1] + 0.5) * res,
                track.spec.origin[1] + (pick[:, 0] + 0.5) * res,
            ],
            axis=1,
        )
        for cx, cy in centres:
            seen = int(
                (
                    (hi[:, 0] >= cx - qh)
                    & (lo[:, 0] <= cx + qh)
                    & (hi[:, 1] >= cy - qh)
                    & (lo[:, 1] <= cy + qh)
                ).sum()
            )
            self.assertLessEqual(seen, budget.k_tile)

    def test_a_bigger_body_never_lowers_the_budget(self):
        walls = extract_walls(blobs_grid(), 0.05)
        counts = [tile_budget(walls, qh).k_tile for qh in (0.2, 0.33, 0.6, 1.0)]
        self.assertEqual(counts, sorted(counts))


class TestQueryExtent(unittest.TestCase):
    def test_half_extent_is_the_half_diagonal(self):
        self.assertAlmostEqual(query_half_extent(0.58, 0.31), math.hypot(0.58, 0.31) / 2)

    def test_nominal_body_without_randomization(self):
        nominal = query_half_extent(F1TENTH.length, F1TENTH.width)
        self.assertAlmostEqual(widest_query_half_extent(F1TENTH, None), nominal)

    def test_randomized_body_widens_the_extent(self):
        # widest_params() widens limit fields for the observation bounds and leaves
        # the body alone, so the budget must read the DR range endpoints instead.
        dr = DomainRandomizationConfig(
            enabled=True,
            low=F1TENTH.with_updates(length=0.55, width=0.30),
            high=F1TENTH.with_updates(length=0.70, width=0.40),
        )
        self.assertAlmostEqual(
            widest_query_half_extent(F1TENTH, dr), math.hypot(0.70, 0.40) / 2
        )

    def test_disabled_randomization_is_ignored(self):
        dr = DomainRandomizationConfig(
            enabled=False,
            low=F1TENTH.with_updates(length=0.55),
            high=F1TENTH.with_updates(length=0.70),
        )
        self.assertAlmostEqual(
            widest_query_half_extent(F1TENTH, dr),
            query_half_extent(F1TENTH.length, F1TENTH.width),
        )

    def test_a_randomized_budget_is_never_smaller(self):
        walls = extract_walls(blobs_grid(), 0.05)
        dr = DomainRandomizationConfig(
            enabled=True,
            low=F1TENTH.with_updates(length=0.55, width=0.30),
            high=F1TENTH.with_updates(length=0.90, width=0.60),
        )
        nominal = tile_budget(walls, widest_query_half_extent(F1TENTH, None))
        widened = tile_budget(walls, widest_query_half_extent(F1TENTH, dr))
        self.assertGreaterEqual(widened.k_tile, nominal.k_tile)


class TestDegenerate(unittest.TestCase):
    def test_empty_walls_give_a_zero_budget(self):
        walls = extract_walls(np.full((40, 40), 255.0), 0.05)
        budget = tile_budget(walls, 0.33)
        self.assertTrue(walls.is_empty)
        self.assertEqual(budget.k_tile, 0)
        self.assertEqual(budget.tile_shape, (0, 0))
        self.assertEqual(budget.table_bytes, 0)

    def test_blank_track_gives_a_zero_budget(self):
        track = Track.from_track_name("Spielberg_blank", 1.0)
        self.assertEqual(track_budget(track, F1TENTH).k_tile, 0)

    def test_bad_arguments_raise(self):
        walls = extract_walls(blobs_grid(), 0.05)
        for tile in (0.0, -1.0, float("nan")):
            with self.assertRaises(ValueError):
                tile_budget(walls, 0.33, tile_size=tile)
        for qh in (0.0, -1.0, float("inf")):
            with self.assertRaises(ValueError):
                tile_budget(walls, qh)
        for margin in (0.9, float("nan")):
            with self.assertRaises(ValueError):
                tile_budget(walls, 0.33, margin=margin)


class TestReportedSizes(unittest.TestCase):
    def test_margin_and_table_size_are_consistent(self):
        walls = extract_walls(blobs_grid(), 0.05)
        budget = tile_budget(walls, 0.33, tile_size=0.5, margin=1.25)
        self.assertEqual(budget.k_tile_safe, math.ceil(budget.k_tile * 1.25))
        rows, cols = budget.tile_shape
        self.assertEqual(budget.table_bytes, rows * cols * budget.k_tile_safe * 4)
        self.assertEqual(budget.n_segments, len(walls))

    def test_the_grid_covers_every_expanded_segment_box(self):
        walls = extract_walls(blobs_grid(), 0.05)
        qh = 0.33
        budget = tile_budget(walls, qh, tile_size=0.5)
        hi = np.maximum(walls.a, walls.b).astype(np.float64) + qh
        rows, cols = budget.tile_shape
        self.assertLessEqual(hi[:, 0].max(), budget.origin[0] + cols * budget.tile_size)
        self.assertLessEqual(hi[:, 1].max(), budget.origin[1] + rows * budget.tile_size)

    def test_scaling_the_track_scales_the_tile_grid(self):
        small = track_budget(Track.from_track_name("Monza", 1.0), F1TENTH)
        large = track_budget(Track.from_track_name("Monza", 4.0), F1TENTH)
        self.assertEqual(small.n_segments, large.n_segments)
        self.assertGreater(large.tile_shape[0], small.tile_shape[0])

    def test_budget_is_deterministic(self):
        track = Track.from_track_name("Monza", 1.0)
        self.assertEqual(track_budget(track, F1TENTH), track_budget(track, F1TENTH))


class TestMemoryGuard(unittest.TestCase):
    """A 0.25 m tile on a map_scale=10 track projects to gigabytes."""

    def test_a_tiny_tile_is_refused_before_allocating(self):
        walls = extract_walls(blobs_grid(), 0.05)
        with self.assertRaises(MemoryError) as caught:
            tile_budget(walls, QH_TEST, tile_size=0.001)
        message = str(caught.exception)
        self.assertIn("0.001", message)
        self.assertIn("max_bytes", message)

    def test_the_cap_is_honoured_exactly(self):
        walls = extract_walls(blobs_grid(), 0.05)
        with self.assertRaises(MemoryError):
            tile_budget(walls, QH_TEST, tile_size=0.5, max_bytes=1)
        self.assertGreater(tile_budget(walls, QH_TEST, tile_size=0.5).k_tile, 0)

    def test_raising_the_cap_permits_a_deliberate_large_build(self):
        walls = extract_walls(blobs_grid(), 0.05)
        with self.assertRaises(MemoryError):
            tile_budget(walls, QH_TEST, tile_size=0.02, max_bytes=100_000)
        self.assertGreater(
            tile_budget(walls, QH_TEST, tile_size=0.02, max_bytes=64 * 1024**2).k_tile, 0
        )

    def test_track_budget_forwards_the_cap(self):
        track = Track.from_track_name("Monza", 1.0)
        with self.assertRaises(MemoryError):
            track_budget(track, F1TENTH, max_bytes=1)

    def test_an_empty_track_is_never_refused(self):
        track = Track.from_track_name("Spielberg_blank", 1.0)
        self.assertEqual(track_budget(track, F1TENTH, max_bytes=1).k_tile, 0)


if __name__ == "__main__":
    unittest.main()
