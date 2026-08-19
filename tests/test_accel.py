import unittest

import numpy as np

from f1tenth_gym.envs.dynamic_models import F1TENTH_VEHICLE_PARAMETERS as F1TENTH
from f1tenth_gym.envs.track import Track
from f1tenth_gym.envs.track.accel import (
    brute_force_candidates,
    build_for_track,
    build_tile_index,
    gather,
    tile_coords,
)
from f1tenth_gym.envs.track.budget import query_half_extent, tile_budget
from f1tenth_gym.envs.track.walls import extract_walls

QH = query_half_extent(0.58, 0.31)


def blobs_grid(size=120):
    grid = np.full((size, size), 255.0)
    grid[20:30, 15:70] = 0.0
    grid[60:64, 30:100] = 0.0
    grid[80:100, 80:86] = 0.0
    return grid


def built(grid=None, tile_size=0.5, qh=QH):
    walls = extract_walls(blobs_grid() if grid is None else grid, 0.05)
    budget = tile_budget(walls, qh, tile_size=tile_size)
    return walls, budget, build_tile_index(walls, budget)


def sample_points(budget, count, seed=0):
    rng = np.random.default_rng(seed)
    rows, cols = budget.tile_shape
    return np.stack(
        [
            rng.uniform(budget.origin[0], budget.origin[0] + cols * budget.tile_size, count),
            rng.uniform(budget.origin[1], budget.origin[1] + rows * budget.tile_size, count),
        ],
        axis=1,
    )


class TestNeverMisses(unittest.TestCase):
    """The one property a broad phase must have: a superset, never a subset."""

    def test_synthetic_grid_over_many_tile_sizes(self):
        for tile_size in (0.25, 0.5, 1.0, 2.0):
            walls, budget, index = built(tile_size=tile_size)
            points = sample_points(budget, 600, seed=1)
            got = gather(points, index)
            for i, point in enumerate(points):
                truth = set(brute_force_candidates(walls, point, QH).tolist())
                cand = {int(x) for x in got[i] if x >= 0}
                self.assertTrue(
                    truth <= cand,
                    f"tile_size={tile_size}: missed {sorted(truth - cand)}",
                )

    def test_real_map(self):
        track = Track.from_track_name("Monza", 1.0)
        walls, budget, index = build_for_track(track, F1TENTH)
        qh = budget.query_half_extent
        rng = np.random.default_rng(7)
        free = np.argwhere(track.occupancy_map != 0.0)
        pick = free[rng.choice(len(free), 800, replace=False)]
        res = float(track.spec.resolution)
        points = np.stack(
            [
                track.spec.origin[0] + (pick[:, 1] + 0.5) * res,
                track.spec.origin[1] + (pick[:, 0] + 0.5) * res,
            ],
            axis=1,
        )
        got = gather(points, index)
        for i, point in enumerate(points):
            truth = set(brute_force_candidates(walls, point, qh).tolist())
            self.assertTrue(truth <= {int(x) for x in got[i] if x >= 0})

    def test_a_query_on_a_tile_seam_still_sees_everything(self):
        walls, budget, index = built(tile_size=0.5)
        rows, cols = budget.tile_shape
        seams = np.array(
            [
                [budget.origin[0] + c * budget.tile_size, budget.origin[1] + r * budget.tile_size]
                for r in range(0, rows, 3)
                for c in range(0, cols, 3)
            ]
        )
        got = gather(seams, index)
        for i, point in enumerate(seams):
            truth = set(brute_force_candidates(walls, point, QH).tolist())
            self.assertTrue(truth <= {int(x) for x in got[i] if x >= 0})


class TestTableShape(unittest.TestCase):
    def test_table_matches_the_budget(self):
        _walls, budget, index = built()
        rows, cols = budget.tile_shape
        self.assertEqual(index.table.shape, (rows, cols, budget.k_tile_safe))
        self.assertEqual(index.table.dtype, np.int32)
        self.assertEqual(index.k, budget.k_tile_safe)
        self.assertEqual(index.origin, budget.origin)
        self.assertEqual(index.tile_size, budget.tile_size)

    def test_entries_are_valid_indices_or_padding(self):
        walls, _budget, index = built()
        flat = index.table.reshape(-1)
        self.assertTrue(((flat == -1) | ((flat >= 0) & (flat < len(walls)))).all())

    def test_padding_is_right_aligned_within_each_tile(self):
        # Occupied slots must be contiguous from 0, so a kernel can stop at the
        # first -1 instead of scanning the whole row.
        _walls, _budget, index = built()
        occupied = index.table >= 0
        self.assertTrue((np.diff(occupied.astype(np.int8), axis=2) <= 0).all())

    def test_no_tile_overflows_its_budget(self):
        _walls, budget, index = built()
        for r in range(index.table.shape[0]):
            for c in range(index.table.shape[1]):
                used = int((index.table[r, c] >= 0).sum())
                self.assertLessEqual(used, budget.k_tile)

    def test_a_budget_from_different_walls_is_rejected(self):
        _walls, budget, _index = built()
        other = extract_walls(np.pad(blobs_grid(), 4, constant_values=255.0), 0.05)
        with self.assertRaises(ValueError):
            build_tile_index(other, budget)


class TestOffMapAndEmpty(unittest.TestCase):
    def test_off_map_queries_are_clamped_not_crashing(self):
        _walls, _budget, index = built()
        far = np.array([[1e4, 1e4], [-1e4, -1e4], [0.0, 1e4], [-1e4, 0.0]])
        rows, cols = index.table.shape[0], index.table.shape[1]
        r, c = tile_coords(far, index)
        self.assertTrue(((r >= 0) & (r < rows)).all())
        self.assertTrue(((c >= 0) & (c < cols)).all())
        self.assertEqual(gather(far, index).shape, (4, index.k))

    def test_empty_walls_gather_to_all_padding(self):
        walls = extract_walls(np.full((40, 40), 255.0), 0.05)
        budget = tile_budget(walls, QH)
        index = build_tile_index(walls, budget)
        self.assertTrue(index.is_empty or (index.table < 0).all())
        self.assertTrue((gather(np.array([[0.0, 0.0], [9.0, 3.0]]), index) < 0).all())

    def test_blank_track_builds_without_raising(self):
        track = Track.from_track_name("Spielberg_blank", 1.0)
        walls, budget, index = build_for_track(track, F1TENTH)
        self.assertTrue(walls.is_empty)
        self.assertEqual(budget.k_tile, 0)
        self.assertTrue((gather(np.zeros((3, 2)), index) < 0).all())


class TestTrackIntegration(unittest.TestCase):
    def test_cache_returns_the_same_index(self):
        track = Track.from_track_name("Monza", 1.0)
        self.assertIs(build_for_track(track, F1TENTH)[2], build_for_track(track, F1TENTH)[2])

    def test_a_wider_body_rebuilds_the_index(self):
        from f1tenth_gym.envs.env_config import DomainRandomizationConfig

        track = Track.from_track_name("Monza", 1.0)
        nominal = build_for_track(track, F1TENTH)[2]
        dr = DomainRandomizationConfig(
            enabled=True,
            low=F1TENTH.with_updates(length=0.55, width=0.30),
            high=F1TENTH.with_updates(length=0.95, width=0.65),
        )
        widened = build_for_track(track, F1TENTH, dr)[2]
        self.assertIsNot(nominal, widened)
        self.assertGreater(widened.query_half_extent, nominal.query_half_extent)

    def test_build_is_deterministic(self):
        walls = extract_walls(blobs_grid(), 0.05)
        budget = tile_budget(walls, QH)
        first = build_tile_index(walls, budget)
        second = build_tile_index(walls, budget)
        self.assertTrue(np.array_equal(first.table, second.table))


class TestMemoryGuard(unittest.TestCase):
    def test_the_table_is_refused_before_allocating(self):
        walls, budget, _index = built()
        with self.assertRaises(MemoryError) as caught:
            build_tile_index(walls, budget, max_bytes=1)
        self.assertIn("build_tile_index", str(caught.exception))

    def test_build_for_track_forwards_the_cap(self):
        track = Track.from_track_name("Monza", 1.0)
        with self.assertRaises(MemoryError):
            build_for_track(track, F1TENTH, max_bytes=1)

    def test_an_empty_track_is_never_refused(self):
        track = Track.from_track_name("Spielberg_blank", 1.0)
        _walls, _budget, index = build_for_track(track, F1TENTH, max_bytes=1)
        self.assertTrue((gather(np.zeros((2, 2)), index) < 0).all())


if __name__ == "__main__":
    unittest.main()
