import unittest

import numpy as np

from f1tenth_gym.envs.track import Track
from f1tenth_gym.envs.track.walls import (
    WallSegments,
    _occupied_at,
    extract_walls,
    has_subpixel_edges,
    wall_segments,
)


def points_inside_rings(walls, points):
    """Even-odd crossing test against the segment set itself.

    Independent of how the normals were derived, so it can adjudicate them.
    """
    a, b = walls.a.astype(np.float64), walls.b.astype(np.float64)
    count = np.zeros(len(points), dtype=int)
    for k in range(len(a)):
        straddles = (a[k, 1] > points[:, 1]) != (b[k, 1] > points[:, 1])
        if not straddles.any():
            continue
        dy = b[k, 1] - a[k, 1]
        x_cross = a[k, 0] + (points[:, 1] - a[k, 1]) * (b[k, 0] - a[k, 0]) / (dy + 1e-30)
        count += (straddles & (points[:, 0] < x_cross)).astype(int)
    return count % 2 == 1


def inverted_normal_count(walls):
    """Segments whose normal points into the enclosed (occupied) region."""
    mid = 0.5 * (walls.a.astype(np.float64) + walls.b.astype(np.float64))
    n = walls.n.astype(np.float64)
    eps = 1e-5
    return int((points_inside_rings(walls, mid + eps * n)
                & ~points_inside_rings(walls, mid - eps * n)).sum())


def half_plane(angle_deg, size=300, resolution=0.05):
    """Binary occupancy grid split by a line, free border so the ring closes."""
    theta = np.radians(angle_deg)
    rows, cols = np.mgrid[0:size, 0:size]
    grid = np.full((size, size), 255.0)
    grid[(cols - size / 2) * np.sin(theta) - (rows - size / 2) * np.cos(theta) < 0] = 0.0
    grid[:3, :] = grid[-3:, :] = grid[:, :3] = grid[:, -3:] = 255.0
    return grid, theta, resolution


def antialiased_half_plane(angle_deg, size=300):
    """Greyscale grid whose edge pixels carry a one-pixel coverage ramp."""
    theta = np.radians(angle_deg)
    rows, cols = np.mgrid[0:size, 0:size]
    signed = (cols - size / 2) * np.sin(theta) - (rows - size / 2) * np.cos(theta)
    grey = (np.clip(signed + 0.5, 0.0, 1.0) * 255.0)
    grey[:3, :] = grey[-3:, :] = grey[:, :3] = grey[:, -3:] = 255.0
    grey = grey.astype(np.uint8)
    return np.where(grey <= 140, 0.0, 255.0).astype(np.float32), grey, theta


def slam_style_map(size=300):
    """ROS map_saver output: exactly 0 occupied / 205 unknown / 254 free."""
    grey = np.full((size, size), 254, dtype=np.uint8)
    rows, cols = np.mgrid[0:size, 0:size]
    grey[(cols > 40) & (cols < 260) & (rows > 150) & (rows < 158)] = 0
    grey[(cols - 70) ** 2 + (rows - 70) ** 2 < 45**2] = 205
    return np.where(grey <= 140, 0.0, 255.0).astype(np.float32), grey


class TestSpielbergWalls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.track = Track.from_track_name("Spielberg", 1.0)
        cls.walls = wall_segments(cls.track)
        cls.res = float(cls.track.spec.resolution)
        cls.origin = tuple(float(v) for v in cls.track.spec.origin[:3])

    def test_uses_the_subpixel_path(self):
        self.assertTrue(has_subpixel_edges(self.track.occupancy_grey))
        self.assertEqual(len(self.walls), 833)

    def test_dtypes(self):
        self.assertEqual(self.walls.a.dtype, np.float32)
        self.assertEqual(self.walls.n.dtype, np.float32)
        self.assertEqual(self.walls.length.dtype, np.float32)
        self.assertEqual(self.walls.adj.dtype, np.int32)

    def test_normals_are_unit(self):
        self.assertTrue(np.allclose(np.linalg.norm(self.walls.n, axis=1), 1.0, atol=1e-6))

    def test_normals_are_perpendicular_to_their_own_segment(self):
        # Bound is float32 eps amplified by (world magnitude / shortest segment).
        edge = (self.walls.b - self.walls.a).astype(np.float64)
        edge /= np.linalg.norm(edge, axis=1, keepdims=True)
        self.assertLess(np.abs((edge * self.walls.n).sum(axis=1)).max(), 5e-4)

    def test_no_normal_points_into_the_wall(self):
        self.assertEqual(inverted_normal_count(self.walls), 0)

    def test_the_boundary_is_smooth_not_a_staircase(self):
        # Binarising first gives a 45 deg turn at every step; sub-pixel gives a few.
        nxt = self.walls.adj[:, 1]
        edge = (self.walls.b - self.walls.a).astype(np.float64)
        edge /= np.linalg.norm(edge, axis=1, keepdims=True)
        turn = np.degrees(np.arccos(np.clip((edge * edge[nxt]).sum(axis=1), -1.0, 1.0)))
        self.assertLess(float(np.median(turn)), 10.0)

    def test_adjacency_forms_closed_rings(self):
        nxt = self.walls.adj[:, 1]
        self.assertTrue((nxt >= 0).all())
        visited = np.zeros(len(self.walls), dtype=bool)
        rings = 0
        for start in range(len(self.walls)):
            if visited[start]:
                continue
            rings += 1
            cur, steps = start, 0
            while True:
                visited[cur] = True
                cur = int(nxt[cur])
                steps += 1
                self.assertLessEqual(steps, len(self.walls))
                if cur == start:
                    break
        self.assertTrue(visited.all())
        self.assertEqual(rings, 4)

    def test_adjacency_is_geometrically_consistent(self):
        nxt, prev = self.walls.adj[:, 1], self.walls.adj[:, 0]
        self.assertTrue(np.allclose(self.walls.b, self.walls.a[nxt], atol=1e-5))
        self.assertTrue(np.allclose(self.walls.a, self.walls.b[prev], atol=1e-5))

    def test_lengths_match_the_endpoints(self):
        expected = np.linalg.norm(self.walls.b - self.walls.a, axis=1)
        self.assertTrue(np.allclose(self.walls.length, expected, atol=1e-5))
        self.assertGreater(self.walls.length.min(), 0.0)

    def test_segments_lie_in_world_frame_not_pixel_frame(self):
        self.assertLess(self.walls.a[:, 0].min(), 0.0)
        height, width = self.track.occupancy_map.shape
        for axis, extent in ((0, width * self.res), (1, height * self.res)):
            low = self.origin[axis]
            self.assertGreaterEqual(self.walls.a[:, axis].min(), low - 1e-3)
            self.assertLessEqual(self.walls.a[:, axis].max(), low + extent + 1e-3)


class TestToleranceRobustness(unittest.TestCase):
    """tol_px is public; a probe-based orientation inverted 17% of normals at 2.0."""

    def test_normals_stay_outward_at_every_tolerance(self):
        for name in ("Spielberg", "Monza"):
            track = Track.from_track_name(name, 1.0)
            for tol in (0.25, 1.0, 2.0, 4.0):
                walls = wall_segments(track, tol_px=tol)
                self.assertEqual(
                    inverted_normal_count(walls), 0,
                    f"{name} at tol_px={tol}: normals point into the wall",
                )

    def test_segment_count_is_monotone_in_tolerance(self):
        track = Track.from_track_name("Monza", 1.0)
        counts = [len(wall_segments(track, tol_px=t)) for t in (0.25, 0.5, 1.0, 2.0, 4.0)]
        self.assertEqual(counts, sorted(counts, reverse=True))

    def test_a_small_obstacle_survives_a_coarse_tolerance(self):
        grid = np.full((20, 20), 255.0)
        grid[9, 9] = 0.0
        for tol in (0.25, 1.0, 3.0):
            self.assertGreaterEqual(
                len(extract_walls(grid, 0.05, tol_px=tol)), 3,
                f"single-pixel obstacle deleted at tol_px={tol}",
            )


class TestAwkwardTopology(unittest.TestCase):
    def test_corner_touching_cells_keep_valid_normals(self):
        # 4-connected contouring splits a diagonal pinch into two islands whose
        # facing facets have an obstacle on both sides, so no normal is correct.
        grid = np.full((9, 9), 255.0)
        grid[3, 3] = grid[4, 4] = 0.0
        walls = extract_walls(grid, 0.05)
        mid = 0.5 * (walls.a + walls.b)
        hit = _occupied_at(mid + 0.7 * 0.05 * walls.n, grid == 0.0, 0.05, (0.0, 0.0, 0.0))
        self.assertEqual(int(hit.sum()), 0)

    def test_obstacle_touching_the_border_closes_into_a_ring(self):
        grid = np.full((20, 20), 255.0)
        grid[0:6, 0:6] = 0.0
        walls = extract_walls(grid, 0.05)
        self.assertFalse(walls.is_empty)
        self.assertTrue((walls.adj >= 0).all(), "border obstacle left an open chain")

    def test_separate_obstacles_never_share_adjacency(self):
        grid = np.full((40, 40), 255.0)
        grid[5:10, 5:10] = 0.0
        grid[25:33, 25:33] = 0.0
        walls = extract_walls(grid, 0.05)
        self.assertTrue(np.allclose(walls.b, walls.a[walls.adj[:, 1]], atol=1e-9))


class TestSubpixelDetection(unittest.TestCase):
    def test_antialiased_grid_recovers_the_analytic_normal(self):
        for angle in (7.0, 15.0, 30.0, 45.0, 60.0):
            occ, grey, theta = antialiased_half_plane(angle)
            walls = extract_walls(occ, 0.05, grayscale=grey, occupied_thresh=0.45)
            want = np.array([np.sin(theta), -np.cos(theta)])
            mid = 0.5 * (walls.a.astype(np.float64) + walls.b.astype(np.float64))
            col, row = mid[:, 0] / 0.05 - 0.5, mid[:, 1] / 0.05 - 0.5
            face = np.abs((col - 150) * np.sin(theta) - (row - 150) * np.cos(theta)) < 2.0
            if face.sum() == 0:
                continue
            weight = walls.length[face].astype(np.float64)
            got = (walls.n[face].astype(np.float64) * weight[:, None]).sum(axis=0)
            got /= np.linalg.norm(got)
            err = np.degrees(np.arccos(np.clip(abs(got @ want), -1.0, 1.0)))
            self.assertLess(err, 1.0, f"{angle} deg: aggregate normal off by {err}")

    def test_binarising_the_same_wall_gives_far_more_segments(self):
        occ, grey, _theta = antialiased_half_plane(30.0)
        binary = extract_walls(occ, 0.05)
        subpixel = extract_walls(occ, 0.05, grayscale=grey, occupied_thresh=0.45)
        self.assertGreater(len(binary), 4 * len(subpixel))

    def test_slam_style_map_falls_back_to_binary(self):
        occ, grey = slam_style_map()
        self.assertFalse(has_subpixel_edges(grey))
        walls = extract_walls(occ, 0.05, grayscale=grey, occupied_thresh=0.45)
        self.assertEqual(len(walls), len(extract_walls(occ, 0.05)))
        self.assertEqual(inverted_normal_count(walls), 0)

    def test_detector_says_no_without_a_greyscale(self):
        self.assertFalse(has_subpixel_edges(None))


class TestBinaryFallbackNormals(unittest.TestCase):
    def test_all_binary_normals_point_outward(self):
        for angle in (0.0, 15.0, 45.0, 75.0):
            grid, _theta, res = half_plane(angle)
            walls = extract_walls(grid, res)
            mid = 0.5 * (walls.a + walls.b)
            hit = _occupied_at(mid + 0.7 * res * walls.n, grid == 0.0, res, (0.0, 0.0, 0.0))
            self.assertEqual(int(hit.sum()), 0, f"{angle} deg wall has inward normals")


class TestDegenerateGrids(unittest.TestCase):
    def test_all_free_grid_is_empty(self):
        walls = extract_walls(np.full((50, 50), 255.0), 0.05)
        self.assertTrue(walls.is_empty)
        self.assertEqual(walls.a.shape, (0, 2))
        self.assertEqual(walls.adj.shape, (0, 2))

    def test_all_occupied_grid_is_empty(self):
        self.assertTrue(extract_walls(np.zeros((50, 50)), 0.05).is_empty)

    def test_blank_map_is_empty(self):
        self.assertTrue(wall_segments(Track.from_track_name("Spielberg_blank", 1.0)).is_empty)

    def test_synthetic_track_is_empty(self):
        track = Track.from_refline(
            x=np.linspace(0.0, 10.0, 50), y=np.zeros(50), velx=np.ones(50)
        )
        self.assertIsNone(track.occupancy_grey)
        self.assertTrue(wall_segments(track).is_empty)

    def test_malformed_input_raises_rather_than_emptying(self):
        for bad in (0.0, float("nan")):
            with self.assertRaises(ValueError):
                extract_walls(np.zeros((4, 4)), bad)
        for tol in (float("nan"), float("inf"), -1.0):
            with self.assertRaises(ValueError):
                extract_walls(np.zeros((4, 4)), 0.05, tol_px=tol)
        with self.assertRaises(ValueError):
            extract_walls(np.zeros(10), 0.05)

    def test_a_huge_origin_raises_instead_of_collapsing_endpoints(self):
        grid = np.full((60, 60), 255.0)
        grid[20:40, 20:40] = 0.0
        with self.assertRaises(ValueError):
            extract_walls(grid, 0.05, origin=(1.0e8, 1.0e8, 0.0))


class TestTrackCache(unittest.TestCase):
    def test_second_call_returns_the_same_object(self):
        track = Track.from_track_name("Monza", 1.0)
        first = wall_segments(track)
        self.assertIs(wall_segments(track), first)

    def test_scaled_track_does_not_reuse_the_cache(self):
        base = Track.from_track_name("Monza", 1.0)
        scaled = Track.from_track_name("Monza", 4.0)
        small, large = wall_segments(base), wall_segments(scaled)
        self.assertIsNot(small, large)
        self.assertEqual(len(small), len(large))
        self.assertAlmostEqual(float(large.length.sum() / small.length.sum()), 4.0, places=3)

    def test_tolerance_change_does_not_reuse_the_cache(self):
        track = Track.from_track_name("Monza", 1.0)
        self.assertIsNot(wall_segments(track), wall_segments(track, tol_px=2.0))

    def test_walls_are_a_namedtuple_of_arrays(self):
        walls = wall_segments(Track.from_track_name("Monza", 1.0))
        self.assertIsInstance(walls, WallSegments)
        self.assertEqual(len(walls._fields), 5)


if __name__ == "__main__":
    unittest.main()
