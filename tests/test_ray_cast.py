"""Opponent occlusion: which beams `ray_cast` bothers to test.

The body's angular extent has to be recovered as an arc on the circle. Taking the
min and max of the four corner bearings instead assumes the body does not straddle
the ends of the scan, and an opponent directly behind does exactly that -- which used
to collapse the bounds to the first and last beam and sweep all 1080 for nothing.
"""

import unittest

import numpy as np

from f1tenth_gym.envs.collision_models import get_vertices
from f1tenth_gym.envs.lidar.laser_models import get_blocked_view_ranges, ray_cast

LENGTH, WIDTH = 0.58, 0.31
EGO = np.array([0.0, 0.0, 0.0])
CLEAR = 30.0


def angles_for(num_beams=1080, fov=4.7123889):
    return np.linspace(-fov / 2, fov / 2, num_beams)


def swept(pose, verts, scan_angles):
    lo_a, hi_a, lo_b, hi_b = get_blocked_view_ranges(pose, verts, scan_angles)
    return max(0, hi_a - lo_a + 1) + max(0, hi_b - lo_b + 1)


def brute_force(pose, scan_angles, verts):
    """Every beam tested, no culling: the ground truth `ray_cast` must reproduce."""
    from f1tenth_gym.envs.lidar.laser_models import get_range

    looped = np.vstack([verts, verts[0:1]])
    scan = np.full(scan_angles.shape[0], CLEAR)
    for i in range(scan_angles.shape[0]):
        for j in range(4):
            reach = get_range(pose, pose[2] + scan_angles[i], looped[j], looped[j + 1])
            if reach < scan[i]:
                scan[i] = reach
    return scan


class TestBodyBehindIsCulled(unittest.TestCase):
    """The regression: a body no beam can reach must cost no beams."""

    def setUp(self):
        self.angles = angles_for()

    def test_a_body_directly_behind_sweeps_nothing(self):
        verts = get_vertices(np.array([-3.0, 0.0, 0.0]), LENGTH, WIDTH)
        self.assertEqual(swept(EGO, verts, self.angles), 0)

    def test_a_body_behind_and_touching_sweeps_nothing(self):
        verts = get_vertices(np.array([-0.6, 0.0, 0.0]), LENGTH, WIDTH)
        self.assertEqual(swept(EGO, verts, self.angles), 0)

    def test_the_scan_is_untouched_by_a_body_behind(self):
        verts = get_vertices(np.array([-3.0, 0.0, 0.0]), LENGTH, WIDTH)
        scan = np.full(1080, CLEAR)
        np.testing.assert_array_equal(
            ray_cast(EGO, scan, self.angles, verts), np.full(1080, CLEAR))

    def test_a_visible_body_still_sweeps_a_tight_range(self):
        for offset, most in (([3.0, 0.0, 0.0], 40), ([2.1, 2.1, 0.0], 60),
                             ([0.0, 3.0, 0.0], 60)):
            verts = get_vertices(np.array(offset), LENGTH, WIDTH)
            count = swept(EGO, verts, self.angles)
            self.assertGreater(count, 0, str(offset))
            self.assertLess(count, most, str(offset))


class TestNothingIsMissed(unittest.TestCase):
    """Culling may only remove beams that could not have been shortened."""

    def _sweep_matches_brute_force(self, num_beams, fov, spread, trials=500, seed=3):
        angles = angles_for(num_beams, fov)
        rng = np.random.default_rng(seed)
        checked = 0
        for _ in range(trials):
            pose = np.array([rng.uniform(-spread, spread), rng.uniform(-spread, spread),
                             rng.uniform(-np.pi, np.pi)])
            if np.hypot(pose[0], pose[1]) < 0.35:
                continue
            verts = get_vertices(pose, LENGTH, WIDTH)
            got = ray_cast(EGO, np.full(num_beams, CLEAR), angles, verts)
            np.testing.assert_allclose(
                got, brute_force(EGO, angles, verts), atol=1e-9,
                err_msg=f"pose {pose} beams {num_beams} fov {fov}")
            checked += 1
        self.assertGreater(checked, trials // 2)

    def test_default_scan(self):
        self._sweep_matches_brute_force(1080, 4.7123889, 6.0)

    def test_close_quarters(self):
        self._sweep_matches_brute_force(1080, 4.7123889, 1.2)

    def test_few_beams(self):
        self._sweep_matches_brute_force(108, 4.7123889, 6.0)

    def test_full_circle_scan(self):
        self._sweep_matches_brute_force(1080, 2 * np.pi, 6.0)

    def test_narrow_scan(self):
        self._sweep_matches_brute_force(720, np.pi, 6.0)


class TestTheWrapIsSplitNotUnioned(unittest.TestCase):
    """A body across the scan's ends gives two tails; unioning them sweeps everything."""

    def test_a_full_circle_scan_never_sweeps_every_beam(self):
        angles = angles_for(1080, 2 * np.pi)
        rng = np.random.default_rng(5)
        worst = 0
        for _ in range(400):
            pose = np.array([rng.uniform(-6, 6), rng.uniform(-6, 6),
                             rng.uniform(-np.pi, np.pi)])
            if np.hypot(pose[0], pose[1]) < 0.35:
                continue
            worst = max(worst, swept(EGO, get_vertices(pose, LENGTH, WIDTH), angles))
        self.assertGreater(worst, 0)
        self.assertLess(worst, 1080)

    def test_a_body_behind_a_full_circle_scan_uses_both_tails(self):
        angles = angles_for(1080, 2 * np.pi)
        verts = get_vertices(np.array([-3.0, 0.0, 0.0]), LENGTH, WIDTH)
        lo_a, hi_a, lo_b, hi_b = get_blocked_view_ranges(EGO, verts, angles)
        self.assertLessEqual(lo_a, hi_a, "lower tail should be live")
        self.assertLessEqual(lo_b, hi_b, "upper tail should be live")
        self.assertLess(hi_a - lo_a + 1 + hi_b - lo_b + 1, 1080)


class TestCullingIsWorthIt(unittest.TestCase):
    def test_a_rear_body_costs_far_fewer_beams_than_a_front_one(self):
        angles = angles_for()
        front = swept(EGO, get_vertices(np.array([3.0, 0.0, 0.0]), LENGTH, WIDTH), angles)
        rear = swept(EGO, get_vertices(np.array([-3.0, 0.0, 0.0]), LENGTH, WIDTH), angles)
        self.assertEqual(rear, 0)
        self.assertGreater(front, 0)

    def test_the_swept_total_stays_well_under_a_full_sweep(self):
        angles = angles_for()
        rng = np.random.default_rng(9)
        total = 0
        poses = 0
        for _ in range(600):
            pose = np.array([rng.uniform(-6, 6), rng.uniform(-6, 6),
                             rng.uniform(-np.pi, np.pi)])
            if np.hypot(pose[0], pose[1]) < 0.35:
                continue
            total += swept(EGO, get_vertices(pose, LENGTH, WIDTH), angles)
            poses += 1
        self.assertLess(total / poses, 100.0)


if __name__ == "__main__":
    unittest.main()
