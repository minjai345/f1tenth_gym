"""The analytic scan backend: exactness, differentiability, and wiring.

The raster backend measures to the nearest occupied cell centre, so it reads long by
up to half a cell diagonal and cannot be differentiated. These pin that the segment
backend has neither problem, and that switching to it changes nothing else.
"""

import math
import unittest

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from f1tenth_gym.envs.env_config import EnvConfig, SimulationConfig
from f1tenth_gym.envs.lidar.config import LiDARConfig, ScanBackend
from f1tenth_gym.envs.lidar.kernels import scan
from f1tenth_gym.envs.lidar.laser_models import ScanSimulator2D
from f1tenth_gym.envs.lidar.segment_scan import SegmentScanSimulator2D
from f1tenth_gym.envs.track import Track
from f1tenth_gym.envs.track.ray_tiles import build_ray_tiles, build_for_track, candidates
from f1tenth_gym.envs.track.walls import wall_segments

HALF_FOV = 2.3561945
SCAN_KWARGS = dict(angle_min=-HALF_FOV, angle_max=HALF_FOV, std_dev=0.0,
                   min_range=0.0, max_range=30.0)


def unit_box():
    """Counter-clockwise unit box centred on the origin, as segment endpoints."""
    corner = np.array([[1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0]])
    return jnp.asarray(corner), jnp.asarray(np.roll(corner, -1, axis=0))


class TestKernel(unittest.TestCase):
    def test_a_box_reads_its_own_half_width(self):
        a, b = unit_box()
        angles = jnp.array([0.0, math.pi / 2, math.pi, -math.pi / 2])
        got = scan(jnp.array([0.0, 0.0, 0.0]), angles, a, b, 30.0)
        np.testing.assert_allclose(np.asarray(got), 1.0, atol=1e-6)

    def test_a_miss_returns_max_range(self):
        a, b = unit_box()
        got = scan(jnp.array([50.0, 50.0, 0.0]), jnp.array([0.0]), a, b, 30.0)
        self.assertAlmostEqual(float(got[0]), 30.0, places=5)

    def test_a_degenerate_segment_is_rejected(self):
        """Padding relies on this: zero length gives a zero denominator."""
        point = jnp.array([[1.0, -1.0]])
        got = scan(jnp.array([0.0, 0.0, 0.0]), jnp.array([0.0]), point, point, 30.0)
        self.assertAlmostEqual(float(got[0]), 30.0, places=5)

    def test_a_parallel_ray_is_rejected(self):
        along = jnp.array([[1.0, 0.0]])
        to = jnp.array([[5.0, 0.0]])
        got = scan(jnp.array([0.0, 0.0, 0.0]), jnp.array([0.0]), along, to, 30.0)
        self.assertAlmostEqual(float(got[0]), 30.0, places=5)

    def test_the_nearest_hit_wins(self):
        near = jnp.array([[2.0, -1.0]]), jnp.array([[2.0, 1.0]])
        far = jnp.array([[5.0, -1.0]]), jnp.array([[5.0, 1.0]])
        a = jnp.concatenate([far[0], near[0]])
        b = jnp.concatenate([far[1], near[1]])
        got = scan(jnp.array([0.0, 0.0, 0.0]), jnp.array([0.0]), a, b, 30.0)
        self.assertAlmostEqual(float(got[0]), 2.0, places=5)

    def test_a_segment_behind_the_sensor_is_not_hit(self):
        a, b = jnp.array([[-3.0, -1.0]]), jnp.array([[-3.0, 1.0]])
        got = scan(jnp.array([0.0, 0.0, 0.0]), jnp.array([0.0]), a, b, 30.0)
        self.assertAlmostEqual(float(got[0]), 30.0, places=5)


class TestDifferentiable(unittest.TestCase):
    """What the distance-transform tracer cannot offer at all."""

    def test_the_range_gradient_is_exact(self):
        a, b = unit_box()

        def loss(pose):
            return scan(pose, jnp.array([0.0]), a, b, 30.0)[0]

        grad = np.asarray(jax.grad(loss)(jnp.array([0.0, 0.0, 0.0])))
        # Moving +x closes on the +x wall one for one; y and yaw do not move it.
        np.testing.assert_allclose(grad, [-1.0, 0.0, 0.0], atol=1e-5)

    def test_the_gradient_matches_central_differences(self):
        """One long wall, so no beam crosses an endpoint.

        Against a box the yaw derivative has real kinks where a beam sweeps past a
        corner and switches faces, and central differences straddle them; that is
        geometry, not a gradient defect.
        """
        angles = jnp.linspace(-1.0, 1.0, 32)
        a, b = jnp.array([[5.0, -50.0]]), jnp.array([[5.0, 50.0]])

        def total(pose):
            return jnp.sum(scan(pose, angles, a, b, 100.0))

        base = np.array([0.13, -0.07, 0.21])
        analytic = np.asarray(jax.grad(total)(jnp.asarray(base)))
        step = 1e-4
        for axis in range(3):
            plus, minus = base.copy(), base.copy()
            plus[axis] += step
            minus[axis] -= step
            fd = (float(total(jnp.asarray(plus)))
                  - float(total(jnp.asarray(minus)))) / (2 * step)
            self.assertLess(abs(analytic[axis] - fd) / (abs(fd) + 1e-6), 5e-3,
                            msg=f"axis {axis}: analytic {analytic[axis]} fd {fd}")

    def test_a_corner_is_where_the_derivative_genuinely_breaks(self):
        """Worth pinning: the kink is in the geometry, not in the kernel."""
        a, b = unit_box()
        angles = jnp.linspace(-1.0, 1.0, 32)

        def total(pose):
            return jnp.sum(scan(pose, angles, a, b, 30.0))

        base = np.array([0.13, -0.07, 0.21])
        analytic = float(jax.grad(total)(jnp.asarray(base))[2])
        step = 1e-3
        plus, minus = base.copy(), base.copy()
        plus[2] += step
        minus[2] -= step
        fd = (float(total(jnp.asarray(plus)))
              - float(total(jnp.asarray(minus)))) / (2 * step)
        self.assertGreater(abs(analytic - fd) / (abs(fd) + 1e-6), 0.05)

    def test_it_jits_and_vmaps_over_poses(self):
        a, b = unit_box()
        angles = jnp.linspace(-HALF_FOV, HALF_FOV, 64)
        run = jax.jit(jax.vmap(lambda p: scan(p, angles, a, b, 30.0)))
        out = run(jnp.zeros((8, 3)))
        self.assertEqual(out.shape, (8, 64))
        self.assertTrue(bool(jnp.all(jnp.isfinite(out))))


class TestRayTiles(unittest.TestCase):
    def setUp(self):
        self.track = Track.from_track_name("Spielberg", 1.0)

    def test_a_tile_lists_every_segment_within_range(self):
        """Over-including is safe; under-including silently loses hits."""
        walls, index = build_for_track(self.track, 30.0)
        seg_a = walls.a.astype(np.float64)
        seg_b = walls.b.astype(np.float64)
        rng = np.random.default_rng(3)
        rows, cols = index.table.shape[0], index.table.shape[1]
        for _ in range(40):
            point = np.array([
                index.origin[0] + rng.uniform(0, cols * index.tile_size),
                index.origin[1] + rng.uniform(0, rows * index.tile_size)])
            listed = set(int(i) for i in candidates(point, index) if i < len(walls))
            edge = seg_b - seg_a
            t = np.clip(((point - seg_a) * edge).sum(1)
                        / np.maximum((edge * edge).sum(1), 1e-12), 0, 1)
            closest = seg_a + t[:, None] * edge
            near = np.flatnonzero(np.hypot(*(closest - point).T) <= 30.0)
            self.assertTrue(set(near.tolist()) <= listed, "a reachable segment was missed")

    def test_padding_points_one_past_the_end(self):
        walls, index = build_for_track(self.track, 30.0)
        self.assertLessEqual(int(index.table.max()), len(walls))
        self.assertEqual(index.n_segments, len(walls))

    def test_an_obstacle_free_map_is_empty_not_a_crash(self):
        blank = Track.from_track_name("Spielberg_blank", 1.0)
        _walls, index = build_for_track(blank, 30.0)
        self.assertTrue(index.is_empty)

    def test_the_index_caches_on_the_track(self):
        first = build_for_track(self.track, 30.0)[1]
        self.assertIs(build_for_track(self.track, 30.0)[1], first)

    def test_bad_arguments_raise(self):
        walls = wall_segments(self.track)
        for bad in (0.0, -1.0, float("nan")):
            with self.assertRaises(ValueError):
                build_ray_tiles(walls, bad)
            with self.assertRaises(ValueError):
                build_ray_tiles(walls, 30.0, tile_size=bad)

    def test_an_oversized_table_is_refused_before_allocating(self):
        walls = wall_segments(self.track)
        with self.assertRaises(MemoryError):
            build_ray_tiles(walls, 30.0, tile_size=0.05, max_bytes=1000)


class TestAgainstBruteForce(unittest.TestCase):
    """The tile gather must not change a single range."""

    def _brute(self, origin, angles, walls, max_range):
        seg_a = walls.a.astype(np.float64)
        edge = walls.b.astype(np.float64) - seg_a
        best = np.full(angles.shape[0], max_range)
        for i, ang in enumerate(angles):
            d = np.array([math.cos(ang), math.sin(ang)])
            den = d[0] * edge[:, 1] - d[1] * edge[:, 0]
            q = seg_a - origin
            with np.errstate(divide="ignore", invalid="ignore"):
                t = (q[:, 0] * edge[:, 1] - q[:, 1] * edge[:, 0]) / den
                u = (q[:, 0] * d[1] - q[:, 1] * d[0]) / den
            ok = (np.abs(den) > 1e-12) & (t >= 0) & (u >= 0) & (u <= 1)
            if ok.any():
                best[i] = min(best[i], t[ok].min())
        return best

    def test_the_gathered_scan_equals_a_full_sweep(self):
        track = Track.from_track_name("Spielberg", 1.0)
        walls = wall_segments(track)
        sim = SegmentScanSimulator2D(120, 4.7123889, **SCAN_KWARGS)
        sim.set_map(track)
        angles = np.linspace(-HALF_FOV, HALF_FOV, 120)
        rng = np.random.default_rng(11)
        for k in rng.choice(len(track.raceline.xs), 6, replace=False):
            pose = np.array([track.raceline.xs[k], track.raceline.ys[k],
                             track.raceline.yaws[k]])
            got = sim.scan(pose)
            want = self._brute(pose[:2], pose[2] + angles, walls, 30.0)
            np.testing.assert_allclose(got, want, atol=2e-3)


class TestBackendWiring(unittest.TestCase):
    def _env(self, backend, **kwargs):
        return gym.make("f1tenth_gym:f1tenth-v0", config=EnvConfig(
            map_name="Spielberg",
            simulation_config=SimulationConfig(max_laps=None),
            lidar_config=LiDARConfig(backend=backend, **kwargs),
            render_enabled=False))

    def test_the_default_is_the_segment_backend(self):
        self.assertIs(LiDARConfig().backend, ScanBackend.SEGMENT)

    def test_selecting_segment_builds_the_segment_simulator(self):
        env = self._env(ScanBackend.SEGMENT)
        self.assertIsInstance(env.unwrapped.sim.scan_sims[0], SegmentScanSimulator2D)
        env.close()
        env = self._env(ScanBackend.RASTER)
        self.assertIsInstance(env.unwrapped.sim.scan_sims[0], ScanSimulator2D)
        env.close()

    def test_a_rollout_produces_a_sane_scan(self):
        env = self._env(ScanBackend.SEGMENT)
        obs, _ = env.reset(seed=1)
        for _ in range(30):
            obs, _r, _t, _tr, _i = env.step(np.array([[0.0, 2.0]], dtype=np.float32))
        scan_values = obs["agent_0"]["scan"]
        self.assertEqual(scan_values.shape, (1080,))
        self.assertTrue(bool(np.all(np.isfinite(scan_values))))
        self.assertGreater(float(scan_values.min()), 0.0)
        self.assertLessEqual(float(scan_values.max()), 30.0 + 1e-3)
        env.close()

    def test_it_reads_shorter_than_the_raster_backend(self):
        """The raster bias is outward; removing it must not make ranges longer."""
        track = Track.from_track_name("Spielberg", 1.0)
        raster = ScanSimulator2D(1080, 4.7123889, **SCAN_KWARGS)
        raster.set_map(track)
        segment = SegmentScanSimulator2D(1080, 4.7123889, **SCAN_KWARGS)
        segment.set_map(track)
        deltas = []
        for k in range(0, len(track.raceline.xs), 137):
            pose = np.array([track.raceline.xs[k], track.raceline.ys[k],
                             track.raceline.yaws[k]])
            both = raster.scan(pose, None), segment.scan(pose)
            live = (both[0] < 29.0) & (both[1] < 29.0)
            deltas.append(np.median(both[0][live] - both[1][live]))
        self.assertGreater(float(np.median(deltas)), 0.0)

    def test_an_obstacle_free_map_reports_max_range(self):
        env = gym.make("f1tenth_gym:f1tenth-v0", config=EnvConfig(
            map_name="Spielberg_blank",
            simulation_config=SimulationConfig(max_laps=None),
            lidar_config=LiDARConfig(backend=ScanBackend.SEGMENT, noise_std=0.0),
            render_enabled=False))
        obs, _ = env.reset(seed=1)
        np.testing.assert_allclose(obs["agent_0"]["scan"], 30.0, atol=1e-4)
        env.close()

    def test_scanning_before_a_map_is_set_raises(self):
        sim = SegmentScanSimulator2D(16, 4.7123889, **SCAN_KWARGS)
        with self.assertRaises(ValueError):
            sim.scan(np.zeros(3))

    def test_noise_is_applied_only_when_a_generator_is_given(self):
        track = Track.from_track_name("Spielberg", 1.0)
        sim = SegmentScanSimulator2D(64, 4.7123889, angle_min=-HALF_FOV,
                                     angle_max=HALF_FOV, std_dev=0.05,
                                     min_range=0.0, max_range=30.0)
        sim.set_map(track)
        pose = np.array([track.raceline.xs[0], track.raceline.ys[0],
                         track.raceline.yaws[0]])
        np.testing.assert_array_equal(sim.scan(pose), sim.scan(pose))
        noisy = sim.scan(pose, np.random.default_rng(0))
        self.assertGreater(float(np.abs(noisy - sim.scan(pose)).max()), 0.0)


class TestConfigValidation(unittest.TestCase):
    def test_the_backend_is_coerced_from_an_int(self):
        self.assertIs(LiDARConfig(backend=2).backend, ScanBackend.SEGMENT)

    def test_an_unknown_backend_raises(self):
        for bad in (0, 99, "raster"):
            with self.assertRaises(ValueError):
                LiDARConfig(backend=bad)

    def test_only_the_two_known_devices_are_accepted(self):
        for good in ("cpu", "gpu"):
            self.assertEqual(LiDARConfig(scan_device=good).scan_device, good)
        for bad in ("tpu", "", None):
            with self.assertRaises(ValueError):
                LiDARConfig(scan_device=bad)

    def test_the_section_stays_hashable(self):
        self.assertIsInstance(hash(LiDARConfig(backend=ScanBackend.SEGMENT)), int)

    def test_with_updates_revalidates(self):
        cfg = LiDARConfig().with_updates(backend=ScanBackend.SEGMENT)
        self.assertIs(cfg.backend, ScanBackend.SEGMENT)
        with self.assertRaises(ValueError):
            cfg.with_updates(scan_device="tpu")


if __name__ == "__main__":
    unittest.main()
