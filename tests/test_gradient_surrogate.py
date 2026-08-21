"""The differentiable contact surrogate: `deepest_depth`.

Pins the surrogate's value against the manifold, its gradient against central
finite differences, and its float32 behaviour at track coordinates.
"""

import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from f1tenth_gym.envs.collision_models import get_vertices
from f1tenth_gym.envs.contact.kernels import deepest_depth, segment_contact
from f1tenth_gym.envs.dynamic_models import F1TENTH_VEHICLE_PARAMETERS as F1TENTH
from f1tenth_gym.envs.track import Track
from f1tenth_gym.envs.track.accel import build_for_track

CPU = jax.devices("cpu")[0]
LENGTH, WIDTH = float(F1TENTH.length), float(F1TENTH.width)
STEP = 1e-4


def rect(pose, dtype=jnp.float32):
    """Body corners for a pose, differentiably (get_vertices is numpy)."""
    cos, sin = jnp.cos(pose[2]), jnp.sin(pose[2])
    corner = jnp.array(
        [[-LENGTH / 2, -WIDTH / 2], [LENGTH / 2, -WIDTH / 2],
         [LENGTH / 2, WIDTH / 2], [-LENGTH / 2, WIDTH / 2]], dtype)
    return corner @ jnp.array([[cos, -sin], [sin, cos]], dtype).T + pose[:2]


class TestAgainstTheManifold(unittest.TestCase):
    """The surrogate must report the same depth the physics does."""

    def setUp(self):
        self.verts = jnp.asarray(get_vertices(np.array([0.0, 0.0, 0.0]), LENGTH, WIDTH))
        self.a = jnp.array([0.2, -5.0])
        self.b = jnp.array([0.2, 5.0])
        self.n = jnp.array([-1.0, 0.0])

    def test_it_equals_the_deepest_manifold_point(self):
        manifold = segment_contact(self.verts, self.a, self.b, self.n)
        self.assertAlmostEqual(
            float(deepest_depth(self.verts, self.a, self.b, self.n)),
            float(manifold.depths.max()), places=6)

    def test_a_clear_body_reports_nothing(self):
        clear = jnp.asarray(get_vertices(np.array([-3.0, 0.0, 0.0]), LENGTH, WIDTH))
        self.assertEqual(float(deepest_depth(clear, self.a, self.b, self.n)), 0.0)

    def test_an_invalid_slot_reports_nothing(self):
        self.assertEqual(
            float(deepest_depth(self.verts, self.a, self.b, self.n, valid=False)), 0.0)

    def test_a_body_behind_the_face_is_not_in_contact(self):
        """One-sided, like segment_contact: the far side of a wall is clear."""
        behind = jnp.asarray(get_vertices(np.array([3.0, 0.0, 0.0]), LENGTH, WIDTH))
        self.assertEqual(float(deepest_depth(behind, self.a, self.b, self.n)), 0.0)


class TestSoftMin(unittest.TestCase):
    def setUp(self):
        self.verts = jnp.asarray(get_vertices(np.array([0.0, 0.0, 0.0]), LENGTH, WIDTH))
        self.a, self.b = jnp.array([0.2, -5.0]), jnp.array([0.2, 5.0])
        self.n = jnp.array([-1.0, 0.0])

    def test_zero_softness_is_the_hard_minimum(self):
        self.assertAlmostEqual(
            float(deepest_depth(self.verts, self.a, self.b, self.n, softness=0.0)),
            float(segment_contact(self.verts, self.a, self.b, self.n).depths.max()),
            places=6)

    def test_softening_never_reports_less_depth(self):
        """A soft minimum sits below the hard one, so the depth sits above it."""
        hard = float(deepest_depth(self.verts, self.a, self.b, self.n, softness=0.0))
        for softness in (0.001, 0.005, 0.02):
            soft = float(deepest_depth(self.verts, self.a, self.b, self.n,
                                       softness=softness))
            self.assertGreaterEqual(soft, hard - 1e-6)

    def test_it_converges_to_the_hard_minimum(self):
        hard = float(deepest_depth(self.verts, self.a, self.b, self.n, softness=0.0))
        errors = [abs(float(deepest_depth(self.verts, self.a, self.b, self.n,
                                          softness=s)) - hard)
                  for s in (0.02, 0.005, 0.001)]
        self.assertEqual(errors, sorted(errors, reverse=True))


class TestFloat32AtTrackCoordinates(unittest.TestCase):
    """Depth is a difference of two O(100) projections at real track coordinates.

    The gradient has to stay usable out there, in the float32 the gym runs in.
    """

    FAR = 340.0

    def _loss(self, offset):
        a = jnp.array([offset + 0.2, offset - 5.0], jnp.float32)
        b = jnp.array([offset + 0.2, offset + 5.0], jnp.float32)
        n = jnp.array([-1.0, 0.0], jnp.float32)
        return lambda pose: deepest_depth(rect(pose), a, b, n)

    def test_the_gradient_survives_far_from_the_origin(self):
        for offset in (0.0, 10.0, self.FAR):
            loss = self._loss(offset)
            pose = jnp.array([offset, offset, 0.0], jnp.float32)
            self.assertAlmostEqual(float(jax.grad(loss)(pose)[0]), 1.0, places=3,
                                   msg=f"offset {offset} m")

    def test_finite_differences_are_not_quantised_away(self):
        """Divide by the step float32 can actually represent, not the one asked for.

        At 340 m the spacing is 3.05e-5, so a nominal 1e-4 step rounds to 9.16e-5;
        dividing by the nominal step would report 0.9155 and look like an error.
        """
        loss = jax.jit(self._loss(self.FAR))
        plus = np.array([self.FAR + STEP, self.FAR, 0.0], np.float32)
        minus = np.array([self.FAR - STEP, self.FAR, 0.0], np.float32)
        actual_step = float(plus[0] - minus[0])
        self.assertGreater(actual_step, 0.0, "the step vanished entirely")
        fd = (float(loss(jnp.asarray(plus))) - float(loss(jnp.asarray(minus)))) / actual_step
        self.assertNotEqual(fd, 0.0)
        self.assertAlmostEqual(fd, 1.0, places=2)

    def test_depth_is_accurate_far_from_the_origin(self):
        loss = self._loss(self.FAR)
        pose = jnp.array([self.FAR, self.FAR, 0.0], jnp.float32)
        self.assertAlmostEqual(float(loss(pose)), 0.09, places=4)


class TestGradientAgainstFiniteDifferences(unittest.TestCase):
    """Per-component check over real track geometry, as the plan requires."""

    @classmethod
    def setUpClass(cls):
        track = Track.from_track_name("Spielberg", 1.0)
        cls.walls, _, index = build_for_track(track, F1TENTH, None)
        with jax.default_device(CPU):
            cls.a = jnp.asarray(cls.walls.a)
            cls.b = jnp.asarray(cls.walls.b)
            cls.n = jnp.asarray(cls.walls.n)
            cls.table = jnp.asarray(index.table)
        cls.origin = np.asarray(index.origin, np.float64)
        cls.tile = float(index.tile_size)

    def _loss_fn(self, softness):
        a, b, n, table = self.a, self.b, self.n, self.table
        origin = jnp.asarray(self.origin, jnp.float32)
        tile = self.tile
        rows, cols = int(table.shape[0]), int(table.shape[1])

        def loss(pose):
            verts = rect(pose)
            col = jnp.clip(((pose[0] - origin[0]) / tile).astype(jnp.int32), 0, cols - 1)
            row = jnp.clip(((pose[1] - origin[1]) / tile).astype(jnp.int32), 0, rows - 1)
            cand = table[row, col]
            ok = cand >= 0
            idx = jnp.where(ok, cand, 0)
            return jax.vmap(
                lambda k, live: deepest_depth(verts, a[k], b[k], n[k], live, softness)
            )(idx, ok).sum()

        return loss

    def _probes(self, penetration, jitter, count=120):
        rng = np.random.default_rng(4)
        step = max(1, len(self.walls) // count)
        poses = []
        for k in range(0, len(self.walls), step):
            mid = 0.5 * (np.asarray(self.walls.a[k]) + np.asarray(self.walls.b[k]))
            centre = mid - np.asarray(self.walls.n[k]) * penetration
            edge = np.asarray(self.walls.b[k]) - np.asarray(self.walls.a[k])
            yaw = math.atan2(edge[1], edge[0]) + rng.uniform(-jitter, jitter)
            poses.append([centre[0], centre[1], yaw])
        return np.array(poses, np.float32)

    def _compare(self, softness, penetration, jitter):
        with jax.default_device(CPU):
            loss = self._loss_fn(softness)
            value = jax.jit(jax.vmap(loss))
            grad = jax.jit(jax.vmap(jax.grad(loss)))
            poses = self._probes(penetration, jitter)
            analytic = np.asarray(grad(jnp.asarray(poses)), np.float64)
            errors = []
            for axis in range(3):
                plus, minus = poses.copy(), poses.copy()
                plus[:, axis] += STEP
                minus[:, axis] -= STEP
                fd = (np.asarray(value(jnp.asarray(plus)), np.float64)
                      - np.asarray(value(jnp.asarray(minus)), np.float64)) / (2 * STEP)
                live = np.abs(fd) > 1e-3
                self.assertGreater(live.sum(), 20, f"no FD signal on axis {axis}")
                rel = np.abs(analytic[live, axis] - fd[live]) / (np.abs(fd[live]) + 1e-6)
                errors.append(float(np.median(rel)))
            return errors

    def test_position_gradients_match_central_differences(self):
        dx, dy, _ = self._compare(softness=0.003, penetration=0.005, jitter=0.6)
        self.assertLess(dx, 0.05)
        self.assertLess(dy, 0.05)

    def test_yaw_gradient_matches_when_the_body_is_tilted(self):
        _, _, dpsi = self._compare(softness=0.003, penetration=0.005, jitter=0.6)
        self.assertLess(dpsi, 0.05)

    def _flush_yaw_gradient(self, softness):
        """Median |d/dpsi| with the body exactly parallel to each wall segment.

        Two vertices tie there. The hard minimum breaks the tie arbitrarily and
        reports a large one-sided derivative; the true two-sided one is near zero.
        """
        with jax.default_device(CPU):
            grad = jax.jit(jax.vmap(jax.grad(self._loss_fn(softness))))
            poses = self._probes(penetration=0.005, jitter=0.0)
            return float(np.median(np.abs(np.asarray(grad(jnp.asarray(poses)))[:, 2])))

    def test_the_hard_minimum_invents_a_yaw_gradient_when_flush(self):
        self.assertGreater(self._flush_yaw_gradient(0.0), 0.1)

    def test_softening_removes_the_invented_yaw_gradient(self):
        hard = self._flush_yaw_gradient(0.0)
        soft = self._flush_yaw_gradient(0.003)
        self.assertLess(soft, 0.01)
        self.assertLess(soft, hard / 100.0)


if __name__ == "__main__":
    unittest.main()
