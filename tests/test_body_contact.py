import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from f1tenth_gym.envs.collision_models import get_vertices
from f1tenth_gym.envs.contact import (
    ContactParams,
    body_contact,
    contact_velocity,
    resolve_pair,
)
from tests.gjk_oracle import collision

CAR_L, CAR_W, MASS, INERTIA = 0.58, 0.31, 3.74, 0.04712


def body(x, y, yaw):
    return jnp.asarray(get_vertices(np.array([x, y, yaw]), CAR_L, CAR_W), jnp.float32)


def momentum(v_a, v_b):
    return MASS * (np.asarray(v_a) + np.asarray(v_b))


def angular_momentum(v, w, centre):
    centre, v = np.asarray(centre), np.asarray(v)
    return INERTIA * float(w) + MASS * (centre[0] * v[1] - centre[1] * v[0])


class TestSeparatingAxis(unittest.TestCase):
    def test_it_agrees_with_the_test_only_gjk_oracle(self):
        rng = np.random.default_rng(0)
        count = 4000
        poses_a = np.stack([rng.uniform(-0.8, 0.8, count), rng.uniform(-0.8, 0.8, count),
                            rng.uniform(-np.pi, np.pi, count)], axis=1)
        poses_b = np.stack([rng.uniform(-0.8, 0.8, count), rng.uniform(-0.8, 0.8, count),
                            rng.uniform(-np.pi, np.pi, count)], axis=1)
        verts_a = np.stack([get_vertices(p, CAR_L, CAR_W) for p in poses_a])
        verts_b = np.stack([get_vertices(p, CAR_L, CAR_W) for p in poses_b])
        gjk = np.array([bool(collision(verts_a[i], verts_b[i])) for i in range(count)])
        manifolds = jax.jit(jax.vmap(body_contact))(
            jnp.asarray(verts_a, jnp.float32), jnp.asarray(verts_b, jnp.float32)
        )
        sat = np.asarray(manifolds.depths).max(axis=1) > 0
        self.assertEqual(int((gjk == sat).sum()), count)

    def test_a_separated_pair_is_empty(self):
        manifold = body_contact(body(-2.0, 0.0, 0.0), body(2.0, 0.0, 0.0))
        self.assertEqual(int(manifold.count), 0)
        self.assertTrue(np.allclose(np.asarray(manifold.normal), 0.0))

    def test_the_normal_points_from_a_to_b(self):
        manifold = body_contact(body(-0.28, 0.0, 0.0), body(0.28, 0.0, 0.0))
        self.assertGreater(int(manifold.count), 0)
        self.assertTrue(np.allclose(np.asarray(manifold.normal), [1.0, 0.0], atol=1e-5))

    def test_face_to_face_gives_two_points(self):
        manifold = body_contact(body(-0.28, 0.0, 0.0), body(0.28, 0.0, 0.0))
        self.assertEqual(int(manifold.count), 2)

    def test_contact_points_lie_between_the_bodies(self):
        manifold = body_contact(body(-0.28, 0.0, 0.0), body(0.28, 0.05, 0.3))
        live = np.asarray(manifold.depths) > 0
        points = np.asarray(manifold.points)[live]
        self.assertGreater(len(points), 0)
        for point in points:
            self.assertLess(abs(point[0]), CAR_L)
            self.assertLess(abs(point[1]), CAR_L)


class TestTwoBodySolver(unittest.TestCase):
    @staticmethod
    def _head_on(restitution):
        manifold = body_contact(body(-0.28, 0.0, 0.0), body(0.28, 0.0, 0.0))
        params = ContactParams(restitution=restitution, friction=0.0)
        return resolve_pair(
            jnp.array([3.0, 0.0]), 0.0, jnp.array([-3.0, 0.0]), 0.0,
            MASS, INERTIA, manifold.points, manifold.depths, manifold.normal,
            jnp.array([-0.28, 0.0]), jnp.array([0.28, 0.0]), params,
        )

    def test_a_dead_stop_at_zero_restitution(self):
        v_a, _w_a, v_b, _w_b, _sep = self._head_on(0.0)
        self.assertAlmostEqual(float(v_a[0]), 0.0, places=3)
        self.assertAlmostEqual(float(v_b[0]), 0.0, places=3)

    def test_restitution_bounces_both_symmetrically(self):
        v_a, _w_a, v_b, _w_b, _sep = self._head_on(0.5)
        self.assertAlmostEqual(float(v_a[0]), -1.5, places=2)
        self.assertAlmostEqual(float(v_b[0]), +1.5, places=2)

    def test_momentum_is_conserved(self):
        """Impulses between two bodies are internal; nothing may leak."""
        rng = np.random.default_rng(0)
        worst_linear = worst_angular = 0.0
        pairs = 0
        for _ in range(200):
            pose_a = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.4, 0.4),
                               rng.uniform(-np.pi, np.pi)])
            pose_b = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.4, 0.4),
                               rng.uniform(-np.pi, np.pi)])
            manifold = body_contact(body(*pose_a), body(*pose_b))
            if float(np.asarray(manifold.depths).max()) <= 0:
                continue
            pairs += 1
            centre_a = jnp.asarray(pose_a[:2], jnp.float32)
            centre_b = jnp.asarray(pose_b[:2], jnp.float32)
            v0a = jnp.asarray(rng.uniform(-4, 4, 2), jnp.float32)
            v0b = jnp.asarray(rng.uniform(-4, 4, 2), jnp.float32)
            w0a, w0b = float(rng.uniform(-3, 3)), float(rng.uniform(-3, 3))
            v_a, w_a, v_b, w_b, _sep = resolve_pair(
                v0a, w0a, v0b, w0b, MASS, INERTIA, manifold.points, manifold.depths,
                manifold.normal, centre_a, centre_b,
                ContactParams(restitution=0.0, friction=0.5),
            )
            worst_linear = max(worst_linear,
                               float(np.abs(momentum(v_a, v_b) - momentum(v0a, v0b)).max()))
            before = angular_momentum(v0a, w0a, centre_a) + angular_momentum(v0b, w0b, centre_b)
            after = angular_momentum(v_a, w_a, centre_a) + angular_momentum(v_b, w_b, centre_b)
            worst_angular = max(worst_angular, abs(after - before))
        self.assertGreater(pairs, 50, "scenario produced too few overlaps")
        self.assertLess(worst_linear, 1e-3)
        self.assertLess(worst_angular, 1e-3)

    def test_the_contact_stops_closing(self):
        manifold = body_contact(body(-0.28, 0.0, 0.0), body(0.28, 0.0, 0.0))
        v_a, w_a, v_b, w_b, _sep = self._head_on(0.0)
        offsets_a = manifold.points - jnp.array([-0.28, 0.0])
        offsets_b = manifold.points - jnp.array([0.28, 0.0])
        for k in range(2):
            closing = float(jnp.dot(
                contact_velocity(v_b, w_b, offsets_b[k])
                - contact_velocity(v_a, w_a, offsets_a[k]),
                manifold.normal,
            ))
            self.assertGreater(closing, -1e-2)

    def test_the_push_out_is_shared(self):
        manifold = body_contact(body(-0.26, 0.0, 0.0), body(0.26, 0.0, 0.0))
        _v_a, _w_a, _v_b, _w_b, separation = resolve_pair(
            jnp.zeros(2), 0.0, jnp.zeros(2), 0.0, MASS, INERTIA,
            manifold.points, manifold.depths, manifold.normal,
            jnp.array([-0.26, 0.0]), jnp.array([0.26, 0.0]),
            ContactParams(baumgarte=0.4, slop=0.002),
        )
        self.assertGreater(float(separation[0]), 0.0)

    def test_it_runs_under_jit(self):
        manifold = body_contact(body(-0.28, 0.0, 0.0), body(0.28, 0.0, 0.0))
        run = jax.jit(lambda va, vb: resolve_pair(
            va, 0.0, vb, 0.0, MASS, INERTIA, manifold.points, manifold.depths,
            manifold.normal, jnp.array([-0.28, 0.0]), jnp.array([0.28, 0.0]),
            ContactParams(),
        ))
        out = run(jnp.array([3.0, 0.0]), jnp.array([-3.0, 0.0]))
        self.assertEqual(len(out), 5)


class TestInTheGym(unittest.TestCase):
    """Two cars meet head-on through the production contact response."""

    @staticmethod
    def _run(mode, restitution=0.0):
        import gymnasium as gym

        from f1tenth_gym.envs.action import LongitudinalActionType, SteerActionType
        from f1tenth_gym.envs.contact import ContactConfig
        from f1tenth_gym.envs.env_config import (
            ControlConfig,
            EnvConfig,
            SimulationConfig,
            TerminationConfig,
        )

        env = gym.make("f1tenth_gym:f1tenth-v0", config=EnvConfig(
            num_agents=2,
            simulation_config=SimulationConfig(max_laps=None),
            termination_config=TerminationConfig(terminate_on_collision=False),
            control_config=ControlConfig(
                longitudinal_mode=LongitudinalActionType.ACCL,
                steering_mode=SteerActionType.STEERING_SPEED,
            ),
            contact_config=ContactConfig(friction=0.3, restitution=restitution),
            collision_check=mode,
            render_enabled=False,
        ))
        try:
            env.reset(seed=1)
            states = np.zeros((2, 7))
            states[0] = [0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0]
            states[1] = [1.2, 0.0, 0.0, 3.0, math.pi, 0.0, 0.0]
            env.reset(seed=1, options={"states": states})
            sim = env.unwrapped.sim
            action = np.zeros((2, 2), dtype=np.float32)
            hit_at = None
            for i in range(120):
                _o, _r, _t, _tr, info = env.step(action)
                if info["collisions"].any() and hit_at is None:
                    hit_at = i
                if hit_at is not None and i - hit_at > 3:
                    break
            speeds = [float(sim.state.standard_state[k][3]) for k in (0, 1)]
            return hit_at, speeds
        finally:
            env.close()

    def test_segment_contact_stops_them(self):
        from f1tenth_gym.envs.collision_models import CollisionCheckMode

        hit_at, speeds = self._run(CollisionCheckMode.SEGMENT_CONTACT)
        self.assertIsNotNone(hit_at)
        self.assertLess(abs(speeds[0]), 0.5)
        self.assertLess(abs(speeds[1]), 0.5)


if __name__ == "__main__":
    unittest.main()
