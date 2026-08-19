import functools
import itertools
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from f1tenth_gym.envs.contact import (
    ContactParams,
    contact_velocity,
    resolve,
    segment_contact,
    speculative_clamp,
    speculative_gap,
)

CAR_L, CAR_W, MASS, INERTIA, DT = 0.58, 0.31, 3.74, 0.04712, 0.01
WALL_A = jnp.array([-5.0, 0.0])
WALL_B = jnp.array([5.0, 0.0])
WALL_N = jnp.array([0.0, 1.0])
NORMALS = jnp.broadcast_to(WALL_N, (2, 2))
_LOCAL = jnp.array([[-CAR_L / 2, CAR_W / 2], [-CAR_L / 2, -CAR_W / 2],
                    [CAR_L / 2, -CAR_W / 2], [CAR_L / 2, CAR_W / 2]])


def body(x, y, yaw):
    c, s = jnp.cos(yaw), jnp.sin(yaw)
    return _LOCAL @ jnp.array([[c, s], [-s, c]]) + jnp.array([x, y])


def resting_manifold(penetration=0.005):
    centre_y = CAR_W / 2 - penetration
    manifold = segment_contact(body(0.0, centre_y, 0.0), WALL_A, WALL_B, WALL_N)
    return manifold, jnp.array([0.0, centre_y])


@functools.partial(jax.jit, static_argnums=(2,))
def rollout(state, params, steps, gravity):
    """Contact-resolved free flight, compiled as one scan so tests stay fast."""

    def step(carry, _):
        x, y, yaw, v, w = carry
        verts = body(x, y, yaw)
        manifold = segment_contact(verts, WALL_A, WALL_B, WALL_N)
        gap = speculative_gap(verts, WALL_A, WALL_B, WALL_N)
        v = speculative_clamp(v, jnp.array([gap]), WALL_N[None], DT)
        v, w, correction = resolve(
            v, w, MASS, INERTIA, manifold.points, manifold.depths,
            NORMALS, jnp.array([x, y]), params,
        )
        x = x + v[0] * DT + correction[0]
        y = y + v[1] * DT + correction[1]
        yaw = yaw + w * DT
        v = v + jnp.array([0.0, -gravity * DT])
        return (x, y, yaw, v, w), 0.5 * MASS * jnp.dot(v, v) + 0.5 * INERTIA * w * w

    return jax.lax.scan(step, state, None, length=steps)


class TestRestitution(unittest.TestCase):
    def test_the_separating_speed_matches_the_coefficient(self):
        manifold, centre = resting_manifold()
        offsets = manifold.points - centre
        for e in (0.0, 0.3, 0.8):
            params = ContactParams(restitution=e, friction=0.0)
            v, w, _ = resolve(jnp.array([0.0, -5.0]), 0.0, MASS, INERTIA,
                              manifold.points, manifold.depths, NORMALS, centre, params)
            for k in range(2):
                got = float(jnp.dot(contact_velocity(v, w, offsets[k]), WALL_N))
                self.assertAlmostEqual(got, e * 5.0, places=2, msg=f"e={e}")

    def test_a_slow_contact_never_gains_energy(self):
        """3.8: a 1 cm/s scrape must not keep earning a bounce."""
        for e in (0.0, 0.3):
            params = ContactParams(restitution=e, restitution_threshold=0.6, friction=0.6)
            start = (0.0, CAR_W / 2 - 0.001, 0.0, jnp.array([0.0, -0.01]), 0.0)
            before = 0.5 * MASS * float(jnp.dot(start[3], start[3]))
            (_x, _y, _yaw, v, w), _ = rollout(start, params, 400, 0.0)
            after = 0.5 * MASS * float(jnp.dot(v, v)) + 0.5 * INERTIA * float(w) ** 2
            self.assertLessEqual(after, before + 1e-9, f"e={e} gained energy")


class TestRestingContact(unittest.TestCase):
    """6.4: dropped on a flat wall with e=0, come to rest and stay there."""

    @classmethod
    def setUpClass(cls):
        params = ContactParams(restitution=0.0, friction=0.6)
        start = (0.0, CAR_W / 2 + 0.05, 0.0, jnp.array([0.0, -2.0]), 0.0)
        cls.final, energy = rollout(start, params, 2000, 9.81)
        cls.energy = np.asarray(energy)
        cls.params = params

    def test_it_settles_inside_the_slop(self):
        penetration = CAR_W / 2 - float(self.final[1])
        self.assertGreaterEqual(penetration, -1e-6)
        self.assertLessEqual(penetration, self.params.slop + 1e-6)

    def test_it_does_not_drift_or_spin(self):
        self.assertAlmostEqual(float(self.final[2]), 0.0, places=6)
        self.assertAlmostEqual(float(self.final[4]), 0.0, places=6)

    def test_the_energy_does_not_grow(self):
        tail = self.energy[-500:]
        self.assertLessEqual(float(tail.max()), float(tail.mean()) * 1.05 + 1e-9)


class TestSpeculativeClamp(unittest.TestCase):
    """3.7: clamp the normal component, never the whole step vector."""

    def test_tangential_speed_is_preserved(self):
        verts = body(0.0, CAR_W / 2 + 0.05, 0.0)
        gap = speculative_gap(verts, WALL_A, WALL_B, WALL_N)
        before = jnp.array([3.0, -20.0])
        after = speculative_clamp(before, jnp.array([gap]), WALL_N[None], DT)
        self.assertEqual(float(after[0]), float(before[0]))
        self.assertAlmostEqual(float(after[1]), -float(gap) / DT, places=5)

    def test_a_slow_approach_is_left_alone(self):
        verts = body(0.0, CAR_W / 2 + 0.05, 0.0)
        gap = speculative_gap(verts, WALL_A, WALL_B, WALL_N)
        before = jnp.array([0.0, -0.5])
        after = speculative_clamp(before, jnp.array([gap]), WALL_N[None], DT)
        self.assertTrue(np.allclose(np.asarray(after), np.asarray(before)))

    def test_it_does_not_fire_once_penetrating(self):
        # A negative gap makes -gap/dt positive; firing there flings the body out.
        verts = body(0.0, CAR_W / 2 - 0.001, 0.0)
        gap = speculative_gap(verts, WALL_A, WALL_B, WALL_N)
        self.assertLess(float(gap), 0.0)
        before = jnp.array([0.0, 0.0])
        after = speculative_clamp(before, jnp.array([gap]), WALL_N[None], DT)
        self.assertTrue(np.allclose(np.asarray(after), np.asarray(before)))


class TestSolverContract(unittest.TestCase):
    def test_more_sweeps_never_increase_the_residual(self):
        manifold, centre = resting_manifold()
        offsets = manifold.points - centre
        params = ContactParams(restitution=0.0, friction=0.0)
        residuals = []
        for iterations in (4, 16, 64, 128):
            v, w, _ = resolve(jnp.array([0.0, -5.0]), 0.0, MASS, INERTIA,
                              manifold.points, manifold.depths, NORMALS, centre,
                              params, iterations)
            residuals.append(max(
                abs(float(jnp.dot(contact_velocity(v, w, offsets[k]), WALL_N)))
                for k in range(2)
            ))
        for earlier, later in itertools.pairwise(residuals):
            self.assertLessEqual(later, earlier + 1e-6)
        self.assertLess(residuals[-1], 1e-3)

    def test_an_empty_manifold_changes_nothing(self):
        v_in, w_in = jnp.array([1.0, -2.0]), 0.7
        v, w, correction = resolve(
            v_in, w_in, MASS, INERTIA, jnp.zeros((2, 2)), jnp.zeros(2),
            NORMALS, jnp.zeros(2), ContactParams(),
        )
        self.assertTrue(np.allclose(np.asarray(v), np.asarray(v_in)))
        self.assertAlmostEqual(float(w), w_in)
        self.assertTrue(np.allclose(np.asarray(correction), 0.0))

    def test_friction_respects_the_cone(self):
        manifold, centre = resting_manifold()
        sliding = jnp.array([4.0, -1.0])
        loose, _w1, _c1 = resolve(sliding, 0.0, MASS, INERTIA, manifold.points,
                                  manifold.depths, NORMALS, centre,
                                  ContactParams(friction=0.0))
        gripping, _w2, _c2 = resolve(sliding, 0.0, MASS, INERTIA, manifold.points,
                                     manifold.depths, NORMALS, centre,
                                     ContactParams(friction=1.0))
        self.assertAlmostEqual(float(loose[0]), 4.0, places=3)
        self.assertLess(float(gripping[0]), float(loose[0]))

    def test_baumgarte_pushes_out_only_beyond_the_slop(self):
        params = ContactParams(baumgarte=0.4, slop=0.002)
        shallow, centre_s = resting_manifold(penetration=0.001)
        deep, centre_d = resting_manifold(penetration=0.050)
        _v, _w, small = resolve(jnp.zeros(2), 0.0, MASS, INERTIA, shallow.points,
                                shallow.depths, NORMALS, centre_s, params)
        _v2, _w2, large = resolve(jnp.zeros(2), 0.0, MASS, INERTIA, deep.points,
                                  deep.depths, NORMALS, centre_d, params)
        self.assertAlmostEqual(float(small[1]), 0.0, places=6)
        self.assertGreater(float(large[1]), 0.0)

    def test_it_runs_under_jit_and_vmap(self):
        manifold, centre = resting_manifold()
        run = jax.jit(jax.vmap(
            lambda v: resolve(v, 0.0, MASS, INERTIA, manifold.points, manifold.depths,
                              NORMALS, centre, ContactParams())
        ))
        v, w, correction = run(jnp.array([[0.0, -5.0], [0.0, -1.0], [2.0, -3.0]]))
        self.assertEqual(v.shape, (3, 2))
        self.assertEqual(w.shape, (3,))
        self.assertEqual(correction.shape, (3, 2))


if __name__ == "__main__":
    unittest.main()
