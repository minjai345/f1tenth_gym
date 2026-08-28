"""Numerical and transformation contracts for the pure JAX dynamics seam."""

from dataclasses import replace
from functools import partial
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from f1tenth_gym.envs.dynamic_models import (
    F1TENTH_VEHICLE_PARAMETERS,
    vehicle_dynamics_st,
)
from f1tenth_gym.envs.dynamic_models.kinematic import vehicle_dynamics_ks_cog
from f1tenth_gym.jax import (
    DynamicsParams,
    euler_step,
    integrate_substeps,
    kinematic_single_track,
    rk4_step,
    single_track,
)


class TestJaxDynamics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host_params = F1TENTH_VEHICLE_PARAMETERS
        cls.params = DynamicsParams.from_vehicle_parameters(cls.host_params)
        cls.param_array = cls.host_params.to_array()

    def test_ks_cog_matches_numpy_reference_under_jit(self):
        state = np.array([1.0, 2.0, 0.2, 3.0, -0.4], dtype=np.float32)
        control = np.array([0.1, 1.5], dtype=np.float32)
        expected = vehicle_dynamics_ks_cog(state, control, self.param_array)
        actual = jax.jit(kinematic_single_track)(state, control, self.params)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

    def test_st_matches_reference_at_reverse_zero_and_dynamic_speeds(self):
        compiled = jax.jit(single_track)
        control = np.array([0.1, 0.3], dtype=np.float32)
        for speed in (-2.0, 0.0, 0.49, 0.5, 3.0):
            state = np.array(
                [1.0, 2.0, 0.15, speed, -0.4, 0.03, 0.02],
                dtype=np.float32,
            )
            expected = vehicle_dynamics_st(state, control, self.param_array)
            actual = compiled(state, control, self.params)
            np.testing.assert_allclose(
                actual, expected, rtol=3e-5, atol=3e-5, err_msg=f"speed={speed}"
            )

    def test_reverse_motion_and_zero_speed_jacobian_are_finite(self):
        reverse = jnp.array([0.0, 0.0, 0.2, -2.0, 0.4, 0.0, 0.0])
        derivative = single_track(reverse, jnp.zeros(2), self.params)
        heading = jnp.array([jnp.cos(reverse[4]), jnp.sin(reverse[4])])
        self.assertLess(float(jnp.dot(derivative[:2], heading)), 0.0)

        stopped = reverse.at[3].set(0.0)
        jacobian = jax.jacrev(single_track)(stopped, jnp.zeros(2), self.params)
        self.assertTrue(bool(jnp.all(jnp.isfinite(jacobian))))

    def test_parameters_are_traced_pytree_leaves_under_vmap(self):
        batched = jax.tree.map(
            lambda value: jnp.asarray([value, value]), self.params
        )
        batched = replace(batched, lr=jnp.asarray([self.params.lr, 1.2 * self.params.lr]))
        states = jnp.asarray(
            [[0.0, 0.0, 0.2, 3.0, 0.0], [0.0, 0.0, 0.2, 3.0, 0.0]]
        )
        controls = jnp.zeros((2, 2))
        result = jax.jit(jax.vmap(kinematic_single_track))(
            states, controls, batched
        )
        self.assertEqual(result.shape, (2, 5))
        self.assertNotEqual(float(result[0, 4]), float(result[1, 4]))

        gradients = jax.grad(
            lambda params: jnp.sum(
                kinematic_single_track(states[0], controls[0], params)
            )
        )(self.params)
        for leaf in jax.tree.leaves(gradients):
            self.assertTrue(bool(jnp.all(jnp.isfinite(leaf))))

    def test_integrators_match_reference_arithmetic_and_compile_with_substeps(self):
        state = jnp.array([0.0, 0.0, 0.1, 2.0, 0.2])
        control = jnp.array([0.05, 0.2])
        dt = 0.01
        derivative = kinematic_single_track(state, control, self.params)
        expected_euler = state + dt * derivative
        actual_euler = euler_step(
            kinematic_single_track, state, control, dt, self.params
        )
        np.testing.assert_allclose(actual_euler, expected_euler, rtol=2e-6, atol=2e-6)

        rollout = jax.jit(
            partial(
                integrate_substeps,
                rk4_step,
                kinematic_single_track,
                num_substeps=4,
            )
        )
        result = rollout(state, control, 0.04, self.params)
        self.assertEqual(result.shape, (5,))
        self.assertTrue(bool(jnp.all(jnp.isfinite(result))))

    def test_substep_count_is_validated_before_tracing(self):
        with self.assertRaisesRegex(ValueError, "num_substeps must be >= 1"):
            integrate_substeps(
                euler_step,
                kinematic_single_track,
                jnp.zeros(5),
                jnp.zeros(2),
                0.01,
                self.params,
                num_substeps=0,
            )


if __name__ == "__main__":
    unittest.main()
