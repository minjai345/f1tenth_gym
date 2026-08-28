"""Contracts for JAX control, FIFO state, and compiled free-flight rollouts."""

from dataclasses import replace
from functools import partial
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from f1tenth_gym.envs.action import (
    LongitudinalActionType,
    SteerActionType,
    speed_action,
    steering_angle_action,
)
from f1tenth_gym.envs.collision_models import CollisionCheckMode
from f1tenth_gym.envs.dynamic_models import (
    DynamicModel,
    F1TENTH_VEHICLE_PARAMETERS,
    vehicle_dynamics_st,
)
from f1tenth_gym.envs.env_config import ControlConfig, EnvConfig, SimulationConfig
from f1tenth_gym.envs.dynamic_models.utils import pid_accl, pid_steer
from f1tenth_gym.envs.integrators import IntegratorType, rk4_integration
from f1tenth_gym.envs.lidar import LiDARConfig
from f1tenth_gym.envs.simulator import F110Simulator
from f1tenth_gym.jax import (
    DynamicsConfig,
    DynamicsParams,
    EpisodeParams,
    LongitudinalControlMode,
    SteeringControlMode,
    adapt_actions,
    kinematic_single_track,
    make_dynamics_state,
    rk4_step,
    rollout_dynamics,
    single_track,
    speed_control,
    steering_angle_control,
    step_dynamics,
)


class TestJaxControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = F1TENTH_VEHICLE_PARAMETERS
        cls.params = DynamicsParams.from_vehicle_parameters(cls.host)

    def test_speed_controller_matches_all_numpy_quadrants(self):
        states = np.zeros((8, 7), dtype=np.float32)
        states[:, 3] = [-3.0, -3.0, 0.0, 0.0, 2.0, 2.0, 7.0, 7.0]
        targets = np.array([-5.0, 1.0, -2.0, 2.0, 1.0, 4.0, 3.0, 9.0])
        expected = np.array(
            [
                pid_accl(t, s[3], self.host.a_max, self.host.v_max, self.host.v_min)
                for t, s in zip(targets, states)
            ]
        )
        actual = jax.jit(speed_control)(targets, states, self.params)
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

    def test_steering_controller_matches_p_and_relay_modes(self):
        states = np.zeros((5, 5), dtype=np.float32)
        states[:, 2] = [-0.2, 0.0, 0.1, 0.2, 0.2]
        targets = np.array([0.4, 0.0, 0.11, -0.4, 0.20005], dtype=np.float32)
        for gain in (12.0, 0.0, -1.0):
            expected = np.array(
                [pid_steer(t, s[2], self.host.sv_max, gain) for t, s in zip(targets, states)]
            )
            actual = jax.jit(steering_angle_control)(
                targets, states, self.params, jnp.asarray(gain)
            )
            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

    def test_action_order_and_direct_modes_are_preserved(self):
        states = jnp.zeros((2, 5))
        actions = jnp.asarray([[0.3, 2.0], [-0.4, -1.0]])
        efforts = adapt_actions(
            actions,
            states,
            self.params,
            longitudinal_mode=LongitudinalControlMode.ACCELERATION,
            steering_mode=SteeringControlMode.STEERING_RATE,
            steer_kp=jnp.asarray(1.0),
        )
        np.testing.assert_array_equal(efforts, actions)


class TestJaxDynamicsState(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.host = F1TENTH_VEHICLE_PARAMETERS
        cls.params = DynamicsParams.from_vehicle_parameters(cls.host)

    @staticmethod
    def _st_config(**updates):
        values = dict(
            num_agents=2,
            state_dim=7,
            dynamics_fn=single_track,
            integrator_fn=rk4_step,
            num_substeps=2,
            longitudinal_mode=LongitudinalControlMode.TARGET_SPEED,
            steering_mode=SteeringControlMode.TARGET_ANGLE,
        )
        values.update(updates)
        return DynamicsConfig(**values)

    def test_one_step_matches_the_numpy_controller_and_integrator(self):
        config = self._st_config()
        initial = np.array(
            [
                [1.0, 2.0, 0.1, 3.0, -0.2, 0.04, 0.02],
                [-1.0, 0.5, -0.2, -1.0, 0.3, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        actions = np.array([[0.25, 5.0], [-0.1, -2.0]], dtype=np.float32)
        episode = EpisodeParams(dynamics=self.params, timestep=0.02)
        state = make_dynamics_state(initial, config)
        actual = jax.jit(step_dynamics, static_argnums=3)(
            jax.random.key(0), state, actions, config, episode
        )

        expected = initial.astype(np.float64)
        params_array = self.host.to_array()
        dt = 0.01
        for index in range(config.num_agents):
            effort = np.array(
                [
                    steering_angle_action(actions[index, 0], expected[index], self.host),
                    speed_action(actions[index, 1], expected[index], self.host),
                ]
            )
            for _ in range(config.num_substeps):
                expected[index] = rk4_integration(
                    vehicle_dynamics_st, expected[index], effort, dt, params_array
                )
            expected[index, 4] = (expected[index, 4] + np.pi) % (2 * np.pi) - np.pi

        np.testing.assert_allclose(actual.model, expected, rtol=4e-5, atol=4e-5)
        np.testing.assert_array_equal(actual.control_input, actions)
        self.assertAlmostEqual(float(actual.sim_time), 0.02, places=7)

    def test_controlled_rollouts_match_the_host_simulator(self):
        """Cross the pure seam against the live mutable simulator for KS/ST."""
        for model, state_dim, dynamics in (
            (DynamicModel.KS, 5, kinematic_single_track),
            (DynamicModel.ST, 7, single_track),
        ):
            with self.subTest(model=model.name):
                host_config = EnvConfig(
                    num_agents=2,
                    simulation_config=SimulationConfig(
                        timestep=0.02,
                        integrator_timestep=0.01,
                        integrator=IntegratorType.RK4,
                        dynamics_model=model,
                        compute_frenet_frame=False,
                        max_laps=None,
                    ),
                    control_config=ControlConfig(
                        longitudinal_mode=LongitudinalActionType.SPEED,
                        steering_mode=SteerActionType.STEERING_ANGLE,
                    ),
                    lidar_config=LiDARConfig(enabled=False),
                    collision_check=CollisionCheckMode.NONE,
                    render_enabled=False,
                )
                simulator = F110Simulator(
                    env_config=host_config,
                    vehicle_params=self.host,
                    model=model,
                    dynamics_fn=model.f_dynamics,
                    integrator_fn=rk4_integration,
                    longitudinal_type=LongitudinalActionType.SPEED,
                    steering_type=SteerActionType.STEERING_ANGLE,
                    track=None,
                    seed=3,
                )
                initial = np.zeros((2, state_dim), dtype=np.float32)
                initial[:, :5] = np.array(
                    [
                        [0.0, 0.0, 0.1, 2.0, 0.2],
                        [1.0, -1.0, -0.15, -0.5, -0.3],
                    ]
                )
                if state_dim == 7:
                    initial[:, 5:] = [[0.02, 0.01], [-0.01, 0.0]]
                simulator.reset(initial, option="state", noise_seed=3)

                config = DynamicsConfig(
                    num_agents=2,
                    state_dim=state_dim,
                    dynamics_fn=dynamics,
                    integrator_fn=rk4_step,
                    num_substeps=2,
                )
                device_state = make_dynamics_state(initial, config)
                episode = EpisodeParams(dynamics=self.params, timestep=0.02)
                compiled = jax.jit(step_dynamics, static_argnums=3)
                for step_index in range(25):
                    actions = np.array(
                        [
                            [0.2 + 0.03 * np.sin(step_index), 4.0],
                            [-0.1, -1.5 + 0.02 * step_index],
                        ],
                        dtype=np.float32,
                    )
                    simulator.step(actions)
                    device_state = compiled(
                        jax.random.fold_in(jax.random.key(3), step_index),
                        device_state,
                        actions,
                        config,
                        episode,
                    )
                np.testing.assert_allclose(
                    device_state.model,
                    simulator.state.state,
                    rtol=1.0e-4,
                    atol=1.0e-4,
                )

    def test_fifo_delays_match_the_mutable_simulator_order(self):
        config = DynamicsConfig(
            num_agents=1,
            state_dim=5,
            dynamics_fn=kinematic_single_track,
            integrator_fn=rk4_step,
            longitudinal_mode=LongitudinalControlMode.ACCELERATION,
            steering_mode=SteeringControlMode.STEERING_RATE,
            steer_delay_steps=2,
            throttle_delay_steps=3,
        )
        state = make_dynamics_state(jnp.zeros((1, 5)), config)
        actions = jnp.asarray(
            [[[1.0, 10.0]], [[2.0, 20.0]], [[3.0, 30.0]], [[4.0, 40.0]]]
        )
        episode = EpisodeParams(dynamics=self.params, timestep=0.01)
        final, history = jax.jit(rollout_dynamics, static_argnums=3)(
            jax.random.key(2), state, actions, config, episode
        )
        expected = np.array(
            [[[0.0, 0.0]], [[0.0, 0.0]], [[1.0, 0.0]], [[2.0, 10.0]]]
        )
        np.testing.assert_array_equal(history.control_input, expected)
        np.testing.assert_array_equal(final.steer_delay_head, [0])
        np.testing.assert_array_equal(final.throttle_delay_head, [1])

    def test_noise_uses_only_the_explicit_key(self):
        config = self._st_config(num_substeps=1)
        state = make_dynamics_state(jnp.zeros((2, 7)), config)
        episode = EpisodeParams(
            dynamics=self.params,
            timestep=0.01,
            steer_noise_std=0.2,
            accel_noise_std=0.7,
        )
        step = partial(step_dynamics, config=config, episode=episode)
        compiled = jax.jit(step)
        actions = jnp.zeros((2, 2))
        first = compiled(jax.random.key(9), state, actions)
        replay = compiled(jax.random.key(9), state, actions)
        other = compiled(jax.random.key(10), state, actions)
        np.testing.assert_array_equal(first.control_input, replay.control_input)
        self.assertFalse(np.array_equal(first.control_input, other.control_input))

        quiet = replace(episode, steer_noise_std=0.0, accel_noise_std=0.0)
        a = step_dynamics(jax.random.key(1), state, actions, config, quiet)
        b = step_dynamics(jax.random.key(2), state, actions, config, quiet)
        np.testing.assert_array_equal(a.model, b.model)

    def test_vmap_accepts_different_episode_parameters(self):
        config = DynamicsConfig(
            num_agents=1,
            state_dim=5,
            dynamics_fn=kinematic_single_track,
            integrator_fn=rk4_step,
        )
        base_state = make_dynamics_state(
            jnp.asarray([[0.0, 0.0, 0.2, 3.0, 0.0]]), config
        )
        states = jax.tree.map(lambda value: jnp.stack((value, value)), base_state)
        params = jax.tree.map(
            lambda value: jnp.asarray([value, value]), self.params
        )
        params = replace(
            params,
            lr=jnp.asarray([self.params.lr, 1.25 * self.params.lr]),
        )
        episodes = EpisodeParams(
            dynamics=params,
            timestep=jnp.asarray([0.01, 0.01]),
            steer_kp=jnp.zeros(2),
            steer_noise_std=jnp.zeros(2),
            accel_noise_std=jnp.zeros(2),
        )
        keys = jax.random.split(jax.random.key(4), 2)
        actions = jnp.asarray([[[0.2, 3.0]], [[0.2, 3.0]]])

        vmapped = jax.jit(
            jax.vmap(lambda key, state, action, ep: step_dynamics(
                key, state, action, config, ep
            ))
        )(keys, states, actions, episodes)
        self.assertEqual(vmapped.model.shape, (2, 1, 5))
        self.assertNotEqual(float(vmapped.model[0, 0, 4]), float(vmapped.model[1, 0, 4]))

    def test_new_domain_randomization_values_do_not_retrace(self):
        config = DynamicsConfig(
            num_agents=1,
            state_dim=5,
            dynamics_fn=kinematic_single_track,
            integrator_fn=rk4_step,
        )
        state = make_dynamics_state(
            jnp.asarray([[0.0, 0.0, 0.2, 3.0, 0.0]]), config
        )
        traces = []

        def transition(current, episode):
            traces.append(None)
            return step_dynamics(
                jax.random.key(0),
                current,
                jnp.asarray([[0.1, 4.0]]),
                config,
                episode,
            )

        compiled = jax.jit(transition)
        nominal = EpisodeParams(dynamics=self.params, timestep=0.01)
        randomized = replace(
            nominal,
            dynamics=replace(
                self.params,
                mu=0.8 * self.params.mu,
                lr=1.1 * self.params.lr,
            ),
            steer_noise_std=0.05,
        )
        compiled(state, nominal).model.block_until_ready()
        compiled(state, randomized).model.block_until_ready()
        self.assertEqual(len(traces), 1)

    def test_scan_eval_shape_and_action_gradient_are_finite(self):
        config = DynamicsConfig(
            num_agents=1,
            state_dim=5,
            dynamics_fn=kinematic_single_track,
            integrator_fn=rk4_step,
            num_substeps=2,
            longitudinal_mode=LongitudinalControlMode.ACCELERATION,
            steering_mode=SteeringControlMode.STEERING_RATE,
        )
        state = make_dynamics_state(
            jnp.asarray([[0.0, 0.0, 0.1, 2.0, 0.2]]), config
        )
        episode = EpisodeParams(dynamics=self.params, timestep=0.02)
        actions = jnp.zeros((12, 1, 2))
        shaped = jax.eval_shape(
            lambda s, a: rollout_dynamics(
                jax.random.key(0), s, a, config, episode
            ),
            state,
            actions,
        )
        self.assertEqual(shaped[1].model.shape, (12, 1, 5))

        def loss(commands):
            final, _history = rollout_dynamics(
                jax.random.key(0), state, commands, config, episode
            )
            return final.model[0, 0] + final.model[0, 1]

        gradient = jax.jit(jax.grad(loss))(actions)
        self.assertTrue(bool(jnp.all(jnp.isfinite(gradient))))

    def test_structural_shape_errors_fail_before_compilation(self):
        config = self._st_config()
        with self.assertRaisesRegex(ValueError, "model_state must have shape"):
            make_dynamics_state(jnp.zeros((1, 7)), config)
        state = make_dynamics_state(jnp.zeros((2, 7)), config)
        episode = EpisodeParams(dynamics=self.params, timestep=0.01)
        with self.assertRaisesRegex(ValueError, "actions must have shape"):
            step_dynamics(jax.random.key(0), state, jnp.zeros((1, 2)), config, episode)


if __name__ == "__main__":
    unittest.main()
