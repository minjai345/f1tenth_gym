"""Gymnasium lifecycle contracts for the JAX-backed environment adapter."""

import unittest
from unittest import mock

import gymnasium as gym
import numpy as np

from f1tenth_gym.envs.action import (
    LongitudinalActionType,
    SteerActionType,
)
from f1tenth_gym.envs.collision_models import CollisionCheckMode
from f1tenth_gym.envs.dynamic_models import (
    DynamicModel,
    F1TENTH_VEHICLE_PARAMETERS,
)
from f1tenth_gym.envs.env_config import (
    ControlConfig,
    DomainRandomizationConfig,
    EnvConfig,
    ObservationConfig,
    ResetConfig,
    RewardConfig,
    RewardMode,
    SimulationConfig,
    TerminationConfig,
)
from f1tenth_gym.envs.f110_env import F110Env
from f1tenth_gym.envs.integrators import IntegratorType
from f1tenth_gym.envs.lidar import LiDARConfig
from f1tenth_gym.envs.observation import ObservationType
from f1tenth_gym.envs.reset import ResetStrategy
from f1tenth_gym.envs.track import Track
from f1tenth_gym.envs.wrappers import SingleAgentWrapper
from f1tenth_gym.envs.jax_env import JaxF110Env


VEHICLE = F1TENTH_VEHICLE_PARAMETERS


def circle_track(count: int = 64, radius: float = 6.0) -> Track:
    """Build a small closed track without map downloads."""
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return Track.from_refline(
        x=radius * np.cos(theta),
        y=radius * np.sin(theta),
        velx=np.full(count, 4.0),
    )


def lightweight_config(track: Track, **changes) -> EnvConfig:
    """Return a fast, offline config with unused geometry disabled."""
    defaults = {
        "map_name": track,
        "simulation_config": SimulationConfig(
            dynamics_model=DynamicModel.KS,
            integrator=IntegratorType.EULER,
            max_laps=None,
        ),
        "lidar_config": LiDARConfig(enabled=False, num_beams=7),
        "collision_check": CollisionCheckMode.NONE,
        "render_enabled": False,
    }
    defaults.update(changes)
    return EnvConfig(**defaults)


def assert_nested_equal(
    case: unittest.TestCase,
    actual,
    expected,
) -> None:
    """Assert exact equality for the nested Dict observation layouts."""
    case.assertEqual(set(actual), set(expected))
    for key in actual:
        if isinstance(actual[key], dict):
            assert_nested_equal(case, actual[key], expected[key])
        else:
            np.testing.assert_array_equal(actual[key], expected[key])


def assert_nested_allclose(
    case: unittest.TestCase,
    actual,
    expected,
    *,
    rtol: float = 2.0e-5,
    atol: float = 2.0e-5,
) -> None:
    """Compare backend observations within the production float32 budget."""
    case.assertEqual(set(actual), set(expected))
    for key in actual:
        if isinstance(actual[key], dict):
            assert_nested_allclose(
                case, actual[key], expected[key], rtol=rtol, atol=atol
            )
        else:
            np.testing.assert_allclose(
                actual[key], expected[key], rtol=rtol, atol=atol
            )


class TestJaxF110Env(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.track = circle_track()

    def make_env(self, config: EnvConfig | None = None) -> JaxF110Env:
        env = JaxF110Env(config or lightweight_config(self.track))
        self.addCleanup(env.close)
        return env

    def test_unwrapped_environment_passes_the_gymnasium_checker(self):
        from gymnasium.utils.env_checker import check_env

        env = self.make_env()
        check_env(env, skip_render_check=True)

    def test_reset_and_step_return_the_native_contract_and_copied_info(self):
        config = lightweight_config(self.track, num_agents=2)
        env = self.make_env(config)

        observation, reset_info = env.reset(seed=11)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(set(observation), {"agent_0", "agent_1"})
        self.assertEqual(
            set(reset_info),
            {"lap_times", "lap_counts", "sim_time", "terminated_agents"},
        )
        for name in ("lap_times", "lap_counts", "terminated_agents"):
            self.assertIsInstance(reset_info[name], np.ndarray)
            self.assertEqual(reset_info[name].shape, (2,))
        self.assertEqual(reset_info["lap_times"].dtype, np.float64)
        self.assertEqual(reset_info["lap_counts"].dtype, np.float64)
        self.assertEqual(reset_info["terminated_agents"].dtype, np.bool_)
        self.assertIsInstance(reset_info["sim_time"], float)

        action = np.zeros((2, 2), dtype=np.float32)
        observation, reward, terminated, truncated, info = env.step(action)
        self.assertTrue(env.observation_space.contains(observation))
        self.assertIs(type(reward), float)
        self.assertIs(type(terminated), bool)
        self.assertIs(type(truncated), bool)
        self.assertEqual(
            set(info),
            {
                "lap_times",
                "lap_counts",
                "sim_time",
                "collisions",
                "terminated_agents",
                "progress",
            },
        )
        for name in (
            "lap_times",
            "lap_counts",
            "collisions",
            "terminated_agents",
            "progress",
        ):
            self.assertIsInstance(info[name], np.ndarray)
            self.assertEqual(info[name].shape, (2,))
        self.assertEqual(info["collisions"].dtype, np.float32)
        self.assertEqual(info["terminated_agents"].dtype, np.bool_)
        self.assertEqual(info["lap_times"].dtype, np.float64)
        self.assertEqual(info["lap_counts"].dtype, np.float64)
        self.assertEqual(info["progress"].dtype, np.float64)
        self.assertIsInstance(info["sim_time"], float)

        # The public observation deliberately carries the pre-transition clock,
        # while info exposes the simulator's post-transition clock.
        self.assertEqual(float(observation["agent_0"]["sim_time"]), 0.0)
        self.assertAlmostEqual(info["sim_time"], 0.01, places=7)

        snapshots = {
            name: value.copy()
            for name, value in info.items()
            if isinstance(value, np.ndarray)
        }
        _next_observation, _reward, _terminated, _truncated, next_info = (
            env.step(action)
        )
        for name, expected in snapshots.items():
            np.testing.assert_array_equal(info[name], expected)
            self.assertIsNot(info[name], next_info[name])

        # Returned arrays must not be able to corrupt the live episode carry.
        info["lap_counts"][:] = 99
        info["terminated_agents"][:] = True
        self.assertFalse(np.any(next_info["terminated_agents"]))
        self.assertFalse(np.any(next_info["lap_counts"] == 99))

    def test_actions_are_shape_checked_but_not_rejected_by_the_space(self):
        env = self.make_env()
        with self.assertRaises(gym.error.ResetNeeded):
            env.step(np.zeros((1, 2), dtype=np.float32))
        env.reset(seed=12)

        with self.assertRaises(ValueError):
            env.step(np.zeros((2,), dtype=np.float32))
        with self.assertRaises(ValueError):
            env.step(np.zeros((1, 1), dtype=np.float32))

        outside = np.asarray(
            [[env.action_space.high[0, 0] + 1.0,
              env.action_space.high[0, 1] + 10.0]],
            dtype=np.float32,
        )
        self.assertFalse(env.action_space.contains(outside))
        observation, *_tail = env.step(outside)
        self.assertTrue(env.observation_space.contains(observation))

    def test_seed_replays_stochastic_steps_and_invalid_action_keeps_the_key(self):
        noisy = lightweight_config(
            self.track,
            control_config=ControlConfig(
                steer_noise_std=0.08,
                accl_noise_std=0.25,
            ),
        )
        tested = self.make_env(noisy)
        reference = self.make_env(noisy)
        state = np.asarray(
            [[6.0, 0.0, 0.05, 1.5, np.pi / 2.0]], dtype=np.float32
        )
        action = np.asarray([[0.12, 3.0]], dtype=np.float32)

        tested.reset(seed=123, options={"states": state})
        reference.reset(seed=123, options={"states": state})
        with self.assertRaises(ValueError):
            tested.step(np.zeros((2,), dtype=np.float32))
        tested_step = tested.step(action)
        reference_step = reference.step(action)
        assert_nested_equal(self, tested_step[0], reference_step[0])
        self.assertEqual(tested_step[1:4], reference_step[1:4])
        for key in reference_step[4]:
            if isinstance(reference_step[4][key], np.ndarray):
                np.testing.assert_array_equal(
                    tested_step[4][key], reference_step[4][key]
                )
            else:
                self.assertEqual(tested_step[4][key], reference_step[4][key])

        replay_observation, _ = tested.reset(
            seed=123, options={"states": state}
        )
        replay_step = tested.step(action)
        assert_nested_equal(self, replay_step[0], reference_step[0])
        self.assertNotEqual(
            float(replay_observation["agent_0"]["state"][3]), 0.0
        )

        tested.reset(seed=124, options={"states": state})
        different_step = tested.step(action)
        self.assertFalse(
            np.array_equal(
                different_step[0]["agent_0"]["state"],
                reference_step[0]["agent_0"]["state"],
            )
        )

    def test_builtin_reward_selects_only_the_configured_ego_agent(self):
        config = lightweight_config(
            self.track,
            num_agents=2,
            ego_index=1,
            reward_config=RewardConfig(
                mode=RewardMode.PROGRESS,
                progress_weight=0.0,
                velocity_weight=1.0,
            ),
        )
        env = self.make_env(config)
        states = np.asarray(
            [
                [6.0, 0.0, 0.0, 1.0, np.pi / 2.0],
                [-6.0, 0.0, 0.0, 3.0, -np.pi / 2.0],
            ],
            dtype=np.float32,
        )
        env.reset(seed=8, options={"states": states})
        observation, reward, *_tail = env.step(
            np.asarray([[0.0, 1.0], [0.0, 3.0]], dtype=np.float32)
        )
        ego_speed = float(observation["agent_1"]["std_state"][3])
        opponent_speed = float(observation["agent_0"]["std_state"][3])
        self.assertAlmostEqual(reward, ego_speed, places=6)
        self.assertNotAlmostEqual(reward, opponent_speed, places=5)

    def test_controlled_rollout_matches_the_mutable_gym_boundary(self):
        config = lightweight_config(
            self.track,
            control_config=ControlConfig(
                longitudinal_mode=LongitudinalActionType.ACCL,
                steering_mode=SteerActionType.STEERING_SPEED,
            ),
        )
        functional = self.make_env(config)
        mutable = F110Env(config)
        self.addCleanup(mutable.close)
        state = np.asarray(
            [[6.0, 0.0, 0.04, 1.7, np.pi / 2.0]], dtype=np.float32
        )
        functional_observation, functional_info = functional.reset(
            seed=22, options={"states": state}
        )
        mutable_observation, mutable_info = mutable.reset(
            seed=22, options={"states": state}
        )

        assert_nested_allclose(
            self, functional_observation, mutable_observation
        )
        self.assertEqual(set(functional_info), set(mutable_info))
        action = np.asarray([[0.03, 0.4]], dtype=np.float32)
        for _ in range(6):
            functional_step = functional.step(action)
            mutable_step = mutable.step(action)
            for field in ("state", "std_state", "frenet_pose", "sim_time"):
                np.testing.assert_allclose(
                    functional_step[0]["agent_0"][field],
                    mutable_step[0]["agent_0"][field],
                    rtol=2.0e-5,
                    atol=2.0e-5,
                )
            self.assertAlmostEqual(
                functional_step[1], mutable_step[1], places=7
            )
            self.assertEqual(functional_step[2:4], mutable_step[2:4])
            self.assertEqual(set(functional_step[4]), set(mutable_step[4]))
            for field in functional_step[4]:
                np.testing.assert_allclose(
                    functional_step[4][field],
                    mutable_step[4][field],
                    rtol=2.0e-5,
                    atol=2.0e-5,
                )

    def test_pose_and_state_reset_overrides_preserve_precedence_and_values(self):
        env = self.make_env(lightweight_config(self.track, num_agents=2))
        poses = np.asarray(
            [[6.0, 0.0, np.pi / 2.0], [-6.0, 0.0, -np.pi / 2.0]],
            dtype=np.float64,
        )
        states = np.asarray(
            [
                [1.0, 2.0, 0.2, 3.0, 0.4],
                [-1.0, -2.0, -0.1, -4.0, -0.3],
            ],
            dtype=np.float64,
        )

        observation, _ = env.reset(
            seed=5, options={"poses": poses, "states": states}
        )
        model = np.stack(
            [observation["agent_0"]["state"],
             observation["agent_1"]["state"]]
        )
        np.testing.assert_allclose(model[:, :2], poses[:, :2])
        np.testing.assert_allclose(model[:, 4], poses[:, 2])
        np.testing.assert_array_equal(model[:, 2:4], 0.0)
        self.assertEqual(model.dtype, np.float32)

        observation, _ = env.reset(seed=5, options={"states": states})
        model = np.stack(
            [observation["agent_0"]["state"],
             observation["agent_1"]["state"]]
        )
        np.testing.assert_array_equal(model, states.astype(np.float32))

        with self.assertRaises(ValueError):
            env.reset(options={"poses": np.zeros((2, 2), dtype=np.float32)})
        with self.assertRaises(ValueError):
            env.reset(options={"states": np.zeros((2, 7), dtype=np.float32)})

    def test_custom_reward_runs_last_with_the_final_public_values(self):
        seen = {}

        def reward_fn(obs, action, info, terminated, truncated):
            seen["obs"] = obs
            seen["action"] = np.array(action, copy=True)
            seen["info"] = info
            seen["terminated"] = terminated
            seen["truncated"] = truncated
            return np.float32(7.25)

        config = lightweight_config(
            self.track,
            reward_config=RewardConfig(
                mode=RewardMode.CUSTOM,
                reward_fn=reward_fn,
            ),
        )
        env = self.make_env(config)
        state = np.asarray(
            [[6.0, 0.0, 0.0, 1.0, np.pi / 2.0]], dtype=np.float32
        )
        env.reset(seed=15, options={"states": state})
        action = np.asarray([[0.0, 2.0]], dtype=np.float64)
        observation, reward, terminated, truncated, info = env.step(action)

        self.assertEqual(reward, 7.25)
        self.assertIs(type(reward), float)
        np.testing.assert_array_equal(seen["action"], action)
        self.assertIs(seen["info"], info)
        self.assertIs(seen["terminated"], terminated)
        self.assertIs(seen["truncated"], truncated)
        self.assertEqual(
            set(seen["info"]),
            {
                "lap_times",
                "lap_counts",
                "sim_time",
                "collisions",
                "terminated_agents",
                "progress",
            },
        )
        self.assertEqual(
            float(seen["obs"]["agent_0"]["sim_time"]), 0.0
        )
        self.assertGreater(seen["info"]["sim_time"], 0.0)
        assert_nested_equal(self, seen["obs"], observation)

    def test_domain_randomization_draw_is_seeded_shared_and_nominal_is_stable(self):
        low = VEHICLE.with_updates(m=3.0)
        high = VEHICLE.with_updates(m=4.0)
        config = lightweight_config(
            self.track,
            num_agents=2,
            domain_randomization_config=DomainRandomizationConfig(
                enabled=True, low=low, high=high
            ),
        )
        env = self.make_env(config)
        poses = np.asarray(
            [[6.0, 0.0, np.pi / 2.0], [-6.0, 0.0, -np.pi / 2.0]],
            dtype=np.float32,
        )
        values = []
        for seed in range(6):
            env.reset(seed=seed, options={"poses": poses})
            sampled = env.episode_vehicle_params
            self.assertGreaterEqual(sampled.m, 3.0)
            self.assertLessEqual(sampled.m, 4.0)
            values.append(sampled.m)
        self.assertGreater(len(set(values)), 1)
        self.assertEqual(env.vehicle_params.m, VEHICLE.m)
        self.assertEqual(env.env_config.params.m, VEHICLE.m)

        env.reset(seed=19, options={"poses": poses})
        first = env.episode_vehicle_params
        env.reset(seed=19, options={"poses": poses})
        second = env.episode_vehicle_params
        np.testing.assert_array_equal(first.to_array(), second.to_array())

    def test_configure_rebuilds_topology_and_rearms_the_config_seed(self):
        reset_config = ResetConfig(strategy=ResetStrategy.RL_RANDOM_STATIC)
        config = lightweight_config(
            self.track, seed=77, reset_config=reset_config
        )
        env = self.make_env(config)

        first, _ = env.reset()
        second, _ = env.reset()
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(
                first["agent_0"]["state"], second["agent_0"]["state"]
            )
        env.configure(config)
        replay, _ = env.reset()
        np.testing.assert_array_equal(
            replay["agent_0"]["state"], first["agent_0"]["state"]
        )

        changed = lightweight_config(
            self.track,
            num_agents=2,
            observation_config=ObservationConfig(
                type=ObservationType.KINEMATIC_STATE
            ),
        )
        env.configure(changed)
        self.assertEqual(env.num_agents, 2)
        self.assertEqual(env.agent_ids, ["agent_0", "agent_1"])
        self.assertEqual(env.action_space.shape, (2, 2))
        observation, _ = env.reset(seed=4)
        self.assertEqual(set(observation), {"agent_0", "agent_1"})
        self.assertTrue(env.observation_space.contains(observation))

        installed_config = env.env_config
        installed_simulator = env.sim
        installed_state = env.core_state
        invalid = changed.with_updates(map_name=object())
        with self.assertRaisesRegex(TypeError, "map must be"):
            env.configure(invalid)
        self.assertIs(env.env_config, installed_config)
        self.assertIs(env.sim, installed_simulator)
        self.assertIs(env.core_state, installed_state)

        with self.assertRaises(TypeError):
            env.configure(object())

    def test_update_params_rebuilds_spaces_and_keeps_updates_shared(self):
        env = self.make_env()
        before_action = float(env.action_space.high[0, 1])
        before_state = float(
            env.observation_space["agent_0"]["std_state"].high[3]
        )
        updated = VEHICLE.with_updates(v_max=30.0)

        with self.assertRaises(NotImplementedError):
            env.update_params(updated, index=0)
        with self.assertRaises(TypeError):
            env.update_params(object())

        env.reset(seed=9)
        state = env.core_state
        transition_key = env._transition_key
        env.update_params(updated)
        self.assertIs(env.core_state, state)
        self.assertIs(env._transition_key, transition_key)
        self.assertEqual(env.vehicle_params, updated)
        self.assertEqual(env.env_config.params, updated)
        self.assertGreater(float(env.action_space.high[0, 1]), before_action)
        self.assertEqual(float(env.action_space.high[0, 1]), 30.0)
        self.assertGreater(
            float(env.observation_space["agent_0"]["std_state"].high[3]),
            before_state,
        )
        env.reset(seed=9)
        self.assertEqual(env.episode_vehicle_params, updated)

    def test_update_map_rebuilds_against_the_new_resolved_track(self):
        env = self.make_env()
        env.reset(seed=8)
        replacement = circle_track(count=72, radius=9.0)

        env.update_map(replacement)

        self.assertIs(env.track, replacement)
        self.assertIs(env.env_config.map_name, replacement)
        self.assertIsNone(env.core_state)
        observation, _info = env.reset(seed=8)
        self.assertTrue(env.observation_space.contains(observation))

    def test_renderer_reuses_the_default_view_and_closes_idempotently(self):
        renderer = mock.Mock()
        frame = np.zeros((12, 10, 3), dtype=np.uint8)
        renderer.render.return_value = frame
        config = lightweight_config(
            self.track,
            observation_config=ObservationConfig(
                type=ObservationType.KINEMATIC_STATE
            ),
            render_enabled=True,
        )
        with mock.patch(
            "f1tenth_gym.envs.jax_env.make_renderer",
            return_value=(renderer, config.render_config),
        ):
            env = JaxF110Env(config, render_mode="rgb_array")

        observation, _info = env.reset(seed=18)
        self.assertNotIn("std_state", observation["agent_0"])
        self.assertIn("std_state", env.render_obs["agent_0"])
        self.assertIn("state", env.render_obs["agent_0"])
        renderer.update_params.assert_called_once_with(
            env.episode_vehicle_params
        )

        actual = env.render()
        self.assertIs(actual, frame)
        renderer.update.assert_called_once_with(obs=env.render_obs)
        renderer.render.assert_called_once_with()

        callback = mock.Mock()
        env.add_render_callback(callback)
        renderer.add_renderer_callback.assert_called_once_with(callback)
        env.close()
        env.close()
        renderer.close.assert_called_once_with()

    def test_timeout_is_truncation_and_reset_clears_the_episode(self):
        config = lightweight_config(
            self.track,
            termination_config=TerminationConfig(max_episode_steps=2),
        )
        env = self.make_env(config)
        env.reset(seed=21)
        action = np.asarray([[0.0, 2.0]], dtype=np.float32)

        _first_observation, _reward, terminated, truncated, info = env.step(
            action
        )
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertAlmostEqual(info["sim_time"], 0.01, places=7)
        second_observation, _reward, terminated, truncated, info = env.step(
            action
        )
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertAlmostEqual(info["sim_time"], 0.02, places=7)

        # Neither the core nor the Gym adapter freezes or auto-resets.  A
        # caller-controlled third transition advances the same timed-out state.
        observation, _reward, terminated, truncated, info = env.step(action)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertAlmostEqual(
            float(observation["agent_0"]["sim_time"]), 0.02, places=7
        )
        self.assertAlmostEqual(info["sim_time"], 0.03, places=7)
        self.assertFalse(
            np.array_equal(
                observation["agent_0"]["state"],
                second_observation["agent_0"]["state"],
            )
        )

        observation, reset_info = env.reset(seed=21)
        self.assertEqual(float(observation["agent_0"]["sim_time"]), 0.0)
        self.assertEqual(reset_info["sim_time"], 0.0)
        self.assertFalse(np.any(reset_info["terminated_agents"]))

    def test_single_agent_and_flatten_wrappers_pass_the_checker(self):
        from gymnasium.utils.env_checker import check_env

        config = lightweight_config(
            self.track,
            observation_config=ObservationConfig(
                type=ObservationType.KINEMATIC_STATE
            ),
        )
        base = self.make_env(config)
        env = gym.wrappers.FlattenObservation(SingleAgentWrapper(base))
        check_env(env, skip_render_check=True)

        observation, _info = env.reset(seed=31)
        self.assertIsInstance(observation, np.ndarray)
        self.assertEqual(observation.dtype, np.float32)
        self.assertEqual(observation.ndim, 1)
        next_observation, reward, terminated, truncated, _info = env.step(
            np.zeros((2,), dtype=np.float32)
        )
        self.assertEqual(next_observation.shape, observation.shape)
        self.assertIs(type(reward), float)
        self.assertIs(type(terminated), bool)
        self.assertIs(type(truncated), bool)


if __name__ == "__main__":
    unittest.main()
