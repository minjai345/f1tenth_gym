"""Focused gates for the optional device-batched SBX adapter."""

import importlib.util
import unittest

import jax
import numpy as np

from f1tenth_gym.envs.batching import PolicyField, PolicyLayout
from f1tenth_gym.envs.collision_models import CollisionCheckMode
from f1tenth_gym.envs.dynamic_models import (
    DynamicModel,
    F1TENTH_VEHICLE_PARAMETERS,
)
from f1tenth_gym.envs.env_config import (
    DomainRandomizationConfig,
    EnvConfig,
    ResetConfig,
    SimulationConfig,
    TerminationConfig,
)
from f1tenth_gym.envs.integrators import IntegratorType
from f1tenth_gym.envs.jax_simulator import JaxSimulator
from f1tenth_gym.envs.lidar import LiDARConfig
from f1tenth_gym.envs.reset import ResetStrategy
from f1tenth_gym.envs.sbx import F110SBXVecEnv
from f1tenth_gym.envs.track import Track


_HAS_SB3 = importlib.util.find_spec("stable_baselines3") is not None
_HAS_SBX = importlib.util.find_spec("sbx") is not None


def _circle_track() -> Track:
    theta = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    return Track.from_refline(
        x=6.0 * np.cos(theta),
        y=6.0 * np.sin(theta),
        velx=np.full(theta.shape, 3.0),
    )


def _config(
    track: Track,
    *,
    max_episode_steps: int = 8,
    randomization: DomainRandomizationConfig | None = None,
) -> EnvConfig:
    return EnvConfig(
        map_name=track,
        simulation_config=SimulationConfig(
            dynamics_model=DynamicModel.KS,
            integrator=IntegratorType.EULER,
            max_laps=None,
        ),
        reset_config=ResetConfig(strategy=ResetStrategy.RL_RANDOM_STATIC),
        lidar_config=LiDARConfig(enabled=False),
        termination_config=TerminationConfig(
            max_episode_steps=max_episode_steps
        ),
        domain_randomization_config=(
            DomainRandomizationConfig()
            if randomization is None
            else randomization
        ),
        collision_check=CollisionCheckMode.NONE,
        render_enabled=False,
    )


@unittest.skipUnless(_HAS_SB3, "stable-baselines3 is an optional dependency")
class TestF110SBXVecEnv(unittest.TestCase):
    def test_constructs_from_simulator_with_a_selected_policy_layout(self):
        track = _circle_track()
        simulator = JaxSimulator(_config(track), track, device="cpu")
        env = F110SBXVecEnv(
            simulator,
            3,
            seed=5,
            policy_layout=PolicyLayout((PolicyField.KINEMATIC_STATE,)),
        )
        self.addCleanup(env.close)

        observation = env.reset()

        self.assertEqual(observation.shape, (3, 5))
        self.assertTrue(env.observation_space.contains(observation[0]))
        self.assertIsNotNone(env.batch_state)
        with self.assertRaisesRegex(ValueError, "shape"):
            env.step_async(np.zeros((3, 1), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "within"):
            env.step_async(np.full((3, 2), 1.1, dtype=np.float32))

    def test_from_config_preserves_terminal_observations_and_writable_arrays(self):
        track = _circle_track()
        env = F110SBXVecEnv.from_config(
            _config(track, max_episode_steps=1),
            track,
            4,
            device="cpu",
            seed=7,
        )
        self.addCleanup(env.close)
        env.seed(11)
        observation = env.reset()

        next_observation, rewards, dones, infos = env.step(
            np.zeros((4, 2), dtype=np.float32)
        )

        self.assertEqual(observation.shape, (4, 8))
        self.assertEqual(next_observation.shape, (4, 8))
        self.assertEqual(rewards.shape, (4,))
        self.assertTrue(next_observation.flags.writeable)
        self.assertTrue(rewards.flags.writeable)
        self.assertTrue(np.all(dones))
        for info in infos:
            self.assertTrue(info["TimeLimit.truncated"])
            terminal = info["terminal_observation"]
            self.assertEqual(terminal.shape, (8,))
            self.assertTrue(terminal.flags.writeable)
            self.assertTrue(env.observation_space.contains(terminal))

    def test_device_domain_randomization_resamples_autoreset_rows(self):
        track = _circle_track()
        vehicle = F1TENTH_VEHICLE_PARAMETERS
        randomization = DomainRandomizationConfig(
            enabled=True,
            low=vehicle.with_updates(m=3.0),
            high=vehicle.with_updates(m=4.0),
        )
        env = F110SBXVecEnv.from_config(
            _config(
                track,
                max_episode_steps=1,
                randomization=randomization,
            ),
            track,
            16,
            device="cpu",
            seed=13,
        )
        self.addCleanup(env.close)
        env.reset()
        initial_state = env.batch_state
        self.assertIsNotNone(initial_state)
        initial_masses = np.asarray(
            jax.device_get(initial_state.params.dynamics.vehicle.m)
        )

        _observation, _rewards, dones, _infos = env.step(
            np.zeros((16, 2), dtype=np.float32)
        )
        reset_state = env.batch_state
        self.assertIsNotNone(reset_state)
        reset_masses = np.asarray(
            jax.device_get(reset_state.params.dynamics.vehicle.m)
        )

        self.assertTrue(np.all(dones))
        self.assertTrue(np.all((3.0 <= initial_masses) & (initial_masses <= 4.0)))
        self.assertTrue(np.all((3.0 <= reset_masses) & (reset_masses <= 4.0)))
        self.assertGreater(float(np.ptp(initial_masses)), 0.0)
        self.assertFalse(np.array_equal(initial_masses, reset_masses))

    @unittest.skipUnless(_HAS_SBX, "sbx-rl is an optional dependency")
    def test_sbx_ppo_completes_a_timeout_rollout(self):
        from sbx import PPO

        track = _circle_track()
        env = F110SBXVecEnv.from_config(
            _config(track, max_episode_steps=1),
            track,
            4,
            device="cpu",
            seed=17,
        )
        self.addCleanup(env.close)
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=2,
            batch_size=4,
            n_epochs=1,
            policy_kwargs={"net_arch": [16]},
            seed=17,
            verbose=0,
        )

        model.learn(total_timesteps=8)

        self.assertGreaterEqual(model.num_timesteps, 8)


if __name__ == "__main__":
    unittest.main()
