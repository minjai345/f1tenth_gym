"""Optional SB3/SBX compatibility gates for the JAX Gymnasium adapter."""

import importlib.util
import unittest

import gymnasium as gym
import numpy as np

from f1tenth_gym.envs.collision_models import CollisionCheckMode
from f1tenth_gym.envs.dynamic_models import DynamicModel
from f1tenth_gym.envs.env_config import (
    EnvConfig,
    ObservationConfig,
    SimulationConfig,
    TerminationConfig,
)
from f1tenth_gym.envs.integrators import IntegratorType
from f1tenth_gym.envs.lidar import LiDARConfig
from f1tenth_gym.envs.observation import ObservationType
from f1tenth_gym.envs.track import Track
from f1tenth_gym.envs.wrappers import SingleAgentWrapper
from f1tenth_gym.envs.jax_env import JaxF110Env

_HAS_SB3 = importlib.util.find_spec("stable_baselines3") is not None
_HAS_SBX = importlib.util.find_spec("sbx") is not None


def _training_env():
    theta = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    track = Track.from_refline(
        x=6.0 * np.cos(theta),
        y=6.0 * np.sin(theta),
        velx=np.full(theta.shape, 3.0),
    )
    config = EnvConfig(
        map_name=track,
        simulation_config=SimulationConfig(
            dynamics_model=DynamicModel.KS,
            integrator=IntegratorType.EULER,
            max_laps=None,
        ),
        observation_config=ObservationConfig(
            type=ObservationType.KINEMATIC_STATE
        ),
        lidar_config=LiDARConfig(enabled=False),
        termination_config=TerminationConfig(max_episode_steps=16),
        collision_check=CollisionCheckMode.NONE,
        render_enabled=False,
    )
    return gym.wrappers.FlattenObservation(
        SingleAgentWrapper(JaxF110Env(config))
    )


class TestOptionalJaxRlCompatibility(unittest.TestCase):
    @unittest.skipUnless(_HAS_SB3, "stable-baselines3 is an optional dependency")
    def test_stable_baselines3_environment_checker(self):
        from stable_baselines3.common.env_checker import check_env

        env = _training_env()
        self.addCleanup(env.close)
        check_env(env, warn=True, skip_render_check=True)

    @unittest.skipUnless(_HAS_SBX, "sbx-rl is an optional dependency")
    def test_sbx_ppo_completes_one_update(self):
        from sbx import PPO

        env = _training_env()
        self.addCleanup(env.close)
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=8,
            batch_size=4,
            n_epochs=1,
            policy_kwargs={"net_arch": [16]},
            seed=4,
            verbose=0,
        )

        model.learn(total_timesteps=8)

        self.assertGreaterEqual(model.num_timesteps, 8)
        observation, _info = env.reset(seed=4)
        action, _state = model.predict(observation, deterministic=True)
        self.assertTrue(env.action_space.contains(action))
        next_observation, reward, terminated, truncated, info = env.step(action)
        self.assertTrue(env.observation_space.contains(next_observation))
        self.assertTrue(np.isfinite(reward))
        self.assertIs(type(terminated), bool)
        self.assertIs(type(truncated), bool)
        self.assertIsInstance(info, dict)


if __name__ == "__main__":
    unittest.main()
