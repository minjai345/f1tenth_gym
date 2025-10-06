import unittest

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box

from f1tenth_gym.envs.env_config import EnvConfig, ObservationConfig
from f1tenth_gym.envs.observation import (
    observation_factory,
    ObservationType,
    ObservationFeature,
)

class TestObservationInterface(unittest.TestCase):
    def test_original_obs_space(self):
        """
        Check backward compatibility with the original observation space.
        """
        env = gym.make(
            "f1tenth_gym:f1tenth-v0", 
            config=EnvConfig(observation_config=ObservationConfig(type=ObservationType.ORIGINAL))
        )

        obs, _ = env.reset()

        obs_keys = [
            "ego_idx",
            "scans",
            "poses_x",
            "poses_y",
            "poses_theta",
            "linear_vels_x",
            "linear_vels_y",
            "ang_vels_z",
            "collisions",
            "lap_times",
            "lap_counts",
            "sim_time",
        ]

        self.assertTrue(
            all(
                [
                    isinstance(env.observation_space.spaces[k], Box)
                    for k in obs_keys
                    if k != "ego_idx"
                ]
            )
        )
        self.assertTrue(
            all(
                [
                    env.observation_space.spaces[k].dtype == np.float32
                    for k in obs_keys
                    if k != "ego_idx"
                ]
            )
        )

        self.assertTrue(isinstance(obs, dict))
        self.assertTrue(all([k in obs for k in obs_keys]))
        self.assertTrue(all([k in obs_keys for k in obs]))
        self.assertTrue(env.observation_space.contains(obs))
        env.close()

    def test_features_observation(self):
        """
        Check the FeatureObservation allows selecting an arbitrary subset of features.
        """
        features = (
            ObservationFeature.POSE_X,
            ObservationFeature.POSE_Y,
            ObservationFeature.POSE_THETA,
        )

        env = gym.make(
            "f1tenth_gym:f1tenth-v0", 
            config=EnvConfig(
                observation_config=ObservationConfig(type=ObservationType.FEATURES, features=features)
            )
        )

        self.assertTrue(isinstance(env.observation_space, gym.spaces.Dict))

        expected_keys = [feature.value for feature in features]
        for agent_id in env.unwrapped.agent_ids:
            space = env.observation_space.spaces[agent_id].spaces
            self.assertTrue(all(key in space for key in expected_keys))
            self.assertTrue(all(key in expected_keys for key in space))
            self.assertTrue(all(isinstance(space[key], Box) for key in expected_keys))
            self.assertTrue(all(space[key].dtype == np.float32 for key in expected_keys))

        obs, _ = env.reset()
        obs, _, _, _, _ = env.step(env.action_space.sample())

        sim_state = env.unwrapped.sim.state
        for i, agent_id in enumerate(env.unwrapped.agent_ids):
            pose_x, pose_y, pose_theta = sim_state.poses[i]
            obs_x, obs_y, obs_theta = (
                obs[agent_id][ObservationFeature.POSE_X.value],
                obs[agent_id][ObservationFeature.POSE_Y.value],
                obs[agent_id][ObservationFeature.POSE_THETA.value],
            )
            for ground_truth, observation in zip(
                [pose_x, pose_y, pose_theta], [obs_x, obs_y, obs_theta]
            ):
                self.assertTrue(np.allclose(ground_truth, observation))
        env.close()

    def test_nonexistent_obs_space(self):
        """
        Check that an error is raised when a nonexistent observation type is requested.
        """
        env = gym.make("f1tenth_gym:f1tenth-v0", config=EnvConfig())
        with self.assertRaises(TypeError):
            observation_factory(env, vehicle_id=0, type="nonexistent_obs_type")
        env.close()

    def test_kinematic_obs_space(self):
        """
        Check the kinematic state observation space contains the correct features [x, y, theta, v].
        """
        env = gym.make(
            "f1tenth_gym:f1tenth-v0", 
            config=EnvConfig(
                observation_config=ObservationConfig(type=ObservationType.KINEMATIC_STATE)
            )
        )

        expected_features = (
            ObservationFeature.POSE_X,
            ObservationFeature.POSE_Y,
            ObservationFeature.POSE_THETA,
            ObservationFeature.LINEAR_VEL_X,
            ObservationFeature.STEER_ANGLE,
        )
        expected_keys = [feature.value for feature in expected_features]

        for agent_id in env.unwrapped.agent_ids:
            space = env.observation_space.spaces[agent_id].spaces
            self.assertTrue(all(key in space for key in expected_keys))
            self.assertTrue(all(key in expected_keys for key in space))

        obs, _ = env.reset()
        obs, _, _, _, _ = env.step(env.action_space.sample())

        sim_state = env.unwrapped.sim.state
        for i, agent_id in enumerate(env.unwrapped.agent_ids):
            std_state = sim_state.standard_state[i]
            pose_x, pose_y, pose_theta = std_state[0], std_state[1], std_state[4]
            vel = std_state[3]
            beta = std_state[6]
            obs_x, obs_y, obs_theta = (
                obs[agent_id][ObservationFeature.POSE_X.value],
                obs[agent_id][ObservationFeature.POSE_Y.value],
                obs[agent_id][ObservationFeature.POSE_THETA.value],
            )
            obs_velx = obs[agent_id][ObservationFeature.LINEAR_VEL_X.value]

            ground_truth = np.asarray([pose_x, pose_y, pose_theta, vel * np.cos(beta)], dtype=np.float32)
            observed = np.asarray([obs_x, obs_y, obs_theta, obs_velx], dtype=np.float32)
            self.assertTrue(np.allclose(ground_truth, observed))
        env.close()

    def test_dynamic_obs_space(self):
        """
        Check the dynamic state observation space contains the correct features.
        """
        env = gym.make(
            "f1tenth_gym:f1tenth-v0", 
            config=EnvConfig(
                observation_config=ObservationConfig(type=ObservationType.DYNAMIC_STATE)
            )
        )

        expected_features = (
            ObservationFeature.POSE_X,
            ObservationFeature.POSE_Y,
            ObservationFeature.POSE_THETA,
            ObservationFeature.LINEAR_VEL_MAGNITUDE,
            ObservationFeature.ANGULAR_VEL_Z,
            ObservationFeature.STEER_ANGLE,
            ObservationFeature.SLIP_ANGLE,
        )
        expected_keys = [feature.value for feature in expected_features]

        for agent_id in env.unwrapped.agent_ids:
            space = env.observation_space.spaces[agent_id].spaces
            self.assertTrue(all(key in space for key in expected_keys))
            self.assertTrue(all(key in expected_keys for key in space))

        obs, _ = env.reset()
        obs, _, _, _, _ = env.step(env.action_space.sample())

        sim_state = env.unwrapped.sim.state
        for i, agent_id in enumerate(env.unwrapped.agent_ids):
            agent_obs = obs[agent_id]
            observed = np.array([
                agent_obs[ObservationFeature.POSE_X.value],
                agent_obs[ObservationFeature.POSE_Y.value],
                agent_obs[ObservationFeature.STEER_ANGLE.value],
                agent_obs[ObservationFeature.LINEAR_VEL_MAGNITUDE.value],
                agent_obs[ObservationFeature.POSE_THETA.value],
                agent_obs[ObservationFeature.ANGULAR_VEL_Z.value],
                agent_obs[ObservationFeature.SLIP_ANGLE.value],
            ])
            std_state = sim_state.standard_state[i]
            ground_truth = np.array([
                std_state[0],
                std_state[1],
                std_state[2],
                std_state[3],
                std_state[4],
                std_state[5],
                std_state[6],
            ], dtype=np.float32)

            self.assertTrue(np.allclose(ground_truth, observed))
        env.close()

    def test_consistency_observe_space(self):
        obs_types = [
            ObservationType.KINEMATIC_STATE,
            ObservationType.DYNAMIC_STATE,
            ObservationType.DIRECT,
            ObservationType.ORIGINAL,
        ]

        env = gym.make("f1tenth_gym:f1tenth-v0", config=EnvConfig())
        env.reset()

        for obs_type in obs_types:
            obs_impl = observation_factory(env, type=obs_type)
            space = obs_impl.space()
            observation = obs_impl.observe()

            self.assertTrue(
                space.contains(observation),
                f"Observation {obs_type} is not contained in its space",
            )
        env.close()

    def test_gymnasium_api(self):
        from gymnasium.utils.env_checker import check_env

        obs_types = [
            ObservationType.DIRECT,
            ObservationType.ORIGINAL,
            ObservationType.KINEMATIC_STATE,
            ObservationType.DYNAMIC_STATE,
        ]

        for obs_type in obs_types:
            env = gym.make(
                "f1tenth_gym:f1tenth-v0", 
                config=EnvConfig(observation_config=ObservationConfig(type=obs_type))
            )
            check_env(
                env.unwrapped,
                skip_render_check=True,
            )
            env.close()

