import unittest
from dataclasses import replace
from typing import Callable

import gymnasium as gym
import numpy as np

from f1tenth_gym.envs.action import LongitudinalActionType, SteerActionType
from f1tenth_gym.envs.env_config import EnvConfig
from f1tenth_gym.envs.observation import ObservationType
from f1tenth_gym.envs.reset import ResetStrategy


def with_params(cfg: EnvConfig, **updates: float) -> EnvConfig:
    return replace(cfg, params=cfg.params.with_updates(**updates))


def with_control_modes(
    cfg: EnvConfig,
    *,
    longitudinal: LongitudinalActionType,
    steering: SteerActionType,
) -> EnvConfig:
    return replace(
        cfg,
        control=replace(
            cfg.control,
            longitudinal_mode=longitudinal,
            steering_mode=steering,
        ),
    )


def with_num_agents(cfg: EnvConfig, num_agents: int) -> EnvConfig:
    return replace(cfg, num_agents=num_agents)


def with_observation(cfg: EnvConfig, obs_type: ObservationType) -> EnvConfig:
    return replace(cfg, observation=replace(cfg.observation, type=obs_type, features=None))


def with_reset_strategy(cfg: EnvConfig, strategy: ResetStrategy) -> EnvConfig:
    return replace(cfg, reset=replace(cfg.reset, strategy=strategy))


class TestEnvInterface(unittest.TestCase):
    @staticmethod
    def _make_env(modifier: Callable[[EnvConfig], EnvConfig] | None = None):
        cfg = EnvConfig()
        if modifier is not None:
            cfg = modifier(cfg)
        env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg)
        return env

    def test_gymnasium_api(self):
        from gymnasium.utils.env_checker import check_env

        env = self._make_env()
        check_env(env.unwrapped, skip_render_check=True)
        env.close()

    def test_configure_method(self):
        """
        Test that the configure method works as expected, and that the parameters are
        correctly updated in the simulator and agents.
        """

        def widen(cfg: EnvConfig) -> EnvConfig:
            return with_params(cfg, width=15.0)

        base_env = self._make_env()
        base_env.unwrapped.configure(config=widen(base_env.unwrapped.env_config))

        extended_env = self._make_env(modifier=widen)

        base_params = base_env.unwrapped.vehicle_params
        extended_params = extended_env.unwrapped.vehicle_params
        self.assertEqual(base_params, extended_params)

        np.testing.assert_allclose(
            base_env.unwrapped.sim.params_array,
            extended_env.unwrapped.sim.params_array,
        )

        obs0, _ = base_env.reset(options={"poses": np.array([[0.0, 0.0, np.pi / 2]])})
        obs1, _ = extended_env.reset(
            options={"poses": np.array([[0.0, 0.0, np.pi / 2]])}
        )
        done0 = done1 = False
        t = 0

        while not done0 and not done1:
            for agent in obs0:
                for observation in obs0[agent]:
                    if not np.allclose(obs0[agent][observation], obs1[agent][observation]):
                        print(
                            f"Observation {observation} should be the same for agent {agent}, "
                            f"got {obs0[agent][observation]} != {obs1[agent][observation]}"
                        )
                    self.assertTrue(
                        np.allclose(obs0[agent][observation], obs1[agent][observation]),
                        f"Observation {observation} should be the same for agent {agent}",
                    )
            self.assertTrue(done0 == done1, "Done should be the same")
            action = base_env.action_space.sample()
            obs0, _, done0, _, _ = base_env.step(action)
            obs1, _, done1, _, _ = extended_env.step(action)
            base_env.render()
            extended_env.render()
            t += 1

        print(f"Done after {t} steps")

        base_env.close()
        extended_env.close()

    def test_configure_action_space(self):
        """
        Try to change the upper bound of the action space, and check that the
        action space is correctly updated.
        """
        base_env = self._make_env()
        action_space_low = base_env.action_space.low
        action_space_high = base_env.action_space.high

        new_v_max = 5.0
        base_env.unwrapped.configure(
            config=with_params(base_env.unwrapped.env_config, v_max=new_v_max)
        )
        new_action_space_low = base_env.action_space.low
        new_action_space_high = base_env.action_space.high

        self.assertTrue(
            (action_space_low == new_action_space_low).all(),
            "Steering action space should be the same",
        )
        self.assertTrue(
            action_space_high[0][0] == new_action_space_high[0][0],
            "Steering action space should be the same",
        )
        self.assertTrue(
            new_action_space_high[0][1] == new_v_max,
            f"Speed action high should be {new_v_max}",
        )
        base_env.close()

    def test_acceleration_action_space(self):
        """
        Test that the acceleration action space is correctly configured.
        """
        env = self._make_env(
            modifier=lambda cfg: with_control_modes(
                cfg,
                longitudinal=LongitudinalActionType.ACCL,
                steering=SteerActionType.STEERING_SPEED,
            )
        )
        params = env.unwrapped.vehicle_params
        action_space_low = env.action_space.low
        action_space_high = env.action_space.high

        self.assertTrue(
            (action_space_low[0][0] - params.sv_min) < 1e-6,
            "lower sv does not match min steering velocity",
        )
        self.assertTrue(
            (action_space_high[0][0] - params.sv_max) < 1e-6,
            "upper sv does not match max steering velocity",
        )
        self.assertTrue(
            (action_space_low[0][1] + params.a_max) < 1e-6,
            "lower acceleration bound does not match a_min",
        )
        self.assertTrue(
            (action_space_high[0][1] - params.a_max) < 1e-6,
            "upper acceleration bound does not match a_max",
        )
        env.close()

    def test_manual_reset_options_in_synch_vec_env(self):
        """
        Test that the environment can be used in a vectorized environment.
        """
        num_envs, num_agents = 3, 2
        cfg = EnvConfig()
        cfg = with_num_agents(cfg, num_agents)
        cfg = with_observation(cfg, ObservationType.KINEMATIC_STATE)
        vec_env = gym.make_vec(
            "f1tenth_gym:f1tenth-v0", asynchronous=False, config=cfg, num_envs=num_envs
        )

        rnd_poses = np.random.random((2, 3))
        obss, infos = vec_env.reset(options={"poses": rnd_poses})

        for i, agent_id in enumerate(obss):
            for ie in range(num_envs):
                agent_obs = obss[agent_id]
                agent_pose = np.array(
                    [
                        agent_obs["pose_x"][ie],
                        agent_obs["pose_y"][ie],
                        agent_obs["pose_theta"][ie],
                    ]
                )
                self.assertTrue(
                    np.allclose(agent_pose, rnd_poses[i]),
                    f"pose of agent {agent_id} in env {ie} should be {rnd_poses[i]}, got {agent_pose}",
                )
        vec_env.close()

    def test_manual_reset_options_in_asynch_vec_env(self):
        """
        Test that the environment can be used in a vectorized environment.
        """
        num_envs, num_agents = 3, 2
        cfg = EnvConfig()
        cfg = with_num_agents(cfg, num_agents)
        cfg = with_observation(cfg, ObservationType.KINEMATIC_STATE)
        vec_env = gym.make_vec(
            "f1tenth_gym:f1tenth-v0", vectorization_mode="async", config=cfg, num_envs=num_envs
        )

        rnd_poses = np.random.random((2, 3))
        obss, infos = vec_env.reset(options={"poses": rnd_poses})

        for i, agent_id in enumerate(obss):
            for ie in range(num_envs):
                agent_obs = obss[agent_id]
                agent_pose = np.array(
                    [
                        agent_obs["pose_x"][ie],
                        agent_obs["pose_y"][ie],
                        agent_obs["pose_theta"][ie],
                    ]
                )
                self.assertTrue(
                    np.allclose(agent_pose, rnd_poses[i]),
                    f"pose of agent {agent_id} in env {ie} should be {rnd_poses[i]}, got {agent_pose}",
                )
        vec_env.close()

    def test_auto_reset_options_in_synch_vec_env(self):
        """
        Test that the environment can be used in a vectorized environment without explicit poses.
        """
        num_envs, num_agents = 3, 2
        cfg = EnvConfig()
        cfg = with_num_agents(cfg, num_agents)
        cfg = with_observation(cfg, ObservationType.KINEMATIC_STATE)
        cfg = with_reset_strategy(cfg, ResetStrategy.RL_RANDOM_RANDOM)
        vec_env = gym.make_vec(
            "f1tenth_gym:f1tenth-v0", vectorization_mode="sync", config=cfg, num_envs=num_envs,
        )

        obss, infos = vec_env.reset()

        for i, agent_id in enumerate(obss):
            agent_pose0 = np.array(
                [
                    obss[agent_id]["pose_x"][0],
                    obss[agent_id]["pose_y"][0],
                    obss[agent_id]["pose_theta"][0],
                ]
            )
            for ie in range(1, num_envs):
                agent_obs = obss[agent_id]
                agent_pose = np.array(
                    [
                        agent_obs["pose_x"][ie],
                        agent_obs["pose_y"][ie],
                        agent_obs["pose_theta"][ie],
                    ]
                )
                self.assertFalse(
                    np.allclose(agent_pose, agent_pose0),
                    f"pose of agent {agent_id} in env {ie} should be random, got same {agent_pose} == {agent_pose0}",
                )

        all_dones_once = [False] * num_envs
        all_dones_twice = [False] * num_envs

        max_steps = 1000
        while not all(all_dones_twice) and max_steps > 0:
            actions = vec_env.action_space.sample()
            obss, rewards, dones, truncations, infos = vec_env.step(actions)

            all_dones_once = [all_dones_once[i] or dones[i] for i in range(num_envs)]
            all_dones_twice = [
                all_dones_twice[i] or all_dones_once[i] for i in range(num_envs)
            ]
            max_steps -= 1

        vec_env.close()
        self.assertTrue(
            all(all_dones_twice),
            f"All envs should be done twice, got {all_dones_twice}",
        )

