from __future__ import annotations
from abc import abstractmethod
from typing import List, Optional

import gymnasium as gym
import numpy as np

from .observation_types import ObservationType
from ..simulator import SimpleSimulator




class Observation:
    """Base class for environment observations."""

    def __init__(self, env):
        self.env = env

    @abstractmethod
    def space(self):
        raise NotImplementedError()

    @abstractmethod
    def observe(self):
        raise NotImplementedError()

    @property
    def _sim(self) -> SimpleSimulator:
        return self.env.unwrapped.sim

    @property
    def _state(self):
        return self._sim.state


def _scan_space(sim: SimpleSimulator) -> gym.spaces.Box:
    beam_count = sim.scan_num_beams
    max_range = sim.scan_max_range if sim.scan_enabled else 1.0
    shape = (beam_count,) if beam_count > 0 else (0,)
    low = 0.0 if beam_count > 0 else np.array([], dtype=np.float32)
    high = max_range if beam_count > 0 else np.array([], dtype=np.float32)
    return gym.spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)


class DirectObservation(Observation):
    def space(self):
        sim = self._sim
        state_dim = sim.state_dim
        scan_space = _scan_space(sim)
        large_num = 1e30

        complete_space = {}
        for agent_id in self.env.unwrapped.agent_ids:
            agent_dict = {
                "scan": scan_space,
                "std_state": gym.spaces.Box(
                    low=-large_num, high=large_num, shape=(7,), dtype=np.float32
                ),
                "state": gym.spaces.Box(
                    low=-large_num, high=large_num, shape=(state_dim,), dtype=np.float32
                ),
                "collision": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(), dtype=np.float32
                ),
                "lap_time": gym.spaces.Box(
                    low=0.0, high=large_num, shape=(), dtype=np.float32
                ),
                "lap_count": gym.spaces.Box(
                    low=0.0, high=large_num, shape=(), dtype=np.float32
                ),
                "sim_time": gym.spaces.Box(
                    low=0.0, high=large_num, shape=(), dtype=np.float32
                ),
            }
            if self.env.unwrapped.config["compute_frenet"]:
                agent_dict["frenet_pose"] = gym.spaces.Box(
                    low=-large_num, high=large_num, shape=(3,), dtype=np.float32
                )
            complete_space[agent_id] = gym.spaces.Dict(agent_dict)
        return gym.spaces.Dict(complete_space)

    def observe(self):
        sim = self._sim
        state = self._state
        beam_count = sim.scan_num_beams
        obs = {}
        for idx, agent_id in enumerate(self.env.unwrapped.agent_ids):
            scan = state.scans[idx, :beam_count] if beam_count > 0 else np.empty((0,), dtype=np.float32)
            agent_obs = {
                "scan": scan.astype(np.float32),
                "std_state": state.standard_state[idx].astype(np.float32),
                "state": state.state[idx].astype(np.float32),
                "collision": np.array(state.collisions[idx], dtype=np.float32),
                "lap_time": np.array(self.env.unwrapped.lap_times[idx], dtype=np.float32),
                "lap_count": np.array(self.env.unwrapped.lap_counts[idx], dtype=np.float32),
                "sim_time": np.array(self.env.unwrapped.sim_time, dtype=np.float32),
            }
            if self.env.unwrapped.config["compute_frenet"]:
                agent_obs["frenet_pose"] = state.frenet[idx].astype(np.float32)
            obs[agent_id] = agent_obs
        return obs


class OriginalObservation(Observation):
    def space(self):
        sim = self._sim
        num_agents = self.env.unwrapped.num_agents
        scan_space = _scan_space(sim)
        large_num = 1e30
        scan_high = sim.scan_max_range + 0.5 if sim.scan_enabled else 1.0
        obs_space = gym.spaces.Dict(
            {
                "ego_idx": gym.spaces.Discrete(num_agents),
                "scans": gym.spaces.Box(
                    low=0.0,
                    high=scan_high,
                    shape=(num_agents, scan_space.shape[0]),
                    dtype=np.float32,
                ),
                "poses_x": gym.spaces.Box(
                    low=-large_num,
                    high=large_num,
                    shape=(num_agents,),
                    dtype=np.float32,
                ),
                "poses_y": gym.spaces.Box(
                    low=-large_num,
                    high=large_num,
                    shape=(num_agents,),
                    dtype=np.float32,
                ),
                "poses_theta": gym.spaces.Box(
                    low=-large_num,
                    high=large_num,
                    shape=(num_agents,),
                    dtype=np.float32,
                ),
                "linear_vels_x": gym.spaces.Box(
                    low=-large_num,
                    high=large_num,
                    shape=(num_agents,),
                    dtype=np.float32,
                ),
                "linear_vels_y": gym.spaces.Box(
                    low=-large_num,
                    high=large_num,
                    shape=(num_agents,),
                    dtype=np.float32,
                ),
                "ang_vels_z": gym.spaces.Box(
                    low=-large_num,
                    high=large_num,
                    shape=(num_agents,),
                    dtype=np.float32,
                ),
                "collisions": gym.spaces.Box(
                    low=0.0, high=1.0, shape=(num_agents,), dtype=np.float32
                ),
                "lap_times": gym.spaces.Box(
                    low=0.0, high=large_num, shape=(num_agents,), dtype=np.float32
                ),
                "lap_counts": gym.spaces.Box(
                    low=0.0, high=large_num, shape=(num_agents,), dtype=np.float32
                ),
                "sim_time": gym.spaces.Box(
                    low=0.0, high=large_num, shape=(), dtype=np.float32
                ),
            }
        )
        return obs_space

    def observe(self):
        sim = self._sim
        state = self._state
        beam_count = sim.scan_num_beams
        num_agents = self.env.unwrapped.num_agents
        std_state = state.standard_state
        speed = std_state[:, 3]
        beta = std_state[:, 6]
        vx = speed * np.cos(beta)
        vy = speed * np.sin(beta)

        scans = (
            state.scans[:, :beam_count].astype(np.float32)
            if beam_count > 0
            else np.zeros((num_agents, 0), dtype=np.float32)
        )

        observations = {
            "ego_idx": getattr(self._sim, "ego_idx", self.env.unwrapped.ego_idx),
            "scans": scans,
            "poses_x": std_state[:, 0].astype(np.float32),
            "poses_y": std_state[:, 1].astype(np.float32),
            "poses_theta": std_state[:, 4].astype(np.float32),
            "linear_vels_x": vx.astype(np.float32),
            "linear_vels_y": vy.astype(np.float32),
            "ang_vels_z": std_state[:, 5].astype(np.float32),
            "collisions": state.collisions.astype(np.float32),
            "lap_times": self.env.unwrapped.lap_times.astype(np.float32),
            "lap_counts": self.env.unwrapped.lap_counts.astype(np.float32),
            "sim_time": np.array(self.env.unwrapped.sim_time, dtype=np.float32),
        }

        return observations


class FeaturesObservation(Observation):
    def __init__(self, env, features: List[str]):
        super().__init__(env)
        self.features = features

    def space(self):
        sim = self._sim
        scan_space = _scan_space(sim)
        large_num = 1e30
        feature_spaces = {
            "scan": scan_space,
            "pose_x": gym.spaces.Box(low=-large_num, high=large_num, shape=(), dtype=np.float32),
            "pose_y": gym.spaces.Box(low=-large_num, high=large_num, shape=(), dtype=np.float32),
            "pose_theta": gym.spaces.Box(low=-large_num, high=large_num, shape=(), dtype=np.float32),
            "linear_vel_x": gym.spaces.Box(low=-large_num, high=large_num, shape=(), dtype=np.float32),
            "linear_vel_y": gym.spaces.Box(low=-large_num, high=large_num, shape=(), dtype=np.float32),
            "linear_vel_magnitude": gym.spaces.Box(low=-large_num, high=large_num, shape=(), dtype=np.float32),
            "ang_vel_z": gym.spaces.Box(low=-large_num, high=large_num, shape=(), dtype=np.float32),
            "delta": gym.spaces.Box(low=-large_num, high=large_num, shape=(), dtype=np.float32),
            "beta": gym.spaces.Box(low=-large_num, high=large_num, shape=(), dtype=np.float32),
            "collision": gym.spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32),
            "lap_time": gym.spaces.Box(low=0.0, high=large_num, shape=(), dtype=np.float32),
            "lap_count": gym.spaces.Box(low=0.0, high=large_num, shape=(), dtype=np.float32),
        }
        return gym.spaces.Dict(
            {
                agent_id: gym.spaces.Dict({k: feature_spaces[k] for k in self.features})
                for agent_id in self.env.unwrapped.agent_ids
            }
        )

    def observe(self):
        sim = self._sim
        state = self._state
        beam_count = sim.scan_num_beams
        obs = {}
        for idx, agent_id in enumerate(self.env.unwrapped.agent_ids):
            std_state = state.standard_state[idx]
            speed = std_state[3]
            beta = std_state[6]
            vx = speed * np.cos(beta)
            vy = speed * np.sin(beta)
            agent_obs = {
                "scan": state.scans[idx, :beam_count].astype(np.float32) if beam_count > 0 else np.empty((0,), dtype=np.float32),
                "pose_x": np.array(std_state[0], dtype=np.float32),
                "pose_y": np.array(std_state[1], dtype=np.float32),
                "pose_theta": np.array(std_state[4], dtype=np.float32),
                "linear_vel_magnitude": np.array(speed, dtype=np.float32),
                "linear_vel_x": np.array(vx, dtype=np.float32),
                "linear_vel_y": np.array(vy, dtype=np.float32),
                "ang_vel_z": np.array(std_state[5], dtype=np.float32),
                "delta": np.array(std_state[2], dtype=np.float32),
                "beta": np.array(beta, dtype=np.float32),
                "collision": np.array(state.collisions[idx], dtype=np.float32),
                "lap_time": np.array(self.env.unwrapped.lap_times[idx], dtype=np.float32),
                "lap_count": np.array(self.env.unwrapped.lap_counts[idx], dtype=np.float32),
            }
            obs[agent_id] = {k: agent_obs[k] for k in self.features}
        return obs


def observation_factory(env, type: Optional[ObservationType] = None, **kwargs) -> Observation:
    obs_type = type or ObservationType.ORIGINAL
    if not isinstance(obs_type, ObservationType):
        raise TypeError("observation_factory 'type' must be an ObservationType")
    if obs_type is ObservationType.ORIGINAL:
        return OriginalObservation(env)
    if obs_type is ObservationType.DIRECT:
        return DirectObservation(env)
    if obs_type is ObservationType.FEATURES:
        raw_features = kwargs.get("features", ())
        feature_list = [str(item) for item in raw_features]
        return FeaturesObservation(env, features=feature_list)
    if obs_type is ObservationType.KINEMATIC_STATE:
        features = ["pose_x", "pose_y", "delta", "linear_vel_x", "pose_theta"]
        return FeaturesObservation(env, features=features)
    if obs_type is ObservationType.DYNAMIC_STATE:
        features = [
            "pose_x",
            "pose_y",
            "delta",
            "linear_vel_magnitude",
            "pose_theta",
            "ang_vel_z",
            "beta",
        ]
        return FeaturesObservation(env, features=features)
    if obs_type is ObservationType.FRENET_DYNAMIC_STATE:
        features = [
            "pose_x",
            "pose_y",
            "delta",
            "linear_vel_x",
            "linear_vel_y",
            "pose_theta",
            "ang_vel_z",
            "beta",
        ]
        return FeaturesObservation(env, features=features)
    raise ValueError(f"Invalid observation type {obs_type}.")

