from __future__ import annotations

import gymnasium as gym
import numpy as np

from .base import Observation, scan_space

__all__ = ["DirectObservation"]

class DirectObservation(Observation):
    def space(self) -> gym.Space:
        sim = self._sim
        state_dim = sim.state_dim
        lidar_space = scan_space(sim)
        large_num = 1e30

        complete_space: dict[str, gym.Space] = {}
        for agent_id in self.env.unwrapped.agent_ids:
            agent_dict: dict[str, gym.Space] = {
                "scan": lidar_space,
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
            if self.env.unwrapped.compute_frenet:
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
            if self.env.unwrapped.compute_frenet:
                agent_obs["frenet_pose"] = state.frenet[idx].astype(np.float32)
            obs[agent_id] = agent_obs
        return obs
