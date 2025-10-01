from __future__ import annotations

import gymnasium as gym
import numpy as np

from .base import Observation, scan_space

__all__ = ["OriginalObservation"]


class OriginalObservation(Observation):
    def space(self) -> gym.Space:
        sim = self._sim
        num_agents = self.env.unwrapped.num_agents
        lidar_space = scan_space(sim)
        large_num = 1e30
        scan_high = sim.scan_max_range + 0.5 if sim.scan_enabled else 1.0
        obs_space = gym.spaces.Dict(
            {
                "ego_idx": gym.spaces.Discrete(num_agents),
                "scans": gym.spaces.Box(
                    low=0.0,
                    high=scan_high,
                    shape=(num_agents, lidar_space.shape[0]),
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
