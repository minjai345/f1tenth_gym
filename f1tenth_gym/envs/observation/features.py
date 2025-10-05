from __future__ import annotations

import gymnasium as gym
import numpy as np

from . import ObservationFeature
from .base import Observation, scan_space

__all__ = ["FeaturesObservation"]


def _scalar_box(low: float, high: float) -> gym.Space:
    return gym.spaces.Box(low=low, high=high, shape=(), dtype=np.float32)


_FEATURE_SPACE_BUILDERS: dict[ObservationFeature, Callable[[object, float], gym.Space]] = {
    ObservationFeature.SCAN: lambda sim, limit: scan_space(sim),
    ObservationFeature.POSE_X: lambda sim, limit: _scalar_box(-limit, limit),
    ObservationFeature.POSE_Y: lambda sim, limit: _scalar_box(-limit, limit),
    ObservationFeature.POSE_THETA: lambda sim, limit: _scalar_box(-limit, limit),
    ObservationFeature.LINEAR_VEL_X: lambda sim, limit: _scalar_box(-limit, limit),
    ObservationFeature.LINEAR_VEL_Y: lambda sim, limit: _scalar_box(-limit, limit),
    ObservationFeature.LINEAR_VEL_MAGNITUDE: lambda sim, limit: _scalar_box(-limit, limit),
    ObservationFeature.ANGULAR_VEL_Z: lambda sim, limit: _scalar_box(-limit, limit),
    ObservationFeature.STEER_ANGLE: lambda sim, limit: _scalar_box(-limit, limit),
    ObservationFeature.SLIP_ANGLE: lambda sim, limit: _scalar_box(-limit, limit),
    ObservationFeature.COLLISION: lambda sim, limit: _scalar_box(0.0, 1.0),
    ObservationFeature.LAP_TIME: lambda sim, limit: _scalar_box(0.0, limit),
    ObservationFeature.LAP_COUNT: lambda sim, limit: _scalar_box(0.0, limit),
}


class FeaturesObservation(Observation):
    def __init__(self, env, features: tuple[ObservationFeature, ...]):
        super().__init__(env)
        self.features = features

    def space(self) -> gym.Space:
        sim = self._sim
        large_num = 1e30
        spaces = {}
        for agent_id in self.env.unwrapped.agent_ids:
            agent_spaces = {
                feature.value: _FEATURE_SPACE_BUILDERS[feature](sim, large_num)
                for feature in self.features
            }
            spaces[agent_id] = gym.spaces.Dict(agent_spaces)
        return gym.spaces.Dict(spaces)

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
            feature_values = {
                ObservationFeature.SCAN: state.scans[idx, :beam_count]
                if beam_count > 0
                else np.empty((0,), dtype=np.float32),
                ObservationFeature.POSE_X: std_state[0],
                ObservationFeature.POSE_Y: std_state[1],
                ObservationFeature.POSE_THETA: std_state[4],
                ObservationFeature.LINEAR_VEL_MAGNITUDE: speed,
                ObservationFeature.LINEAR_VEL_X: vx,
                ObservationFeature.LINEAR_VEL_Y: vy,
                ObservationFeature.ANGULAR_VEL_Z: std_state[5],
                ObservationFeature.STEER_ANGLE: std_state[2],
                ObservationFeature.SLIP_ANGLE: beta,
                ObservationFeature.COLLISION: state.collisions[idx],
                ObservationFeature.LAP_TIME: self.env.unwrapped.lap_times[idx],
                ObservationFeature.LAP_COUNT: self.env.unwrapped.lap_counts[idx],
            }
            agent_obs = {}
            for feature in self.features:
                value = feature_values[feature]
                if feature is ObservationFeature.SCAN:
                    agent_obs[feature.value] = value.astype(np.float32)
                else:
                    agent_obs[feature.value] = np.array(value, dtype=np.float32)
            obs[agent_id] = agent_obs
        return obs
