from __future__ import annotations

import gymnasium as gym
import numpy as np

from . import ObservationFeature
from .base import Observation, scan_space

__all__ = ["FeaturesObservation"]


_FEATURE_SPACE_BUILDERS: dict[ObservationFeature, callable] = {}


def _register(feature: ObservationFeature):
    def decorator(builder):
        _FEATURE_SPACE_BUILDERS[feature] = builder
        return builder

    return decorator


@_register(ObservationFeature.SCAN)
def _(sim, large_num):
    return scan_space(sim)


@_register(ObservationFeature.POSE_X)
def _(sim, large_num):
    return gym.spaces.Box(low=-large_num, high=large_num, shape=(), dtype=np.float32)


@_register(ObservationFeature.POSE_Y)
def _(sim, large_num):
    return gym.spaces.Box(low=-large_num, high=large_num, shape=(), dtype=np.float32)


@_register(ObservationFeature.POSE_THETA)
def _(sim, large_num):
    return gym.spaces.Box(low=-large_num, high=large_num, shape=(), dtype=np.float32)


@_register(ObservationFeature.LINEAR_VEL_X)
def _(sim, large_num):
    return gym.spaces.Box(low=-large_num, high=large_num, shape=(), dtype=np.float32)


@_register(ObservationFeature.LINEAR_VEL_Y)
def _(sim, large_num):
    return gym.spaces.Box(low=-large_num, high=large_num, shape=(), dtype=np.float32)


@_register(ObservationFeature.LINEAR_VEL_MAGNITUDE)
def _(sim, large_num):
    return gym.spaces.Box(low=-large_num, high=large_num, shape=(), dtype=np.float32)


@_register(ObservationFeature.ANGULAR_VEL_Z)
def _(sim, large_num):
    return gym.spaces.Box(low=-large_num, high=large_num, shape=(), dtype=np.float32)


@_register(ObservationFeature.STEER_ANGLE)
def _(sim, large_num):
    return gym.spaces.Box(low=-large_num, high=large_num, shape=(), dtype=np.float32)


@_register(ObservationFeature.SLIP_ANGLE)
def _(sim, large_num):
    return gym.spaces.Box(low=-large_num, high=large_num, shape=(), dtype=np.float32)


@_register(ObservationFeature.COLLISION)
def _(sim, large_num):
    return gym.spaces.Box(low=0.0, high=1.0, shape=(), dtype=np.float32)


@_register(ObservationFeature.LAP_TIME)
def _(sim, large_num):
    return gym.spaces.Box(low=0.0, high=large_num, shape=(), dtype=np.float32)


@_register(ObservationFeature.LAP_COUNT)
def _(sim, large_num):
    return gym.spaces.Box(low=0.0, high=large_num, shape=(), dtype=np.float32)


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
            agent_obs = {
                feature.value: np.array(feature_values[feature], dtype=np.float32)
                if feature is not ObservationFeature.SCAN
                else feature_values[feature].astype(np.float32)
                for feature in self.features
            }
            obs[agent_id] = agent_obs
        return obs
