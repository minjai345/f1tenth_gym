"""Gymnasium wrappers offering clean interfaces onto the multi-agent F110Env.

Thin adapters: the native env keeps its multi-agent
``dict[agent -> dict[field -> value]]`` observation.
"""
from __future__ import annotations

import copy
from collections import deque

import gymnasium as gym
import numpy as np

__all__ = ["SingleAgentWrapper", "ObservationDelayWrapper"]


class SingleAgentWrapper(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Flat single-agent interface over a 1-agent F110Env.

    Unwraps ``obs["agent_0"]`` and reshapes the action from ``(1, 2)`` to ``(2,)``.
    Compose with ``gymnasium.wrappers.FlattenObservation`` for a flat ``Box``.
    Requires ``num_agents == 1``; ``info`` passes through unchanged.
    """

    def __init__(self, env: gym.Env):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)
        num_agents = env.unwrapped.num_agents
        if num_agents != 1:
            raise ValueError(
                f"SingleAgentWrapper requires num_agents == 1, got {num_agents}"
            )
        self._agent_id = env.unwrapped.agent_ids[0]
        self.observation_space = env.observation_space[self._agent_id]
        act = env.action_space  # Box(shape=(1, 2))
        self.action_space = gym.spaces.Box(
            low=np.asarray(act.low[0]), high=np.asarray(act.high[0]), dtype=act.dtype
        )

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return obs[self._agent_id], info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(1, -1)
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs[self._agent_id], reward, terminated, truncated, info


class ObservationDelayWrapper(gym.Wrapper, gym.utils.RecordConstructorArgs):
    """Delay the observation by a fixed number of steps (sensor/perception lag).

    ``step`` returns the observation from ``delay_steps`` steps ago while reward,
    termination and ``info`` stay current; the reset observation is repeated until
    enough history exists. Works on any observation shape, space unchanged.
    """

    def __init__(self, env: gym.Env, delay_steps: int = 1):
        gym.utils.RecordConstructorArgs.__init__(self, delay_steps=delay_steps)
        gym.Wrapper.__init__(self, env)
        if delay_steps < 0:
            raise ValueError(f"delay_steps must be >= 0, got {delay_steps}")
        self._delay_steps = int(delay_steps)
        # holds delay_steps+1 frames; the leftmost is delay_steps steps old
        self._buffer: deque = deque(maxlen=self._delay_steps + 1)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._buffer.clear()
        for _ in range(self._delay_steps + 1):
            self._buffer.append(copy.deepcopy(obs))
        return copy.deepcopy(self._buffer[0]), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._buffer.append(copy.deepcopy(obs))
        return copy.deepcopy(self._buffer[0]), reward, terminated, truncated, info
