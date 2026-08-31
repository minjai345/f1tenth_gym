from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import gymnasium as gym
import numpy as np

if TYPE_CHECKING:
    from ..simulator import F110Simulator

__all__ = ["Observation", "scan_space", "scan_space_from"]


class Observation(ABC):
    """Base class for environment observations."""

    def __init__(self, env):
        self.env = env

    @abstractmethod
    def space(self) -> gym.Space:
        """Return the Gymnasium space describing this observation."""

    @abstractmethod
    def observe(self):
        """Compute the current observation."""

    @property
    def _sim(self) -> "F110Simulator":
        return self.env.unwrapped.sim

    @property
    def _state(self):
        return self._sim.state


def scan_space_from(
    enabled: bool,
    num_beams: int,
    range_max: float,
) -> gym.spaces.Box:
    """Build a scan space from explicit LiDAR components.

    Keeping this helper independent of a simulator lets other environment
    adapters reuse the native observation contract without constructing the
    mutable simulator.  Disabled LiDAR has the same empty observation shape as
    :func:`scan_space`, regardless of its configured beam count.
    """

    beam_count = num_beams if enabled else 0
    max_range = range_max if enabled else 1.0
    shape = (beam_count,) if beam_count > 0 else (0,)
    low = 0.0 if beam_count > 0 else np.array([], dtype=np.float32)
    high = max_range if beam_count > 0 else np.array([], dtype=np.float32)
    return gym.spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)


def scan_space(sim: "F110Simulator") -> gym.spaces.Box:
    """Build a scan space that adapts to the active LiDAR configuration."""

    return scan_space_from(
        enabled=sim.scan_enabled,
        num_beams=sim.scan_num_beams,
        range_max=sim.scan_max_range,
    )
