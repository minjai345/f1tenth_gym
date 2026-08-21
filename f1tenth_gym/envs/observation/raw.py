from __future__ import annotations

from typing import Callable, NamedTuple

import gymnasium as gym
import numpy as np

from .base import Observation, scan_space
from .full import field_space, physical_bounds

__all__ = ["RawObservation"]


def _batched(space: gym.spaces.Box, num_agents: int) -> gym.spaces.Box:
    """A per-agent Box tiled into an agent-batched Box."""
    shape = (num_agents, *space.shape)
    low = np.broadcast_to(space.low, shape).astype(np.float32)
    high = np.broadcast_to(space.high, shape).astype(np.float32)
    return gym.spaces.Box(low=low, high=high, dtype=np.float32)


class _Key(NamedTuple):
    """One raw output key: how to size it, and how to read it.

    Space builder and value getter live side by side so a new key cannot reach
    ``observation_space`` without also reaching ``observe()``.
    """

    space: Callable[..., gym.Space]
    value: Callable[..., np.ndarray]


# Batched per-agent fields, borrowing FullObservation's physically-derived
# bounds rather than duplicating the math. `sim_time` is the one scalar.
_KEYS: dict[str, _Key] = {
    "scans": _Key(
        space=lambda sim, n, bounds: _batched(scan_space(sim), n),
        value=lambda sim, state, env: state.scans[:, : sim.scan_num_beams].astype(np.float32),
    ),
    "state": _Key(
        space=lambda sim, n, bounds: _batched(field_space("state", sim, bounds), n),
        value=lambda sim, state, env: state.state.astype(np.float32),
    ),
    "standard_state": _Key(
        space=lambda sim, n, bounds: _batched(field_space("std_state", sim, bounds), n),
        value=lambda sim, state, env: state.standard_state.astype(np.float32),
    ),
    "collisions": _Key(
        space=lambda sim, n, bounds: gym.spaces.Box(
            low=0.0, high=1.0, shape=(n,), dtype=np.float32
        ),
        value=lambda sim, state, env: state.collisions.astype(np.float32),
    ),
    "frenet": _Key(
        space=lambda sim, n, bounds: _batched(field_space("frenet_pose", sim, bounds), n),
        value=lambda sim, state, env: state.frenet.astype(np.float32),
    ),
    "lap_times": _Key(
        space=lambda sim, n, bounds: _batched(field_space("lap_time", sim, bounds), n),
        value=lambda sim, state, env: np.asarray(env.lap_times, dtype=np.float32).copy(),
    ),
    "lap_counts": _Key(
        space=lambda sim, n, bounds: _batched(field_space("lap_time", sim, bounds), n),
        value=lambda sim, state, env: np.asarray(env.lap_counts, dtype=np.float32).copy(),
    ),
    "sim_time": _Key(
        space=lambda sim, n, bounds: field_space("sim_time", sim, bounds),
        value=lambda sim, state, env: np.asarray(env.sim_time, dtype=np.float32),
    ),
}


class RawObservation(Observation):
    """Raw agent-batched arrays straight off the simulator's SoA state.

    Every array is a copy: the SoA buffers are overwritten in place each step, so
    live views would corrupt anything stored.
    """

    def _selected_keys(self) -> tuple[str, ...]:
        keys = ["state", "standard_state", "collisions", "lap_times", "lap_counts", "sim_time"]
        if self._sim.scan_enabled:
            keys.insert(0, "scans")
        if self.env.unwrapped.compute_frenet:
            keys.append("frenet")
        return tuple(keys)

    def _handler(self, key: str) -> _Key:
        try:
            return _KEYS[key]
        except KeyError:
            raise ValueError(f"no handler for raw observation key {key!r}") from None

    def space(self) -> gym.Space:
        sim = self._sim
        bounds = physical_bounds(self.env)
        return gym.spaces.Dict(
            {
                key: self._handler(key).space(sim, sim.num_agents, bounds)
                for key in self._selected_keys()
            }
        )

    def observe(self):
        """Read the selected raw arrays off the SoA state.

        Returns:
            A flat dict of arrays batched over agents::

                state          (N, state_dim)   model-native state rows
                standard_state (N, 7)           CoG-anchored standardized rows
                scans          (N, num_beams)   only when the LiDAR is enabled
                collisions     (N,)
                frenet         (N, 3)           only when the Frenet frame is on
                lap_times      (N,)
                lap_counts     (N,)
                sim_time       ()
        """
        sim = self._sim
        state = self._state
        env = self.env.unwrapped
        return {
            key: self._handler(key).value(sim, state, env)
            for key in self._selected_keys()
        }
