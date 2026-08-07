from __future__ import annotations

import gymnasium as gym
import numpy as np

from .base import Observation, scan_space
from .full import FullObservation

__all__ = ["RawObservation"]


def _batched(space: gym.spaces.Box, num_agents: int) -> gym.spaces.Box:
    """A per-agent Box tiled into an agent-batched Box."""
    shape = (num_agents, *space.shape)
    low = np.broadcast_to(space.low, shape).astype(np.float32)
    high = np.broadcast_to(space.high, shape).astype(np.float32)
    return gym.spaces.Box(low=low, high=high, dtype=np.float32)


class RawObservation(Observation):
    """Raw agent-batched arrays straight off the simulator's SoA state.

    ``observe()`` returns a flat dict of arrays batched over agents::

        state          (N, state_dim)   model-native state rows
        standard_state (N, 7)           CoG-anchored standardized rows
        scans          (N, num_beams)   only when the LiDAR is enabled
        collisions     (N,)
        frenet         (N, 3)           only when the Frenet frame is computed
        lap_times      (N,)
        lap_counts     (N,)
        sim_time       ()

    Every array is a **copy** (``.astype(np.float32)``) — the SoA buffers are
    overwritten in place each step, so returning live views would silently
    corrupt anything stored (RL replay buffers most of all).
    """

    def _bounds_provider(self) -> FullObservation:
        # reuse FullObservation's physically-derived per-field bounds rather
        # than duplicating the math here
        return FullObservation(self.env)

    def _selected_keys(self) -> tuple[str, ...]:
        keys = ["state", "standard_state", "collisions", "lap_times", "lap_counts", "sim_time"]
        if self._sim.scan_enabled:
            keys.insert(0, "scans")
        if self.env.unwrapped.compute_frenet:
            keys.append("frenet")
        return tuple(keys)

    def space(self) -> gym.Space:
        sim = self._sim
        n = sim.num_agents
        full = self._bounds_provider()
        bounds = full._physical_bounds()
        per_key: dict[str, gym.Space] = {}
        for key in self._selected_keys():
            if key == "scans":
                per_key[key] = _batched(scan_space(sim), n)
            elif key == "state":
                per_key[key] = _batched(full._field_space("state", sim, bounds), n)
            elif key == "standard_state":
                per_key[key] = _batched(full._field_space("std_state", sim, bounds), n)
            elif key == "collisions":
                per_key[key] = gym.spaces.Box(low=0.0, high=1.0, shape=(n,), dtype=np.float32)
            elif key == "frenet":
                per_key[key] = _batched(full._field_space("frenet_pose", sim, bounds), n)
            elif key in ("lap_times", "lap_counts"):
                per_key[key] = _batched(full._field_space("lap_time", sim, bounds), n)
            elif key == "sim_time":
                per_key[key] = full._field_space("sim_time", sim, bounds)
        return gym.spaces.Dict(per_key)

    def observe(self):
        sim = self._sim
        state = self._state
        env = self.env.unwrapped
        out: dict[str, np.ndarray] = {}
        for key in self._selected_keys():
            if key == "scans":
                out[key] = state.scans[:, : sim.scan_num_beams].astype(np.float32)
            elif key == "state":
                out[key] = state.state.astype(np.float32)
            elif key == "standard_state":
                out[key] = state.standard_state.astype(np.float32)
            elif key == "collisions":
                out[key] = state.collisions.astype(np.float32)
            elif key == "frenet":
                out[key] = state.frenet.astype(np.float32)
            elif key == "lap_times":
                out[key] = np.asarray(env.lap_times, dtype=np.float32).copy()
            elif key == "lap_counts":
                out[key] = np.asarray(env.lap_counts, dtype=np.float32).copy()
            elif key == "sim_time":
                out[key] = np.asarray(env.sim_time, dtype=np.float32)
        return out
