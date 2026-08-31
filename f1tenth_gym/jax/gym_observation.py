"""Host-only Gymnasium observation packaging for the functional JAX core.

This module is intentionally a deep import.  It owns device-to-host transfer,
NumPy copies, public field selection, and Gymnasium spaces; none of those
concerns belong in the pure :mod:`f1tenth_gym.jax` transition.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import warnings

import gymnasium as gym
import jax
import numpy as np

from f1tenth_gym.envs.dynamic_models import DynamicModel
from f1tenth_gym.envs.env_config import EnvConfig, ObservationConfig
from f1tenth_gym.envs.observation import (
    FEATURE_PRESETS,
    ObservationType,
)
from f1tenth_gym.envs.observation.base import scan_space_from
from f1tenth_gym.envs.observation.fields import (
    ALL_FIELDS,
    BASE_FIELDS,
    DERIVED_FIELDS,
)
from f1tenth_gym.envs.observation.full import (
    field_space_from,
    physical_bounds_from,
)
from f1tenth_gym.envs.track import Track

from .builder import CoreBundle
from .environment import CoreConfig, CoreObservation


_ALL_FIELDS = frozenset(ALL_FIELDS)
_DERIVED_FIELDS = frozenset(DERIVED_FIELDS)
_BASE_DEPENDENCIES = {
    "scan": "scans",
    "std_state": "standard_state",
    "state": "state",
    "collision": "collisions",
    "lap_time": "lap_times",
    "lap_count": "lap_counts",
    "sim_time": "sim_time",
    "frenet_pose": "frenet",
}
_RAW_BASE_KEYS = (
    "state",
    "standard_state",
    "collisions",
    "lap_times",
    "lap_counts",
    "sim_time",
)


def _copy_float32(value) -> np.ndarray:
    """Return one independent public float32 leaf, including 0-d scalars."""
    return np.array(value, dtype=np.float32, copy=True)


def _batched(space: gym.spaces.Box, num_agents: int) -> gym.spaces.Box:
    shape = (num_agents, *space.shape)
    low = np.broadcast_to(space.low, shape).astype(np.float32)
    high = np.broadcast_to(space.high, shape).astype(np.float32)
    return gym.spaces.Box(low=low, high=high, dtype=np.float32)


def _resolve_fields(
    config: EnvConfig,
    *,
    scan_enabled: bool,
    frenet_enabled: bool,
) -> tuple[str, ...] | None:
    observation = config.observation_config
    selected = observation.type
    if not isinstance(selected, ObservationType):
        raise TypeError("observation type must be an ObservationType")

    if selected is ObservationType.DIRECT:
        warnings.warn(
            "ObservationType.DIRECT changed meaning in v1.0.0: it now returns "
            "raw agent-batched arrays. Use ObservationType.DEFAULT for the "
            "packaged per-agent dict.",
            stacklevel=3,
        )
        return None
    if selected is ObservationType.DEFAULT:
        fields = BASE_FIELDS
        if not scan_enabled:
            fields = tuple(field for field in fields if field != "scan")
        if not frenet_enabled:
            fields = tuple(field for field in fields if field != "frenet_pose")
        return fields
    if selected is ObservationType.FEATURES:
        if observation.features is None:
            raise ValueError("FullObservation requires 'features' to be specified")
        fields = tuple(observation.features)
        if not fields:
            raise ValueError("FullObservation requires at least one feature")
        unknown = next((field for field in fields if field not in _ALL_FIELDS), None)
        if unknown is not None:
            raise ValueError(f"Unknown observation feature: {unknown!r}")
    elif selected in FEATURE_PRESETS:
        fields = FEATURE_PRESETS[selected]
    else:
        raise ValueError(f"Unsupported observation type: {selected}")

    if "frenet_pose" in fields and not frenet_enabled:
        raise ValueError(
            "frenet_pose requested but environment does not compute the Frenet frame"
        )
    if "scan" in fields and not scan_enabled:
        raise ValueError(
            "scan requested but the LiDAR is disabled "
            "(lidar_config.enabled=False)"
        )
    return fields


def _raw_keys(
    *, scan_enabled: bool, frenet_enabled: bool
) -> tuple[str, ...]:
    keys = list(_RAW_BASE_KEYS)
    if scan_enabled:
        keys.insert(0, "scans")
    if frenet_enabled:
        keys.append("frenet")
    return tuple(keys)


def _dependencies(fields: tuple[str, ...] | None, raw_keys: tuple[str, ...]):
    if fields is None:
        return raw_keys
    dependencies: list[str] = []
    for field in fields:
        dependency = (
            "standard_state"
            if field in _DERIVED_FIELDS
            else _BASE_DEPENDENCIES[field]
        )
        if dependency not in dependencies:
            dependencies.append(dependency)
    return tuple(dependencies)


def _field_space(
    field: str,
    config: EnvConfig,
    core_config: CoreConfig,
    bounds: dict,
) -> gym.Space:
    lidar = config.lidar_config
    return field_space_from(
        field=field,
        state_dim=core_config.dynamics.state_dim,
        scan_enabled=core_config.scan_enabled,
        scan_num_beams=lidar.num_beams,
        scan_max_range=lidar.range_max,
        bounds=bounds,
    )


def _direct_space(
    keys: tuple[str, ...],
    config: EnvConfig,
    core_config: CoreConfig,
    bounds: dict,
) -> gym.Space:
    num_agents = config.num_agents
    lidar = config.lidar_config
    spaces: dict[str, gym.Space] = {}
    for key in keys:
        if key == "scans":
            spaces[key] = _batched(
                scan_space_from(True, lidar.num_beams, lidar.range_max),
                num_agents,
            )
        elif key == "state":
            spaces[key] = _batched(
                _field_space("state", config, core_config, bounds), num_agents
            )
        elif key == "standard_state":
            spaces[key] = _batched(
                _field_space("std_state", config, core_config, bounds),
                num_agents,
            )
        elif key == "collisions":
            spaces[key] = gym.spaces.Box(
                low=0.0, high=1.0, shape=(num_agents,), dtype=np.float32
            )
        elif key == "frenet":
            spaces[key] = _batched(
                _field_space("frenet_pose", config, core_config, bounds),
                num_agents,
            )
        elif key in ("lap_times", "lap_counts"):
            spaces[key] = _batched(
                _field_space("lap_time", config, core_config, bounds),
                num_agents,
            )
        elif key == "sim_time":
            spaces[key] = _field_space(
                "sim_time", config, core_config, bounds
            )
        else:  # pragma: no cover - every static raw key is handled above
            raise ValueError(f"no space builder for raw observation key {key!r}")
    return gym.spaces.Dict(spaces)


def _validate_topology(config: EnvConfig, core_config: CoreConfig) -> None:
    if not isinstance(config, EnvConfig):
        raise TypeError("config must be an EnvConfig instance")
    if not isinstance(core_config, CoreConfig):
        raise TypeError("core_config must be a CoreConfig instance")
    expected_dim = {
        DynamicModel.KS: 5,
        DynamicModel.ST: 7,
    }.get(config.simulation_config.dynamics_model)
    if expected_dim is None:
        raise ValueError(
            "Gym observation packaging supports only KS and ST dynamics"
        )
    if (
        core_config.dynamics.num_agents != config.num_agents
        or core_config.dynamics.state_dim != expected_dim
    ):
        raise ValueError("EnvConfig and CoreConfig dynamics topology do not match")
    lidar = config.lidar_config
    expected_beams = lidar.num_beams if lidar.enabled else 1
    if (
        core_config.scan_enabled != lidar.enabled
        or core_config.scan.num_agents != config.num_agents
        or core_config.scan.num_beams != expected_beams
        or not math.isclose(
            core_config.scan.angle_min, lidar.angle_min, rel_tol=0.0, abs_tol=1e-12
        )
        or not math.isclose(
            core_config.scan.angle_max, lidar.angle_max, rel_tol=0.0, abs_tol=1e-12
        )
    ):
        raise ValueError("EnvConfig and CoreConfig LiDAR topology do not match")
    if (
        core_config.frenet_enabled
        != config.simulation_config.compute_frenet_frame
    ):
        raise ValueError("EnvConfig and CoreConfig Frenet topology do not match")


def _validate_bundle(config: EnvConfig, bundle: CoreBundle) -> None:
    if not isinstance(bundle, CoreBundle):
        raise TypeError("bundle must be a CoreBundle instance")
    if not isinstance(bundle.track, Track):
        raise TypeError("bundle.track must be a resolved Track instance")
    _validate_topology(config, bundle.config)
    configured_range = float(config.lidar_config.range_max)
    core_range = float(np.asarray(jax.device_get(bundle.params.scan.range_max)))
    if not math.isclose(
        core_range, configured_range, rel_tol=1.0e-6, abs_tol=1.0e-6
    ):
        raise ValueError(
            "EnvConfig and CoreBundle LiDAR range do not match: "
            f"{configured_range} != {core_range}"
        )


@dataclass(frozen=True)
class GymObservationAdapter:
    """Static field layout, finite space, and host packer for one core."""

    fields: tuple[str, ...] | None
    raw_keys: tuple[str, ...]
    agent_ids: tuple[str, ...]
    observation_space: gym.Space
    scan_enabled: bool
    frenet_enabled: bool
    dependencies: tuple[str, ...]
    state_dim: int
    scan_num_beams: int

    @classmethod
    def from_bundle(
        cls,
        bundle: CoreBundle,
        observation_config: ObservationConfig | None = None,
    ) -> "GymObservationAdapter":
        """Resolve one immutable public layout for a paired core bundle."""
        if not isinstance(bundle, CoreBundle):
            raise TypeError("bundle must be a CoreBundle instance")
        if not isinstance(bundle.env_config, EnvConfig):
            raise TypeError("bundle.env_config must be an EnvConfig instance")
        if observation_config is not None and not isinstance(
            observation_config, ObservationConfig
        ):
            raise TypeError(
                "observation_config must be an ObservationConfig or None"
            )
        config = (
            bundle.env_config
            if observation_config is None
            else bundle.env_config.with_updates(
                observation_config=observation_config
            )
        )
        _validate_bundle(config, bundle)
        core_config = bundle.config
        track = bundle.track
        fields = _resolve_fields(
            config,
            scan_enabled=core_config.scan_enabled,
            frenet_enabled=core_config.frenet_enabled,
        )
        raw_keys = _raw_keys(
            scan_enabled=core_config.scan_enabled,
            frenet_enabled=core_config.frenet_enabled,
        )
        widest = config.domain_randomization_config.widest_params(config.params)
        bounds = physical_bounds_from(
            widest,
            track,
            config.simulation_config.integrator_timestep,
        )
        agent_ids = tuple(f"agent_{index}" for index in range(config.num_agents))
        if fields is None:
            observation_space = _direct_space(
                raw_keys, config, core_config, bounds
            )
        else:
            observation_space = gym.spaces.Dict(
                {
                    agent_id: gym.spaces.Dict(
                        {
                            field: _field_space(
                                field, config, core_config, bounds
                            )
                            for field in fields
                        }
                    )
                    for agent_id in agent_ids
                }
            )
        return cls(
            fields=fields,
            raw_keys=raw_keys,
            agent_ids=agent_ids,
            observation_space=observation_space,
            scan_enabled=core_config.scan_enabled,
            frenet_enabled=core_config.frenet_enabled,
            dependencies=_dependencies(fields, raw_keys),
            state_dim=core_config.dynamics.state_dim,
            scan_num_beams=(config.lidar_config.num_beams if core_config.scan_enabled else 1),
        )

    def _expected_shape(self, dependency: str) -> tuple[int, ...]:
        num_agents = len(self.agent_ids)
        return {
            "scans": (num_agents, self.scan_num_beams),
            "state": (num_agents, self.state_dim),
            "standard_state": (num_agents, 7),
            "collisions": (num_agents,),
            "frenet": (num_agents, 3),
            "lap_times": (num_agents,),
            "lap_counts": (num_agents,),
            "sim_time": (),
        }[dependency]

    def _host_dependencies(
        self, observation: CoreObservation
    ) -> dict[str, np.ndarray]:
        if not isinstance(observation, CoreObservation):
            raise TypeError("observation must be a CoreObservation")
        device_values = tuple(
            getattr(observation, name) for name in self.dependencies
        )
        host_values = jax.device_get(device_values)
        projected = dict(zip(self.dependencies, host_values, strict=True))
        for name, value in projected.items():
            if np.shape(value) != self._expected_shape(name):
                raise ValueError(
                    f"CoreObservation.{name} must have shape "
                    f"{self._expected_shape(name)}, got {np.shape(value)}"
                )
        return projected

    @staticmethod
    def _derived_values(standard_state: np.ndarray) -> dict[str, np.generic]:
        speed = standard_state[3]
        beta = standard_state[6]
        vx = speed * np.cos(beta)
        vy = speed * np.sin(beta)
        return {
            "pose_x": standard_state[0],
            "pose_y": standard_state[1],
            "pose_theta": standard_state[4],
            "linear_vel_x": vx,
            "linear_vel_y": vy,
            "linear_vel_magnitude": np.hypot(vx, vy),
            "ang_vel_z": standard_state[5],
            "delta": standard_state[2],
            "beta": beta,
        }

    def package(self, observation: CoreObservation):
        """Transfer only selected dependencies and return independent copies."""
        host = self._host_dependencies(observation)
        if self.fields is None:
            return {key: _copy_float32(host[key]) for key in self.raw_keys}

        packaged: dict[str, dict[str, np.ndarray]] = {}
        needs_derived = any(field in _DERIVED_FIELDS for field in self.fields)
        for index, agent_id in enumerate(self.agent_ids):
            agent: dict[str, np.ndarray] = {}
            derived = (
                self._derived_values(host["standard_state"][index])
                if needs_derived else {}
            )
            for field in self.fields:
                if field in _DERIVED_FIELDS:
                    agent[field] = _copy_float32(derived[field])
                else:
                    dependency = _BASE_DEPENDENCIES[field]
                    value = (
                        host[dependency]
                        if dependency == "sim_time"
                        else host[dependency][index]
                    )
                    agent[field] = _copy_float32(value)
            packaged[agent_id] = agent
        return packaged


__all__ = ["GymObservationAdapter"]
