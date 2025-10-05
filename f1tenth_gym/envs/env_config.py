"""Typed configuration structures for the simplified F1TENTH gym environment."""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from collections.abc import Mapping
from typing import Callable, Optional, Type, TypeVar, TYPE_CHECKING, Any

import numpy as np

from .integrators import IntegratorType
from .dynamic_models import (
    DynamicModel,
    VehicleParameters,
    F1TENTH_VEHICLE_PARAMETERS,
)
from .action import LongitudinalActionType, SteerActionType
from .observation import ObservationType, ObservationFeature
from .reset import ResetStrategy
from .lidar import LiDARConfig
from .collision_models import CollisionCheckMode

if TYPE_CHECKING:
    from .track import Track

# Type aliases for callables used throughout the simulator
DynamicsFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
StandardStateFn = Callable[[np.ndarray], np.ndarray]
IntegratorFn = Callable[[DynamicsFn, np.ndarray, np.ndarray, float, np.ndarray], np.ndarray]


class LoopCounterMode(IntEnum):
    TOGGLE = 1
    FRENET_BASED = 2
    WINDING_ANGLE = 3


@dataclass(frozen=True, slots=True)
class ControlConfig:
    longitudinal_mode: LongitudinalActionType = LongitudinalActionType.SPEED
    steering_mode: SteerActionType = SteerActionType.STEERING_ANGLE
    steer_delay_steps: int = 0

    def with_updates(self, **changes: Any) -> "ControlConfig":
        return replace(self, **changes)


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    timestep: float = 0.01
    integrator_timestep: float = 0.01
    integrator: IntegratorType = IntegratorType.RK4
    dynamics_model: DynamicModel = DynamicModel.ST
    loop_counter: LoopCounterMode = LoopCounterMode.FRENET_BASED
    compute_frenet_frame: bool = True
    max_laps: Optional[int] = 1

    def with_updates(self, **changes: Any) -> "SimulationConfig":
        updated = replace(self, **changes)
        if updated.loop_counter is LoopCounterMode.FRENET_BASED and not updated.compute_frenet_frame:
            updated = replace(updated, compute_frenet_frame=True)
        return updated


@dataclass(frozen=True, slots=True)
class ObservationConfig:
    type: ObservationType = ObservationType.DIRECT
    features: Optional[tuple[ObservationFeature, ...]] = None

    def with_updates(self, **changes: Any) -> "ObservationConfig":
        return replace(self, **changes)


@dataclass(frozen=True, slots=True)
class ResetConfig:
    strategy: ResetStrategy = ResetStrategy.RL_GRID_STATIC

    def with_updates(self, **changes: Any) -> "ResetConfig":
        return replace(self, **changes)


@dataclass(frozen=True, slots=True, init=False)
class EnvConfig:
    seed: int = 12345
    map_name: "Track | str" = "Spielberg"
    map_scale: float = 1.0
    params: VehicleParameters = F1TENTH_VEHICLE_PARAMETERS
    num_agents: int = 1
    ego_index: int = 0
    control: ControlConfig = ControlConfig()
    simulation: SimulationConfig = SimulationConfig()
    observation: ObservationConfig = ObservationConfig()
    reset: ResetConfig = ResetConfig()
    lidar: LiDARConfig = LiDARConfig()
    collision_check: CollisionCheckMode = CollisionCheckMode.LIDAR_SCAN
    render_enabled: bool = True

    def __init__(
        self,
        *,
        seed: int = 12345,
        map_name: "Track | str" = "Spielberg",
        map_scale: float = 1.0,
        params: VehicleParameters | Mapping[str, object] = F1TENTH_VEHICLE_PARAMETERS,
        num_agents: int = 1,
        ego_index: int = 0,
        control: ControlConfig | None = None,
        simulation: SimulationConfig | None = None,
        observation: ObservationConfig | None = None,
        reset: ResetConfig | None = None,
        lidar: LiDARConfig | None = None,
        collision_check: CollisionCheckMode = CollisionCheckMode.LIDAR_SCAN,
        render_enabled: bool = True,
        **overrides: object,
    ) -> None:
        if not isinstance(params, VehicleParameters):
            raise TypeError("params must be a VehicleParameters instance or mapping")

        control_cfg = control or ControlConfig()
        simulation_cfg = simulation or SimulationConfig()
        observation_cfg = observation or ObservationConfig()
        reset_cfg = reset or ResetConfig()
        lidar_cfg = lidar or LiDARConfig()

        overrides = dict(overrides)

        control_updates = {key: overrides.pop(key) for key in list(overrides) if key in {"longitudinal_mode", "steering_mode", "steer_delay_steps"}}
        if control_updates:
            control_cfg = control_cfg.with_updates(**control_updates)

        simulation_updates = {key: overrides.pop(key) for key in list(overrides) if key in {"timestep", "integrator_timestep", "integrator", "dynamics_model", "loop_counter", "compute_frenet_frame", "max_laps"}}
        if "dynamics_model" in simulation_updates:
            simulation_updates["dynamics_model"] = _expect_enum(simulation_updates["dynamics_model"], DynamicModel, "dynamics_model")
        if "loop_counter" in simulation_updates:
            simulation_updates["loop_counter"] = _expect_enum(simulation_updates["loop_counter"], LoopCounterMode, "loop_counter")
        if "integrator" in simulation_updates:
            simulation_updates["integrator"] = _expect_enum(simulation_updates["integrator"], IntegratorType, "integrator")
        if simulation_updates:
            simulation_cfg = simulation_cfg.with_updates(**simulation_updates)

        if "observation_type" in overrides:
            value = overrides.pop("observation_type")
            observation_cfg = observation_cfg.with_updates(
                type=_expect_enum(value, ObservationType, "observation_type"),
                features=observation_cfg.features if value is ObservationType.FEATURES else None,
            )
        if "observation_features" in overrides:
            value = overrides.pop("observation_features")
            if value is None:
                features = None
            else:
                features = tuple(ObservationFeature.from_value(item) for item in value)
            observation_cfg = observation_cfg.with_updates(features=features)

        if "reset_strategy" in overrides:
            value = overrides.pop("reset_strategy")
            reset_cfg = reset_cfg.with_updates(
                strategy=_expect_enum(value, ResetStrategy, "reset_strategy")
            )

        lidar_updates = {key: overrides.pop(key) for key in list(overrides) if key in {"lidar_enabled", "lidar_num_beams", "lidar_field_of_view", "lidar_maximum_range", "lidar_noise_std"}}
        if lidar_updates:
            mapped_updates = {}
            if "lidar_enabled" in lidar_updates:
                mapped_updates["enabled"] = bool(lidar_updates["lidar_enabled"])
            if "lidar_num_beams" in lidar_updates:
                mapped_updates["num_beams"] = int(lidar_updates["lidar_num_beams"])
            if "lidar_field_of_view" in lidar_updates:
                mapped_updates["field_of_view"] = float(lidar_updates["lidar_field_of_view"])
            if "lidar_maximum_range" in lidar_updates:
                mapped_updates["maximum_range"] = float(lidar_updates["lidar_maximum_range"])
            if "lidar_noise_std" in lidar_updates:
                mapped_updates["noise_std"] = float(lidar_updates["lidar_noise_std"])
            lidar_cfg = lidar_cfg.with_updates(**mapped_updates)

        if overrides:
            unknown = ", ".join(sorted(overrides.keys()))
            raise TypeError(f"Unknown configuration keywords: {unknown}")

        object.__setattr__(self, "seed", int(seed))
        object.__setattr__(self, "map_name", map_name)
        object.__setattr__(self, "map_scale", float(map_scale))
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "num_agents", int(num_agents))
        object.__setattr__(self, "ego_index", int(ego_index))
        object.__setattr__(self, "control", control_cfg)
        object.__setattr__(self, "simulation", simulation_cfg)
        object.__setattr__(self, "observation", observation_cfg)
        object.__setattr__(self, "reset", reset_cfg)
        object.__setattr__(self, "lidar", lidar_cfg)
        object.__setattr__(self, "collision_check", collision_check)
        object.__setattr__(self, "render_enabled", bool(render_enabled))

        self.__post_init__()

    def __post_init__(self) -> None:
        if (
            self.simulation.loop_counter is LoopCounterMode.FRENET_BASED
            and not self.simulation.compute_frenet_frame
        ):
            object.__setattr__(self, "simulation", self.simulation.with_updates(compute_frenet_frame=True))

    def with_updates(self, **changes: Any) -> "EnvConfig":
        return replace(self, **changes)


EnumType = TypeVar("EnumType", bound=IntEnum)


def _expect_enum(value: object, enum_cls: Type[EnumType], field_name: str) -> EnumType:
    if isinstance(value, enum_cls):
        return value
    raise TypeError(
        f"{field_name} must be an instance of {enum_cls.__name__}, got {value!r}"
    )


def default_env_config() -> EnvConfig:
    return EnvConfig()