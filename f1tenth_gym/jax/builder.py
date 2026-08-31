"""Host conversion from ``EnvConfig`` and ``Track`` to the functional JAX core.

This module is intentionally a deep import: unlike :mod:`f1tenth_gym.jax`, it
loads host configuration and map types and performs device selection.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from f1tenth_gym.envs.action import (
    LongitudinalActionType,
    SteerActionType,
)
from f1tenth_gym.envs.collision_models import CollisionCheckMode
from f1tenth_gym.envs.contact.solver import ContactParams
from f1tenth_gym.envs.dynamic_models import (
    PARAMETER_ORDER,
    DynamicModel,
    VehicleParameters,
)
from f1tenth_gym.envs.env_config import (
    EnvConfig,
    LoopCounterMode,
    RewardMode,
)
from f1tenth_gym.envs.integrators import IntegratorType
from f1tenth_gym.envs.reset import ReferenceLine, ResetStrategy
from f1tenth_gym.envs.termination import AgentTerminationMode
from f1tenth_gym.envs.track import Track
from f1tenth_gym.envs.track.budget import DEFAULT_MAX_BYTES
from f1tenth_gym.envs.track.walls import DEFAULT_TOL_PX

from .contact import WallContactConfig
from .controls import LongitudinalControlMode, SteeringControlMode
from .core import DynamicsConfig, EpisodeParams
from .dynamics import (
    DynamicsParams,
    kinematic_single_track,
    single_track,
)
from .environment import CoreConfig, CoreParams, CoreTables
from .episode import (
    BookkeepingParams,
    BuiltinRewardMode,
    EpisodeConfig,
    TerminationMode,
)
from .geometry import BodyParams
from .integrators import euler_step, rk4_step
from .lidar import ScanConfig, ScanParams
from .pairs import PairContactConfig, PairTable
from .preprocess import (
    build_pair_table,
    build_reset_table,
    build_scan_params,
    build_track_table,
)
from .reset import ResetConfig
from .track import FrenetProjectionConfig, TrackTable


@dataclass(frozen=True)
class CoreBundle:
    """Paired resolved track and device arrays ready for core/adapters."""

    env_config: EnvConfig
    config: CoreConfig
    tables: CoreTables
    params: CoreParams
    device: Any
    track: Track


def _float32_tree(tree: Any):
    """Normalize continuous production parameters independently of JAX x64."""
    return jax.tree.map(
        lambda value: np.asarray(value, dtype=np.float32),
        tree,
    )


def _disabled_pair_table(num_agents: int) -> PairTable:
    """Return constant-size masked topology when pair contact cannot execute."""
    return PairTable(
        indices=jnp.zeros((1, 2), dtype=jnp.int32),
        mask=jnp.zeros((1,), dtype=jnp.bool_),
        num_agents=num_agents,
    )


def _substeps(config: EnvConfig) -> int:
    simulation = config.simulation_config
    ratio = simulation.timestep / simulation.integrator_timestep
    count = max(1, int(round(ratio)))
    if not np.isclose(ratio, count, rtol=0.0, atol=1.0e-9):
        raise ValueError(
            f"timestep ({simulation.timestep}) must be an integer multiple of "
            f"integrator_timestep ({simulation.integrator_timestep}), got a "
            f"ratio of {ratio}"
        )
    return count


def _dynamics(config: EnvConfig) -> tuple[int, Any]:
    model = config.simulation_config.dynamics_model
    if model is DynamicModel.KS:
        return 5, kinematic_single_track
    if model is DynamicModel.ST:
        return 7, single_track
    raise ValueError(f"unsupported functional dynamics model: {model!r}")


def _integrator(config: EnvConfig):
    selected = config.simulation_config.integrator
    if selected is IntegratorType.EULER:
        return euler_step
    if selected is IntegratorType.RK4:
        return rk4_step
    raise ValueError(f"unsupported functional integrator: {selected!r}")


def _longitudinal(config: EnvConfig) -> LongitudinalControlMode:
    selected = config.control_config.longitudinal_mode
    if selected is LongitudinalActionType.ACCL:
        return LongitudinalControlMode.ACCELERATION
    if selected is LongitudinalActionType.SPEED:
        return LongitudinalControlMode.TARGET_SPEED
    raise ValueError(f"unsupported longitudinal control mode: {selected!r}")


def _steering(config: EnvConfig) -> SteeringControlMode:
    selected = config.control_config.steering_mode
    if selected is SteerActionType.STEERING_ANGLE:
        return SteeringControlMode.TARGET_ANGLE
    if selected is SteerActionType.STEERING_SPEED:
        return SteeringControlMode.STEERING_RATE
    raise ValueError(f"unsupported steering control mode: {selected!r}")


def _termination(config: EnvConfig) -> TerminationMode:
    selected = config.termination_config.agent_mode
    if selected is AgentTerminationMode.EGO:
        return TerminationMode.EGO
    if selected is AgentTerminationMode.ANY:
        return TerminationMode.ANY
    if selected is AgentTerminationMode.ALL:
        return TerminationMode.ALL
    raise ValueError(f"unsupported agent termination mode: {selected!r}")


def _reward(
    config: EnvConfig,
    custom_reward_fallback: BuiltinRewardMode | None = None,
) -> BuiltinRewardMode:
    """Map the host reward mode to compiled dispatch.

    A Python ``CUSTOM`` callback cannot execute in the functional transition.
    Host adapters that run that callback after device-to-host conversion may
    provide an explicit built-in fallback whose result they discard.  Keeping
    this opt-in makes the standalone core builder reject callbacks by default
    instead of silently changing their meaning.
    """
    if custom_reward_fallback is not None and not isinstance(
        custom_reward_fallback, BuiltinRewardMode
    ):
        raise TypeError(
            "custom_reward_fallback must be a BuiltinRewardMode or None"
        )
    selected = config.reward_config.mode
    if selected is not RewardMode.CUSTOM and custom_reward_fallback is not None:
        raise ValueError(
            "custom_reward_fallback only applies to RewardMode.CUSTOM"
        )
    if selected is RewardMode.SURVIVAL:
        return BuiltinRewardMode.SURVIVAL
    if selected is RewardMode.PROGRESS:
        return BuiltinRewardMode.PROGRESS
    if selected is RewardMode.CUSTOM:
        if custom_reward_fallback is not None:
            return custom_reward_fallback
        raise ValueError(
            "RewardMode.CUSTOM is adapter-only; the functional core accepts "
            "built-in rewards or a future pure-JAX reward callable"
        )
    raise ValueError(f"unsupported reward mode: {selected!r}")


def _reset_settings(
    config: EnvConfig,
) -> tuple[ResetConfig, ReferenceLine, float, float, float | None]:
    host = config.reset_config
    strategy = host.strategy
    if strategy is ResetStrategy.MAP_RANDOM_STATIC:
        raise ValueError(
            "MAP_RANDOM_STATIC is not supported by the functional core; use "
            "an RL_* strategy or an explicit reset override"
        )
    grid = strategy in (
        ResetStrategy.RL_GRID_STATIC,
        ResetStrategy.RL_GRID_RANDOM,
    )
    if strategy not in (
        ResetStrategy.RL_GRID_STATIC,
        ResetStrategy.RL_RANDOM_STATIC,
        ResetStrategy.RL_GRID_RANDOM,
        ResetStrategy.RL_RANDOM_RANDOM,
    ):
        raise ValueError(f"unsupported functional reset strategy: {strategy!r}")
    default_shuffle = strategy in (
        ResetStrategy.RL_GRID_RANDOM,
        ResetStrategy.RL_RANDOM_RANDOM,
    )
    shuffle = default_shuffle if host.shuffle is None else bool(host.shuffle)
    move_laterally = (
        False if host.move_laterally is None else bool(host.move_laterally)
    )
    reference = host.reference_line or ReferenceLine.RACELINE
    minimum = 1.5 if host.min_dist is None else float(host.min_dist)
    maximum = 2.5 if host.max_dist is None else float(host.max_dist)
    start_width = (
        1.0 if host.start_width is None else float(host.start_width)
    ) if grid else None
    return (
        ResetConfig(
            num_agents=config.num_agents,
            move_laterally=move_laterally,
            shuffle=shuffle,
        ),
        reference,
        minimum,
        maximum,
        start_width,
    )


def _validate_host_surface(config: EnvConfig) -> None:
    if not isinstance(config, EnvConfig):
        raise TypeError("config must be an EnvConfig instance")
    if config.simulation_config.loop_counter is not LoopCounterMode.FRENET_BASED:
        raise ValueError(
            "the functional core currently supports FRENET_BASED lap counting only"
        )
    if (
        config.collision_check is CollisionCheckMode.SEGMENT_CONTACT
        and not math.isclose(
            config.contact_config.wall_tolerance_px,
            DEFAULT_TOL_PX,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError(
            "the functional preprocessor currently supports only the default "
            f"wall_tolerance_px={DEFAULT_TOL_PX}"
        )


def build_core_config(
    config: EnvConfig,
    *,
    custom_reward_fallback: BuiltinRewardMode | None = None,
) -> CoreConfig:
    """Translate validated host enums and topology into one static core config."""
    _validate_host_surface(config)
    state_dim, dynamics_fn = _dynamics(config)
    reset, _reference, _minimum, _maximum, _start_width = _reset_settings(config)
    control = config.control_config
    lidar = config.lidar_config
    contact = config.contact_config
    simulation = config.simulation_config
    return CoreConfig(
        dynamics=DynamicsConfig(
            num_agents=config.num_agents,
            state_dim=state_dim,
            dynamics_fn=dynamics_fn,
            integrator_fn=_integrator(config),
            num_substeps=_substeps(config),
            longitudinal_mode=_longitudinal(config),
            steering_mode=_steering(config),
            steer_delay_steps=control.steer_delay_steps,
            throttle_delay_steps=control.throttle_delay_steps,
            derive_steer_kp=control.steer_kp is None,
        ),
        reset=reset,
        scan=ScanConfig(
            config.num_agents,
            lidar.num_beams if lidar.enabled else 1,
            lidar.angle_min,
            lidar.angle_max,
        ),
        wall_contact=WallContactConfig(
            config.num_agents,
            state_dim,
            solver_iterations=contact.solver_iterations,
        ),
        pair_contact=PairContactConfig(
            config.num_agents,
            state_dim,
            solver_iterations=contact.solver_iterations,
        ),
        episode=EpisodeConfig(
            num_agents=config.num_agents,
            ego_index=config.ego_index,
            count_partial_first_lap=simulation.count_partial_first_lap,
            termination_mode=_termination(config),
            reward_mode=_reward(config, custom_reward_fallback),
        ),
        frenet=FrenetProjectionConfig(),
        contact_enabled=(
            config.collision_check is CollisionCheckMode.SEGMENT_CONTACT
        ),
        scan_enabled=lidar.enabled,
        frenet_enabled=simulation.compute_frenet_frame,
    )


def build_core_tables(
    config: EnvConfig,
    track: Track,
    *,
    vehicle_params: VehicleParameters | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> CoreTables:
    """Preprocess one resolved track for selected reset and geometry modes.

    ``vehicle_params`` is the effective episode draw when supplied. Contact
    allocation also includes the config's complete domain-randomization bounds.
    Disabled geometry receives constant-size masked placeholders.
    """
    _validate_host_surface(config)
    if not isinstance(track, Track):
        raise TypeError("track must be a resolved Track instance")
    explicit_draw = vehicle_params is not None
    if vehicle_params is None:
        vehicle_params = config.params
    if not isinstance(vehicle_params, VehicleParameters):
        raise TypeError("vehicle_params must be a VehicleParameters instance")
    if explicit_draw:
        _validate_vehicle_draw(config, vehicle_params)
    _reset, reference, minimum, maximum, start_width = _reset_settings(config)
    line = track.centerline if reference is ReferenceLine.CENTERLINE else track.raceline
    contact_enabled = config.collision_check is CollisionCheckMode.SEGMENT_CONTACT
    scan_enabled = config.lidar_config.enabled
    track_table = build_track_table(
        track,
        vehicle_params,
        domain_randomization=config.domain_randomization_config,
        contact_tile_size=config.contact_config.tile_size,
        contact_margin=config.contact_config.margin,
        contact_enabled=contact_enabled,
        ray_max_range=(config.lidar_config.range_max if scan_enabled else None),
        max_bytes=max_bytes,
    )
    return CoreTables(
        reset=build_reset_table(
            line,
            min_dist=minimum,
            max_dist=maximum,
            start_width=start_width,
        ),
        track=track_table,
        pairs=(
            build_pair_table(config.num_agents)
            if contact_enabled
            else _disabled_pair_table(config.num_agents)
        ),
    )


def _validate_vehicle_draw(config: EnvConfig, params: VehicleParameters) -> None:
    randomization = config.domain_randomization_config
    if not randomization.randomized_fields():
        return
    low_params = randomization.low
    high_params = randomization.high
    if low_params is None or high_params is None:
        raise ValueError("enabled domain randomization requires low/high bounds")
    # The mutable environment samples every finite field, including fields whose
    # endpoints agree. Checking the complete supported surface prevents a caller
    # from smuggling an out-of-range fixed body or dynamics value past a table
    # sized from the configured bounds.
    for name in PARAMETER_ORDER[:20]:
        value = getattr(params, name)
        low = getattr(low_params, name)
        high = getattr(high_params, name)
        if not low <= value <= high:
            raise ValueError(
                f"vehicle_params.{name}={value} lies outside DR bounds "
                f"[{low}, {high}]"
            )


def build_core_params(
    config: EnvConfig,
    track_table: TrackTable,
    *,
    vehicle_params: VehicleParameters | None = None,
    custom_reward_fallback: BuiltinRewardMode | None = None,
) -> CoreParams:
    """Translate one nominal or already-sampled vehicle parameter episode.

    Continuous production leaves are normalized to float32, counters to int32
    and enable flags to booleans independently of process-wide JAX x64 mode.
    """
    _validate_host_surface(config)
    _reward(config, custom_reward_fallback)
    if not isinstance(track_table, TrackTable):
        raise TypeError("track_table must be a TrackTable instance")
    explicit_draw = vehicle_params is not None
    if vehicle_params is None:
        vehicle_params = config.params
    if not isinstance(vehicle_params, VehicleParameters):
        raise TypeError("vehicle_params must be a VehicleParameters instance")
    if explicit_draw:
        _validate_vehicle_draw(config, vehicle_params)
    control = config.control_config
    contact = config.contact_config
    lidar = config.lidar_config
    simulation = config.simulation_config
    termination = config.termination_config
    reward = config.reward_config
    scan = (
        build_scan_params(lidar, track_table)
        if lidar.enabled
        else ScanParams.from_lidar_config(lidar)
    )
    dynamics = _float32_tree(
        DynamicsParams.from_vehicle_parameters(vehicle_params)
    )
    body = _float32_tree(BodyParams.from_vehicle_parameters(vehicle_params))
    contact_params = _float32_tree(
        ContactParams(
            restitution=contact.restitution,
            friction=contact.friction,
            restitution_threshold=contact.restitution_threshold,
            baumgarte=contact.baumgarte,
            slop=contact.slop,
        )
    )
    return CoreParams(
        transition=EpisodeParams(
            dynamics=dynamics,
            timestep=np.float32(simulation.timestep),
            steer_kp=np.float32(
                0.0 if control.steer_kp is None else control.steer_kp
            ),
            steer_noise_std=np.float32(control.steer_noise_std),
            accel_noise_std=np.float32(control.accl_noise_std),
        ),
        body=body,
        contact=contact_params,
        scan=_float32_tree(scan),
        bookkeeping=BookkeepingParams(
            terminate_on_collision=np.bool_(termination.terminate_on_collision),
            lap_limit_enabled=np.bool_(simulation.max_laps is not None),
            max_laps=np.int32(
                0 if simulation.max_laps is None else simulation.max_laps
            ),
            step_limit_enabled=np.bool_(
                termination.max_episode_steps is not None
            ),
            max_episode_steps=np.int32(
                0
                if termination.max_episode_steps is None
                else termination.max_episode_steps
            ),
            progress_weight=np.float32(reward.progress_weight),
            velocity_weight=np.float32(reward.velocity_weight),
            timestep_weight=np.float32(reward.timestep_weight),
            collision_penalty=np.float32(reward.collision_penalty),
        ),
    )


def _core_device(config: EnvConfig):
    requested = []
    if config.collision_check is CollisionCheckMode.SEGMENT_CONTACT:
        requested.append(config.contact_config.device)
    if config.lidar_config.enabled:
        requested.append(config.lidar_config.scan_device)
    unique = set(requested)
    if len(unique) > 1:
        raise ValueError(
            "the composed core needs one device, but contact and LiDAR request "
            f"different devices: {sorted(unique)}"
        )
    backend = requested[0] if requested else "cpu"
    try:
        devices = jax.devices(backend)
    except RuntimeError as exc:
        raise RuntimeError(
            f"requested JAX {backend!r} device is unavailable"
        ) from exc
    if not devices:
        raise RuntimeError(f"requested JAX {backend!r} device is unavailable")
    return devices[0]


def _put_on_device(tree: Any, device: Any):
    return jax.tree.map(
        lambda value: jax.device_put(jnp.asarray(value), device),
        tree,
    )


def build_core(
    config: EnvConfig,
    track: Track,
    *,
    vehicle_params: VehicleParameters | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    custom_reward_fallback: BuiltinRewardMode | None = None,
) -> CoreBundle:
    """Build and place one complete functional core from a resolved track.

    Varying domain-randomization bounds require an explicit sampled
    ``vehicle_params`` until key-driven sampling is part of the core. Active
    contact and LiDAR must select the same available JAX device. A host adapter
    that executes a Python custom reward after conversion may opt into a
    ``custom_reward_fallback`` whose compiled result it ignores.
    """
    _validate_host_surface(config)
    if not isinstance(track, Track):
        raise TypeError("track must be a resolved Track instance")
    if (
        config.domain_randomization_config.randomized_fields()
        and vehicle_params is None
    ):
        raise ValueError(
            "domain randomization requires an explicit sampled vehicle_params "
            "until the key-driven core sampler is composed"
        )
    core_config = build_core_config(
        config,
        custom_reward_fallback=custom_reward_fallback,
    )
    device = _core_device(config)
    tables = build_core_tables(
        config,
        track,
        vehicle_params=vehicle_params,
        max_bytes=max_bytes,
    )
    params = build_core_params(
        config,
        tables.track,
        vehicle_params=vehicle_params,
        custom_reward_fallback=custom_reward_fallback,
    )
    return CoreBundle(
        env_config=config,
        config=core_config,
        tables=_put_on_device(tables, device),
        params=_put_on_device(params, device),
        device=device,
        track=track,
    )


__all__ = [
    "CoreBundle",
    "build_core",
    "build_core_config",
    "build_core_params",
    "build_core_tables",
]
