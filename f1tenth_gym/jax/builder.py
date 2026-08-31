"""Host conversion from ``EnvConfig`` and ``Track`` to the functional JAX core.

This module is intentionally a deep import: unlike :mod:`f1tenth_gym.jax`, it
loads host configuration and map types and performs device selection.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Any, Iterable

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
from .indexed import IndexedCoreTables, stack_core_tables
from .lidar import ScanConfig, ScanParams
from .pairs import PairContactConfig, PairTable
from .preprocess import (
    build_pair_table,
    build_reset_table,
    build_scan_params,
    build_track_table,
)
from .randomization import (
    ACTIVE_VEHICLE_FIELDS,
    ActiveVehicleParams,
    VehicleRandomizationParams,
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
    randomization: VehicleRandomizationParams
    device: Any
    track: Track


@dataclass(frozen=True)
class IndexedCoreBucket:
    """One exact-shape map executable and its source-row routing metadata.

    ``tracks`` and the leading table axis use the same bucket-local ordering.
    ``source_indices`` identifies rows in the caller's original environment
    batch, while ``map_indices`` selects one of those unique local maps for
    each source row.  Callers can therefore compile once per bucket and stitch
    observations or state leaves back with ``source_indices``.
    """

    tables: IndexedCoreTables
    tracks: tuple[Track, ...]
    source_indices: tuple[int, ...]
    map_indices: jax.Array

    def __post_init__(self) -> None:
        if not isinstance(self.tables, IndexedCoreTables):
            raise TypeError("tables must be an IndexedCoreTables instance")
        tracks = tuple(self.tracks)
        if len(tracks) != self.tables.num_maps:
            raise ValueError(
                "tracks must contain one entry per indexed map, got "
                f"{len(tracks)} and {self.tables.num_maps}"
            )
        if any(not isinstance(track, Track) for track in tracks):
            raise TypeError("every bucket track must be a Track instance")
        source_indices = tuple(int(index) for index in self.source_indices)
        if not source_indices:
            raise ValueError("an indexed core bucket must route at least one row")
        if any(index < 0 for index in source_indices):
            raise ValueError("source_indices must be non-negative")
        if len(set(source_indices)) != len(source_indices):
            raise ValueError("source_indices must be unique within a bucket")
        map_indices = jnp.asarray(self.map_indices)
        if map_indices.shape != (len(source_indices),):
            raise ValueError(
                "map_indices must have one entry per source row, got "
                f"{map_indices.shape} and {len(source_indices)}"
            )
        if not jnp.issubdtype(map_indices.dtype, jnp.integer):
            raise TypeError("map_indices must have an integer dtype")
        host_map_indices = np.asarray(map_indices)
        if (
            host_map_indices.size
            and (
                host_map_indices.min() < 0
                or host_map_indices.max() >= self.tables.num_maps
            )
        ):
            raise ValueError(
                f"map_indices must be in [0, {self.tables.num_maps})"
            )
        object.__setattr__(self, "tracks", tracks)
        object.__setattr__(self, "source_indices", source_indices)
        object.__setattr__(self, "map_indices", map_indices)


@dataclass(frozen=True)
class IndexedCoreBundle:
    """Host-orchestrated exact-shape map buckets sharing one core topology."""

    env_config: EnvConfig
    config: CoreConfig
    params: CoreParams
    randomization: VehicleRandomizationParams
    device: Any
    buckets: tuple[IndexedCoreBucket, ...]
    num_environments: int
    num_unique_tracks: int

    def __post_init__(self) -> None:
        buckets = tuple(self.buckets)
        num_environments = int(self.num_environments)
        num_unique_tracks = int(self.num_unique_tracks)
        if num_environments < 1:
            raise ValueError("num_environments must be >= 1")
        if num_unique_tracks < 1:
            raise ValueError("num_unique_tracks must be >= 1")
        if not buckets or any(
            not isinstance(bucket, IndexedCoreBucket) for bucket in buckets
        ):
            raise TypeError(
                "buckets must contain at least one IndexedCoreBucket"
            )
        routed = tuple(
            source
            for bucket in buckets
            for source in bucket.source_indices
        )
        if sorted(routed) != list(range(num_environments)):
            raise ValueError(
                "bucket source_indices must partition every environment row "
                "exactly once"
            )
        if sum(bucket.tables.num_maps for bucket in buckets) != num_unique_tracks:
            raise ValueError(
                "bucket map counts must equal num_unique_tracks"
            )
        object.__setattr__(self, "buckets", buckets)
        object.__setattr__(self, "num_environments", num_environments)
        object.__setattr__(self, "num_unique_tracks", num_unique_tracks)


def _float32_tree(tree: Any):
    """Normalize continuous production parameters independently of JAX x64."""
    return jax.tree.map(
        lambda value: np.asarray(value, dtype=np.float32),
        tree,
    )


def _active_vehicle_params(params: VehicleParameters) -> ActiveVehicleParams:
    """Copy the supported host ABI prefix into finite float32 leaves."""
    return ActiveVehicleParams(
        **{
            name: np.float32(getattr(params, name))
            for name in ACTIVE_VEHICLE_FIELDS
        }
    )


def _validate_active_values(
    params: VehicleParameters,
    *,
    prefix: str,
) -> None:
    """Reject active values that can make supported kernels undefined."""
    for name in ACTIVE_VEHICLE_FIELDS:
        value = getattr(params, name)
        if not math.isfinite(value):
            raise ValueError(f"{prefix}.{name} must be finite, got {value!r}")

    for name in ("lf", "lr", "m", "I", "v_switch", "width", "length"):
        value = getattr(params, name)
        if value <= 0.0:
            raise ValueError(f"{prefix}.{name} must be > 0, got {value!r}")
    for name in ("mu", "C_Sf", "C_Sr", "h", "a_max"):
        value = getattr(params, name)
        if value < 0.0:
            raise ValueError(f"{prefix}.{name} must be >= 0, got {value!r}")
    for lower_name, upper_name in (
        ("s_min", "s_max"),
        ("sv_min", "sv_max"),
        ("v_min", "v_max"),
    ):
        lower = getattr(params, lower_name)
        upper = getattr(params, upper_name)
        if lower > upper:
            raise ValueError(
                f"{prefix}.{lower_name} must be <= {prefix}.{upper_name}, "
                f"got {lower!r} > {upper!r}"
            )


def _validate_randomization_intervals(
    low: VehicleParameters,
    high: VehicleParameters,
) -> None:
    """Validate bounds and cross-field ordering for every possible draw."""
    _validate_active_values(low, prefix="domain_randomization.low")
    _validate_active_values(high, prefix="domain_randomization.high")
    for name in ACTIVE_VEHICLE_FIELDS:
        lower = getattr(low, name)
        upper = getattr(high, name)
        if lower > upper:
            raise ValueError(
                f"domain_randomization.low.{name} must be <= "
                f"domain_randomization.high.{name}, got "
                f"{lower!r} > {upper!r}"
            )

    # Fields are sampled independently. Endpoint validation alone does not
    # prevent a draw whose lower limit is above its independently drawn upper
    # limit, so require the complete intervals to remain ordered.
    for lower_name, upper_name in (
        ("s_min", "s_max"),
        ("sv_min", "sv_max"),
        ("v_min", "v_max"),
    ):
        lower = getattr(high, lower_name)
        upper = getattr(low, upper_name)
        if lower > upper:
            raise ValueError(
                "domain-randomization intervals must preserve "
                f"{lower_name} <= {upper_name} for every draw, but "
                f"high.{lower_name}={lower!r} > low.{upper_name}={upper!r}"
            )


def build_vehicle_randomization_params(
    config: EnvConfig,
) -> VehicleRandomizationParams:
    """Build the pure-JAX active vehicle and per-episode DR bounds.

    Disabled randomization and enabled-but-constant bounds collapse to one
    nominal vehicle with a false enable flag. A varying configuration keeps
    all twenty active bounds so dynamics and collision geometry are sampled
    from one correlated vehicle draw on device.
    """
    if not isinstance(config, EnvConfig):
        raise TypeError("config must be an EnvConfig instance")
    if tuple(PARAMETER_ORDER[:20]) != ACTIVE_VEHICLE_FIELDS:
        raise RuntimeError(
            "the functional active-vehicle fields no longer match the first "
            "20 entries of the VehicleParameters ABI"
        )

    nominal_host = config.params
    _validate_active_values(nominal_host, prefix="params")
    nominal = _active_vehicle_params(nominal_host)
    randomization = config.domain_randomization_config
    varying = bool(randomization.randomized_fields())
    if not varying:
        return VehicleRandomizationParams(
            nominal=nominal,
            low=nominal,
            high=nominal,
            enabled=np.bool_(False),
        )

    low_host = randomization.low
    high_host = randomization.high
    if low_host is None or high_host is None:  # guarded by host config
        raise ValueError("enabled domain randomization requires low/high bounds")
    _validate_randomization_intervals(low_host, high_host)
    return VehicleRandomizationParams(
        nominal=nominal,
        low=_active_vehicle_params(low_host),
        high=_active_vehicle_params(high_host),
        enabled=np.bool_(True),
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
    _validate_active_values(params, prefix="vehicle_params")
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
    for name in ACTIVE_VEHICLE_FIELDS:
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


def _platform_device(platform: str):
    """Resolve the first available device for one explicit JAX platform."""
    try:
        devices = jax.devices(platform)
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(
            f"requested JAX {platform!r} device is unavailable"
        ) from exc
    if not devices:
        raise RuntimeError(f"requested JAX {platform!r} device is unavailable")
    return devices[0]


def _core_device(config: EnvConfig, target_device: Any = None):
    if target_device is not None:
        if isinstance(target_device, str):
            return _platform_device(target_device)
        if isinstance(target_device, jax.Device):
            return target_device
        raise TypeError(
            "target_device must be a JAX platform string, jax.Device, or None"
        )

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
    return _platform_device(backend)


def _put_on_device(tree: Any, device: Any):
    return jax.tree.map(
        lambda value: jax.device_put(jnp.asarray(value), device),
        tree,
    )


def _core_table_signature(table: CoreTables) -> tuple:
    """Return the exact complete-table leaf shape and dtype signature."""
    if not isinstance(table, CoreTables):
        raise TypeError("table must be a CoreTables instance")
    return tuple(
        (tuple(leaf.shape), np.dtype(leaf.dtype).str)
        for leaf in jax.tree.leaves(table)
    )


def _deduplicate_tracks(
    tracks: Iterable[Track],
) -> tuple[tuple[Track, ...], tuple[int, ...]]:
    """Validate rows and map repeated object identities to one host track."""
    source_tracks = tuple(tracks)
    if not source_tracks:
        raise ValueError("at least one resolved Track is required")
    for index, track in enumerate(source_tracks):
        if not isinstance(track, Track):
            raise TypeError(
                f"tracks[{index}] must be a resolved Track instance"
            )
        if track.centerline is None or track.raceline is None:
            raise ValueError(
                f"tracks[{index}] must define both centerline and raceline"
            )

    unique_tracks: list[Track] = []
    identity_to_index: dict[int, int] = {}
    source_to_unique: list[int] = []
    for track in source_tracks:
        identity = id(track)
        unique_index = identity_to_index.get(identity)
        if unique_index is None:
            unique_index = len(unique_tracks)
            identity_to_index[identity] = unique_index
            unique_tracks.append(track)
        source_to_unique.append(unique_index)
    return tuple(unique_tracks), tuple(source_to_unique)


def build_indexed_core(
    config: EnvConfig,
    tracks: Iterable[Track],
    *,
    vehicle_params: VehicleParameters | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    custom_reward_fallback: BuiltinRewardMode | None = None,
    target_device: str | jax.Device | None = None,
) -> IndexedCoreBundle:
    """Build exact-shape indexed-map buckets for one environment topology.

    ``tracks`` has one entry per desired environment row. Repeated object
    identities are preprocessed once. Distinct maps are then grouped by every
    leaf shape and dtype in their complete ``CoreTables`` values, including
    reset and pair topology as well as track geometry. Each returned bucket is
    ready for the pure functions in :mod:`f1tenth_gym.jax.indexed`; callers
    slice inputs by ``source_indices`` and pass ``map_indices`` unchanged.

    Core topology, nominal parameters and randomization bounds are shared by
    all buckets because this builder accepts one ``EnvConfig`` and one optional
    episode vehicle draw. Heterogeneous configuration topologies belong in
    separate calls.
    """
    _validate_host_surface(config)
    unique_tracks, source_to_unique = _deduplicate_tracks(tracks)
    randomization = build_vehicle_randomization_params(config)
    core_config = build_core_config(
        config,
        custom_reward_fallback=custom_reward_fallback,
    )
    device = _core_device(config, target_device)

    unique_tables = tuple(
        build_core_tables(
            config,
            track,
            vehicle_params=vehicle_params,
            max_bytes=max_bytes,
        )
        for track in unique_tracks
    )
    params = build_core_params(
        config,
        unique_tables[0].track,
        vehicle_params=vehicle_params,
        custom_reward_fallback=custom_reward_fallback,
    )

    groups: dict[tuple, list[int]] = defaultdict(list)
    for unique_index, table in enumerate(unique_tables):
        groups[_core_table_signature(table)].append(unique_index)

    buckets: list[IndexedCoreBucket] = []
    for unique_indices in groups.values():
        local_index = {
            unique_index: index
            for index, unique_index in enumerate(unique_indices)
        }
        source_indices = tuple(
            source_index
            for source_index, unique_index in enumerate(source_to_unique)
            if unique_index in local_index
        )
        map_indices = np.asarray(
            [
                local_index[source_to_unique[source_index]]
                for source_index in source_indices
            ],
            dtype=np.int32,
        )
        indexed_tables = stack_core_tables(
            unique_tables[unique_index] for unique_index in unique_indices
        )
        buckets.append(
            IndexedCoreBucket(
                tables=_put_on_device(indexed_tables, device),
                tracks=tuple(unique_tracks[index] for index in unique_indices),
                source_indices=source_indices,
                map_indices=jax.device_put(map_indices, device),
            )
        )

    return IndexedCoreBundle(
        env_config=config,
        config=core_config,
        params=_put_on_device(params, device),
        randomization=_put_on_device(randomization, device),
        device=device,
        buckets=tuple(buckets),
        num_environments=len(source_to_unique),
        num_unique_tracks=len(unique_tracks),
    )


def build_core(
    config: EnvConfig,
    track: Track,
    *,
    vehicle_params: VehicleParameters | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    custom_reward_fallback: BuiltinRewardMode | None = None,
    target_device: str | jax.Device | None = None,
) -> CoreBundle:
    """Build and place one complete functional core from a resolved track.

    With no explicit episode draw, ``params`` contains the nominal vehicle and
    ``randomization`` carries device-sampleable active bounds. An explicit
    ``vehicle_params`` remains useful for host-managed episodes and is checked
    against the same bounds. Active contact and LiDAR must select the same
    available JAX device unless authoritative ``target_device`` is supplied.
    A host adapter that executes a Python custom reward after conversion may
    opt into a ``custom_reward_fallback`` whose compiled result it ignores.
    """
    _validate_host_surface(config)
    if not isinstance(track, Track):
        raise TypeError("track must be a resolved Track instance")
    randomization = build_vehicle_randomization_params(config)
    core_config = build_core_config(
        config,
        custom_reward_fallback=custom_reward_fallback,
    )
    device = _core_device(config, target_device)
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
        randomization=_put_on_device(randomization, device),
        device=device,
        track=track,
    )


__all__ = [
    "CoreBundle",
    "IndexedCoreBucket",
    "IndexedCoreBundle",
    "build_core",
    "build_core_config",
    "build_core_params",
    "build_core_tables",
    "build_indexed_core",
    "build_vehicle_randomization_params",
]
