"""Host-side construction for the functional JAX simulator.

The public entry point is :class:`JaxSimulator`.  Static topology, fixed-shape
tables, and traced parameters remain separate internally, but callers no
longer need to assemble those implementation stages themselves.
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

from .contact.functional import WallContactConfig
from .action_jax import LongitudinalControlMode, SteeringControlMode
from .dynamic_models.jax_core import DynamicsConfig, DynamicsRuntimeParams
from .dynamic_models.jax import (
    DynamicsParams,
    kinematic_single_track,
    single_track,
)
from .jax_core import (
    CoreConfig,
    CoreObservation,
    CoreParams,
    CoreState,
    CoreTables,
    reset_core,
    reset_core_from_poses,
    reset_core_from_state,
    step_core,
)
from .episode import (
    BuiltinRewardMode,
    EpisodeConfig,
    EpisodeParams,
    TerminationMode,
)
from .contact.geometry import BodyParams
from .integrators_jax import euler_step, rk4_step
from .indexed_batching import IndexedCoreTables, stack_core_tables
from .lidar.functional import ScanConfig, ScanParams
from .contact.pairs import PairContactConfig, PairTable, make_pair_table
from .dynamic_models.randomization import (
    ACTIVE_VEHICLE_FIELDS,
    ActiveVehicleParams,
    VehicleRandomizationParams,
)
from .reset.functional import ResetSamplingConfig
from .reset.preprocessing import preprocess_reset
from .track.functional import FrenetProjectionConfig, TrackTable
from .track.preprocessing import preprocess_track


_RESET = jax.jit(reset_core, static_argnums=2)
_RESET_FROM_POSES = jax.jit(reset_core_from_poses, static_argnums=3)
_RESET_FROM_STATE = jax.jit(reset_core_from_state, static_argnums=3)
_STEP = jax.jit(step_core, static_argnums=4)


class JaxSimulator:
    """Configured functional simulator with device-ready core components.

    ``CoreConfig``, ``CoreTables``, and ``CoreParams`` deliberately remain
    distinct because JAX treats static topology and traced values differently.
    This host object hides their construction without hiding that execution
    contract from code that calls the pure reset and step functions.
    """

    env_config: EnvConfig
    config: CoreConfig
    tables: CoreTables
    params: CoreParams
    randomization: VehicleRandomizationParams
    device: jax.Device
    track: Track
    effective_vehicle_params: VehicleParameters
    space_vehicle_params: VehicleParameters

    def __init__(
        self,
        config: EnvConfig,
        track: Track,
        *,
        device: str | jax.Device | None = None,
        vehicle_params: VehicleParameters | None = None,
        max_table_bytes: int = DEFAULT_MAX_BYTES,
        _custom_reward_fallback: BuiltinRewardMode | None = None,
    ) -> None:
        _validate_host_surface(config)
        if not isinstance(track, Track):
            raise TypeError("track must be a resolved Track instance")

        effective_vehicle = _effective_vehicle(config, vehicle_params)
        randomization = _make_randomization(config, effective_vehicle)
        space_vehicle = _space_vehicle_params(config, effective_vehicle)
        core_config = _make_config(config, _custom_reward_fallback)
        core_device = _core_device(config, device)
        tables = _make_tables(
            config,
            track,
            effective_vehicle,
            max_bytes=max_table_bytes,
        )
        params = _make_params(config, tables.track, effective_vehicle)

        self.env_config = config
        self.config = core_config
        self.tables = _put_on_device(tables, core_device)
        self.params = _put_on_device(params, core_device)
        self.randomization = _put_on_device(randomization, core_device)
        self.device = core_device
        self.track = track
        self.effective_vehicle_params = effective_vehicle
        self.space_vehicle_params = space_vehicle
        self._parameter_track_table = tables.track

    def params_for_vehicle(self, vehicle: VehicleParameters) -> CoreParams:
        """Build device parameters for one validated episode vehicle draw."""
        if not isinstance(vehicle, VehicleParameters):
            raise TypeError("vehicle must be a VehicleParameters instance")
        _validate_vehicle_draw(
            self.env_config,
            vehicle,
            effective_vehicle=self.effective_vehicle_params,
        )
        params = _make_params(
            self.env_config,
            self._parameter_track_table,
            vehicle,
        )
        return _put_on_device(params, self.device)

    def reset(
        self,
        key: jax.Array,
        *,
        params: CoreParams | None = None,
    ) -> tuple[CoreObservation, CoreState]:
        """Reset from the configured sampler using an explicit JAX key."""
        return _RESET(
            key,
            self.tables,
            self.config,
            self.params if params is None else params,
        )

    def reset_from_poses(
        self,
        key: jax.Array,
        poses: jax.Array,
        *,
        params: CoreParams | None = None,
    ) -> tuple[CoreObservation, CoreState]:
        """Reset from explicit CoG ``[x, y, yaw]`` poses."""
        return _RESET_FROM_POSES(
            key,
            poses,
            self.tables,
            self.config,
            self.params if params is None else params,
        )

    def reset_from_state(
        self,
        key: jax.Array,
        state: jax.Array,
        *,
        params: CoreParams | None = None,
    ) -> tuple[CoreObservation, CoreState]:
        """Reset from complete native KS/ST state rows."""
        return _RESET_FROM_STATE(
            key,
            state,
            self.tables,
            self.config,
            self.params if params is None else params,
        )

    def step(
        self,
        key: jax.Array,
        state: CoreState,
        actions: jax.Array,
        *,
        params: CoreParams | None = None,
    ):
        """Advance one compiled functional transition."""
        return _STEP(
            key,
            state,
            actions,
            self.tables,
            self.config,
            self.params if params is None else params,
        )


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


class IndexedJaxSimulator:
    """Exact-shape indexed-map buckets sharing one simulator topology."""

    env_config: EnvConfig
    config: CoreConfig
    params: CoreParams
    randomization: VehicleRandomizationParams
    device: jax.Device
    tracks: tuple[Track, ...]
    buckets: tuple[IndexedCoreBucket, ...]
    num_environments: int
    num_unique_tracks: int
    effective_vehicle_params: VehicleParameters
    space_vehicle_params: VehicleParameters

    def __init__(
        self,
        config: EnvConfig,
        tracks: Iterable[Track],
        *,
        device: str | jax.Device | None = None,
        vehicle_params: VehicleParameters | None = None,
        max_table_bytes: int = DEFAULT_MAX_BYTES,
        _custom_reward_fallback: BuiltinRewardMode | None = None,
    ) -> None:
        _validate_host_surface(config)
        unique_tracks, source_to_unique = _deduplicate_tracks(tracks)
        effective_vehicle = _effective_vehicle(config, vehicle_params)
        randomization = _make_randomization(config, effective_vehicle)
        space_vehicle = _space_vehicle_params(config, effective_vehicle)
        core_config = _make_config(config, _custom_reward_fallback)
        core_device = _core_device(config, device)
        unique_tables = tuple(
            _make_tables(
                config,
                track,
                effective_vehicle,
                max_bytes=max_table_bytes,
            )
            for track in unique_tracks
        )
        params = _make_params(config, unique_tables[0].track, effective_vehicle)
        buckets = _indexed_buckets(
            unique_tracks,
            source_to_unique,
            unique_tables,
            core_device,
        )

        self.env_config = config
        self.config = core_config
        self.params = _put_on_device(params, core_device)
        self.randomization = _put_on_device(randomization, core_device)
        self.device = core_device
        self.tracks = unique_tracks
        self.buckets = buckets
        self.num_environments = len(source_to_unique)
        self.num_unique_tracks = len(unique_tracks)
        self.effective_vehicle_params = effective_vehicle
        self.space_vehicle_params = space_vehicle
        self._parameter_track_table = unique_tables[0].track

    def params_for_vehicle(self, vehicle: VehicleParameters) -> CoreParams:
        """Build device parameters for one validated episode vehicle draw."""
        if not isinstance(vehicle, VehicleParameters):
            raise TypeError("vehicle must be a VehicleParameters instance")
        _validate_vehicle_draw(
            self.env_config,
            vehicle,
            effective_vehicle=self.effective_vehicle_params,
        )
        params = _make_params(
            self.env_config,
            self._parameter_track_table,
            vehicle,
        )
        return _put_on_device(params, self.device)


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


def _make_randomization(
    config: EnvConfig,
    effective_vehicle: VehicleParameters,
) -> VehicleRandomizationParams:
    """Create the active vehicle and per-episode randomization bounds.

    Disabled randomization and enabled-but-constant bounds collapse to one
    nominal vehicle with a false enable flag. A varying configuration keeps
    all twenty active bounds so dynamics and collision geometry are sampled
    from one correlated vehicle draw on device.
    """
    if tuple(PARAMETER_ORDER[:20]) != ACTIVE_VEHICLE_FIELDS:
        raise RuntimeError(
            "the functional active-vehicle fields no longer match the first "
            "20 entries of the VehicleParameters ABI"
        )

    nominal_host = effective_vehicle
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
    provide an explicit built-in fallback whose result they discard. Keeping
    this opt-in makes standalone construction reject callbacks by default
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
) -> tuple[ResetSamplingConfig, ReferenceLine, float, float, float | None]:
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
        ResetSamplingConfig(
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


def _make_config(
    config: EnvConfig,
    custom_reward_fallback: BuiltinRewardMode | None = None,
) -> CoreConfig:
    """Translate validated host values into one static core configuration."""
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


def _make_tables(
    config: EnvConfig,
    track: Track,
    vehicle_params: VehicleParameters,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> CoreTables:
    """Preprocess one resolved track for enabled reset and geometry modes."""
    _reset, reference, minimum, maximum, start_width = _reset_settings(config)
    line = track.centerline if reference is ReferenceLine.CENTERLINE else track.raceline
    contact_enabled = config.collision_check is CollisionCheckMode.SEGMENT_CONTACT
    scan_enabled = config.lidar_config.enabled
    track_table = preprocess_track(
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
        reset=preprocess_reset(
            line,
            min_dist=minimum,
            max_dist=maximum,
            start_width=start_width,
        ),
        track=track_table,
        pairs=(
            make_pair_table(config.num_agents)
            if contact_enabled
            else _disabled_pair_table(config.num_agents)
        ),
    )


def _effective_vehicle(
    config: EnvConfig,
    vehicle_params: VehicleParameters | None,
) -> VehicleParameters:
    """Return the nominal vehicle or validate one explicit episode draw."""
    effective = config.params if vehicle_params is None else vehicle_params
    if not isinstance(effective, VehicleParameters):
        raise TypeError("vehicle_params must be a VehicleParameters instance")
    _validate_active_values(effective, prefix="vehicle_params")
    return effective


def _space_vehicle_params(
    config: EnvConfig,
    effective_vehicle: VehicleParameters,
) -> VehicleParameters:
    """Return one host vehicle whose limits cover every runtime vehicle."""
    randomization = config.domain_randomization_config
    if not randomization.randomized_fields():
        return effective_vehicle
    low = randomization.low
    high = randomization.high
    if low is None or high is None:  # guarded by host config
        raise ValueError("enabled domain randomization requires low/high bounds")

    widest = randomization.widest_params(effective_vehicle)
    widen_at_low = ("v_min", "s_min", "sv_min", "lf", "lr")
    widen_at_high = ("v_max", "s_max", "sv_max", "a_max")
    changes = {
        name: min(
            getattr(effective_vehicle, name),
            getattr(low, name),
            getattr(high, name),
        )
        for name in widen_at_low
    }
    changes.update(
        {
            name: max(
                getattr(effective_vehicle, name),
                getattr(low, name),
                getattr(high, name),
            )
            for name in widen_at_high
        }
    )
    return widest.with_updates(**changes)


def _validate_vehicle_draw(
    config: EnvConfig,
    params: VehicleParameters,
    *,
    effective_vehicle: VehicleParameters,
) -> None:
    _validate_active_values(params, prefix="vehicle_params")
    randomization = config.domain_randomization_config
    if not randomization.randomized_fields():
        for name in ACTIVE_VEHICLE_FIELDS:
            value = getattr(params, name)
            expected = getattr(effective_vehicle, name)
            if value != expected:
                raise ValueError(
                    f"vehicle_params.{name}={value} does not match this "
                    f"simulator's fixed vehicle envelope ({expected}); "
                    "construct a new JaxSimulator or IndexedJaxSimulator "
                    "with vehicle_params=... to rebuild fixed tables"
                )
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
        effective = getattr(effective_vehicle, name)
        low = min(effective, getattr(low_params, name))
        high = max(effective, getattr(high_params, name))
        if not low <= value <= high:
            raise ValueError(
                f"vehicle_params.{name}={value} lies outside the runtime "
                "envelope formed by the effective vehicle and DR bounds "
                f"[{low}, {high}]; construct a new JaxSimulator or "
                "IndexedJaxSimulator with a compatible domain-randomization "
                "envelope to rebuild fixed tables"
            )


def _make_params(
    config: EnvConfig,
    track_table: TrackTable,
    vehicle_params: VehicleParameters,
) -> CoreParams:
    """Translate one validated vehicle into traced core parameters.

    Continuous production leaves are normalized to float32, counters to int32
    and enable flags to booleans independently of process-wide JAX x64 mode.
    """
    if not isinstance(track_table, TrackTable):
        raise TypeError("track_table must be a TrackTable instance")
    control = config.control_config
    contact = config.contact_config
    lidar = config.lidar_config
    simulation = config.simulation_config
    termination = config.termination_config
    reward = config.reward_config
    if lidar.enabled:
        reach = float(np.asarray(track_table.ray_tiles.reach))
        requested = float(lidar.range_max)
        if requested > reach + 1.0e-6:
            raise ValueError(
                f"LiDAR range_max ({requested}) exceeds the ray-table reach "
                f"({reach}); preprocess the track for at least the configured "
                "range"
            )
    scan = ScanParams.from_lidar_config(lidar)
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
        dynamics=DynamicsRuntimeParams(
            vehicle=dynamics,
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
        episode=EpisodeParams(
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


def _core_device(config: EnvConfig, device: Any = None):
    if device is not None:
        if isinstance(device, str):
            return _platform_device(device)
        if isinstance(device, jax.Device):
            return device
        raise TypeError(
            "device must be a JAX platform string, jax.Device, or None"
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


def _indexed_buckets(
    unique_tracks: tuple[Track, ...],
    source_to_unique: tuple[int, ...],
    unique_tables: tuple[CoreTables, ...],
    device: jax.Device,
) -> tuple[IndexedCoreBucket, ...]:
    """Group complete tables by exact shape and build their routing metadata."""
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
    return tuple(buckets)


__all__ = [
    "IndexedCoreBucket",
    "IndexedJaxSimulator",
    "JaxSimulator",
]
