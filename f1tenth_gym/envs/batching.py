"""Pure shared-map batching and policy views for device-native rollouts.

This module adds an environment batch axis around
:mod:`f1tenth_gym.envs.jax_core`. One compiled batch shares structural
configuration and map tables while keeping core state and traced parameters
independent for every environment. It does not perform host conversion or
framework-specific packaging.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable

import jax
import jax.numpy as jnp

from .action_jax import LongitudinalControlMode, SteeringControlMode
from .jax_core import (
    CoreConfig,
    CoreMetrics,
    CoreObservation,
    CoreParams,
    CoreState,
    CoreTables,
    reset_core,
    reset_core_from_poses,
    reset_core_from_state,
    step_core,
)
from .episode import EpisodeEvents
from .dynamic_models.randomization import (
    VehicleRandomizationParams,
    domain_randomization_key,
    sample_core_params,
)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BatchState:
    """Device carry for a batch of environments sharing one map topology."""

    core: CoreState
    params: CoreParams


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BatchStep:
    """One raw batched transition without implicit reset."""

    observation: CoreObservation
    state: BatchState
    rewards: jax.Array
    events: EpisodeEvents
    metrics: CoreMetrics


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AutoResetBatchStep:
    """One transition plus selectively chosen next-episode carries.

    ``transition_observation`` and all transition outputs describe the terminal
    step.  ``next_observation`` and ``state`` contain a fresh reset only in rows
    for which ``reset`` is true.
    """

    transition_observation: CoreObservation
    next_observation: CoreObservation
    state: BatchState
    rewards: jax.Array
    events: EpisodeEvents
    metrics: CoreMetrics
    reset: jax.Array


class PolicyField(IntEnum):
    """Static channel groups available to a device-native policy."""

    KINEMATIC_STATE = 0
    DYNAMIC_STATE = 1
    NATIVE_STATE = 2
    SCAN = 3
    COLLISION = 4
    FRENET = 5
    LAP_TIME = 6
    LAP_COUNT = 7
    SIM_TIME = 8


@dataclass(frozen=True)
class PolicyLayout:
    """Ordered, static policy channel selection.

    Each selected group is concatenated in the given order.  The result keeps
    decentralized environment and agent axes as ``(batch, agents, features)``.
    """

    fields: tuple[PolicyField, ...]

    def __post_init__(self) -> None:
        fields = tuple(self.fields)
        if not fields:
            raise ValueError("a policy layout requires at least one field")
        for field in fields:
            if not isinstance(field, PolicyField):
                raise TypeError("policy layout fields must be PolicyField values")
        object.__setattr__(self, "fields", fields)


RewardFn = Callable[
    [CoreObservation, jax.Array, EpisodeEvents, CoreMetrics, CoreParams],
    jax.Array,
]


def _batched_core_params(
    state_or_params: BatchState | CoreParams,
    config: CoreConfig,
) -> tuple[CoreParams, int]:
    """Resolve and validate one row of core parameters per environment."""
    if not isinstance(config, CoreConfig):
        raise TypeError("config must be a CoreConfig")
    if isinstance(state_or_params, BatchState):
        params = state_or_params.params
    elif isinstance(state_or_params, CoreParams):
        params = state_or_params
    else:
        raise TypeError("state_or_params must be a BatchState or CoreParams")

    timestep = jnp.asarray(params.dynamics.timestep)
    if timestep.ndim != 1 or timestep.shape[0] < 1:
        raise ValueError(
            "params must have one leading row per environment; "
            f"got timestep shape {timestep.shape}"
        )
    batch_size = timestep.shape[0]
    dynamics = params.dynamics.vehicle
    for name in (
        "s_min",
        "s_max",
        "sv_min",
        "sv_max",
        "a_max",
        "v_min",
        "v_max",
    ):
        shape = jnp.shape(getattr(dynamics, name))
        if shape != (batch_size,):
            raise ValueError(
                "state_or_params dynamics leaves must have one leading row "
                f"per environment; {name} has shape {shape}, expected "
                f"{(batch_size,)}"
            )

    if isinstance(state_or_params, BatchState):
        expected_model = (
            batch_size,
            config.dynamics.num_agents,
            config.dynamics.state_dim,
        )
        if state_or_params.core.dynamics.model.shape != expected_model:
            raise ValueError(
                "state.core.dynamics.model must have shape "
                f"{expected_model}, got "
                f"{state_or_params.core.dynamics.model.shape}"
            )
    return params, batch_size


def batch_action_bounds(
    state_or_params: BatchState | CoreParams,
    config: CoreConfig,
) -> tuple[jax.Array, jax.Array]:
    """Return active physical action bounds as ``(B, A, 2)`` arrays.

    Vehicle parameters vary across environments and are shared by all agents
    in one environment. The selected controller modes determine whether each
    action column represents a setpoint or a direct model effort.
    """
    params, batch_size = _batched_core_params(state_or_params, config)
    dynamics = params.dynamics.vehicle

    if config.dynamics.steering_mode is SteeringControlMode.TARGET_ANGLE:
        steering_low = dynamics.s_min
        steering_high = dynamics.s_max
    elif config.dynamics.steering_mode is SteeringControlMode.STEERING_RATE:
        steering_low = dynamics.sv_min
        steering_high = dynamics.sv_max
    else:
        raise ValueError(
            "unsupported steering control mode: "
            f"{config.dynamics.steering_mode!r}"
        )

    if config.dynamics.longitudinal_mode is LongitudinalControlMode.TARGET_SPEED:
        longitudinal_low = dynamics.v_min
        longitudinal_high = dynamics.v_max
    elif config.dynamics.longitudinal_mode is LongitudinalControlMode.ACCELERATION:
        longitudinal_low = -dynamics.a_max
        longitudinal_high = dynamics.a_max
    else:
        raise ValueError(
            "unsupported longitudinal control mode: "
            f"{config.dynamics.longitudinal_mode!r}"
        )

    low = jnp.stack((steering_low, longitudinal_low), axis=-1)
    high = jnp.stack((steering_high, longitudinal_high), axis=-1)
    shape = (batch_size, config.dynamics.num_agents, 2)
    return (
        jnp.broadcast_to(low[:, None, :], shape),
        jnp.broadcast_to(high[:, None, :], shape),
    )


def scale_normalized_actions(
    actions: jax.Array,
    state_or_params: BatchState | CoreParams,
    config: CoreConfig,
) -> jax.Array:
    """Affinely map normalized ``(B, A, 2)`` actions to physical commands.

    The function deliberately does not clip. A policy should impose its own
    bounded distribution (for example, a tanh transform); out-of-range inputs
    are extrapolated by the same affine map so policy bugs remain visible.
    """
    low, high = batch_action_bounds(state_or_params, config)
    actions = jnp.asarray(actions, dtype=low.dtype)
    if actions.shape != low.shape:
        raise ValueError(
            f"actions must have shape {low.shape}, got {actions.shape}"
        )
    weight = (actions + 1.0) * 0.5
    return (1.0 - weight) * low + weight * high


def _key_batch_size(keys: jax.Array, name: str) -> int:
    keys = jnp.asarray(keys)
    if keys.ndim == 0:
        raise ValueError(f"{name} must have a leading environment batch axis")
    try:
        key_data = jax.random.key_data(keys)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain JAX PRNG keys") from error
    if key_data.ndim != 2:
        raise ValueError(f"{name} must contain one PRNG key per environment")
    if key_data.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one PRNG key")
    return key_data.shape[0]


def _validate_base_inputs(
    keys: jax.Array,
    config: CoreConfig,
    name: str = "keys",
) -> int:
    if not isinstance(config, CoreConfig):
        raise TypeError("config must be a CoreConfig")
    return _key_batch_size(keys, name)


def _reset_rows(
    keys: jax.Array,
    tables: CoreTables,
    config: CoreConfig,
    base_params: CoreParams,
    randomization: VehicleRandomizationParams,
    *,
    reset_fn: Callable[..., tuple[CoreObservation, CoreState]] | None = None,
    overrides: jax.Array | None = None,
    tables_batched: bool = False,
) -> tuple[CoreObservation, BatchState]:
    """Sample parameters and reset rows against shared or selected tables."""
    def reset_one(key, override, environment_tables):
        params, _vehicle = sample_core_params(
            domain_randomization_key(key), base_params, randomization
        )
        if reset_fn is None:
            observation, core = reset_core(
                key, environment_tables, config, params
            )
        else:
            observation, core = reset_fn(
                key, override, environment_tables, config, params
            )
        return observation, core, params

    override_rows = keys if overrides is None else overrides
    observation, core, params = jax.vmap(
        reset_one,
        in_axes=(0, 0, 0 if tables_batched else None),
    )(keys, override_rows, tables)
    return observation, BatchState(core=core, params=params)


def reset_batch(
    keys: jax.Array,
    tables: CoreTables,
    config: CoreConfig,
    base_params: CoreParams,
    randomization: VehicleRandomizationParams,
) -> tuple[CoreObservation, BatchState]:
    """Sample parameters and reset ``B`` environments against shared tables."""
    _validate_base_inputs(keys, config)
    return _reset_rows(keys, tables, config, base_params, randomization)


def reset_batch_from_poses(
    keys: jax.Array,
    poses: jax.Array,
    tables: CoreTables,
    config: CoreConfig,
    base_params: CoreParams,
    randomization: VehicleRandomizationParams,
) -> tuple[CoreObservation, BatchState]:
    """Reset a batch from explicit CoG poses with per-environment parameters."""
    batch_size = _validate_base_inputs(keys, config)
    poses = jnp.asarray(poses)
    expected = (batch_size, config.dynamics.num_agents, 3)
    if poses.shape != expected:
        raise ValueError(f"poses must have shape {expected}, got {poses.shape}")

    return _reset_rows(
        keys, tables, config, base_params, randomization,
        reset_fn=reset_core_from_poses,
        overrides=poses,
    )


def reset_batch_from_state(
    keys: jax.Array,
    model_state: jax.Array,
    tables: CoreTables,
    config: CoreConfig,
    base_params: CoreParams,
    randomization: VehicleRandomizationParams,
) -> tuple[CoreObservation, BatchState]:
    """Reset from complete batched native KS/ST state rows."""
    batch_size = _validate_base_inputs(keys, config)
    model_state = jnp.asarray(model_state)
    expected = (
        batch_size,
        config.dynamics.num_agents,
        config.dynamics.state_dim,
    )
    if model_state.shape != expected:
        raise ValueError(
            f"model_state must have shape {expected}, got {model_state.shape}"
        )

    return _reset_rows(
        keys, tables, config, base_params, randomization,
        reset_fn=reset_core_from_state,
        overrides=model_state,
    )


def _validate_step_inputs(
    keys: jax.Array,
    state: BatchState,
    actions: jax.Array,
    config: CoreConfig,
    *,
    key_name: str = "keys",
) -> int:
    batch_size = _validate_base_inputs(keys, config, key_name)
    if not isinstance(state, BatchState):
        raise TypeError("state must be a BatchState")
    actions = jnp.asarray(actions)
    expected = (batch_size, config.dynamics.num_agents, 2)
    if actions.shape != expected:
        raise ValueError(f"actions must have shape {expected}, got {actions.shape}")
    model = state.core.dynamics.model
    expected_model = (
        batch_size,
        config.dynamics.num_agents,
        config.dynamics.state_dim,
    )
    if model.shape != expected_model:
        raise ValueError(
            "state.core.dynamics.model must have shape "
            f"{expected_model}, got {model.shape}"
        )
    if jnp.shape(state.params.dynamics.timestep) != (batch_size,):
        raise ValueError("state.params must have one leading row per environment")
    return batch_size


def _step_rows(
    keys: jax.Array,
    state: BatchState,
    actions: jax.Array,
    tables: CoreTables,
    config: CoreConfig,
    reward_fn: RewardFn | None,
    *,
    tables_batched: bool = False,
) -> BatchStep:
    """Execute rows and package one common shared/indexed step result."""
    actions = jnp.asarray(actions)
    table_axis = 0 if tables_batched else None

    def step_one(key, core, action, params, environment_tables):
        return step_core(
            key, core, action, environment_tables, config, params
        )

    observation, core, rewards, events, metrics = jax.vmap(
        step_one,
        in_axes=(0, 0, 0, 0, table_axis),
    )(keys, state.core, actions, state.params, tables)
    if reward_fn is not None:
        rewards = jax.vmap(reward_fn)(
            observation, actions, events, metrics, state.params
        )
        expected = (actions.shape[0], config.dynamics.num_agents)
        if rewards.shape != expected:
            raise ValueError(
                f"reward_fn must return shape {(config.dynamics.num_agents,)}, "
                f"giving batched shape {expected}; got {rewards.shape}"
            )
    return BatchStep(
        observation=observation,
        state=BatchState(core=core, params=state.params),
        rewards=rewards,
        events=events,
        metrics=metrics,
    )


def step_batch(
    keys: jax.Array,
    state: BatchState,
    actions: jax.Array,
    tables: CoreTables,
    config: CoreConfig,
    reward_fn: RewardFn | None = None,
) -> BatchStep:
    """Step a batch once without freezing or resetting terminal rows.

    A custom ``reward_fn`` is vmapped over environments and receives
    ``(observation, actions, events, metrics, active_params)``.  It must be a
    pure JAX callable returning one reward per agent.
    """
    _validate_step_inputs(keys, state, actions, config)
    return _step_rows(keys, state, actions, tables, config, reward_fn)


def _select_batch_rows(mask: jax.Array, selected: Any, other: Any) -> Any:
    """Select complete pytree rows with one scalar predicate per environment."""

    def choose(selected_leaf, other_leaf):
        selected_leaf = jnp.asarray(selected_leaf)
        other_leaf = jnp.asarray(other_leaf)
        row_mask = jnp.reshape(mask, (mask.shape[0],) + (1,) * (other_leaf.ndim - 1))
        return jnp.where(row_mask, selected_leaf, other_leaf)

    return jax.tree.map(choose, selected, other)


def _validate_autoreset_inputs(
    step_keys: jax.Array,
    reset_keys: jax.Array,
    state: BatchState,
    actions: jax.Array,
    config: CoreConfig,
) -> int:
    """Validate common step/reset batch axes once for autoreset callers."""
    batch_size = _validate_step_inputs(
        step_keys, state, actions, config, key_name="step_keys"
    )
    reset_batch_size = _validate_base_inputs(reset_keys, config, "reset_keys")
    if reset_batch_size != batch_size:
        raise ValueError(
            "step_keys and reset_keys must contain the same number of environments"
        )
    return batch_size


def _step_and_autoreset_rows(
    step_keys: jax.Array,
    reset_keys: jax.Array,
    state: BatchState,
    actions: jax.Array,
    tables: CoreTables,
    config: CoreConfig,
    base_params: CoreParams,
    randomization: VehicleRandomizationParams,
    reward_fn: RewardFn | None,
    *,
    tables_batched: bool = False,
) -> AutoResetBatchStep:
    """Run common shared/indexed transition and selective reset composition."""
    transition = _step_rows(
        step_keys, state, actions, tables, config, reward_fn,
        tables_batched=tables_batched,
    )
    reset_observation, reset_state = _reset_rows(
        reset_keys, tables, config, base_params, randomization,
        tables_batched=tables_batched,
    )
    reset = (
        transition.metrics.status.terminated
        | transition.metrics.status.truncated
    )
    return AutoResetBatchStep(
        transition_observation=transition.observation,
        next_observation=_select_batch_rows(
            reset, reset_observation, transition.observation
        ),
        state=_select_batch_rows(reset, reset_state, transition.state),
        rewards=transition.rewards,
        events=transition.events,
        metrics=transition.metrics,
        reset=reset.astype(jnp.bool_),
    )


def step_batch_autoreset(
    step_keys: jax.Array,
    reset_keys: jax.Array,
    state: BatchState,
    actions: jax.Array,
    tables: CoreTables,
    config: CoreConfig,
    base_params: CoreParams,
    randomization: VehicleRandomizationParams,
    reward_fn: RewardFn | None = None,
) -> AutoResetBatchStep:
    """Step and reset whole environment rows that terminate or truncate.

    Reset candidates use independent explicit keys.  Terminal transition
    outputs are retained while only ``next_observation``, state, and sampled
    parameters select the reset candidate for completed rows.
    """
    _validate_autoreset_inputs(
        step_keys,
        reset_keys,
        state,
        actions,
        config,
    )
    return _step_and_autoreset_rows(
        step_keys, reset_keys, state, actions, tables, config,
        base_params, randomization, reward_fn,
    )


def select_ego_rewards(
    rewards: jax.Array,
    config_or_index: CoreConfig | int = 0,
) -> jax.Array:
    """Select one ego reward per environment from ``(..., agents)`` rewards."""
    rewards = jnp.asarray(rewards)
    if rewards.ndim < 1:
        raise ValueError("rewards must have a trailing agent axis")
    if isinstance(config_or_index, CoreConfig):
        ego_index = config_or_index.episode.ego_index
    else:
        ego_index = int(config_or_index)
    if not 0 <= ego_index < rewards.shape[-1]:
        raise ValueError(
            f"ego index must be in [0, {rewards.shape[-1]}), got {ego_index}"
        )
    return rewards[..., ego_index]


def _policy_channel(
    observation: CoreObservation,
    field: PolicyField,
) -> jax.Array:
    if field is PolicyField.KINEMATIC_STATE:
        return observation.standard_state[..., :5]
    if field is PolicyField.DYNAMIC_STATE:
        return observation.standard_state
    if field is PolicyField.NATIVE_STATE:
        return observation.state
    if field is PolicyField.SCAN:
        return observation.scans
    if field is PolicyField.COLLISION:
        return observation.collisions[..., None]
    if field is PolicyField.FRENET:
        return observation.frenet
    if field is PolicyField.LAP_TIME:
        return observation.lap_times[..., None]
    if field is PolicyField.LAP_COUNT:
        return observation.lap_counts[..., None]
    if field is PolicyField.SIM_TIME:
        batch_size, num_agents = observation.state.shape[:2]
        time = jnp.reshape(observation.sim_time, (batch_size, 1, 1))
        return jnp.broadcast_to(time, (batch_size, num_agents, 1))
    raise ValueError(f"unsupported policy field: {field!r}")


def policy_observation(
    observation: CoreObservation,
    config: CoreConfig,
    layout: PolicyLayout,
) -> jax.Array:
    """Pack ordered decentralized policy features as ``(B, A, F)``."""
    if not isinstance(config, CoreConfig):
        raise TypeError("config must be a CoreConfig")
    if not isinstance(layout, PolicyLayout):
        raise TypeError("layout must be a PolicyLayout")
    if observation.state.ndim != 3:
        raise ValueError("observation must carry leading batch and agent axes")
    if observation.state.shape[1] != config.dynamics.num_agents:
        raise ValueError("observation and config must use the same agent count")
    if PolicyField.SCAN in layout.fields and not config.scan_enabled:
        raise ValueError("SCAN requested but sensing is disabled")
    if PolicyField.FRENET in layout.fields and not config.frenet_enabled:
        raise ValueError("FRENET requested but the Frenet frame is disabled")
    channels = tuple(_policy_channel(observation, field) for field in layout.fields)
    expected_prefix = observation.state.shape[:2]
    for channel in channels:
        if channel.ndim != 3 or channel.shape[:2] != expected_prefix:
            raise ValueError("policy observation fields have incompatible batch axes")
    return jnp.concatenate(channels, axis=-1)


def flatten_joint_observation(observation: jax.Array) -> jax.Array:
    """Flatten ``(B, A, F)`` into an explicit centralized ``(B, A*F)`` view."""
    observation = jnp.asarray(observation)
    if observation.ndim != 3:
        raise ValueError(
            "joint policy observation must have shape (batch, agents, features)"
        )
    return jnp.reshape(
        observation,
        (observation.shape[0], observation.shape[1] * observation.shape[2]),
    )


__all__ = [
    "AutoResetBatchStep",
    "BatchState",
    "BatchStep",
    "PolicyField",
    "PolicyLayout",
    "batch_action_bounds",
    "flatten_joint_observation",
    "policy_observation",
    "reset_batch",
    "reset_batch_from_poses",
    "reset_batch_from_state",
    "scale_normalized_actions",
    "select_ego_rewards",
    "step_batch",
    "step_batch_autoreset",
]
