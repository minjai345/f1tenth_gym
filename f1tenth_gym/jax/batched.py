"""Pure shared-map batching and policy views for device-native rollouts.

This module adds an environment batch axis around :mod:`.environment`.  One
compiled batch shares structural configuration and map tables while keeping
core state and traced parameters independent for every environment.  It does
not perform host conversion or framework-specific packaging.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable

import jax
import jax.numpy as jnp

from .environment import (
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
from .randomization import (
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


def _sample_and_reset(
    key: jax.Array,
    tables: CoreTables,
    config: CoreConfig,
    base_params: CoreParams,
    randomization: VehicleRandomizationParams,
) -> tuple[CoreObservation, CoreState, CoreParams]:
    params, _vehicle = sample_core_params(
        domain_randomization_key(key), base_params, randomization
    )
    observation, core = reset_core(key, tables, config, params)
    return observation, core, params


def reset_batch(
    keys: jax.Array,
    tables: CoreTables,
    config: CoreConfig,
    base_params: CoreParams,
    randomization: VehicleRandomizationParams,
) -> tuple[CoreObservation, BatchState]:
    """Sample parameters and reset ``B`` environments against shared tables."""
    _validate_base_inputs(keys, config)
    observation, core, params = jax.vmap(
        lambda key: _sample_and_reset(
            key, tables, config, base_params, randomization
        )
    )(keys)
    return observation, BatchState(core=core, params=params)


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

    def reset_one(key, environment_poses):
        params, _vehicle = sample_core_params(
            domain_randomization_key(key), base_params, randomization
        )
        observation, core = reset_core_from_poses(
            key, environment_poses, tables, config, params
        )
        return observation, core, params

    observation, core, params = jax.vmap(reset_one)(keys, poses)
    return observation, BatchState(core=core, params=params)


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

    def reset_one(key, environment_state):
        params, _vehicle = sample_core_params(
            domain_randomization_key(key), base_params, randomization
        )
        observation, core = reset_core_from_state(
            key, environment_state, tables, config, params
        )
        return observation, core, params

    observation, core, params = jax.vmap(reset_one)(keys, model_state)
    return observation, BatchState(core=core, params=params)


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
    if jnp.shape(state.params.transition.timestep) != (batch_size,):
        raise ValueError("state.params must have one leading row per environment")
    return batch_size


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
    batch_size = _validate_step_inputs(keys, state, actions, config)
    actions = jnp.asarray(actions)

    def step_one(key, core, action, params):
        return step_core(key, core, action, tables, config, params)

    observation, core, rewards, events, metrics = jax.vmap(step_one)(
        keys, state.core, actions, state.params
    )
    if reward_fn is not None:
        rewards = jax.vmap(reward_fn)(
            observation, actions, events, metrics, state.params
        )
        expected = (batch_size, config.dynamics.num_agents)
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


def _select_batch_rows(mask: jax.Array, selected: Any, other: Any) -> Any:
    """Select complete pytree rows with one scalar predicate per environment."""

    def choose(selected_leaf, other_leaf):
        selected_leaf = jnp.asarray(selected_leaf)
        other_leaf = jnp.asarray(other_leaf)
        row_mask = jnp.reshape(mask, (mask.shape[0],) + (1,) * (other_leaf.ndim - 1))
        return jnp.where(row_mask, selected_leaf, other_leaf)

    return jax.tree.map(choose, selected, other)


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
    batch_size = _validate_step_inputs(
        step_keys, state, actions, config, key_name="step_keys"
    )
    reset_batch_size = _validate_base_inputs(reset_keys, config, "reset_keys")
    if reset_batch_size != batch_size:
        raise ValueError(
            "step_keys and reset_keys must contain the same number of environments"
        )
    transition = step_batch(
        step_keys, state, actions, tables, config, reward_fn=reward_fn
    )
    reset_observation, reset_state = reset_batch(
        reset_keys, tables, config, base_params, randomization
    )
    reset = (
        transition.metrics.status.terminated
        | transition.metrics.status.truncated
    )
    next_observation = _select_batch_rows(
        reset, reset_observation, transition.observation
    )
    next_state = _select_batch_rows(reset, reset_state, transition.state)
    return AutoResetBatchStep(
        transition_observation=transition.observation,
        next_observation=next_observation,
        state=next_state,
        rewards=transition.rewards,
        events=transition.events,
        metrics=transition.metrics,
        reset=reset.astype(jnp.bool_),
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
    "flatten_joint_observation",
    "policy_observation",
    "reset_batch",
    "reset_batch_from_poses",
    "reset_batch_from_state",
    "select_ego_rewards",
    "step_batch",
    "step_batch_autoreset",
]
