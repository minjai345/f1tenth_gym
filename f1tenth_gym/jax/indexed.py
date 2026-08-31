"""Pure exact-shape indexed-map batching for device-native rollouts.

The shared-map functions in :mod:`.batched` close over one ``CoreTables``
value.  This module keeps the same state and result contracts while selecting
one complete, exact-shape table row for every environment.  Different table
shapes still belong in separate host-orchestrated compilation buckets.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from .batched import (
    AutoResetBatchStep,
    BatchState,
    BatchStep,
    RewardFn,
    _select_batch_rows,
    _validate_base_inputs,
    _validate_step_inputs,
)
from .environment import (
    CoreConfig,
    CoreParams,
    CoreTables,
    reset_core,
    reset_core_from_poses,
    reset_core_from_state,
    step_core,
)
from .randomization import (
    VehicleRandomizationParams,
    domain_randomization_key,
    sample_core_params,
)


@partial(
    jax.tree_util.register_dataclass,
    data_fields=("tables",),
    meta_fields=("num_maps",),
)
@dataclass(frozen=True)
class IndexedCoreTables:
    """A stack of complete ``CoreTables`` values with identical leaf shapes."""

    tables: CoreTables
    num_maps: int

    def __post_init__(self) -> None:
        count = int(self.num_maps)
        if count < 1:
            raise ValueError(f"num_maps must be >= 1, got {count}")
        if not isinstance(self.tables, CoreTables):
            raise TypeError("tables must be a CoreTables instance")
        for leaf in jax.tree.leaves(self.tables):
            if jnp.ndim(leaf) < 1 or jnp.shape(leaf)[0] != count:
                raise ValueError(
                    "every indexed table leaf must have a leading map axis "
                    f"of length {count}"
                )
        object.__setattr__(self, "num_maps", count)


def stack_core_tables(tables: Any) -> IndexedCoreTables:
    """Stack non-empty, exact-shape complete table values along a map axis."""
    tables = tuple(tables)
    if not tables:
        raise ValueError("at least one CoreTables value is required")
    if any(not isinstance(table, CoreTables) for table in tables):
        raise TypeError("every table must be a CoreTables instance")

    first_structure = jax.tree.structure(tables[0])
    first_leaves = jax.tree.leaves(tables[0])
    for table in tables[1:]:
        if jax.tree.structure(table) != first_structure:
            raise ValueError("core tables must have identical pytree structure")
        leaves = jax.tree.leaves(table)
        for first, leaf in zip(first_leaves, leaves, strict=True):
            if jnp.shape(leaf) != jnp.shape(first) or jnp.asarray(leaf).dtype != jnp.asarray(first).dtype:
                raise ValueError(
                    "core tables must have identical leaf shapes and dtypes"
                )

    stacked = jax.tree.map(
        lambda *values: jnp.stack(tuple(jnp.asarray(value) for value in values)),
        *tables,
    )
    return IndexedCoreTables(tables=stacked, num_maps=len(tables))


def _validate_map_indices(
    map_indices: jax.Array,
    batch_size: int,
    indexed: IndexedCoreTables,
) -> jax.Array:
    if not isinstance(indexed, IndexedCoreTables):
        raise TypeError("indexed must be an IndexedCoreTables instance")
    indices = jnp.asarray(map_indices)
    if indices.shape != (batch_size,):
        raise ValueError(
            f"map_indices must have shape {(batch_size,)}, got {indices.shape}"
        )
    if not jnp.issubdtype(indices.dtype, jnp.integer):
        raise TypeError("map_indices must have an integer dtype")
    return indices


def _table_row(indexed: IndexedCoreTables, map_index: jax.Array) -> CoreTables:
    return jax.tree.map(lambda leaf: leaf[map_index], indexed.tables)


def _sample_and_reset(
    key: jax.Array,
    map_index: jax.Array,
    indexed: IndexedCoreTables,
    config: CoreConfig,
    base_params: CoreParams,
    randomization: VehicleRandomizationParams,
):
    params, _vehicle = sample_core_params(
        domain_randomization_key(key), base_params, randomization
    )
    observation, core = reset_core(
        key, _table_row(indexed, map_index), config, params
    )
    return observation, core, params


def reset_indexed_batch(
    keys: jax.Array,
    map_indices: jax.Array,
    indexed: IndexedCoreTables,
    config: CoreConfig,
    base_params: CoreParams,
    randomization: VehicleRandomizationParams,
):
    """Reset a batch whose rows select among exact-shape map tables."""
    batch_size = _validate_base_inputs(keys, config)
    indices = _validate_map_indices(map_indices, batch_size, indexed)
    observation, core, params = jax.vmap(
        lambda key, map_index: _sample_and_reset(
            key,
            map_index,
            indexed,
            config,
            base_params,
            randomization,
        )
    )(keys, indices)
    return observation, BatchState(core=core, params=params)


def reset_indexed_batch_from_poses(
    keys: jax.Array,
    map_indices: jax.Array,
    poses: jax.Array,
    indexed: IndexedCoreTables,
    config: CoreConfig,
    base_params: CoreParams,
    randomization: VehicleRandomizationParams,
):
    """Reset indexed map rows from explicit CoG poses."""
    batch_size = _validate_base_inputs(keys, config)
    indices = _validate_map_indices(map_indices, batch_size, indexed)
    poses = jnp.asarray(poses)
    expected = (batch_size, config.dynamics.num_agents, 3)
    if poses.shape != expected:
        raise ValueError(f"poses must have shape {expected}, got {poses.shape}")

    def reset_one(key, map_index, environment_poses):
        params, _vehicle = sample_core_params(
            domain_randomization_key(key), base_params, randomization
        )
        observation, core = reset_core_from_poses(
            key,
            environment_poses,
            _table_row(indexed, map_index),
            config,
            params,
        )
        return observation, core, params

    observation, core, params = jax.vmap(reset_one)(keys, indices, poses)
    return observation, BatchState(core=core, params=params)


def reset_indexed_batch_from_state(
    keys: jax.Array,
    map_indices: jax.Array,
    model_state: jax.Array,
    indexed: IndexedCoreTables,
    config: CoreConfig,
    base_params: CoreParams,
    randomization: VehicleRandomizationParams,
):
    """Reset indexed map rows from complete native KS/ST state."""
    batch_size = _validate_base_inputs(keys, config)
    indices = _validate_map_indices(map_indices, batch_size, indexed)
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

    def reset_one(key, map_index, environment_state):
        params, _vehicle = sample_core_params(
            domain_randomization_key(key), base_params, randomization
        )
        observation, core = reset_core_from_state(
            key,
            environment_state,
            _table_row(indexed, map_index),
            config,
            params,
        )
        return observation, core, params

    observation, core, params = jax.vmap(reset_one)(
        keys, indices, model_state
    )
    return observation, BatchState(core=core, params=params)


def step_indexed_batch(
    keys: jax.Array,
    map_indices: jax.Array,
    state: BatchState,
    actions: jax.Array,
    indexed: IndexedCoreTables,
    config: CoreConfig,
    reward_fn: RewardFn | None = None,
) -> BatchStep:
    """Step indexed map rows once without reset or freezing."""
    batch_size = _validate_step_inputs(keys, state, actions, config)
    indices = _validate_map_indices(map_indices, batch_size, indexed)
    actions = jnp.asarray(actions)

    def step_one(key, map_index, core, action, params):
        return step_core(
            key,
            core,
            action,
            _table_row(indexed, map_index),
            config,
            params,
        )

    observation, core, rewards, events, metrics = jax.vmap(step_one)(
        keys, indices, state.core, actions, state.params
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


def step_indexed_batch_autoreset(
    step_keys: jax.Array,
    reset_keys: jax.Array,
    map_indices: jax.Array,
    state: BatchState,
    actions: jax.Array,
    indexed: IndexedCoreTables,
    config: CoreConfig,
    base_params: CoreParams,
    randomization: VehicleRandomizationParams,
    reward_fn: RewardFn | None = None,
) -> AutoResetBatchStep:
    """Step indexed rows and selectively reset them on the same selected map."""
    batch_size = _validate_step_inputs(
        step_keys, state, actions, config, key_name="step_keys"
    )
    reset_size = _validate_base_inputs(reset_keys, config, "reset_keys")
    if reset_size != batch_size:
        raise ValueError(
            "step_keys and reset_keys must contain the same number of environments"
        )
    indices = _validate_map_indices(map_indices, batch_size, indexed)
    transition = step_indexed_batch(
        step_keys,
        indices,
        state,
        actions,
        indexed,
        config,
        reward_fn=reward_fn,
    )
    reset_observation, reset_state = reset_indexed_batch(
        reset_keys,
        indices,
        indexed,
        config,
        base_params,
        randomization,
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


__all__ = [
    "IndexedCoreTables",
    "reset_indexed_batch",
    "reset_indexed_batch_from_poses",
    "reset_indexed_batch_from_state",
    "stack_core_tables",
    "step_indexed_batch",
    "step_indexed_batch_autoreset",
]
