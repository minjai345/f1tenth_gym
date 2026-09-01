"""Pure exact-shape indexed-map batching for device-native rollouts.

The shared-map functions in :mod:`.batching` close over one ``CoreTables``
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

from .batching import (
    AutoResetBatchStep,
    BatchState,
    BatchStep,
    RewardFn,
    _reset_rows,
    _step_and_autoreset_rows,
    _step_rows,
    _validate_autoreset_inputs,
    _validate_base_inputs,
    _validate_step_inputs,
)
from .jax_core import (
    CoreConfig,
    CoreParams,
    CoreTables,
    reset_core_from_poses,
    reset_core_from_state,
)
from .dynamic_models.randomization import VehicleRandomizationParams


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
            if (
                jnp.shape(leaf) != jnp.shape(first)
                or jnp.asarray(leaf).dtype != jnp.asarray(first).dtype
            ):
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


def _table_rows(indexed: IndexedCoreTables, map_indices: jax.Array) -> CoreTables:
    """Gather selected exact-shape table rows for the common batch kernels."""
    return jax.tree.map(lambda leaf: leaf[map_indices], indexed.tables)


def reset_indexed_batch(
    keys: jax.Array,
    map_indices: jax.Array,
    indexed: IndexedCoreTables,
    config: CoreConfig,
    base_params: CoreParams,
    randomization: VehicleRandomizationParams,
):
    """Reset a batch whose rows select among exact-shape map tables.

    ``map_indices`` values must lie in ``[0, indexed.num_maps)``.  The host
    ``IndexedJaxSimulator`` validates that invariant before compilation;
    this pure traced entry point validates only shape and integer dtype.
    """
    batch_size = _validate_base_inputs(keys, config)
    indices = _validate_map_indices(map_indices, batch_size, indexed)
    return _reset_rows(
        keys, _table_rows(indexed, indices), config, base_params, randomization,
        tables_batched=True,
    )


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
    return _reset_rows(
        keys, _table_rows(indexed, indices), config, base_params, randomization,
        reset_fn=reset_core_from_poses,
        overrides=poses,
        tables_batched=True,
    )


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
    return _reset_rows(
        keys, _table_rows(indexed, indices), config, base_params, randomization,
        reset_fn=reset_core_from_state,
        overrides=model_state,
        tables_batched=True,
    )


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
    return _step_rows(
        keys, state, actions, _table_rows(indexed, indices), config, reward_fn,
        tables_batched=True,
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
    batch_size = _validate_autoreset_inputs(
        step_keys,
        reset_keys,
        state,
        actions,
        config,
    )
    indices = _validate_map_indices(map_indices, batch_size, indexed)
    return _step_and_autoreset_rows(
        step_keys, reset_keys, state, actions,
        _table_rows(indexed, indices), config,
        base_params, randomization, reward_fn,
        tables_batched=True,
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
