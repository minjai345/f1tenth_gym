"""Pure key-driven reset sampling over host-preprocessed reference lines."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .core import DynamicsConfig, DynamicsState, make_dynamics_state


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ResetTable:
    """Fixed-shape waypoint and successor candidates for one reset policy."""

    waypoints: jax.Array
    start_indices: jax.Array
    successor_indices: jax.Array
    successor_counts: jax.Array


@dataclass(frozen=True)
class ResetConfig:
    """Structural reset choices shared by every environment in one executable."""

    num_agents: int
    move_laterally: bool = False
    shuffle: bool = False

    def __post_init__(self) -> None:
        if self.num_agents < 1:
            raise ValueError(f"num_agents must be >= 1, got {self.num_agents}")


def sample_reset_poses(
    key: jax.Array,
    table: ResetTable,
    config: ResetConfig,
) -> jax.Array:
    """Sample current RL grid/all-track reset semantics with explicit keys."""
    keys = jax.random.split(key, config.num_agents + 3)
    start_slot = jax.random.randint(keys[0], (), 0, table.start_indices.shape[0])
    start = table.start_indices[start_slot]
    if config.move_laterally:
        lateral_sign = jnp.where(jax.random.bernoulli(keys[1]), 1.0, -1.0)
    else:
        lateral_sign = jnp.asarray(0.0, dtype=table.waypoints.dtype)
    waypoint_count = table.waypoints.shape[0]

    def body(current: jax.Array, inputs: tuple[jax.Array, jax.Array]):
        agent_index, successor_key = inputs
        xy = table.waypoints[current]
        following = table.waypoints[(current + 1) % waypoint_count]
        yaw = jnp.arctan2(following[1] - xy[1], following[0] - xy[0])
        if config.num_agents > 1:
            lateral = (
                lateral_sign
                * jnp.where(agent_index % 2 == 0, 1.0, -1.0)
                / config.num_agents
            )
            xy = xy + lateral * jnp.stack((-jnp.sin(yaw), jnp.cos(yaw)))
        pose = jnp.concatenate((xy, jnp.asarray([yaw], dtype=xy.dtype)))

        count = table.successor_counts[current]
        slot = jax.random.randint(successor_key, (), 0, count)
        next_waypoint = table.successor_indices[current, slot]
        return next_waypoint, pose

    _last, poses = jax.lax.scan(
        body,
        start,
        (jnp.arange(config.num_agents), keys[2 : 2 + config.num_agents]),
    )
    if config.shuffle:
        poses = poses[jax.random.permutation(keys[-1], config.num_agents)]
    return poses


def model_state_from_poses(
    poses: jax.Array,
    dynamics_config: DynamicsConfig,
) -> jax.Array:
    """Build zero-motion native KS/ST rows from CoG ``[x, y, yaw]`` poses."""
    poses = jnp.asarray(poses)
    expected = (dynamics_config.num_agents, 3)
    if poses.shape != expected:
        raise ValueError(f"poses must have shape {expected}, got {poses.shape}")
    model = jnp.zeros(
        (dynamics_config.num_agents, dynamics_config.state_dim),
        dtype=poses.dtype,
    )
    model = model.at[:, :2].set(poses[:, :2])
    return model.at[:, 4].set(poses[:, 2])


def reset_dynamics_state(
    key: jax.Array,
    table: ResetTable,
    reset_config: ResetConfig,
    dynamics_config: DynamicsConfig,
) -> tuple[jax.Array, DynamicsState]:
    """Sample poses and initialize zero-velocity native KS/ST state."""
    if reset_config.num_agents != dynamics_config.num_agents:
        raise ValueError("reset and dynamics configs must use the same num_agents")
    poses = sample_reset_poses(key, table, reset_config)
    model = model_state_from_poses(poses, dynamics_config)
    return poses, make_dynamics_state(model, dynamics_config)


__all__ = [
    "ResetConfig",
    "ResetTable",
    "model_state_from_poses",
    "reset_dynamics_state",
    "sample_reset_poses",
]
