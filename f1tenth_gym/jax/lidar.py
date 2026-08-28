"""Pure exact LiDAR sensing over fixed-shape track and vehicle geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from .geometry import BodyParams, body_vertices, transform_pose
from .lidar_kernels import ray_segment_range
from .track import TrackTable, tile_candidates


@dataclass(frozen=True)
class ScanConfig:
    """Hashable beam topology for one compiled sensor program."""

    num_agents: int
    num_beams: int
    angle_min: float
    angle_max: float

    def __post_init__(self) -> None:
        if self.num_agents < 1:
            raise ValueError(f"num_agents must be >= 1, got {self.num_agents}")
        if self.num_beams < 1:
            raise ValueError(f"num_beams must be >= 1, got {self.num_beams}")
        if self.angle_min >= self.angle_max:
            raise ValueError(
                f"angle_min ({self.angle_min}) must be below angle_max "
                f"({self.angle_max})"
            )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ScanParams:
    """Traced clean-sensor values that may vary between environments."""

    range_max: Any
    offset_x: Any
    offset_y: Any
    offset_yaw: Any

    @classmethod
    def from_lidar_config(cls, config: Any) -> "ScanParams":
        """Copy the traced leaves from a validated host ``LiDARConfig``."""
        x, y, yaw = config.base_link_to_lidar_tf
        return cls(
            range_max=config.range_max,
            offset_x=x,
            offset_y=y,
            offset_yaw=yaw,
        )


def beam_angles(config: ScanConfig, dtype: Any = jnp.float32) -> jax.Array:
    """Return the configured ascending relative beam angles."""
    increment = (config.angle_max - config.angle_min) / max(
        config.num_beams - 1, 1
    )
    return jnp.asarray(config.angle_min, dtype=dtype) + jnp.arange(
        config.num_beams, dtype=dtype
    ) * jnp.asarray(increment, dtype=dtype)


def lidar_poses(model_state: jax.Array, params: ScanParams) -> jax.Array:
    """Transform supported model states' CoG poses to their LiDAR frames."""
    base_poses = model_state[:, jnp.asarray((0, 1, 4))]
    offset = jnp.stack(
        (
            jnp.asarray(params.offset_x, dtype=model_state.dtype),
            jnp.asarray(params.offset_y, dtype=model_state.dtype),
            jnp.asarray(params.offset_yaw, dtype=model_state.dtype),
        )
    )
    return jax.vmap(lambda pose: transform_pose(pose, offset))(base_poses)


def _wall_ranges(
    pose: jax.Array,
    track: TrackTable,
    angles: jax.Array,
    max_range: jax.Array,
) -> jax.Array:
    candidates, tile_mask = tile_candidates(track.ray_tiles, pose[:2])
    valid = tile_mask & track.walls.mask[candidates]
    seg_a = track.walls.a[candidates]
    raw_b = track.walls.b[candidates]
    # Degenerate invalid segments are rejected by the portable kernel. This
    # keeps padding safe even when candidate slot zero is also a real wall.
    seg_b = jnp.where(valid[:, None], raw_b, seg_a)
    bearings = pose[2] + angles
    directions = jnp.stack((jnp.cos(bearings), jnp.sin(bearings)), axis=1)
    return ray_segment_range(pose[:2], directions, seg_a, seg_b, max_range)


def opponent_ranges(
    pose: jax.Array,
    vertices: jax.Array,
    ego_index: jax.Array,
    angles: jax.Array,
    max_range: jax.Array,
) -> jax.Array:
    """Range to the nearest other vehicle body for every beam.

    The current host implementation culls beams by angular extent before its
    edge loop. The device path casts all fixed body edges instead: it has the
    same geometric result while avoiding data-dependent beam ranges.
    """
    seg_a = vertices
    seg_b = jnp.roll(vertices, -1, axis=1)
    opponents = jnp.arange(vertices.shape[0], dtype=jnp.int32) != ego_index
    seg_b = jnp.where(opponents[:, None, None], seg_b, seg_a)
    seg_a = seg_a.reshape((-1, 2))
    seg_b = seg_b.reshape((-1, 2))
    bearings = pose[2] + angles
    directions = jnp.stack((jnp.cos(bearings), jnp.sin(bearings)), axis=1)
    return ray_segment_range(pose[:2], directions, seg_a, seg_b, max_range)


def clean_scan(
    model_state: jax.Array,
    track: TrackTable,
    body: BodyParams,
    config: ScanConfig,
    params: ScanParams,
) -> jax.Array:
    """Compute noise-free wall ranges shortened by opponent bodies."""
    if model_state.ndim != 2 or model_state.shape[0] != config.num_agents:
        raise ValueError(
            "model_state must have shape (num_agents, state_dim), got "
            f"{model_state.shape}"
        )
    if model_state.shape[1] not in (5, 7):
        raise ValueError(
            f"state_dim must be 5 (KS) or 7 (ST), got {model_state.shape[1]}"
        )
    poses = lidar_poses(model_state, params)
    base_poses = model_state[:, jnp.asarray((0, 1, 4))]
    vertices = jax.vmap(lambda pose: body_vertices(pose, body))(base_poses)
    angles = beam_angles(config, model_state.dtype)
    max_range = jnp.asarray(params.range_max, dtype=model_state.dtype)
    walls = jax.vmap(lambda pose: _wall_ranges(pose, track, angles, max_range))(
        poses
    )
    agents = jnp.arange(config.num_agents, dtype=jnp.int32)
    opponents = jax.vmap(
        lambda pose, index: opponent_ranges(
            pose, vertices, index, angles, max_range
        )
    )(poses, agents)
    return jnp.minimum(walls, opponents)


__all__ = [
    "ScanConfig",
    "ScanParams",
    "beam_angles",
    "clean_scan",
    "lidar_poses",
    "opponent_ranges",
]
