"""Pure JAX track tables and reference-line coordinate transforms."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SplineTable:
    """Fixed-shape periodic seven-channel cubic spline data."""

    knots: jax.Array
    coefficients: jax.Array
    points: jax.Array
    knot_mask: jax.Array
    segment_mask: jax.Array
    s_interval: jax.Array
    length: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class WallTable:
    """Padded oriented wall geometry with explicit validity masks."""

    a: jax.Array
    b: jax.Array
    normals: jax.Array
    adjacency: jax.Array
    adjacency_mask: jax.Array
    lengths: jax.Array
    mask: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TileTable:
    """Fixed-width tile candidates for contact or ray queries."""

    indices: jax.Array
    mask: jax.Array
    origin: jax.Array
    tile_size: jax.Array
    reach: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TrackTable:
    """All device arrays produced from one host ``Track``."""

    centerline: SplineTable
    raceline: SplineTable
    walls: WallTable
    contact_tiles: TileTable
    ray_tiles: TileTable


def _wrap_angle(angle: jax.Array) -> jax.Array:
    return jnp.arctan2(jnp.sin(angle), jnp.cos(angle))


def evaluate_spline(table: SplineTable, arclength: jax.Array) -> jax.Array:
    """Evaluate ``[x, y, cos(yaw), sin(yaw), k, vx, ax]`` at arclength."""
    s = jnp.mod(arclength, table.length)
    segment_count = jnp.sum(table.segment_mask, dtype=jnp.int32)
    raw_segment = (
        s / (table.length + table.s_interval) * segment_count
    ).astype(jnp.int32)
    segment = jnp.mod(raw_segment, segment_count)
    delta = s - table.knots[segment]
    powers = jnp.asarray([3, 2, 1, 0], dtype=jnp.int32)
    basis = delta**powers
    return basis @ table.coefficients[:, segment, :]


def frenet_to_cartesian(
    table: SplineTable,
    frenet_pose: jax.Array,
) -> jax.Array:
    """Convert one ``[s, ey, ephi]`` pose using the selected reference line."""
    values = evaluate_spline(table, frenet_pose[0])
    yaw = jnp.arctan2(values[3], values[2])
    x = values[0] - frenet_pose[1] * jnp.sin(yaw)
    y = values[1] + frenet_pose[1] * jnp.cos(yaw)
    return jnp.stack((x, y, _wrap_angle(yaw + frenet_pose[2])))


def cartesian_to_frenet(
    table: SplineTable,
    pose: jax.Array,
) -> jax.Array:
    """Globally project one ``[x, y, yaw]`` pose onto a reference line.

    Stepping will add the current host simulator's local search window later;
    this global fixed-shape form is the reset/teleport contract and projection
    oracle for the device tables.
    """
    starts = table.points[:-1, :2]
    edges = table.points[1:, :2] - starts
    length_sq = jnp.maximum(jnp.sum(edges * edges, axis=1), 1.0e-12)
    offset = pose[:2] - starts
    fraction = jnp.clip(jnp.sum(offset * edges, axis=1) / length_sq, 0.0, 1.0)
    projections = starts + fraction[:, None] * edges
    distances_sq = jnp.sum((pose[:2] - projections) ** 2, axis=1)
    distances_sq = jnp.where(table.segment_mask, distances_sq, jnp.inf)
    segment = jnp.argmin(distances_sq)
    s = table.knots[segment] + fraction[segment] * (
        table.knots[segment + 1] - table.knots[segment]
    )
    s = jnp.mod(s, table.length)

    values = evaluate_spline(table, s)
    yaw = jnp.arctan2(values[3], values[2])
    normal = jnp.stack((-jnp.sin(yaw), jnp.cos(yaw)))
    signed = jnp.sign(jnp.dot(pose[:2] - values[:2], normal))
    ey = jnp.sqrt(distances_sq[segment]) * signed
    return jnp.stack((s, ey, _wrap_angle(pose[2] - yaw)))


def tile_candidates(table: TileTable, points: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Gather padded candidates and masks for world-space query points."""
    rows, cols = table.indices.shape[:2]
    col = jnp.clip(
        ((points[..., 0] - table.origin[0]) / table.tile_size).astype(jnp.int32),
        0,
        cols - 1,
    )
    row = jnp.clip(
        ((points[..., 1] - table.origin[1]) / table.tile_size).astype(jnp.int32),
        0,
        rows - 1,
    )
    return table.indices[row, col], table.mask[row, col]


__all__ = [
    "SplineTable",
    "TileTable",
    "TrackTable",
    "WallTable",
    "cartesian_to_frenet",
    "evaluate_spline",
    "frenet_to_cartesian",
    "tile_candidates",
]
