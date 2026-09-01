"""Host-only preprocessing from current Track geometry to fixed JAX tables."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from f1tenth_gym.envs.track.accel import build_for_track as build_contact_tiles
from f1tenth_gym.envs.track.budget import (
    DEFAULT_MARGIN,
    DEFAULT_MAX_BYTES,
    DEFAULT_TILE_SIZE as CONTACT_TILE_SIZE,
)
from f1tenth_gym.envs.track.ray_tiles import (
    DEFAULT_TILE_SIZE as RAY_TILE_SIZE,
    build_for_track as build_ray_tiles,
)
from f1tenth_gym.envs.track.walls import empty_walls, wall_segments

from .functional import SplineTable, TileTable, TrackTable, WallTable


def _spline_table(reference_line) -> SplineTable:
    spline = reference_line.spline
    knots = np.asarray(spline.s, dtype=np.float32)
    coefficients = np.asarray(spline.spline_c, dtype=np.float32)
    points = np.asarray(spline.points, dtype=np.float32)
    if coefficients.shape != (4, knots.size - 1, 7):
        raise ValueError(
            "reference spline must have coefficients shaped "
            f"(4, {knots.size - 1}, 7), got {coefficients.shape}"
        )
    if points.shape != (knots.size, 7):
        raise ValueError(
            f"reference spline points must have shape ({knots.size}, 7), got {points.shape}"
        )
    return SplineTable(
        knots=jnp.asarray(knots),
        coefficients=jnp.asarray(coefficients),
        points=jnp.asarray(points),
        knot_mask=jnp.ones(knots.shape, dtype=jnp.bool_),
        segment_mask=jnp.ones((knots.size - 1,), dtype=jnp.bool_),
        s_interval=jnp.asarray(spline.s_interval, dtype=jnp.float32),
        length=jnp.asarray(knots[-1], dtype=jnp.float32),
    )


def _wall_table(walls) -> WallTable:
    count = len(walls)
    if count:
        a = np.asarray(walls.a, dtype=np.float32)
        b = np.asarray(walls.b, dtype=np.float32)
        normals = np.asarray(walls.n, dtype=np.float32)
        adjacency_raw = np.asarray(walls.adj, dtype=np.int32)
        lengths = np.asarray(walls.length, dtype=np.float32)
        mask = np.ones((count,), dtype=bool)
    else:
        # Kernels can gather slot zero safely and then apply the false mask.
        a = b = normals = np.zeros((1, 2), dtype=np.float32)
        adjacency_raw = np.full((1, 2), -1, dtype=np.int32)
        lengths = np.zeros((1,), dtype=np.float32)
        mask = np.zeros((1,), dtype=bool)
    adjacency_mask = adjacency_raw >= 0
    adjacency = np.where(adjacency_mask, adjacency_raw, 0)
    return WallTable(
        a=jnp.asarray(a),
        b=jnp.asarray(b),
        normals=jnp.asarray(normals),
        adjacency=jnp.asarray(adjacency),
        adjacency_mask=jnp.asarray(adjacency_mask),
        lengths=jnp.asarray(lengths),
        mask=jnp.asarray(mask),
    )


def _contact_table(index) -> TileTable:
    raw = np.asarray(index.table, dtype=np.int32)
    mask = raw >= 0
    return TileTable(
        indices=jnp.asarray(np.where(mask, raw, 0)),
        mask=jnp.asarray(mask),
        origin=jnp.asarray(index.origin, dtype=jnp.float32),
        tile_size=jnp.asarray(index.tile_size, dtype=jnp.float32),
        reach=jnp.asarray(index.query_half_extent, dtype=jnp.float32),
    )


def _ray_table(index) -> TileTable:
    raw = np.asarray(index.table, dtype=np.int32)
    mask = raw < int(index.n_segments)
    return TileTable(
        indices=jnp.asarray(np.where(mask, raw, 0)),
        mask=jnp.asarray(mask),
        origin=jnp.asarray(index.origin, dtype=jnp.float32),
        tile_size=jnp.asarray(index.tile_size, dtype=jnp.float32),
        reach=jnp.asarray(index.max_range, dtype=jnp.float32),
    )


def _empty_tile_table(tile_size: float) -> TileTable:
    """Return one masked slot for a structurally disabled geometry query."""
    tile_size = float(tile_size)
    if not math.isfinite(tile_size) or tile_size <= 0.0:
        raise ValueError(
            f"disabled geometry tile_size must be finite and > 0, got {tile_size}"
        )
    return TileTable(
        indices=jnp.zeros((1, 1, 1), dtype=jnp.int32),
        mask=jnp.zeros((1, 1, 1), dtype=jnp.bool_),
        origin=jnp.zeros((2,), dtype=jnp.float32),
        tile_size=jnp.asarray(tile_size, dtype=jnp.float32),
        reach=jnp.asarray(0.0, dtype=jnp.float32),
    )


def preprocess_track(
    track,
    vehicle_params,
    *,
    domain_randomization=None,
    contact_tile_size: float = CONTACT_TILE_SIZE,
    contact_margin: float = DEFAULT_MARGIN,
    contact_enabled: bool = True,
    ray_max_range: float | None = 30.0,
    ray_tile_size: float = RAY_TILE_SIZE,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> TrackTable:
    """Extract one current ``Track`` on the host and transfer its tables to JAX."""
    needs_walls = contact_enabled or ray_max_range is not None
    walls = wall_segments(track) if needs_walls else empty_walls()
    if contact_enabled:
        contact_walls, _budget, contact_index = build_contact_tiles(
            track,
            vehicle_params,
            domain_randomization,
            tile_size=contact_tile_size,
            margin=contact_margin,
            max_bytes=max_bytes,
        )
        if contact_walls is not walls and (
            len(contact_walls) != len(walls)
            or not np.array_equal(contact_walls.a, walls.a)
        ):
            raise ValueError("contact preprocessing produced different walls")
        contact_table = _contact_table(contact_index)
    else:
        contact_table = _empty_tile_table(contact_tile_size)

    if ray_max_range is not None:
        ray_walls, ray_index = build_ray_tiles(
            track,
            ray_max_range,
            tile_size=ray_tile_size,
            max_bytes=max_bytes,
        )
        if ray_walls is not walls and (
            len(ray_walls) != len(walls)
            or not np.array_equal(ray_walls.a, walls.a)
        ):
            raise ValueError("ray preprocessing produced different walls")
        ray_table = _ray_table(ray_index)
    else:
        ray_table = _empty_tile_table(ray_tile_size)
    return TrackTable(
        centerline=_spline_table(track.centerline),
        raceline=_spline_table(track.raceline),
        walls=_wall_table(walls),
        contact_tiles=contact_table,
        ray_tiles=ray_table,
    )


__all__ = [
    "preprocess_track",
]
