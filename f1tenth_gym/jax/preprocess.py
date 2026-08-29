"""Host-only preprocessing from current Track geometry to fixed JAX tables."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Iterable

import jax
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

from .lidar import ScanParams
from .pairs import PairTable, make_pair_table
from .reset import ResetTable
from .track import SplineTable, TileTable, TrackTable, WallTable


@dataclass(frozen=True)
class TrackTableBucket:
    """Maps with one exact shape signature, stacked for one compiled program."""

    source_indices: tuple[int, ...]
    tables: TrackTable


@dataclass(frozen=True)
class BatchLayoutReport:
    """Memory comparison used to choose exact-shape buckets over global padding."""

    exact_bytes: int
    global_padded_bytes: int
    bucket_count: int

    @property
    def global_padding_ratio(self) -> float:
        return self.global_padded_bytes / max(self.exact_bytes, 1)


@dataclass(frozen=True)
class TrackTableSet:
    """Deduplicated host layout plus environment-to-unique-map indexes."""

    buckets: tuple[TrackTableBucket, ...]
    map_indices: np.ndarray
    unique_count: int
    layout: BatchLayoutReport


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


def build_track_table(
    track,
    vehicle_params,
    *,
    domain_randomization=None,
    contact_tile_size: float = CONTACT_TILE_SIZE,
    contact_margin: float = DEFAULT_MARGIN,
    ray_max_range: float = 30.0,
    ray_tile_size: float = RAY_TILE_SIZE,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> TrackTable:
    """Extract one current ``Track`` on the host and transfer its tables to JAX."""
    walls, _budget, contact_index = build_contact_tiles(
        track,
        vehicle_params,
        domain_randomization,
        tile_size=contact_tile_size,
        margin=contact_margin,
        max_bytes=max_bytes,
    )
    ray_walls, ray_index = build_ray_tiles(
        track,
        ray_max_range,
        tile_size=ray_tile_size,
        max_bytes=max_bytes,
    )
    if ray_walls is not walls:
        if len(ray_walls) != len(walls) or not np.array_equal(ray_walls.a, walls.a):
            raise ValueError("contact and ray preprocessing produced different walls")
    return TrackTable(
        centerline=_spline_table(track.centerline),
        raceline=_spline_table(track.raceline),
        walls=_wall_table(walls),
        contact_tiles=_contact_table(contact_index),
        ray_tiles=_ray_table(ray_index),
    )


def build_scan_params(lidar_config, track_table: TrackTable) -> ScanParams:
    """Build traced sensor leaves and reject an under-sized ray candidate table."""
    reach = float(np.asarray(track_table.ray_tiles.reach))
    requested = float(lidar_config.range_max)
    if requested > reach + 1.0e-6:
        raise ValueError(
            f"LiDAR range_max ({requested}) exceeds the ray-table reach ({reach}); "
            "rebuild the track table for at least the configured range"
        )
    return ScanParams.from_lidar_config(lidar_config)


def validate_pair_table(table: PairTable, num_agents: int) -> PairTable:
    """Validate one complete pair topology on the host before compilation."""
    if num_agents < 1:
        raise ValueError(f"num_agents must be >= 1, got {num_agents}")
    if table.num_agents != num_agents:
        raise ValueError(
            "pair table and requested agent counts must match, got "
            f"{table.num_agents} and {num_agents}"
        )
    indices = np.asarray(table.indices)
    mask = np.asarray(table.mask)
    if indices.ndim != 2 or indices.shape[1] != 2:
        raise ValueError(f"pair indices must have shape (pairs, 2), got {indices.shape}")
    if mask.shape != (indices.shape[0],):
        raise ValueError(
            f"pair mask must have shape ({indices.shape[0]},), got {mask.shape}"
        )
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(f"pair indices must have an integer dtype, got {indices.dtype}")
    if not np.issubdtype(mask.dtype, np.bool_):
        raise ValueError(f"pair mask must have a boolean dtype, got {mask.dtype}")
    if indices.size and (indices.min() < 0 or indices.max() >= num_agents):
        raise ValueError(f"all pair indexes must be in [0, {num_agents})")
    live = indices[mask]
    if live.size and np.any(live[:, 0] >= live[:, 1]):
        raise ValueError("live pairs must be canonical with first < second")
    if len({tuple(pair) for pair in live.tolist()}) != len(live):
        raise ValueError("live pairs must be unique")
    expected = {
        (left, right)
        for left in range(num_agents)
        for right in range(left + 1, num_agents)
    }
    actual = {tuple(pair) for pair in live.tolist()}
    if actual != expected:
        raise ValueError("live pairs must contain every unordered agent pair")
    if indices.shape[0] < max(len(expected), 1):
        raise ValueError(
            f"pair table capacity must be >= {max(len(expected), 1)}"
        )
    return table


def build_pair_table(num_agents: int, capacity: int | None = None) -> PairTable:
    """Build and host-validate the fixed simultaneous body-pair topology."""
    return validate_pair_table(make_pair_table(num_agents, capacity), num_agents)


def build_reset_table(
    reference_line,
    *,
    min_dist: float,
    max_dist: float,
    start_width: float | None = None,
) -> ResetTable:
    """Precompute current RL reset start/successor choices on the host."""
    if not math.isfinite(min_dist) or min_dist < 0.0:
        raise ValueError(f"min_dist must be finite and >= 0, got {min_dist}")
    if not math.isfinite(max_dist) or max_dist < min_dist:
        raise ValueError(
            f"max_dist must be finite and >= min_dist ({min_dist}), got {max_dist}"
        )
    waypoints = np.stack((reference_line.xs, reference_line.ys), axis=1).astype(
        np.float32
    )
    count = int(reference_line.n)
    if count < 2:
        raise ValueError("a reset reference line needs at least two waypoints")

    if start_width is None:
        start_indices = np.arange(count, dtype=np.int32)
    else:
        if not math.isfinite(start_width) or start_width <= 0.0:
            raise ValueError(f"start_width must be finite and > 0, got {start_width}")
        step_size = float(reference_line.length) / count
        start_count = min(count, max(1, int(start_width / step_size)))
        start_indices = np.arange(start_count, dtype=np.int32)

    successors: list[np.ndarray] = []
    guard_limit = max(4 * count, count + int(max_dist / max(reference_line.length, 1e-6) * count) + 4 * count)
    for waypoint_id in range(count):
        pointer = waypoint_id
        distance = 0.0
        first_id = None
        interval_len = None
        iterations = 0
        while distance <= max_dist:
            current = pointer % count
            previous = (pointer - 1) % count
            distance += float(np.linalg.norm(waypoints[current] - waypoints[previous]))
            if first_id is None and distance >= min_dist:
                first_id = pointer
                interval_len = 0
            if first_id is not None and distance <= max_dist:
                interval_len += 1
            pointer += 1
            iterations += 1
            if iterations > guard_limit:
                raise ValueError("reset successor search did not advance around the line")
        if first_id is None or interval_len is None:
            raise ValueError(f"no successor found for waypoint {waypoint_id}")
        # Preserve the mutable sampler's inclusive randint upper construction:
        # interval_len live waypoints produce interval_len + 1 possible offsets.
        choices = (first_id + np.arange(interval_len + 1)) % count
        successors.append(choices.astype(np.int32))

    width = max(len(values) for values in successors)
    successor_indices = np.zeros((count, width), dtype=np.int32)
    successor_counts = np.zeros((count,), dtype=np.int32)
    for index, values in enumerate(successors):
        successor_indices[index, : len(values)] = values
        successor_counts[index] = len(values)
    return ResetTable(
        waypoints=jnp.asarray(waypoints),
        start_indices=jnp.asarray(start_indices),
        successor_indices=jnp.asarray(successor_indices),
        successor_counts=jnp.asarray(successor_counts),
    )


def _shape_signature(table: TrackTable) -> tuple:
    return tuple((tuple(leaf.shape), str(leaf.dtype)) for leaf in jax.tree.leaves(table))


def bucket_track_tables(tables: Iterable[TrackTable]) -> tuple[TrackTableBucket, ...]:
    """Stack exact-shape groups; heterogeneous groups compile independently."""
    tables = tuple(tables)
    groups: dict[tuple, list[tuple[int, TrackTable]]] = defaultdict(list)
    for index, table in enumerate(tables):
        groups[_shape_signature(table)].append((index, table))
    buckets = []
    for entries in groups.values():
        source_indices, members = zip(*entries)
        stacked = jax.tree.map(lambda *values: jnp.stack(values), *members)
        buckets.append(TrackTableBucket(tuple(source_indices), stacked))
    return tuple(buckets)


def compare_batch_layout(tables: Iterable[TrackTable]) -> BatchLayoutReport:
    """Compare exact/bucket storage with one globally padded map stack."""
    tables = tuple(tables)
    if not tables:
        return BatchLayoutReport(0, 0, 0)
    leaves = [jax.tree.leaves(table) for table in tables]
    exact = sum(int(leaf.size * leaf.dtype.itemsize) for row in leaves for leaf in row)
    global_bytes = 0
    for position in range(len(leaves[0])):
        shapes = [row[position].shape for row in leaves]
        if any(len(shape) != len(shapes[0]) for shape in shapes):
            raise ValueError("track-table leaves have incompatible ranks")
        padded_shape = tuple(max(shape[axis] for shape in shapes) for axis in range(len(shapes[0])))
        itemsize = leaves[0][position].dtype.itemsize
        global_bytes += len(tables) * int(np.prod(padded_shape, dtype=np.int64)) * itemsize
    return BatchLayoutReport(exact, global_bytes, len(bucket_track_tables(tables)))


def build_track_table_set(
    tracks: Iterable,
    vehicle_params,
    **preprocess_kwargs,
) -> TrackTableSet:
    """Preprocess each shared ``Track`` object once and bucket unique maps."""
    unique_tracks = []
    identity_to_index: dict[int, int] = {}
    map_indices = []
    for track in tracks:
        identity = id(track)
        if identity not in identity_to_index:
            identity_to_index[identity] = len(unique_tracks)
            unique_tracks.append(track)
        map_indices.append(identity_to_index[identity])
    tables = tuple(
        build_track_table(track, vehicle_params, **preprocess_kwargs)
        for track in unique_tracks
    )
    return TrackTableSet(
        buckets=bucket_track_tables(tables),
        map_indices=np.asarray(map_indices, dtype=np.int32),
        unique_count=len(tables),
        layout=compare_batch_layout(tables),
    )


__all__ = [
    "BatchLayoutReport",
    "TrackTableBucket",
    "TrackTableSet",
    "bucket_track_tables",
    "build_pair_table",
    "build_scan_params",
    "build_reset_table",
    "build_track_table",
    "build_track_table_set",
    "compare_batch_layout",
    "validate_pair_table",
]
