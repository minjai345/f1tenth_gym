"""Exact fixed-shape buffer sizes for the tile-gather broad phase.

A fixed-shape kernel needs to know, before it runs, the most segments any single
query can see. Sampling underestimates that: 3,000 random poses once reported 59
candidates where the true maximum was 71, and the shortfall is silent.
"""

import math
from typing import NamedTuple

import numpy as np

from .walls import WallSegments

DEFAULT_TILE_SIZE = 0.5
DEFAULT_MARGIN = 1.25
# A tile grid costs rows*cols per entry, so halving tile_size quadruples it and
# map_scale=10 multiplies it by 100. Refuse before allocating rather than after.
DEFAULT_MAX_BYTES = 512 * 1024 * 1024


def _size(n):
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.1f} MB"


def _refuse_if_too_large(count, bytes_each, max_bytes, what, detail):
    """Raise before allocating, rather than after the machine has started swapping."""
    projected = int(count) * int(bytes_each)
    if projected > max_bytes:
        raise MemoryError(
            f"{what} would need {_size(projected)} for {detail}, over the "
            f"{_size(max_bytes)} cap. Use a larger tile_size (cost scales as "
            f"1/tile_size^2) or raise max_bytes."
        )
    return projected


class TileBudget(NamedTuple):
    """Sizes for a tile-gather candidate table over one track and one vehicle.

    Attributes:
        tile_size: Tile side in metres.
        query_half_extent: Half-diagonal of the query body, in metres.
        origin: World ``(x, y)`` of tile (0, 0)'s lower corner.
        tile_shape: ``(rows, cols)`` of the tile grid.
        k_tile: Exact most candidates any one tile can hand a query.
        k_tile_safe: ``k_tile`` scaled by the margin, the value to allocate.
        table_bytes: Size of an int32 ``(rows, cols, k_tile_safe)`` table.
        n_segments: Segments the budget was computed over.
    """

    tile_size: float
    query_half_extent: float
    origin: tuple
    tile_shape: tuple
    k_tile: int
    k_tile_safe: int
    table_bytes: int
    n_segments: int


def query_half_extent(length: float, width: float) -> float:
    """Half-diagonal of the collision rectangle, the radius a query box needs.

    Args:
        length: Body length in metres.
        width: Body width in metres.

    Returns:
        ``hypot(length, width) / 2``.
    """
    return 0.5 * math.hypot(float(length), float(width))


def widest_query_half_extent(vehicle_params, dr_config=None) -> float:
    """Query half-extent at the largest body domain randomization can produce.

    A budget sized for the nominal car overflows the moment DR grows it.

    Args:
        vehicle_params: Nominal ``VehicleParameters``.
        dr_config: A ``DomainRandomizationConfig``, or None for no randomization.

    Returns:
        The half-diagonal to size buffers against.
    """
    # Not widest_params(): that widens limit fields for the observation bounds and
    # leaves the body alone. The body's upper bound is the DR range's own endpoint.
    length, width = float(vehicle_params.length), float(vehicle_params.width)
    if dr_config is not None and getattr(dr_config, "enabled", False):
        for bound in (dr_config.low, dr_config.high):
            if bound is None:
                continue
            if math.isfinite(float(bound.length)):
                length = max(length, float(bound.length))
            if math.isfinite(float(bound.width)):
                width = max(width, float(bound.width))
    return query_half_extent(length, width)


def tile_budget(
    walls: WallSegments,
    query_half_extent: float,
    tile_size: float = DEFAULT_TILE_SIZE,
    margin: float = DEFAULT_MARGIN,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> TileBudget:
    """Count, exactly, the most candidates any tile can hand a single query.

    A segment is a candidate for a query centred in tile ``T`` iff its bounding box
    grown by ``query_half_extent`` overlaps ``T``. Counted by difference-array
    accumulation over every tile it reaches, so the result is a maximum, not an estimate.

    Args:
        walls: Extracted wall segments.
        query_half_extent: Half-diagonal of the querying body, in metres.
        tile_size: Tile side in metres.
        margin: Safety factor applied to ``k_tile`` to get ``k_tile_safe``.
        max_bytes: Ceiling on the accumulator this may allocate.

    Returns:
        A :class:`TileBudget`. All-zero with ``tile_shape == (0, 0)`` when there are
        no walls, which is the case for obstacle-free and synthetic tracks.

    Raises:
        ValueError: On a non-positive tile size or extent, or a margin below 1.
        MemoryError: If the tile grid would exceed ``max_bytes``.
    """
    tile_size = float(tile_size)
    query_half_extent = float(query_half_extent)
    margin = float(margin)
    if not math.isfinite(tile_size) or tile_size <= 0.0:
        raise ValueError(f"tile_size must be finite and > 0, got {tile_size}")
    if not math.isfinite(query_half_extent) or query_half_extent <= 0.0:
        raise ValueError(f"query_half_extent must be finite and > 0, got {query_half_extent}")
    if not math.isfinite(margin) or margin < 1.0:
        raise ValueError(f"margin must be finite and >= 1, got {margin}")

    if walls.is_empty:
        return TileBudget(tile_size, query_half_extent, (0.0, 0.0), (0, 0), 0, 0, 0, 0)

    a, b = walls.a.astype(np.float64), walls.b.astype(np.float64)
    lo = np.minimum(a, b) - query_half_extent
    hi = np.maximum(a, b) + query_half_extent
    origin = (float(lo[:, 0].min()), float(lo[:, 1].min()))

    col0 = np.floor((lo[:, 0] - origin[0]) / tile_size).astype(np.int64)
    row0 = np.floor((lo[:, 1] - origin[1]) / tile_size).astype(np.int64)
    col1 = np.floor((hi[:, 0] - origin[0]) / tile_size).astype(np.int64)
    row1 = np.floor((hi[:, 1] - origin[1]) / tile_size).astype(np.int64)
    rows, cols = int(row1.max()) + 1, int(col1.max()) + 1
    # int64 accumulator plus the two cumsum copies it spawns
    _refuse_if_too_large(
        (rows + 1) * (cols + 1), 24, max_bytes, "tile_budget",
        f"a {rows}x{cols} tile grid at tile_size={tile_size} m",
    )

    # +1 row/col of slack so the -1 corner of each rectangle always lands in-bounds
    diff = np.zeros((rows + 1, cols + 1), dtype=np.int64)
    np.add.at(diff, (row0, col0), 1)
    np.add.at(diff, (row0, col1 + 1), -1)
    np.add.at(diff, (row1 + 1, col0), -1)
    np.add.at(diff, (row1 + 1, col1 + 1), 1)
    counts = diff.cumsum(axis=0).cumsum(axis=1)[:rows, :cols]

    k_tile = int(counts.max())
    k_safe = math.ceil(k_tile * margin)
    return TileBudget(
        tile_size=tile_size,
        query_half_extent=query_half_extent,
        origin=origin,
        tile_shape=(rows, cols),
        k_tile=k_tile,
        k_tile_safe=k_safe,
        table_bytes=rows * cols * k_safe * 4,
        n_segments=len(walls),
    )


def track_budget(
    track,
    vehicle_params,
    dr_config=None,
    tile_size: float = DEFAULT_TILE_SIZE,
    margin: float = DEFAULT_MARGIN,
    tol_px: float | None = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> TileBudget:
    """Tile budget for a track and vehicle, sized at the widest randomized body.

    Args:
        track: A ``Track``; its walls are extracted and cached on first use.
        vehicle_params: Nominal ``VehicleParameters``.
        dr_config: A ``DomainRandomizationConfig``, or None.
        tile_size: Tile side in metres.
        margin: Safety factor on the candidate count.
        tol_px: Wall simplification tolerance, or None for the extraction default.
        max_bytes: Ceiling on the tile grid this may allocate.

    Returns:
        A :class:`TileBudget`.
    """
    from .walls import DEFAULT_TOL_PX, wall_segments

    walls = wall_segments(track, DEFAULT_TOL_PX if tol_px is None else tol_px)
    return tile_budget(
        walls,
        widest_query_half_extent(vehicle_params, dr_config),
        tile_size,
        margin,
        max_bytes,
    )
