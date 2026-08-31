"""Tile-gather broad phase: which wall segments a query box could possibly touch.

One coalesced fetch of a fixed number of candidates per query. Measured against a
BVH on an RTX 3080 at 256/1024/4096 queries: 0.381/0.221/0.137 us against
12.38/2.96/0.732, so the tree is not implemented here.
"""

import math
from dataclasses import dataclass

import jax
import numpy as np

from .budget import (
    DEFAULT_MARGIN,
    DEFAULT_MAX_BYTES,
    DEFAULT_TILE_SIZE,
    TileBudget,
    _refuse_if_too_large,
    tile_budget,
)
from .walls import WallSegments


@dataclass(frozen=True, eq=False)
class TileIndex:
    """Per-tile candidate lists, padded to a fixed width.

    Registered as a pytree whose only leaf is ``table``: the geometry is aux_data, so
    a ``tree_map`` cannot silently rescale the grid out from under the table.

    Attributes:
        table: (rows, cols, k) int32 segment indices, -1 padding.
        origin: World ``(x, y)`` of tile (0, 0)'s lower corner.
        tile_size: Tile side in metres.
        query_half_extent: Conservative query radius the table was built for; a
            body reaching farther from the lookup point can miss candidates.
    """

    table: np.ndarray
    origin: tuple
    tile_size: float
    query_half_extent: float

    @property
    def k(self) -> int:
        return int(self.table.shape[2])

    @property
    def is_empty(self) -> bool:
        return self.table.size == 0 or int(self.table.max()) < 0


jax.tree_util.register_pytree_node(
    TileIndex,
    lambda idx: ((idx.table,), (idx.origin, idx.tile_size, idx.query_half_extent)),
    lambda aux, children: TileIndex(children[0], *aux),
)


def empty_index(tile_size: float = DEFAULT_TILE_SIZE, qh: float = 1.0) -> TileIndex:
    """An index over no walls, shaped so a gather still returns all -1."""
    return TileIndex(
        table=np.full((1, 1, 1), -1, dtype=np.int32),
        origin=(0.0, 0.0),
        tile_size=float(tile_size),
        query_half_extent=float(qh),
    )


def tile_coords(points, index: TileIndex):
    """World points -> clamped ``(row, col)`` tile indices.

    Clamping is what makes an off-map query safe: it lands in an edge tile whose
    candidate list is still a superset of what a box there could reach.

    Args:
        points: (M, 2) world coordinates.
        index: The tile index to look up in.

    Returns:
        ``(rows, cols)`` int arrays of length M.
    """
    rows, cols = index.table.shape[0], index.table.shape[1]
    col = np.floor((points[:, 0] - index.origin[0]) / index.tile_size).astype(np.int64)
    row = np.floor((points[:, 1] - index.origin[1]) / index.tile_size).astype(np.int64)
    return np.clip(row, 0, rows - 1), np.clip(col, 0, cols - 1)


def gather(points, index: TileIndex) -> np.ndarray:
    """Candidate segment indices for each query centre.

    The numpy reference for the jitted gather; one fancy-index, no control flow.

    Args:
        points: (M, 2) query centres in world metres.
        index: The tile index.

    Returns:
        (M, k) int32 segment indices, -1 where a slot is unused.
    """
    row, col = tile_coords(np.asarray(points, dtype=np.float64), index)
    return index.table[row, col]


def build_tile_index(
    walls: WallSegments, budget: TileBudget, max_bytes: int = DEFAULT_MAX_BYTES
) -> TileIndex:
    """Bin every segment into the tiles a query centred there could reach.

    Args:
        walls: Extracted wall segments.
        budget: Sizes from :func:`~f1tenth_gym.envs.track.budget.tile_budget`,
            whose origin and tile size this reuses so the two cannot disagree.
        max_bytes: Ceiling on the candidate table this may allocate.

    Returns:
        A :class:`TileIndex`.

    Raises:
        ValueError: If the budget was computed over a different segment count, or
            if a tile needs more slots than the budget allocated.
        MemoryError: If the candidate table would exceed ``max_bytes``.
    """
    if walls.is_empty or budget.k_tile == 0:
        return empty_index(budget.tile_size, budget.query_half_extent)
    if budget.n_segments != len(walls):
        raise ValueError(
            f"budget covers {budget.n_segments} segments, walls has {len(walls)}"
        )

    qh, tile = budget.query_half_extent, budget.tile_size
    ox, oy = budget.origin
    rows, cols = budget.tile_shape
    k = budget.k_tile_safe
    _refuse_if_too_large(
        rows * cols, k * 4, max_bytes, "build_tile_index",
        f"a {rows}x{cols}x{k} candidate table at tile_size={tile} m",
    )

    a, b = walls.a.astype(np.float64), walls.b.astype(np.float64)
    lo, hi = np.minimum(a, b) - qh, np.maximum(a, b) + qh
    # Equal segment counts do not mean equal geometry; the origin is derived from
    # these same extents, so a mismatch means the budget is for other walls.
    want = (float(lo[:, 0].min()), float(lo[:, 1].min()))
    if not (math.isclose(want[0], ox, abs_tol=1e-6) and math.isclose(want[1], oy, abs_tol=1e-6)):
        raise ValueError(f"budget origin {budget.origin} does not match these walls {want}")
    col0 = np.floor((lo[:, 0] - ox) / tile).astype(np.int64)
    row0 = np.floor((lo[:, 1] - oy) / tile).astype(np.int64)
    col1 = np.floor((hi[:, 0] - ox) / tile).astype(np.int64)
    row1 = np.floor((hi[:, 1] - oy) / tile).astype(np.int64)

    span_r, span_c = row1 - row0 + 1, col1 - col0 + 1
    per_segment = span_r * span_c
    # ~10 live int64 arrays plus an argsort over one entry per (tile, segment) pair;
    # this scales with fill density, not with the table the guard above covered.
    pairs = int(per_segment.sum())
    _refuse_if_too_large(
        pairs, 80, max_bytes, "build_tile_index",
        f"{pairs} (tile, segment) incidence pairs at tile_size={tile} m",
    )
    seg = np.repeat(np.arange(len(walls), dtype=np.int64), per_segment)
    within = np.arange(per_segment.sum()) - np.repeat(
        np.cumsum(per_segment) - per_segment, per_segment
    )
    span_c_rep = np.repeat(span_c, per_segment)
    r = np.repeat(row0, per_segment) + within // span_c_rep
    c = np.repeat(col0, per_segment) + within % span_c_rep

    flat = r * cols + c
    order = np.argsort(flat, kind="stable")
    flat, seg = flat[order], seg[order]
    slot = np.arange(len(flat)) - np.searchsorted(flat, flat, side="left")
    if slot.size and int(slot.max()) >= k:
        raise ValueError(
            f"a tile needs {int(slot.max()) + 1} slots but the budget allocated {k}"
        )

    table = np.full((rows * cols, k), -1, dtype=np.int32)
    table[flat, slot] = seg.astype(np.int32)
    return TileIndex(
        table=table.reshape(rows, cols, k),
        origin=(float(ox), float(oy)),
        tile_size=float(tile),
        query_half_extent=float(qh),
    )


def build_for_track(
    track,
    vehicle_params,
    dr_config=None,
    tile_size: float = DEFAULT_TILE_SIZE,
    margin: float = DEFAULT_MARGIN,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> tuple:
    """Extract walls, size the budget and build the index, caching on the track.

    Args:
        track: A ``Track``.
        vehicle_params: Nominal ``VehicleParameters``.
        dr_config: A ``DomainRandomizationConfig``, or None.
        tile_size: Tile side in metres.
        margin: Safety factor on the candidate count.
        max_bytes: Ceiling on what the budget and table may allocate.

    Returns:
        ``(walls, budget, index)``.
    """
    from .budget import widest_query_half_extent
    from .walls import wall_segments

    walls = wall_segments(track)
    qh = widest_query_half_extent(vehicle_params, dr_config)
    key = (qh, float(tile_size), float(margin), int(max_bytes))
    cached = getattr(track, "_tile_index", None)
    if cached is not None and cached[0] == key and cached[1] is walls:
        return (walls,) + cached[2]

    budget = tile_budget(walls, qh, tile_size, margin, max_bytes)
    index = build_tile_index(walls, budget, max_bytes)
    try:
        track._tile_index = (key, walls, (budget, index))
    except (AttributeError, TypeError):
        pass
    return walls, budget, index


def brute_force_candidates(walls: WallSegments, point, qh: float) -> np.ndarray:
    """Every segment whose expanded box contains ``point``; the gather's oracle.

    Args:
        walls: Extracted wall segments.
        point: One world ``(x, y)``.
        qh: Query half-extent in metres.

    Returns:
        Sorted int array of segment indices.
    """
    a, b = walls.a.astype(np.float64), walls.b.astype(np.float64)
    lo, hi = np.minimum(a, b) - qh, np.maximum(a, b) + qh
    inside = (
        (lo[:, 0] <= point[0])
        & (hi[:, 0] >= point[0])
        & (lo[:, 1] <= point[1])
        & (hi[:, 1] >= point[1])
    )
    return np.flatnonzero(inside)


def max_reach_tiles(qh: float, tile_size: float) -> int:
    """Tiles a single query box can straddle, for sanity-checking a tile size."""
    return int(math.ceil(2.0 * qh / tile_size) + 1) ** 2
