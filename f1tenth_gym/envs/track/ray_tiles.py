"""Per-tile candidate lists for ray casting, sized by the sensor's range.

The contact tile index answers "what could this 0.33 m body touch"; a 30 m ray needs
a different question and a different table. Both cache on the ``Track``.
"""

import math
from dataclasses import dataclass

import numpy as np

from .budget import DEFAULT_MAX_BYTES, _refuse_if_too_large
from .walls import WallSegments, wall_segments

DEFAULT_TILE_SIZE = 5.0


@dataclass(frozen=True, eq=False)
class RayTileIndex:
    """Segments reachable from inside each tile, padded to a fixed width.

    Attributes:
        table: (rows, cols, k) int32 segment indices. Padding is ``len(walls)``, one
            past the end, which addresses a degenerate segment the consumer appends;
            the intersection test rejects it on a zero denominator, so no mask is needed.
        origin: World ``(x, y)`` of tile (0, 0)'s lower corner.
        tile_size: Tile side in metres.
        max_range: Range the table was built for. Casting further can miss segments.
        n_segments: Segments covered, so a consumer can size its padded array.
    """

    table: np.ndarray
    origin: tuple
    tile_size: float
    max_range: float
    n_segments: int

    @property
    def k(self) -> int:
        return int(self.table.shape[2])

    @property
    def is_empty(self) -> bool:
        return self.n_segments == 0


def point_segment_distance(point, seg_a, seg_b):
    """Distance from one point to every segment.

    Args:
        point: (2,) world position.
        seg_a: (S, 2) segment starts.
        seg_b: (S, 2) segment ends.

    Returns:
        (S,) distances in metres.
    """
    edge = seg_b - seg_a
    length_sq = np.maximum((edge * edge).sum(axis=1), 1e-12)
    t = np.clip(((point - seg_a) * edge).sum(axis=1) / length_sq, 0.0, 1.0)
    closest = seg_a + t[:, None] * edge
    return np.hypot(closest[:, 0] - point[0], closest[:, 1] - point[1])


def build_ray_tiles(
    walls: WallSegments,
    max_range: float,
    tile_size: float = DEFAULT_TILE_SIZE,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> RayTileIndex:
    """List, per tile, every segment a ray starting inside it could reach.

    Measured from the tile centre with the half-diagonal added to the radius, so the
    list is a superset for every start point in the tile. Over-including is safe;
    under-including silently loses hits.

    Args:
        walls: Extracted wall segments.
        max_range: Longest ray the table must serve, in metres.
        tile_size: Tile side in metres.
        max_bytes: Ceiling on the table this may allocate.

    Returns:
        A :class:`RayTileIndex`, empty when there are no walls.

    Raises:
        ValueError: On a non-positive range or tile size.
        MemoryError: If the table would exceed ``max_bytes``.
    """
    max_range = float(max_range)
    tile_size = float(tile_size)
    if not math.isfinite(max_range) or max_range <= 0.0:
        raise ValueError(f"max_range must be finite and > 0, got {max_range}")
    if not math.isfinite(tile_size) or tile_size <= 0.0:
        raise ValueError(f"tile_size must be finite and > 0, got {tile_size}")

    if walls.is_empty:
        return RayTileIndex(np.zeros((1, 1, 1), np.int32), (0.0, 0.0),
                            tile_size, max_range, 0)

    seg_a = walls.a.astype(np.float64)
    seg_b = walls.b.astype(np.float64)
    lo = np.minimum(seg_a, seg_b).min(axis=0)
    hi = np.maximum(seg_a, seg_b).max(axis=0)
    cols = max(1, int(math.ceil((hi[0] - lo[0]) / tile_size)))
    rows = max(1, int(math.ceil((hi[1] - lo[1]) / tile_size)))
    radius = max_range + 0.5 * math.hypot(tile_size, tile_size)

    reachable = []
    widest = 0
    for row in range(rows):
        cy = lo[1] + (row + 0.5) * tile_size
        for col in range(cols):
            cx = lo[0] + (col + 0.5) * tile_size
            found = np.flatnonzero(
                point_segment_distance(np.array([cx, cy]), seg_a, seg_b) <= radius)
            reachable.append(found)
            widest = max(widest, found.size)

    k = max(1, widest)
    _refuse_if_too_large(
        rows * cols, k * 4, max_bytes, "build_ray_tiles",
        f"a {rows}x{cols}x{k} candidate table at max_range={max_range} m",
    )
    table = np.full((rows * cols, k), len(walls), dtype=np.int32)
    for i, found in enumerate(reachable):
        table[i, :found.size] = found
    return RayTileIndex(
        table=table.reshape(rows, cols, k),
        origin=(float(lo[0]), float(lo[1])),
        tile_size=tile_size,
        max_range=max_range,
        n_segments=len(walls),
    )


def candidates(point, index: RayTileIndex) -> np.ndarray:
    """Segment indices a ray from ``point`` could reach; the numpy reference gather."""
    rows, cols = index.table.shape[0], index.table.shape[1]
    col = int(np.clip((point[0] - index.origin[0]) // index.tile_size, 0, cols - 1))
    row = int(np.clip((point[1] - index.origin[1]) // index.tile_size, 0, rows - 1))
    return index.table[row, col]


def build_for_track(
    track,
    max_range: float,
    tile_size: float = DEFAULT_TILE_SIZE,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> tuple:
    """Extract walls and build the ray index, caching both on the track.

    Args:
        track: A ``Track``.
        max_range: Longest ray the table must serve.
        tile_size: Tile side in metres.
        max_bytes: Ceiling on the table.

    Returns:
        ``(walls, index)``.
    """
    walls = wall_segments(track)
    key = (float(max_range), float(tile_size), int(max_bytes))
    cached = getattr(track, "_ray_tiles", None)
    if cached is not None and cached[0] == key and cached[1] is walls:
        return walls, cached[2]

    index = build_ray_tiles(walls, max_range, tile_size, max_bytes)
    try:
        track._ray_tiles = (key, walls, index)
    except (AttributeError, TypeError):
        pass
    return walls, index
