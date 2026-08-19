"""Oriented wall segments extracted from a track's occupancy grid.

The segments carry outward normals, which is what separates them from a plain
boundary polyline: contact response needs a direction, and a raster has none.
"""

import logging
import math
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from skimage.measure import approximate_polygon, find_contours

logger = logging.getLogger(__name__)

DEFAULT_TOL_PX = 0.25

_MIN_RING_POINTS = 3
_PROBE_FRACTION = 0.7
# map_saver writes one "unknown" grey (205); an anti-aliased render writes a
# continuum. Spielberg/Austin/Monza have 234 intermediate levels, SLAM output has 1.
_MIN_SUBPIXEL_LEVELS = 16


class WallSegments(NamedTuple):
    """Oriented boundary segments in world metres, as a JAX-compatible pytree.

    Segment ``i`` runs from ``a[i]`` to ``b[i]`` with outward unit normal ``n[i]``.
    Read ``n``; do not recompute it from ``a`` and ``b`` -- endpoints are float32 at
    world magnitudes, so a 4 cm segment 77 m out loses 2e-4 rad in the subtraction.

    Attributes:
        a: (N, 2) float32 start points.
        b: (N, 2) float32 end points.
        n: (N, 2) float32 unit normals, pointing away from occupied space.
        adj: (N, 2) int32 ``[previous, next]`` segment indices; -1 where a chain ends.
        length: (N,) float32 segment lengths.
    """

    a: np.ndarray
    b: np.ndarray
    n: np.ndarray
    adj: np.ndarray
    length: np.ndarray

    def __len__(self) -> int:
        return int(self.a.shape[0])

    @property
    def is_empty(self) -> bool:
        return self.a.shape[0] == 0


def empty_walls() -> WallSegments:
    """An extraction that found no boundary, shaped so kernels need no branch."""
    return WallSegments(
        a=np.zeros((0, 2), dtype=np.float32),
        b=np.zeros((0, 2), dtype=np.float32),
        n=np.zeros((0, 2), dtype=np.float32),
        adj=np.zeros((0, 2), dtype=np.int32),
        length=np.zeros((0,), dtype=np.float32),
    )


def has_subpixel_edges(grayscale: np.ndarray | None) -> bool:
    """True when the image's edge greys are a coverage ramp rather than one flag value.

    Args:
        grayscale: Un-thresholded map greys, or None.

    Returns:
        Whether sub-pixel edge positions can be recovered by interpolation.
    """
    if grayscale is None:
        return False
    levels = np.unique(np.asarray(grayscale))
    return int(((levels > 4) & (levels < 251)).sum()) >= _MIN_SUBPIXEL_LEVELS


def _grid_to_world(rc, resolution, origin):
    """(row, col) index coordinates -> world metres, as (M, 2).

    Index ``r`` addresses the cell whose centre sits half a cell above the
    origin corner, matching ``xy_2_rc``'s ``int((y - orig_y) / resolution)``.
    """
    ox, oy, theta = origin
    gx = (rc[:, 1] + 0.5) * resolution
    gy = (rc[:, 0] + 0.5) * resolution
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    return np.stack(
        [ox + gx * cos_t - gy * sin_t, oy + gx * sin_t + gy * cos_t], axis=1
    )


def _occupied_at(points, occupied, resolution, origin):
    """Boolean per world point: is it inside an occupied cell? Off-grid is False."""
    ox, oy, theta = origin
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    dx = points[:, 0] - ox
    dy = points[:, 1] - oy
    col = np.floor((dx * cos_t + dy * sin_t) / resolution).astype(np.int64)
    row = np.floor((-dx * sin_t + dy * cos_t) / resolution).astype(np.int64)
    height, width = occupied.shape
    inside = (row >= 0) & (row < height) & (col >= 0) & (col < width)
    out = np.zeros(points.shape[0], dtype=bool)
    out[inside] = occupied[row[inside], col[inside]]
    return out


def _winding_normals(a, b):
    """Right perpendicular of each edge; consistent around a contour, sign unresolved."""
    edge = b - a
    n = np.stack([edge[:, 1], -edge[:, 0]], axis=1)
    return n / np.linalg.norm(n, axis=1, keepdims=True)


def _contour_sign(raw_world, occupied, resolution, origin):
    """Whether the winding normal points out (+1) or in (-1), voted over one contour.

    Voting on the *unsimplified* contour is what makes this independent of
    ``tol_px``: those midpoints always sit on the boundary, so the probe is valid
    however far simplification later moves a chord.
    """
    a, b = raw_world[:-1], raw_world[1:]
    keep = np.linalg.norm(b - a, axis=1) > 1e-12
    if not keep.any():
        return 1.0
    a, b = a[keep], b[keep]
    n = _winding_normals(a, b)
    probe = 0.5 * (a + b) + _PROBE_FRACTION * resolution * n
    votes_inward = int(_occupied_at(probe, occupied, resolution, origin).sum())
    return -1.0 if votes_inward * 2 > len(n) else 1.0


def _rings_to_segments(points_world, closed, index_offset):
    """One simplified contour -> (a, b, adj) with adjacency local to the contour."""
    if closed:
        pts = points_world[:-1]
        count = len(pts)
        if count < _MIN_RING_POINTS:
            return None
        a = pts
        b = np.roll(pts, -1, axis=0)
        idx = np.arange(count)
        adj = np.stack([(idx - 1) % count, (idx + 1) % count], axis=1) + index_offset
    else:
        count = len(points_world) - 1
        if count < 1:
            return None
        a = points_world[:-1]
        b = points_world[1:]
        idx = np.arange(count)
        prev = np.where(idx > 0, idx - 1 + index_offset, -1)
        nxt = np.where(idx < count - 1, idx + 1 + index_offset, -1)
        adj = np.stack([prev, nxt], axis=1)
    return a, b, adj


def _drop_repeats(points):
    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.any(np.diff(points, axis=0) != 0.0, axis=1)
    return points[keep]


def extract_walls(
    occupancy_map: np.ndarray,
    resolution: float,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    tol_px: float = DEFAULT_TOL_PX,
    grayscale: np.ndarray | None = None,
    occupied_thresh: float = 0.45,
    negate: bool = False,
) -> WallSegments:
    """Trace the occupied/free boundary into oriented segments.

    Marching squares, Douglas-Peucker at ``tol_px``, normals from the contour winding.
    An anti-aliased ``grayscale`` is traced at its sub-pixel level set instead, avoiding
    the staircase a binarised diagonal gives.

    Args:
        occupancy_map: (H, W) grid where **0.0 is an obstacle** and any other value
            is free, i.e. ``Track.occupancy_map``. Row 0 is the smallest world y.
        resolution: Metres per cell, already scaled by ``map_scale``.
        origin: ``(x, y, theta)`` of the grid's (0, 0) corner, already scaled.
        tol_px: Douglas-Peucker tolerance in pixels. Normals stay correct at any
            tolerance; raising it trades boundary accuracy for segment count.
            Marching squares bevels convex corners inward by up to sqrt(2)/4 cells
            regardless, so obstacles are never over-reported.
        grayscale: Un-thresholded map greys, or None to force the binarised grid.
        occupied_thresh: ROS occupancy threshold, used only for the sub-pixel level.
        negate: ROS negate flag, used only for the sub-pixel level.

    Returns:
        A :class:`WallSegments`, empty when the grid has no boundary at all.

    Raises:
        ValueError: On a non-2D grid, a non-positive resolution, a non-finite or
            negative ``tol_px``, or a world origin so large that float32 endpoints
            collapse.
    """
    occupancy_map = np.asarray(occupancy_map)
    if occupancy_map.ndim != 2:
        raise ValueError(f"occupancy_map must be 2-D, got shape {occupancy_map.shape}")
    resolution = float(resolution)
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise ValueError(f"resolution must be finite and > 0, got {resolution}")
    tol_px = float(tol_px)
    if not math.isfinite(tol_px) or tol_px < 0.0:
        raise ValueError(f"tol_px must be finite and >= 0, got {tol_px}")
    origin = tuple(float(v) for v in origin) + (0.0,) * (3 - len(origin))

    occupied = occupancy_map == 0.0
    if not occupied.any() or occupied.all():
        return empty_walls()

    subpixel = has_subpixel_edges(grayscale)
    if subpixel:
        field = np.asarray(grayscale, dtype=np.float64)
        if field.shape != occupied.shape:
            raise ValueError("grayscale and occupancy_map must have the same shape")
        level = 255.0 * (occupied_thresh if negate else 1.0 - occupied_thresh)
        free_value = 0.0 if negate else 255.0
    else:
        field = occupied.astype(np.float64)
        level, free_value = 0.5, 0.0
    logger.info(
        "wall extraction: %s edges, %d occupied cells",
        "sub-pixel" if subpixel else "binarised",
        int(occupied.sum()),
    )

    # Pad so a border-touching obstacle still closes into a ring. fully_connected
    # keeps a corner-touching pair one island, else its facets have no outward normal.
    padded = np.pad(field, 1, constant_values=free_value)

    a_parts, b_parts, adj_parts, n_parts = [], [], [], []
    total = 0
    for contour in find_contours(padded, level, fully_connected="high"):
        contour = contour - 1.0
        closed = bool(np.allclose(contour[0], contour[-1]))
        raw_world = _grid_to_world(_drop_repeats(contour), resolution, origin)
        if len(raw_world) < 2:
            continue
        sign = _contour_sign(raw_world, occupied, resolution, origin)

        simplified = _drop_repeats(approximate_polygon(contour, tolerance=tol_px))
        if closed and not np.allclose(simplified[0], simplified[-1]):
            simplified = np.vstack([simplified, simplified[0]])
        # A ring simplified below a triangle is a real obstacle that the tolerance
        # erased, not noise; keep its unsimplified outline rather than deleting it.
        if closed and len(simplified) - 1 < _MIN_RING_POINTS:
            simplified = _drop_repeats(contour)

        world = _grid_to_world(simplified, resolution, origin)
        built = _rings_to_segments(world, closed, total)
        if built is None:
            continue
        a, b, adj = built
        a_parts.append(a)
        b_parts.append(b)
        adj_parts.append(adj)
        n_parts.append(sign * _winding_normals(a, b))
        total += len(a)

    if total == 0:
        return empty_walls()

    a = np.concatenate(a_parts)
    b = np.concatenate(b_parts)
    adj = np.concatenate(adj_parts)
    n = np.concatenate(n_parts)
    length = np.linalg.norm(b - a, axis=1)

    a32, b32 = a.astype(np.float32), b.astype(np.float32)
    if np.any(np.all(a32 == b32, axis=1)):
        raise ValueError(
            "float32 endpoints collapsed: the world origin "
            f"{origin[:2]} is too large for this resolution. Re-centre the map."
        )
    return WallSegments(
        a=a32,
        b=b32,
        n=n.astype(np.float32),
        adj=adj.astype(np.int32),
        length=length.astype(np.float32),
    )


def wall_segments(track, tol_px: float = DEFAULT_TOL_PX) -> WallSegments:
    """Extract a track's wall segments, caching them on the track.

    Keyed on resolution, origin, tolerance and grid shape. ``track.occupancy_map``
    must be treated as immutable once this has run: an in-place edit is not detected
    and the stale extraction is returned.

    Args:
        track: A ``Track`` carrying ``occupancy_map`` and ``spec``.
        tol_px: Douglas-Peucker tolerance in pixels.

    Returns:
        A :class:`WallSegments` in world metres.
    """
    spec = track.spec
    origin = tuple(float(v) for v in spec.origin[:3])
    key = (
        float(spec.resolution),
        origin,
        float(tol_px),
        tuple(track.occupancy_map.shape),
    )
    cached = getattr(track, "_wall_segments", None)
    if cached is not None and cached[0] == key:
        return cached[1]

    walls = extract_walls(
        track.occupancy_map,
        spec.resolution,
        origin,
        tol_px,
        grayscale=getattr(track, "occupancy_grey", None),
        occupied_thresh=float(spec.occupied_thresh),
        negate=bool(spec.negate),
    )
    try:
        track._wall_segments = (key, walls)
    except (AttributeError, TypeError):
        pass
    return walls
