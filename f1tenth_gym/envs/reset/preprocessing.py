"""Host preprocessing for functional reset sampling."""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from .functional import ResetTable


def preprocess_reset(
    reference_line,
    *,
    min_dist: float,
    max_dist: float,
    start_width: float | None = None,
) -> ResetTable:
    """Precompute current RL reset start and successor choices on the host."""
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
            raise ValueError(
                f"start_width must be finite and > 0, got {start_width}"
            )
        step_size = float(reference_line.length) / count
        start_count = min(count, max(1, int(start_width / step_size)))
        start_indices = np.arange(start_count, dtype=np.int32)

    successors: list[np.ndarray] = []
    guard_limit = max(
        4 * count,
        count
        + int(max_dist / max(reference_line.length, 1e-6) * count)
        + 4 * count,
    )
    for waypoint_id in range(count):
        pointer = waypoint_id
        distance = 0.0
        first_id = None
        interval_len = None
        iterations = 0
        while distance <= max_dist:
            current = pointer % count
            previous = (pointer - 1) % count
            distance += float(
                np.linalg.norm(waypoints[current] - waypoints[previous])
            )
            if first_id is None and distance >= min_dist:
                first_id = pointer
                interval_len = 0
            if first_id is not None and distance <= max_dist:
                interval_len += 1
            pointer += 1
            iterations += 1
            if iterations > guard_limit:
                raise ValueError(
                    "reset successor search did not advance around the line"
                )
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


__all__ = ["preprocess_reset"]
