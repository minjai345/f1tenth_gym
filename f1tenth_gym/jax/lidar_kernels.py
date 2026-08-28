"""Portable analytic ray-segment scanning kernels."""

import jax.numpy as jnp


PARALLEL_EPS = 1e-12


def ray_segment_range(origin, direction, seg_a, seg_b, max_range):
    """Distance along each ray to its nearest valid segment intersection."""
    edge = seg_b - seg_a
    offset = seg_a[None, :, :] - origin[None, None, :]

    denominator = (
        direction[:, None, 0] * edge[None, :, 1]
        - direction[:, None, 1] * edge[None, :, 0]
    )
    ray_fraction = (
        offset[..., 0] * edge[None, :, 1]
        - offset[..., 1] * edge[None, :, 0]
    )
    segment_fraction = (
        offset[..., 0] * direction[:, None, 1]
        - offset[..., 1] * direction[:, None, 0]
    )

    safe = jnp.where(jnp.abs(denominator) > PARALLEL_EPS, denominator, 1.0)
    ray_fraction = ray_fraction / safe
    segment_fraction = segment_fraction / safe
    hit = (
        (jnp.abs(denominator) > PARALLEL_EPS)
        & (ray_fraction >= 0.0)
        & (segment_fraction >= 0.0)
        & (segment_fraction <= 1.0)
    )
    ranges = jnp.where(hit, ray_fraction, max_range)
    return jnp.minimum(jnp.min(ranges, axis=1), max_range)


def scan(pose, angles, seg_a, seg_b, max_range):
    """Cast a fixed beam-angle sweep from one ``[x, y, yaw]`` pose."""
    bearings = pose[2] + angles
    directions = jnp.stack((jnp.cos(bearings), jnp.sin(bearings)), axis=1)
    return ray_segment_range(pose[:2], directions, seg_a, seg_b, max_range)


__all__ = ["PARALLEL_EPS", "ray_segment_range", "scan"]
