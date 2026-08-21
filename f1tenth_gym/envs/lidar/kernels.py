"""Analytic ray-segment scanning.

Pure JAX: fixed shapes, no data-dependent control flow, ``vmap``-ready, and no gym
imports, matching ``contact/kernels.py``. The ranges are differentiable with respect
to the sensor pose, which sphere-tracing a distance transform cannot be.
"""

import jax.numpy as jnp

# Below this the ray is parallel within float32 noise; dividing by it returns a
# huge t that then wins the min.
PARALLEL_EPS = 1e-12


def ray_segment_range(origin, direction, seg_a, seg_b, max_range):
    """Distance from ``origin`` along ``direction`` to the nearest segment hit.

    Cramer's rule on ``origin + t*direction == seg_a + u*(seg_b - seg_a)``. A
    degenerate segment gives a zero denominator and is rejected, which is what makes
    it usable as padding in a fixed-shape candidate list.

    Args:
        origin: (2,) sensor position in world metres.
        direction: (B, 2) unit ray directions.
        seg_a: (K, 2) segment starts.
        seg_b: (K, 2) segment ends.
        max_range: Scalar clamp, also the value returned where nothing is hit.

    Returns:
        (B,) ranges in metres.
    """
    edge = seg_b - seg_a
    offset = seg_a[None, :, :] - origin[None, None, :]

    # 2-D cross products, broadcast to (B, K).
    denom = (direction[:, None, 0] * edge[None, :, 1]
             - direction[:, None, 1] * edge[None, :, 0])
    t = (offset[..., 0] * edge[None, :, 1] - offset[..., 1] * edge[None, :, 0])
    u = (offset[..., 0] * direction[:, None, 1] - offset[..., 1] * direction[:, None, 0])

    safe = jnp.where(jnp.abs(denom) > PARALLEL_EPS, denom, 1.0)
    t = t / safe
    u = u / safe

    hit = (jnp.abs(denom) > PARALLEL_EPS) & (t >= 0.0) & (u >= 0.0) & (u <= 1.0)
    t = jnp.where(hit, t, max_range)
    return jnp.minimum(jnp.min(t, axis=1), max_range)


def scan(pose, angles, seg_a, seg_b, max_range):
    """One full sweep from a sensor pose.

    Args:
        pose: (3,) sensor ``(x, y, yaw)`` in world metres and radians.
        angles: (B,) beam angles relative to ``yaw``.
        seg_a: (K, 2) segment starts.
        seg_b: (K, 2) segment ends.
        max_range: Scalar clamp.

    Returns:
        (B,) ranges in metres.
    """
    bearing = pose[2] + angles
    direction = jnp.stack([jnp.cos(bearing), jnp.sin(bearing)], axis=1)
    return ray_segment_range(pose[:2], direction, seg_a, seg_b, max_range)
