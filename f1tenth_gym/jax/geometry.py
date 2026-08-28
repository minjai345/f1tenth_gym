"""Pure JAX rigid-body geometry shared by sensing and contact."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BodyParams:
    """Traced collision-body dimensions and its offset from the CoG pose."""

    length: Any
    width: Any
    centre_x: Any
    centre_y: Any

    @classmethod
    def from_vehicle_parameters(cls, params: Any) -> "BodyParams":
        """Copy geometry from a host ``VehicleParameters`` object.

        Vehicle body offsets are configured from ``base_link`` at the rear axle,
        while supported model poses are CoG referenced. Subtracting ``lr`` moves
        the configured body centre into the common CoG frame.
        """
        return cls(
            length=params.length,
            width=params.width,
            centre_x=-params.lr + params.collision_body_center_x,
            centre_y=params.collision_body_center_y,
        )


def transform_pose(pose: jax.Array, offset: jax.Array) -> jax.Array:
    """Apply one planar body-frame ``[x, y, yaw]`` offset to a world pose."""
    cosine = jnp.cos(pose[2])
    sine = jnp.sin(pose[2])
    x = pose[0] + offset[0] * cosine - offset[1] * sine
    y = pose[1] + offset[0] * sine + offset[1] * cosine
    return jnp.stack((x, y, pose[2] + offset[2]))


def collision_body_pose(pose: jax.Array, params: BodyParams) -> jax.Array:
    """Move a CoG-referenced model pose to the collision-body centre."""
    offset = jnp.stack(
        (
            jnp.asarray(params.centre_x, dtype=pose.dtype),
            jnp.asarray(params.centre_y, dtype=pose.dtype),
            jnp.asarray(0.0, dtype=pose.dtype),
        )
    )
    return transform_pose(pose, offset)


def body_vertices(pose: jax.Array, params: BodyParams) -> jax.Array:
    """Return collision-body corners in cyclic rear-left-first order."""
    body_pose = collision_body_pose(pose, params)
    half_length = jnp.asarray(params.length, dtype=pose.dtype) / 2.0
    half_width = jnp.asarray(params.width, dtype=pose.dtype) / 2.0
    local = jnp.stack(
        (
            jnp.stack((-half_length, half_width)),
            jnp.stack((-half_length, -half_width)),
            jnp.stack((half_length, -half_width)),
            jnp.stack((half_length, half_width)),
        )
    )
    cosine = jnp.cos(body_pose[2])
    sine = jnp.sin(body_pose[2])
    rotation = jnp.stack(
        (jnp.stack((cosine, sine)), jnp.stack((-sine, cosine)))
    )
    return local @ rotation + body_pose[:2]


__all__ = [
    "BodyParams",
    "body_vertices",
    "collision_body_pose",
    "transform_pose",
]
