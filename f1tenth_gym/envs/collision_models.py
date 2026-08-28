"""Production collision-mode selection and vehicle body geometry."""

from enum import IntEnum

import numpy as np
from numba import njit


class CollisionCheckMode(IntEnum):
    """Supported collision behavior.

    ``SEGMENT_CONTACT`` detects wall and vehicle contact, applies impulses and
    position correction, and records per-agent flags. ``NONE`` disables both
    detection and response. LiDAR is never used as a collision response.
    """

    NONE = 0
    SEGMENT_CONTACT = 3


@njit(cache=True)
def get_trmtx(pose):
    """Return the homogeneous vehicle-to-world transform for ``pose``."""
    x, y, theta = pose
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return np.array(
        [
            [cos_theta, -sin_theta, 0.0, x],
            [sin_theta, cos_theta, 0.0, y],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


@njit(cache=True)
def get_vertices(pose, length, width):
    """Return body corners in cyclic rear-left to front-left order."""
    transform = get_trmtx(pose)
    rear_left = transform.dot(
        np.asarray([[-length / 2], [width / 2], [0.0], [1.0]])
    ).flatten()
    rear_right = transform.dot(
        np.asarray([[-length / 2], [-width / 2], [0.0], [1.0]])
    ).flatten()
    front_left = transform.dot(
        np.asarray([[length / 2], [width / 2], [0.0], [1.0]])
    ).flatten()
    front_right = transform.dot(
        np.asarray([[length / 2], [-width / 2], [0.0], [1.0]])
    ).flatten()
    rear_left = rear_left / rear_left[3]
    rear_right = rear_right / rear_right[3]
    front_right = front_right / front_right[3]
    front_left = front_left / front_left[3]
    return np.asarray(
        [
            [rear_left[0], rear_left[1]],
            [rear_right[0], rear_right[1]],
            [front_right[0], front_right[1]],
            [front_left[0], front_left[1]],
        ]
    )
