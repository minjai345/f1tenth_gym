"""Pure JAX building blocks for device-resident F1TENTH simulation.

The functional environment is intentionally assembled in layers.  This
namespace currently exposes the dynamics and integration kernels; importing it
does not load maps, construct a Gymnasium environment, or initialize rendering.
"""

from .dynamics import (
    DynamicsParams,
    kinematic_single_track,
    single_track,
)
from .integrators import euler_step, integrate_substeps, rk4_step

__all__ = [
    "DynamicsParams",
    "euler_step",
    "integrate_substeps",
    "kinematic_single_track",
    "rk4_step",
    "single_track",
]
