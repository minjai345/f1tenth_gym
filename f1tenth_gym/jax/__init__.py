"""Pure JAX building blocks for device-resident F1TENTH simulation.

The functional environment is intentionally assembled in layers. This
namespace exposes pure dynamics, control, state, reference-line and reset
kernels. Host-only map extraction lives in :mod:`f1tenth_gym.jax.preprocess`;
importing this package does not load maps, construct a Gymnasium environment,
or initialize rendering.
"""

from .dynamics import (
    DynamicsParams,
    kinematic_single_track,
    single_track,
)
from .controls import (
    LongitudinalControlMode,
    SteeringControlMode,
    adapt_actions,
    speed_control,
    steering_angle_control,
)
from .core import (
    DynamicsConfig,
    DynamicsState,
    EpisodeParams,
    make_dynamics_state,
    rollout_dynamics,
    step_dynamics,
)
from .integrators import euler_step, integrate_substeps, rk4_step
from .reset import (
    ResetConfig,
    ResetTable,
    reset_dynamics_state,
    sample_reset_poses,
)
from .track import (
    SplineTable,
    TileTable,
    TrackTable,
    WallTable,
    cartesian_to_frenet,
    evaluate_spline,
    frenet_to_cartesian,
    tile_candidates,
)

__all__ = [
    "DynamicsConfig",
    "DynamicsParams",
    "DynamicsState",
    "EpisodeParams",
    "LongitudinalControlMode",
    "ResetConfig",
    "ResetTable",
    "SplineTable",
    "SteeringControlMode",
    "TileTable",
    "TrackTable",
    "WallTable",
    "adapt_actions",
    "cartesian_to_frenet",
    "euler_step",
    "integrate_substeps",
    "kinematic_single_track",
    "make_dynamics_state",
    "evaluate_spline",
    "frenet_to_cartesian",
    "reset_dynamics_state",
    "rk4_step",
    "rollout_dynamics",
    "single_track",
    "sample_reset_poses",
    "speed_control",
    "steering_angle_control",
    "step_dynamics",
    "tile_candidates",
]
