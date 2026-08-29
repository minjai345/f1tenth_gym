"""Pure JAX building blocks for device-resident F1TENTH simulation.

The functional environment is intentionally assembled in layers. This
namespace exposes pure dynamics, control, state, reference-line, reset,
geometry and clean-sensing kernels. Host-only map extraction lives in
:mod:`f1tenth_gym.jax.preprocess`; importing this package does not load maps,
construct a Gymnasium environment, or initialize rendering.
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
from .contact import (
    ContactParams,
    WallContactConfig,
    apply_contact_response,
    resolve_wall_contacts,
    world_velocity,
)
from .integrators import euler_step, integrate_substeps, rk4_step
from .geometry import (
    BodyParams,
    body_vertices,
    collision_body_pose,
    transform_pose,
)
from .lidar import (
    ScanConfig,
    ScanParams,
    ScanState,
    beam_angles,
    clean_scan,
    lidar_poses,
    observed_scan,
    opponent_ranges,
    reset_scan_state,
)
from .pairs import (
    PairContactConfig,
    PairTable,
    make_pair_table,
    resolve_contacts,
    resolve_pair_contacts,
    solve_pair_impulses,
)
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
    "BodyParams",
    "ContactParams",
    "DynamicsConfig",
    "DynamicsParams",
    "DynamicsState",
    "EpisodeParams",
    "LongitudinalControlMode",
    "PairContactConfig",
    "PairTable",
    "ResetConfig",
    "ResetTable",
    "ScanConfig",
    "ScanParams",
    "ScanState",
    "SplineTable",
    "SteeringControlMode",
    "TileTable",
    "TrackTable",
    "WallTable",
    "WallContactConfig",
    "adapt_actions",
    "apply_contact_response",
    "beam_angles",
    "body_vertices",
    "cartesian_to_frenet",
    "clean_scan",
    "collision_body_pose",
    "euler_step",
    "integrate_substeps",
    "kinematic_single_track",
    "lidar_poses",
    "make_dynamics_state",
    "make_pair_table",
    "evaluate_spline",
    "frenet_to_cartesian",
    "opponent_ranges",
    "observed_scan",
    "reset_dynamics_state",
    "reset_scan_state",
    "resolve_wall_contacts",
    "resolve_contacts",
    "resolve_pair_contacts",
    "rk4_step",
    "rollout_dynamics",
    "single_track",
    "sample_reset_poses",
    "speed_control",
    "solve_pair_impulses",
    "steering_angle_control",
    "step_dynamics",
    "tile_candidates",
    "transform_pose",
    "world_velocity",
]
