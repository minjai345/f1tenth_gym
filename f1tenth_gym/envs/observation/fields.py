"""The observation field vocabulary.

Single source of truth: both the factory (``observation/__init__.py``) and
``FullObservation`` validate against THESE tuples. They used to carry
duplicate copies, and a field added to one but not the other worked through
``FullObservation(fields=...)`` while being rejected by the factory.
"""

BASE_FIELDS: tuple[str, ...] = (
    "scan",
    "std_state",
    "state",
    "collision",
    "lap_time",
    "lap_count",
    "sim_time",
    "frenet_pose",
)

DERIVED_FIELDS: tuple[str, ...] = (
    "pose_x",
    "pose_y",
    "pose_theta",
    "linear_vel_x",
    "linear_vel_y",
    "linear_vel_magnitude",
    "ang_vel_z",
    "delta",
    "beta",
)

ALL_FIELDS: tuple[str, ...] = BASE_FIELDS + DERIVED_FIELDS
