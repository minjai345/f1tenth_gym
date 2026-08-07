from __future__ import annotations

import warnings
from enum import IntEnum
from typing import Iterable

from .fields import ALL_FIELDS, BASE_FIELDS, DERIVED_FIELDS

__all__ = [
    "ObservationType",
    "ALL_FEATURES",
    "BASE_FIELDS",
    "DERIVED_FIELDS",
    "Observation",
    "scan_space",
    "FullObservation",
    "RawObservation",
    "observation_factory",
]


class ObservationType(IntEnum):
    """Type of observation returned by the environment.

    DEFAULT: Per-agent dict of named fields (the packaged form; the default).
    ORIGINAL: Alias of DEFAULT (backwards compatibility).
    DIRECT: Raw agent-batched SoA arrays (``RawObservation``). Changed meaning
        in v1.0.0 — it used to be the packaged form and warns when selected.
    FEATURES: Custom subset of fields specified via config.
    KINEMATIC_STATE: Kinematic state fields (x, y, delta, vx, theta).
    DYNAMIC_STATE: Dynamic state fields including angular velocity and slip angle.
    FRENET_DYNAMIC_STATE: Dynamic state with separate vx/vy components.
    """

    DIRECT = 1
    DEFAULT = 2
    ORIGINAL = 2  # alias of DEFAULT
    FEATURES = 3
    KINEMATIC_STATE = 4
    DYNAMIC_STATE = 5
    FRENET_DYNAMIC_STATE = 6


ALL_FEATURES: tuple[str, ...] = ALL_FIELDS
_ALLOWED_FIELDS = set(ALL_FIELDS)


from .base import Observation, scan_space  # noqa: E402
from .full import FullObservation  # noqa: E402
from .raw import RawObservation  # noqa: E402


FEATURE_PRESETS: dict[ObservationType, tuple[str, ...]] = {
    ObservationType.KINEMATIC_STATE: (
        "pose_x",
        "pose_y",
        "delta",
        "linear_vel_x",
        "pose_theta",
    ),
    ObservationType.DYNAMIC_STATE: (
        "pose_x",
        "pose_y",
        "delta",
        "linear_vel_magnitude",
        "pose_theta",
        "ang_vel_z",
        "beta",
    ),
    ObservationType.FRENET_DYNAMIC_STATE: (
        "pose_x",
        "pose_y",
        "delta",
        "linear_vel_x",
        "linear_vel_y",
        "pose_theta",
        "ang_vel_z",
        "beta",
    ),
}


def _normalize_fields(fields: Iterable[str] | None) -> tuple[str, ...]:
    if fields is None:
        raise ValueError("FullObservation requires 'features' to be specified")

    normalized = tuple(fields)
    if not normalized:
        raise ValueError("FullObservation requires at least one feature")

    invalid = next((item for item in normalized if item not in _ALLOWED_FIELDS), None)
    if invalid is not None:
        raise ValueError(f"Unknown observation feature: {invalid!r}")

    return normalized


def observation_factory(
    env,
    type: ObservationType | None = None,
    **kwargs,
) -> Observation:
    """Create an Observation instance based on the specified type.

    Args:
        env: The F110Env environment instance.
        type: Observation type to create (default: DEFAULT).
        **kwargs: Additional arguments (e.g., features for FEATURES type).

    Returns:
        Configured Observation instance.
    """
    if type is None:
        obs_type = ObservationType.DEFAULT
    elif isinstance(type, ObservationType):
        obs_type = type
    else:
        raise TypeError("observation_factory 'type' must be an ObservationType")

    if obs_type is ObservationType.DEFAULT:  # covers the ORIGINAL alias
        return FullObservation(env)
    if obs_type is ObservationType.DIRECT:
        warnings.warn(
            "ObservationType.DIRECT changed meaning in v1.0.0: it now returns "
            "raw agent-batched arrays. Use ObservationType.DEFAULT for the "
            "packaged per-agent dict.",
            stacklevel=2,
        )
        return RawObservation(env)
    if obs_type in FEATURE_PRESETS:
        return FullObservation(env, fields=FEATURE_PRESETS[obs_type])
    if obs_type is ObservationType.FEATURES:
        return FullObservation(env, fields=_normalize_fields(kwargs.get("features")))

    raise ValueError(f"Unsupported observation type: {obs_type}")
