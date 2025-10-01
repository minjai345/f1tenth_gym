from __future__ import annotations

from enum import Enum, IntEnum
from typing import Iterable, Tuple

__all__ = [
    "ObservationType",
    "ObservationFeature",
    "Observation",
    "scan_space",
    "DirectObservation",
    "OriginalObservation",
    "FeaturesObservation",
    "observation_factory",
]


class ObservationType(IntEnum):
    DIRECT = 1
    ORIGINAL = 2
    FEATURES = 3
    KINEMATIC_STATE = 4
    DYNAMIC_STATE = 5
    FRENET_DYNAMIC_STATE = 6


class ObservationFeature(Enum):
    SCAN = "scan"
    POSE_X = "pose_x"
    POSE_Y = "pose_y"
    POSE_THETA = "pose_theta"
    LINEAR_VEL_X = "linear_vel_x"
    LINEAR_VEL_Y = "linear_vel_y"
    LINEAR_VEL_MAGNITUDE = "linear_vel_magnitude"
    ANGULAR_VEL_Z = "ang_vel_z"
    STEER_ANGLE = "delta"
    SLIP_ANGLE = "beta"
    COLLISION = "collision"
    LAP_TIME = "lap_time"
    LAP_COUNT = "lap_count"

    @classmethod
    def from_value(cls, value: object) -> "ObservationFeature":
        if isinstance(value, ObservationFeature):
            return value
        raise TypeError(
            "Observation features must be provided as ObservationFeature enums "
            f"(got {value!r})"
        )


from .base import Observation, scan_space  # noqa: E402
from .direct import DirectObservation  # noqa: E402
from .original import OriginalObservation  # noqa: E402
from .features import FeaturesObservation  # noqa: E402


_FEATURE_PRESETS: dict[ObservationType, Tuple[ObservationFeature, ...]] = {
    ObservationType.KINEMATIC_STATE: (
        ObservationFeature.POSE_X,
        ObservationFeature.POSE_Y,
        ObservationFeature.STEER_ANGLE,
        ObservationFeature.LINEAR_VEL_X,
        ObservationFeature.POSE_THETA,
    ),
    ObservationType.DYNAMIC_STATE: (
        ObservationFeature.POSE_X,
        ObservationFeature.POSE_Y,
        ObservationFeature.STEER_ANGLE,
        ObservationFeature.LINEAR_VEL_MAGNITUDE,
        ObservationFeature.POSE_THETA,
        ObservationFeature.ANGULAR_VEL_Z,
        ObservationFeature.SLIP_ANGLE,
    ),
    ObservationType.FRENET_DYNAMIC_STATE: (
        ObservationFeature.POSE_X,
        ObservationFeature.POSE_Y,
        ObservationFeature.STEER_ANGLE,
        ObservationFeature.LINEAR_VEL_X,
        ObservationFeature.LINEAR_VEL_Y,
        ObservationFeature.POSE_THETA,
        ObservationFeature.ANGULAR_VEL_Z,
        ObservationFeature.SLIP_ANGLE,
    ),
}


_OBSERVATION_BUILDERS = {
    ObservationType.ORIGINAL: lambda env, **kwargs: OriginalObservation(env),
    ObservationType.DIRECT: lambda env, **kwargs: DirectObservation(env),
}


def _build_features_observation(env, *, features: Iterable[ObservationFeature] | None = None):
    if features is None:
        raise ValueError("FeaturesObservation requires 'features' to be specified")
    feature_tuple = tuple(ObservationFeature.from_value(item) for item in features)
    if not feature_tuple:
        raise ValueError("FeaturesObservation requires at least one feature")
    return FeaturesObservation(env, features=feature_tuple)


_OBSERVATION_BUILDERS[ObservationType.FEATURES] = _build_features_observation


def observation_factory(env, type: ObservationType | None = None, **kwargs) -> Observation:
    obs_type = type or ObservationType.ORIGINAL
    if not isinstance(obs_type, ObservationType):
        raise TypeError("observation_factory 'type' must be an ObservationType")

    if obs_type in _FEATURE_PRESETS:
        return _build_features_observation(
            env,
            features=_FEATURE_PRESETS[obs_type],
        )

    builder = _OBSERVATION_BUILDERS.get(obs_type)
    if builder is not None:
        return builder(env, **kwargs)

    if obs_type is ObservationType.FEATURES:
        return _build_features_observation(env, features=kwargs.get("features"))

    raise ValueError(f"Unsupported observation type: {obs_type}")
