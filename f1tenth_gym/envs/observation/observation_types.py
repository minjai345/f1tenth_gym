from enum import IntEnum


class ObservationType(IntEnum):
    """Enumerates the observation pipelines supported by the environment."""

    DIRECT = 1
    ORIGINAL = 2
    FEATURES = 3
    KINEMATIC_STATE = 4
    DYNAMIC_STATE = 5
    FRENET_DYNAMIC_STATE = 6
