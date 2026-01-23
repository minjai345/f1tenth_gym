from __future__ import annotations

from dataclasses import dataclass, replace

__all__ = ["LiDARConfig"]


@dataclass(frozen=True)
class LiDARConfig:
    """Configuration for the simulated LiDAR sensor."""

    enabled: bool = True
    num_beams: int = 1080
    field_of_view: float = 4.712389
    maximum_range: float = 30.0
    noise_std: float = 0.01
    # (x, y, yaw) offset from base_link in meters/radians.
    base_link_to_lidar_tf: tuple[float, float, float] = (0.275, 0.0, 0.0)

    def with_updates(self, **changes: object) -> "LiDARConfig":
        return replace(self, **changes)
