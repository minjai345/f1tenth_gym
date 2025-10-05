from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

__all__ = ["LiDARConfig"]


@dataclass(frozen=True, slots=True)
class LiDARConfig:
    """Configuration for the simulated LiDAR sensor."""

    enabled: bool = True
    num_beams: int = 1080
    field_of_view: float = 4.712389
    maximum_range: float = 30.0
    noise_std: float = 0.01

    def as_mapping(self) -> dict[str, float]:
        """Return a mutable mapping representation of the configuration."""

        return {
            "enable_scan": self.enabled,
            "lidar_num_beams": self.num_beams,
            "lidar_fov": self.field_of_view,
            "lidar_range": self.maximum_range,
            "lidar_noise_std": self.noise_std,
        }

    def with_updates(self, **changes: object) -> "LiDARConfig":
        return replace(self, **changes)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> "LiDARConfig":
        """Construct a configuration from a legacy mapping."""

        return cls(
            enabled=bool(raw.get("enable_scan", True)),
            num_beams=int(raw.get("lidar_num_beams", 1080)),
            field_of_view=float(raw.get("lidar_fov", 4.712389)),
            maximum_range=float(raw.get("lidar_range", 30.0)),
            noise_std=float(raw.get("lidar_noise_std", 0.01)),
        )
