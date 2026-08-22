from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import IntEnum

__all__ = ["LiDARConfig", "ScanBackend"]


class ScanBackend(IntEnum):
    """How a range is produced.

    RASTER sphere-traces the distance transform, so it reads long by up to half a
    cell diagonal. SEGMENT intersects the wall segments analytically: exact and
    differentiable, but JAX-backed, so vector envs need ``context="spawn"``.
    """

    RASTER = 1
    SEGMENT = 2


# 270 degrees, the default sweep. Stored as the angle pair, since that is what
# the sensor actually has; field_of_view is derived from it.
_DEFAULT_HALF_FOV = 2.3561945


@dataclass(frozen=True)
class LiDARConfig:
    """Configuration for the simulated LiDAR sensor.

    ``angle_min``/``angle_max`` are the only stored geometry; :attr:`field_of_view`
    is a read-only property derived from them, so the two cannot disagree. Build
    from a sweep with :meth:`from_fov`.

    Attributes:
        enabled: Whether LiDAR scanning is enabled.
        num_beams: Number of laser beams in the scan.
        angle_min: Start angle of scan in radians.
        angle_max: End angle of scan in radians.
        range_min: Minimum range in meters, readings below are clipped.
        range_max: Maximum range in meters, readings above are clipped.
        noise_std: Standard deviation of Gaussian noise on range readings.
        dropout_prob: Per-beam, per-step probability that a beam returns a
            no-return (clamped to range_max), modelling missed detections.
        range_bias_std: Std of a per-beam systematic range bias, drawn once per
            episode (reproducible with reset(seed=...)), modelling calibration
            error. Constant across a rollout, unlike noise_std.
        base_link_to_lidar_tf: (x, y, yaw) offset from base_link in meters/radians.

    All of the above affect the *observed* scan only; collision detection uses
    the clean scan.
    """

    enabled: bool = True
    num_beams: int = 1080
    angle_min: float = -_DEFAULT_HALF_FOV
    angle_max: float = _DEFAULT_HALF_FOV
    range_min: float = 0.0
    range_max: float = 30.0
    noise_std: float = 0.01
    dropout_prob: float = 0.0
    range_bias_std: float = 0.0
    # (x, y, yaw) offset from base_link in meters/radians.
    base_link_to_lidar_tf: tuple[float, float, float] = (0.275, 0.0, 0.0)
    backend: ScanBackend = ScanBackend.SEGMENT
    # SEGMENT only. Per agent per launch it trails RASTER by 10%, and batching is
    # what pays: at 12 agents "gpu" is 7x faster per agent, "cpu" 0.86x.
    scan_device: str = "cpu"

    def __post_init__(self) -> None:
        # Validation
        try:
            object.__setattr__(self, "backend", ScanBackend(self.backend))
        except ValueError as exc:
            raise ValueError(f"backend must be a ScanBackend: {exc}") from exc
        if self.scan_device not in ("cpu", "gpu"):
            raise ValueError(
                f"scan_device must be 'cpu' or 'gpu', got {self.scan_device!r}")
        if self.num_beams < 1:
            raise ValueError(f"num_beams must be >= 1, got {self.num_beams}")
        if self.range_max <= 0:
            raise ValueError(f"range_max must be > 0, got {self.range_max}")
        if self.range_min < 0:
            raise ValueError(f"range_min must be >= 0, got {self.range_min}")
        if self.range_min >= self.range_max:
            raise ValueError(
                f"range_min ({self.range_min}) must be less than range_max ({self.range_max})"
            )
        if self.noise_std < 0:
            raise ValueError(f"noise_std must be >= 0, got {self.noise_std}")
        if not (0.0 <= self.dropout_prob <= 1.0):
            raise ValueError(f"dropout_prob must be in [0, 1], got {self.dropout_prob}")
        if self.range_bias_std < 0:
            raise ValueError(f"range_bias_std must be >= 0, got {self.range_bias_std}")
        if self.angle_min >= self.angle_max:
            raise ValueError(
                f"angle_min ({self.angle_min}) must be less than angle_max ({self.angle_max})"
            )
        if self.angle_min < -math.pi:
            raise ValueError(
                f"angle_min ({self.angle_min}) must be >= -π (-180°). "
                f"Did you pass degrees instead of radians? Use np.deg2rad() to convert."
            )
        if self.angle_max > math.pi:
            raise ValueError(
                f"angle_max ({self.angle_max}) must be <= π (180°). "
                f"Did you pass degrees instead of radians? Use np.deg2rad() to convert."
            )

    @classmethod
    def from_fov(cls, field_of_view: float, **kwargs: object) -> "LiDARConfig":
        """Build a config with a symmetric sweep of ``field_of_view`` radians.

        Angles are the stored geometry, so this computes ``∓field_of_view/2``::

            LiDARConfig.from_fov(np.deg2rad(270))     # the default sweep

        Raises:
            ValueError: if ``field_of_view`` is not positive, or if an explicit
                ``angle_min``/``angle_max`` is also given — that combination is
                over-determined.
        """
        if field_of_view <= 0:
            raise ValueError(f"field_of_view must be > 0, got {field_of_view}")
        if "angle_min" in kwargs or "angle_max" in kwargs:
            raise ValueError(
                "from_fov sets angle_min/angle_max itself; pass either a field "
                "of view or an explicit angle pair, not both"
            )
        half = field_of_view / 2.0
        return cls(angle_min=-half, angle_max=half, **kwargs)

    @property
    def field_of_view(self) -> float:
        """Total angular sweep in radians — always ``angle_max - angle_min``.

        Read-only and derived: the angles are the sensor's real extent, so the
        two can never disagree. Use :meth:`from_fov` to build from a sweep.
        """
        return float(self.angle_max) - float(self.angle_min)

    @property
    def angle_increment(self) -> float:
        """Angular distance between consecutive measurements in radians."""
        if self.num_beams <= 1:
            return 0.0
        return (self.angle_max - self.angle_min) / (self.num_beams - 1)

    @property
    def maximum_range(self) -> float:
        """Alias for range_max for backwards compatibility."""
        return self.range_max

    def with_updates(self, **changes: object) -> "LiDARConfig":
        """Return a re-validated copy with ``changes`` applied.

        A plain ``replace()``: the angles are the only stored geometry, so there
        is nothing to special-case.
        """
        return replace(self, **changes)
