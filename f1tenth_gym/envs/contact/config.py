"""Tuning for the segment-contact response."""

import math
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ContactConfig:
    """Solver and broad-phase settings for ``CollisionCheckMode.SEGMENT_CONTACT``.

    Frozen and hashable, like every other config section; ``with_updates`` re-runs
    validation because ``replace`` re-runs ``__post_init__``.

    Attributes:
        restitution: Bounce coefficient in [0, 1]; 0 is a dead stop.
        friction: Coulomb coefficient bounding the tangential impulse.
        restitution_threshold: Approach speed below which restitution is switched
            off, so a slow scrape cannot keep earning a bounce.
        baumgarte: Fraction of the excess penetration removed positionally per step.
        slop: Penetration left uncorrected, so resting contact does not jitter.
        solver_iterations: Jacobi sweeps per step. Named apart from the simulator's
            integrator ``substeps``, which is an unrelated quantity.
        tile_size: Broad-phase tile side in metres. Cost scales as 1/tile_size^2.
        margin: Safety factor on the exact per-tile candidate count.
        wall_tolerance_px: Douglas-Peucker tolerance for wall extraction.
    """

    restitution: float = 0.0
    friction: float = 0.6
    restitution_threshold: float = 0.6
    baumgarte: float = 0.4
    slop: float = 0.002
    solver_iterations: int = 64
    tile_size: float = 0.5
    margin: float = 1.25
    wall_tolerance_px: float = 0.25

    def __post_init__(self):
        for name in ("restitution", "baumgarte"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1], got {value}")
            object.__setattr__(self, name, value)
        for name in ("friction", "restitution_threshold", "slop"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0, got {value}")
            object.__setattr__(self, name, value)
        for name in ("tile_size", "margin", "wall_tolerance_px"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0, got {value}")
            object.__setattr__(self, name, value)
        if self.margin < 1.0:
            raise ValueError(f"margin must be >= 1, got {self.margin}")
        # Coerce rather than accept 63.5: it is a loop bound.
        iterations = int(self.solver_iterations)
        if iterations < 1:
            raise ValueError(f"solver_iterations must be >= 1, got {iterations}")
        object.__setattr__(self, "solver_iterations", iterations)

    def with_updates(self, **changes) -> "ContactConfig":
        """A copy with fields replaced, re-validated."""
        return replace(self, **changes)
