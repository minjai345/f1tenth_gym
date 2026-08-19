"""Contact geometry and response.

``kernels`` and ``solver`` are pure JAX with no gym imports, so they survive the
migration unchanged.
"""

from .config import ContactConfig
from .kernels import Manifold, segment_contact, speculative_gap
from .solver import ContactParams, contact_velocity, resolve, speculative_clamp

__all__ = [
    "ContactConfig",
    "ContactParams",
    "Manifold",
    "contact_velocity",
    "resolve",
    "segment_contact",
    "speculative_clamp",
    "speculative_gap",
]
