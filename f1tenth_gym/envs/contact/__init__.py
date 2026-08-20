"""Contact geometry and response.

``kernels`` and ``solver`` are pure JAX with no gym imports, so they survive the
migration unchanged.
"""

from .config import ContactConfig
from .kernels import (
    Manifold,
    body_contact,
    deepest_depth,
    segment_contact,
    speculative_gap,
)
from .solver import (
    ContactParams,
    contact_velocity,
    resolve,
    resolve_pair,
    speculative_clamp,
)

__all__ = [
    "ContactConfig",
    "ContactParams",
    "Manifold",
    "body_contact",
    "contact_velocity",
    "deepest_depth",
    "resolve",
    "resolve_pair",
    "segment_contact",
    "speculative_clamp",
    "speculative_gap",
]
