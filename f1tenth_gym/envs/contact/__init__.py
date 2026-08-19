"""Contact geometry and response.

``kernels`` is pure JAX with no gym imports, so it survives the migration unchanged.
"""

from .kernels import Manifold, segment_contact, speculative_gap

__all__ = ["Manifold", "segment_contact", "speculative_gap"]
