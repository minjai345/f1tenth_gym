"""Narrow-phase contact between a convex body and an oriented wall segment.

Pure JAX: fixed shapes, no data-dependent control flow, ``vmap``-ready, and no gym
imports. Each guard below is a defect that was found by measurement and failed
silently; the regression tests in ``tests/test_contact_kernels.py`` pin them.
"""

from typing import NamedTuple

import jax.numpy as jnp

# Overlap of exactly zero is touching, not separated. Using <= 0 here silently kills
# every flat-face resting contact, which is the case the solver most needs.
SEPARATION_EPS = 1e-9
_TINY = 1e-12
# Returned when a body cannot reach a segment at all, so a speculative clamp on it
# can never fire.
NO_CONTACT_GAP = 1e6


class Manifold(NamedTuple):
    """Up to two contact points against one segment.

    A single point cannot resist rotation about itself, so a body resting flat would
    accumulate friction torque with nothing opposing it. Two points are the fix.

    Attributes:
        points: (2, 2) contact positions in world metres, zero where unused.
        depths: (2,) penetration depths, positive inside the wall, zero where unused.
        normal: (2,) the segment's outward unit normal, carried so a solver needs
            nothing else.
    """

    points: jnp.ndarray
    depths: jnp.ndarray
    normal: jnp.ndarray

    @property
    def count(self):
        return jnp.sum(self.depths > 0.0)


def _face_normals(verts):
    """Outward unit normals of a counter-clockwise polygon's edges."""
    edge = jnp.roll(verts, -1, axis=0) - verts
    out = jnp.stack([edge[:, 1], -edge[:, 0]], axis=1)
    return out / (jnp.linalg.norm(out, axis=1, keepdims=True) + _TINY)


def _axis_overlap(verts, seg_a, seg_b, axis):
    """Overlap of the body and the segment projected onto one axis."""
    pv = verts @ axis
    qa, qb = seg_a @ axis, seg_b @ axis
    return jnp.minimum(pv.max(), jnp.maximum(qa, qb)) - jnp.maximum(
        pv.min(), jnp.minimum(qa, qb)
    )


def segment_contact(verts, seg_a, seg_b, normal, valid=True) -> Manifold:
    """Contact manifold between a convex quad and one one-sided wall segment.

    Four gates: the body's two face axes must overlap, the body must straddle the
    segment's plane rather than sit behind it, the incident face is the one most
    opposed to the wall normal, and it is clipped to the segment's tangential span.

    Args:
        verts: (4, 2) body corners in counter-clockwise order, as ``get_vertices``
            returns them.
        seg_a: (2,) segment start in world metres.
        seg_b: (2,) segment end.
        normal: (2,) the segment's outward unit normal. Must be perpendicular to
            ``seg_b - seg_a``, as ``WallSegments`` guarantees; a normal that is not
            makes ``seg_a @ normal`` cease to be a plane and the gates meaningless.
        valid: Boolean; False produces an empty manifold, for padded candidate slots.

    Returns:
        A :class:`Manifold` with one or two live points, or all zeros. Against a
        single isolated segment the reference-face clip can decline an overlap whose
        crossing falls outside the incident face; on a continuous wall the adjacent
        segments cover it, measured 0 misses over 1,950 overlapping poses on Monza.
    """
    # Required separating axes: testing only the wall's normal and tangent is a box
    # test in the contact frame, which invents contacts on 10.6% of random poses.
    body_axes = _face_normals(verts)
    touching = (
        _axis_overlap(verts, seg_a, seg_b, body_axes[0]) >= -SEPARATION_EPS
    ) & (_axis_overlap(verts, seg_a, seg_b, body_axes[1]) >= -SEPARATION_EPS)

    # One-sided: a body wholly behind the face is not in contact with it. Without
    # this the far side of a two-pixel wall reports a deep hit pointing backwards.
    along_normal = verts @ normal
    plane = seg_a @ normal
    touching &= (plane - along_normal.min() > 0.0) & (along_normal.max() > plane)

    incident = jnp.argmin(body_axes @ normal)
    p0 = verts[incident]
    p1 = verts[(incident + 1) % 4]

    tangent = jnp.array([-normal[1], normal[0]])
    span_lo = jnp.minimum(seg_a @ tangent, seg_b @ tangent)
    span_hi = jnp.maximum(seg_a @ tangent, seg_b @ tangent)
    t0, t1 = p0 @ tangent, p1 @ tangent
    touching &= (jnp.minimum(t0, t1) <= span_hi) & (jnp.maximum(t0, t1) >= span_lo)

    denom = jnp.where(jnp.abs(t1 - t0) < _TINY, _TINY, t1 - t0)

    def on_face(t):
        return p0 + (p1 - p0) * jnp.clip((t - t0) / denom, 0.0, 1.0)

    points = jnp.stack([on_face(jnp.clip(t0, span_lo, span_hi)),
                        on_face(jnp.clip(t1, span_lo, span_hi))])
    depths = plane - points @ normal

    live = (depths > 0.0) & touching & valid
    return Manifold(
        points=jnp.where(live[:, None], points, 0.0),
        depths=jnp.where(live, depths, 0.0),
        normal=jnp.where(jnp.any(live), normal, 0.0),
    )


def speculative_gap(verts, seg_a, seg_b, normal, valid=True):
    """Signed clearance along the wall normal; negative once penetrating.

    Feeds the speculative clamp ``v += (-s/dt - v_n) * n``, which scales only the
    normal component. Scaling the whole step vector kills tangential motion, so a
    scraping car stops dead and re-collides forever.

    Args:
        verts: (4, 2) body corners.
        seg_a: (2,) segment start.
        seg_b: (2,) segment end.
        normal: (2,) outward unit normal.
        valid: Boolean; False returns :data:`NO_CONTACT_GAP`.

    Returns:
        Scalar clearance in metres, :data:`NO_CONTACT_GAP` when the body cannot
        reach this segment along the normal at all.
    """
    tangent = jnp.array([-normal[1], normal[0]])
    reachable = valid & (
        _axis_overlap(verts, seg_a, seg_b, tangent) >= -SEPARATION_EPS
    )
    gap = (verts @ normal).min() - seg_a @ normal
    return jnp.where(reachable, gap, NO_CONTACT_GAP)
