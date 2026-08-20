"""Narrow-phase contact between a convex body and an oriented wall segment.

Pure JAX: fixed shapes, no data-dependent control flow, ``vmap``-ready, and no gym
imports. Each guard below is a defect that was found by measurement and failed
silently; the regression tests in ``tests/test_contact_kernels.py`` pin them.
"""

from typing import NamedTuple

import jax
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


def _soft_min(values, softness):
    """``min`` when ``softness`` is 0, a smooth lower bound above it.

    The hard min is only C0 in pose: its gradient jumps where the deepest vertex
    swaps. Softening trades a small bias for a continuous derivative.
    """
    hard = values.min()
    safe = jnp.where(softness > 0.0, softness, 1.0)
    # Shifted by the hard min so every exponent is <= 0 and cannot overflow.
    soft = hard - safe * jnp.log(jnp.sum(jnp.exp(-(values - hard) / safe)))
    return jnp.where(softness > 0.0, soft, hard)


def deepest_depth(verts, seg_a, seg_b, normal, valid=True, softness=0.0):
    """Penetration of the deepest body vertex: the differentiable surrogate.

    Not a substitute for :func:`segment_contact`. **Use the manifold for physics and
    this for learning.** A manifold carries two points so a resting body cannot spin,
    but their summed depth is a worse thing to differentiate: over 417 poses on
    Spielberg in float32 its gradient sits 9.7% from central differences on d/dx
    against 1.4% here, and it goes numerically dead -- no finite-difference signal on
    any axis -- at 12 of those poses where this goes dead at none.

    Everything is projected in a frame centred on the body, which is algebraically a
    no-op: the depth is a difference of two projections that are O(100) at real track
    coordinates, and cancelling the centre before subtracting keeps significant bits
    that would otherwise be lost. Measured against the same maths in world axes, that
    is a 3.8x accuracy gain on d/dx (0.86% against 3.25%), not a correctness fix --
    both variants produce a usable gradient.

    Args:
        verts: (4, 2) body corners, counter-clockwise.
        seg_a: (2,) segment start in world metres.
        seg_b: (2,) segment end.
        normal: (2,) the segment's outward unit normal, perpendicular to
            ``seg_b - seg_a``.
        valid: Boolean; False returns 0.0, for padded candidate slots.
        softness: Metres of smoothing over the vertex minimum. 0 is the exact
            deepest depth; a few mm smooths the kinks where the deepest vertex
            swaps. This matters most flush against a wall, where two vertices tie
            exactly: the hard minimum picks one and reports a spurious median
            ``|d/dpsi|`` of 0.29 where the true derivative is near zero, and 0.5 mm
            of softening cuts that to 0.0004.

    Returns:
        Scalar penetration in metres, positive inside the wall, 0.0 when not in
        contact. Gated on the body's tangential span rather than the incident
        face's, so it can be positive on an overlap the manifold's face clip
        declines; on a continuous wall an adjacent segment covers that case.
    """
    # Centring cancels exactly out of every difference below, so this changes no
    # value -- only how many significant bits survive to reach the difference.
    centre = verts.mean(axis=0)
    local = verts - centre
    local_a = seg_a - centre
    local_b = seg_b - centre

    body_axes = _face_normals(verts)
    touching = valid & (
        _axis_overlap(local, local_a, local_b, body_axes[0]) >= -SEPARATION_EPS
    ) & (_axis_overlap(local, local_a, local_b, body_axes[1]) >= -SEPARATION_EPS)

    tangent = jnp.array([-normal[1], normal[0]])
    touching &= _axis_overlap(local, local_a, local_b, tangent) >= -SEPARATION_EPS

    along = local @ normal
    plane = local_a @ normal
    # One-sided, exactly as segment_contact: a body wholly behind the face is clear.
    touching &= along.max() > plane

    depth = plane - _soft_min(along, softness)
    return jnp.where(touching & (depth > 0.0), depth, 0.0)


def _project(verts, axis):
    proj = verts @ axis
    return proj.min(), proj.max()


def body_contact(verts_a, verts_b, valid=True) -> Manifold:
    """Contact manifold between two convex quads, by separating-axis test.

    The minimum-overlap axis is the minimum translation vector; the reference face
    is the one that produced it and the incident face is the most opposed face on
    the other body, clipped to the reference span exactly as the wall path does.

    Args:
        verts_a: (4, 2) first body's corners, counter-clockwise.
        verts_b: (4, 2) second body's corners, counter-clockwise.
        valid: Boolean; False produces an empty manifold.

    Returns:
        A :class:`Manifold` whose ``normal`` points from ``verts_a`` toward
        ``verts_b``, so ``a`` is pushed along ``-normal`` and ``b`` along ``+normal``.
    """
    axes = jnp.concatenate([_face_normals(verts_a)[:2], _face_normals(verts_b)[:2]])

    def overlap_on(axis):
        lo_a, hi_a = _project(verts_a, axis)
        lo_b, hi_b = _project(verts_b, axis)
        return jnp.minimum(hi_a, hi_b) - jnp.maximum(lo_a, lo_b)

    overlaps = jax.vmap(overlap_on)(axes)
    touching = valid & jnp.all(overlaps >= -SEPARATION_EPS)

    # The winning axis is the MTV direction; the depth comes from the clip below,
    # which is per-point rather than the single scalar overlap.
    best = jnp.argmin(overlaps)
    axis = axes[best]
    # Orient from a to b so the sign of the impulse is unambiguous.
    centre_a = verts_a.mean(axis=0)
    centre_b = verts_b.mean(axis=0)
    normal = jnp.where(jnp.dot(centre_b - centre_a, axis) < 0.0, -axis, axis)

    # The reference face belongs to whichever body owns the winning axis.
    a_owns = best < 2
    reference, incident = jax.lax.cond(
        a_owns, lambda: (verts_a, verts_b), lambda: (verts_b, verts_a)
    )
    ref_normal = jnp.where(a_owns, normal, -normal)

    incident_axes = _face_normals(incident)
    face = jnp.argmin(incident_axes @ ref_normal)
    p0 = incident[face]
    p1 = incident[(face + 1) % 4]

    tangent = jnp.array([-ref_normal[1], ref_normal[0]])
    ref_t = reference @ tangent
    span_lo, span_hi = ref_t.min(), ref_t.max()
    t0, t1 = p0 @ tangent, p1 @ tangent
    denom = jnp.where(jnp.abs(t1 - t0) < _TINY, _TINY, t1 - t0)

    def on_face(t):
        return p0 + (p1 - p0) * jnp.clip((t - t0) / denom, 0.0, 1.0)

    points = jnp.stack([on_face(jnp.clip(t0, span_lo, span_hi)),
                        on_face(jnp.clip(t1, span_lo, span_hi))])
    plane = (reference @ ref_normal).max()
    depths = plane - points @ ref_normal

    live = (depths > 0.0) & touching
    return Manifold(
        points=jnp.where(live[:, None], points, 0.0),
        depths=jnp.where(live, depths, 0.0),
        normal=jnp.where(jnp.any(live), normal, 0.0),
    )
