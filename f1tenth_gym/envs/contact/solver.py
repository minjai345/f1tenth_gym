"""Impulse contact resolution for a planar rigid body.

Pure JAX: fixed shapes, ``fori_loop`` only, no gym imports. Impulses are a velocity
projection, since no dynamics model here has a slot for an external wrench. Expect
violent spin: at a front corner 87% of the contact compliance is rotational.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

_TINY = 1e-12


class ContactParams(NamedTuple):
    """Solver tuning.

    Attributes:
        restitution: Bounce coefficient, 0 for a dead stop.
        friction: Coulomb coefficient bounding the tangential impulse.
        restitution_threshold: Approach speed below which restitution is switched
            off. During a scrape the per-iteration closing speed is ~1 cm/s, and
            applying ``(1 + e)`` to each one injects energy indefinitely.
        baumgarte: Fraction of the excess penetration removed positionally per call.
        slop: Penetration left uncorrected, so resting contact does not jitter.
    """

    restitution: float = 0.0
    friction: float = 0.6
    restitution_threshold: float = 0.6
    baumgarte: float = 0.4
    slop: float = 0.002


def _cross(r, v):
    """z-component of r x v for planar vectors."""
    return r[..., 0] * v[..., 1] - r[..., 1] * v[..., 0]


def contact_velocity(velocity, omega, r):
    """World velocity of the material point at offset ``r`` from the centre of mass."""
    return velocity + omega * jnp.stack([-r[..., 1], r[..., 0]], axis=-1)


def speculative_clamp(velocity, gaps, normals, dt):
    """Stop the body closing more than the available gap in one step.

    Clamps the normal component only. Scaling the whole displacement kills tangential
    motion, so a scraping car stops dead and re-collides forever.

    Args:
        velocity: (2,) world linear velocity.
        gaps: (N,) clearance per slot, large where the slot cannot be reached.
        normals: (N, 2) outward unit normals.
        dt: Timestep in seconds.

    Returns:
        (2,) the clamped velocity.
    """

    def one(v, i):
        n = normals[i]
        v_n = jnp.dot(v, n)
        limit = -gaps[i] / dt
        # Separated slots only. Once penetrating the gap is negative, the limit turns
        # positive, and this would fling the body out at gap/dt instead of clamping.
        fires = (gaps[i] > 0.0) & (v_n < limit)
        return jnp.where(fires, v + (limit - v_n) * n, v), None

    out, _ = jax.lax.scan(one, velocity, jnp.arange(normals.shape[0]))
    return out


def resolve(
    velocity,
    omega,
    mass,
    inertia,
    points,
    depths,
    normals,
    centre,
    params: ContactParams,
    iterations: int = 64,
):
    """Solve normal and friction impulses over a fixed set of contact slots.

    Jacobi, not Gauss-Seidel: contacts solve from the same velocity and their
    corrections are averaged, so a sweep is one vectorised op. On an RTX 3080, 64
    Jacobi sweeps beat 16 sequential ones on both residual and time (0.82 vs 3.55 ms).

    Args:
        velocity: (2,) world linear velocity of the centre of mass.
        omega: Scalar angular velocity, rad/s.
        mass: Body mass, kg.
        inertia: Yaw inertia about the centre of mass, kg m^2.
        points: (N, 2) contact positions; slots with zero depth are ignored.
        depths: (N,) penetration depths, positive inside the wall.
        normals: (N, 2) outward unit normals.
        centre: (2,) centre of mass in world metres.
        params: Solver tuning.
        iterations: Jacobi sweeps; static, it is the loop bound.

    Returns:
        ``(velocity, omega, position_correction)`` where the correction is the mean
        Baumgarte push-out over live contacts, in metres.
    """
    live = depths > 0.0
    r = points - centre
    inv_m = 1.0 / mass
    inv_i = 1.0 / inertia
    n_live = jnp.maximum(jnp.sum(live), 1.0)

    # Bias is e * the (negative) approach speed, so the solver drives v_n to
    # +e*|approach|. Using -approach here drives it the other way: still closing.
    approach = jnp.sum(contact_velocity(velocity, omega, r) * normals, axis=-1)
    bounce = jnp.where(
        -approach > params.restitution_threshold, params.restitution * approach, 0.0
    )
    rn = _cross(r, normals)
    k_n = jnp.maximum(inv_m + rn * rn * inv_i, _TINY)

    def sweep(_, carry):
        v, w, acc_n = carry
        v_c = contact_velocity(v, w, r)
        v_n = jnp.sum(v_c * normals, axis=-1)

        # Accumulated clamp: the total normal impulse may never pull the body in.
        delta = -(v_n + bounce) / k_n
        clamped = jnp.maximum(acc_n + delta, 0.0)
        delta = jnp.where(live, (clamped - acc_n) / n_live, 0.0)
        acc_n = acc_n + delta

        # Friction opposes the slide that exists, not a fixed tangent.
        v_t_vec = v_c - v_n[:, None] * normals
        speed_t = jnp.linalg.norm(v_t_vec, axis=-1, keepdims=True)
        tangent = jnp.where(
            speed_t > _TINY,
            v_t_vec / (speed_t + _TINY),
            jnp.stack([-normals[:, 1], normals[:, 0]], axis=-1),
        )
        rt = _cross(r, tangent)
        k_t = jnp.maximum(inv_m + rt * rt * inv_i, _TINY)
        bound = params.friction * acc_n
        j_t = jnp.clip(-jnp.sum(v_c * tangent, axis=-1) / k_t, -bound, bound)
        j_t = jnp.where(live, j_t / n_live, 0.0)

        impulse = delta[:, None] * normals + j_t[:, None] * tangent
        v = v + jnp.sum(impulse, axis=0) * inv_m
        w = w + jnp.sum(_cross(r, impulse)) * inv_i
        return (v, w, acc_n)

    velocity, omega, _ = jax.lax.fori_loop(
        0, iterations, sweep, (velocity, omega, jnp.zeros_like(depths))
    )

    excess = jnp.maximum(depths - params.slop, 0.0) * live
    push = jnp.sum(excess[:, None] * normals, axis=0)
    correction = params.baumgarte * push / n_live
    return velocity, omega, correction


def resolve_pair(
    velocity_a,
    omega_a,
    velocity_b,
    omega_b,
    mass,
    inertia,
    points,
    depths,
    normal,
    centre_a,
    centre_b,
    params: ContactParams,
    iterations: int = 64,
):
    """Solve impulses between two dynamic bodies sharing one contact normal.

    The wall version treats the other side as infinitely massive; here both bodies
    take the impulse, so the effective mass carries both translational and both
    rotational terms.

    Args:
        velocity_a: (2,) world linear velocity of body a.
        omega_a: Scalar angular velocity of body a.
        velocity_b: (2,) world linear velocity of body b.
        omega_b: Scalar angular velocity of body b.
        mass: Body mass, kg; both bodies are identical vehicles.
        inertia: Yaw inertia, kg m^2.
        points: (N, 2) contact positions.
        depths: (N,) penetration depths; zero marks an unused slot.
        normal: (2,) unit normal pointing from a toward b.
        centre_a: (2,) centre of mass of body a.
        centre_b: (2,) centre of mass of body b.
        params: Solver tuning.
        iterations: Jacobi sweeps; static.

    Returns:
        ``(velocity_a, omega_a, velocity_b, omega_b, separation)`` where separation
        is the positional push-out to apply to b, and its negation to a.
    """
    live = depths > 0.0
    r_a = points - centre_a
    r_b = points - centre_b
    inv_m = 1.0 / mass
    inv_i = 1.0 / inertia
    n_live = jnp.maximum(jnp.sum(live), 1.0)
    normals = jnp.broadcast_to(normal, points.shape)

    def relative(v_a, w_a, v_b, w_b):
        return contact_velocity(v_b, w_b, r_b) - contact_velocity(v_a, w_a, r_a)

    approach = jnp.sum(relative(velocity_a, omega_a, velocity_b, omega_b) * normals, axis=-1)
    bounce = jnp.where(
        -approach > params.restitution_threshold, params.restitution * approach, 0.0
    )
    rn_a = _cross(r_a, normals)
    rn_b = _cross(r_b, normals)
    k_n = jnp.maximum(2.0 * inv_m + (rn_a * rn_a + rn_b * rn_b) * inv_i, _TINY)

    def sweep(_, carry):
        v_a, w_a, v_b, w_b, acc_n = carry
        v_rel = relative(v_a, w_a, v_b, w_b)
        v_n = jnp.sum(v_rel * normals, axis=-1)

        delta = -(v_n + bounce) / k_n
        clamped = jnp.maximum(acc_n + delta, 0.0)
        delta = jnp.where(live, (clamped - acc_n) / n_live, 0.0)
        acc_n = acc_n + delta

        v_t_vec = v_rel - v_n[:, None] * normals
        speed_t = jnp.linalg.norm(v_t_vec, axis=-1, keepdims=True)
        tangent = jnp.where(
            speed_t > _TINY,
            v_t_vec / (speed_t + _TINY),
            jnp.stack([-normals[:, 1], normals[:, 0]], axis=-1),
        )
        rt_a = _cross(r_a, tangent)
        rt_b = _cross(r_b, tangent)
        k_t = jnp.maximum(2.0 * inv_m + (rt_a * rt_a + rt_b * rt_b) * inv_i, _TINY)
        bound = params.friction * acc_n
        j_t = jnp.clip(-jnp.sum(v_rel * tangent, axis=-1) / k_t, -bound, bound)
        j_t = jnp.where(live, j_t / n_live, 0.0)

        impulse = delta[:, None] * normals + j_t[:, None] * tangent
        total = jnp.sum(impulse, axis=0)
        v_a = v_a - total * inv_m
        w_a = w_a - jnp.sum(_cross(r_a, impulse)) * inv_i
        v_b = v_b + total * inv_m
        w_b = w_b + jnp.sum(_cross(r_b, impulse)) * inv_i
        return (v_a, w_a, v_b, w_b, acc_n)

    velocity_a, omega_a, velocity_b, omega_b, _ = jax.lax.fori_loop(
        0, iterations, sweep, (velocity_a, omega_a, velocity_b, omega_b, jnp.zeros_like(depths))
    )

    excess = jnp.maximum(depths - params.slop, 0.0) * live
    # Split the push-out between the two bodies rather than moving one twice as far.
    separation = 0.5 * params.baumgarte * jnp.sum(excess) / n_live * normal
    return velocity_a, omega_a, velocity_b, omega_b, separation
