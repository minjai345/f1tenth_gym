"""The numpy-to-JAX boundary for wall contact.

The only file here that knows the gym is currently numpy. Marshalling and dtype
handling only: anything resembling physics belongs in ``kernels`` or ``solver``, or
the migration seam stops being a deletion.
"""

import jax
import jax.numpy as jnp
import numpy as np

from .kernels import segment_contact, speculative_gap
from .solver import ContactParams, resolve, speculative_clamp


class WallContact:
    """Resolves one track's walls against a body, holding the jitted kernels.

    Rebuild whenever the track or the vehicle changes: the tile index is sized for a
    particular body, and the solver constants are baked into the traced closure.
    """

    def __init__(self, walls, index, params: ContactParams, iterations: int, dt: float):
        """
        Args:
            walls: A ``WallSegments`` for the track.
            index: A ``TileIndex`` built for the widest body in play.
            params: Solver tuning.
            iterations: Jacobi sweeps per call.
            dt: Simulation timestep, for the speculative clamp.
        """
        self.walls = walls
        self.index = index
        self.params = params
        self.iterations = int(iterations)
        self.dt = float(dt)
        self.is_empty = walls.is_empty

        if self.is_empty:
            self._resolve = None
            return

        seg_a = jnp.asarray(walls.a)
        seg_b = jnp.asarray(walls.b)
        seg_n = jnp.asarray(walls.n)
        table = jnp.asarray(index.table)
        origin = jnp.asarray(index.origin, dtype=jnp.float32)
        tile = float(index.tile_size)
        rows, cols = int(table.shape[0]), int(table.shape[1])

        def run(verts, centre, velocity, omega, mass, inertia):
            col = jnp.clip(((centre[0] - origin[0]) / tile).astype(jnp.int32), 0, cols - 1)
            row = jnp.clip(((centre[1] - origin[1]) / tile).astype(jnp.int32), 0, rows - 1)
            cand = table[row, col]
            ok = cand >= 0
            idx = jnp.where(ok, cand, 0)

            manifolds = jax.vmap(
                lambda k, live: segment_contact(verts, seg_a[k], seg_b[k], seg_n[k], live)
            )(idx, ok)
            gaps = jax.vmap(
                lambda k, live: speculative_gap(verts, seg_a[k], seg_b[k], seg_n[k], live)
            )(idx, ok)

            velocity = speculative_clamp(velocity, gaps, seg_n[idx], self.dt)
            points = manifolds.points.reshape(-1, 2)
            depths = manifolds.depths.reshape(-1)
            normals = jnp.repeat(seg_n[idx], 2, axis=0)
            velocity, omega, correction = resolve(
                velocity, omega, mass, inertia, points, depths, normals,
                centre, params, self.iterations,
            )
            return velocity, omega, correction, jnp.any(depths > 0.0)

        self._resolve = jax.jit(run)

    def __call__(self, vertices, centre, velocity, omega, mass, inertia):
        """Resolve one body against the walls.

        Args:
            vertices: (4, 2) collision-body corners, counter-clockwise.
            centre: (2,) centre of mass in world metres.
            velocity: (2,) world linear velocity.
            omega: Scalar angular velocity.
            mass: Body mass, kg.
            inertia: Yaw inertia, kg m^2.

        Returns:
            ``(velocity, omega, correction, in_contact)`` as numpy/float/bool.
        """
        if self._resolve is None:
            return np.asarray(velocity, np.float64), float(omega), np.zeros(2), False
        v, w, correction, hit = self._resolve(
            jnp.asarray(vertices, jnp.float32),
            jnp.asarray(centre, jnp.float32),
            jnp.asarray(velocity, jnp.float32),
            jnp.float32(omega),
            jnp.float32(mass),
            jnp.float32(inertia),
        )
        return (
            np.asarray(v, dtype=np.float64),
            float(w),
            np.asarray(correction, dtype=np.float64),
            bool(hit),
        )


def build(track, vehicle_params, contact_config, dt, dr_config=None) -> WallContact:
    """Extract walls, size the broad phase and compile the kernels for one track.

    Args:
        track: A ``Track``.
        vehicle_params: Nominal ``VehicleParameters``.
        contact_config: A ``ContactConfig``.
        dt: Simulation timestep.
        dr_config: A ``DomainRandomizationConfig``, or None. Sizes the broad phase
            at the widest body randomization can produce.

    Returns:
        A :class:`WallContact`.
    """
    from ..track.accel import build_for_track

    walls, _budget, index = build_for_track(
        track,
        vehicle_params,
        dr_config,
        tile_size=contact_config.tile_size,
        margin=contact_config.margin,
    )
    params = ContactParams(
        restitution=contact_config.restitution,
        friction=contact_config.friction,
        restitution_threshold=contact_config.restitution_threshold,
        baumgarte=contact_config.baumgarte,
        slop=contact_config.slop,
    )
    return WallContact(walls, index, params, contact_config.solver_iterations, dt)
