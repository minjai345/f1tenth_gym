"""The numpy-to-JAX boundary for wall contact.

Marshalling and dtype handling only: physics belongs in ``kernels`` or ``solver``.
Launch cost dwarfs the arithmetic for a one-body solve (``jnp.asarray`` 0.216 ms
against 0.002 ms for ``np``), hence numpy inputs and an explicit device pin.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import SingleDeviceSharding

from .kernels import body_contact, segment_contact, speculative_gap
from .solver import ContactParams, resolve, resolve_pair, speculative_clamp


def resolve_device(name: str):
    """The device the contact kernels run on.

    Availability is a property of the machine, not of the config, so this is checked
    at ``gym.make`` rather than at ``ContactConfig`` build: a config pickled from a
    GPU box to CPU workers must still construct.

    Args:
        name: ``"cpu"`` or ``"gpu"``.

    Returns:
        A ``Device``.

    Raises:
        ValueError: If JAX cannot see that backend here. Silently falling back would
            cost an order of magnitude with nothing said, which is the failure mode
            ``lidar.enabled=False`` already has.
    """
    backend_error = None
    try:
        found = jax.devices(name)
    except RuntimeError as exc:
        # On first discovery JAX initializes every registered plugin. An
        # unrelated plugin can fail after the requested backend was cached
        # (for example CUDA OOM while asking for CPU), so retry the named
        # backend before declaring it absent.
        backend_error = exc
        try:
            found = jax.devices(name)
        except RuntimeError:
            found = []
    if not found:
        try:
            available_devices = jax.devices()
        except RuntimeError:
            available_devices = []
        cached = [device for device in available_devices if device.platform == name]
        if cached:
            return cached[0]
        available = sorted({device.platform for device in available_devices})
        raise ValueError(
            f"contact_config.device={name!r} but JAX sees no {name} backend here; "
            f"available: {available}. Set JAX_PLATFORMS to include it, or choose "
            f"a device that is present."
        ) from backend_error
    return found[0]


def _pin(device):
    """jit kwargs placing the computation on ``device``."""
    return {"out_shardings": SingleDeviceSharding(device)}


def _put(array, device):
    """A closure constant on ``device``."""
    return jax.device_put(array, device)


class WallContact:
    """Resolves one track's walls against a body, holding the jitted kernels.

    Rebuild whenever the track or the vehicle changes: the tile index is sized for a
    particular body, and the solver constants are baked into the traced closure.
    """

    def __init__(self, walls, index, params: ContactParams, iterations: int, dt: float,
                 device: str = "cpu"):
        """
        Args:
            walls: A ``WallSegments`` for the track.
            index: A ``TileIndex`` built for the widest body in play.
            params: Solver tuning.
            iterations: Jacobi sweeps per call.
            dt: Simulation timestep, for the speculative clamp.
            device: ``"cpu"`` or ``"gpu"``; see ``ContactConfig.device``.
        """
        self.walls = walls
        self.index = index
        self.params = params
        self.iterations = int(iterations)
        self.dt = float(dt)
        self.is_empty = walls.is_empty
        dev = resolve_device(device)

        if self.is_empty:
            self._resolve = None
            return

        seg_a = _put(walls.a, dev)
        seg_b = _put(walls.b, dev)
        seg_n = _put(walls.n, dev)
        table = _put(index.table, dev)
        origin = _put(np.asarray(index.origin, dtype=np.float32), dev)
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

        self._resolve = jax.jit(run, **_pin(dev))

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
            np.asarray(vertices, np.float32),
            np.asarray(centre, np.float32),
            np.asarray(velocity, np.float32),
            np.float32(omega),
            np.float32(mass),
            np.float32(inertia),
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
    return WallContact(
        walls, index, params, contact_config.solver_iterations, dt, contact_config.device
    )


class BodyPairContact:
    """Resolves one pair of vehicle bodies, holding the jitted kernel."""

    def __init__(self, params: ContactParams, iterations: int, device: str = "cpu"):
        """
        Args:
            params: Solver tuning.
            iterations: Jacobi sweeps per call.
            device: ``"cpu"`` or ``"gpu"``; see ``ContactConfig.device``.
        """
        dev = resolve_device(device)

        def run(verts_a, verts_b, centre_a, centre_b, v_a, w_a, v_b, w_b, mass, inertia):
            manifold = body_contact(verts_a, verts_b)
            v_a, w_a, v_b, w_b, separation = resolve_pair(
                v_a, w_a, v_b, w_b, mass, inertia,
                manifold.points, manifold.depths, manifold.normal,
                centre_a, centre_b, params, int(iterations),
            )
            return v_a, w_a, v_b, w_b, separation, jnp.any(manifold.depths > 0.0)

        self._run = jax.jit(run, **_pin(dev))

    def __call__(self, verts_a, verts_b, centre_a, centre_b, v_a, w_a, v_b, w_b,
                 mass, inertia):
        """Resolve one body pair.

        Returns:
            ``(v_a, omega_a, v_b, omega_b, separation, in_contact)``; ``separation``
            is applied to b and its negation to a.
        """
        out = self._run(
            np.asarray(verts_a, np.float32), np.asarray(verts_b, np.float32),
            np.asarray(centre_a, np.float32), np.asarray(centre_b, np.float32),
            np.asarray(v_a, np.float32), np.float32(w_a),
            np.asarray(v_b, np.float32), np.float32(w_b),
            np.float32(mass), np.float32(inertia),
        )
        v_a, w_a, v_b, w_b, separation, hit = out
        return (
            np.asarray(v_a, np.float64), float(w_a),
            np.asarray(v_b, np.float64), float(w_b),
            np.asarray(separation, np.float64), bool(hit),
        )
