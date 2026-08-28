"""Simultaneous fixed-pair Jacobi contact for multiple vehicle bodies."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from f1tenth_gym.envs.contact.kernels import body_contact
from f1tenth_gym.envs.contact.solver import ContactParams

from .contact import (
    WallContactConfig,
    apply_contact_response,
    resolve_wall_contacts,
    world_velocity,
)
from .dynamics import DynamicsParams
from .geometry import BodyParams, body_vertices
from .track import TrackTable


@partial(
    jax.tree_util.register_dataclass,
    data_fields=("indices", "mask"),
    meta_fields=("num_agents",),
)
@dataclass(frozen=True)
class PairTable:
    """Fixed-capacity body-pair indexes and their validity mask."""

    indices: jax.Array
    mask: jax.Array
    num_agents: int


@dataclass(frozen=True)
class PairContactConfig:
    """Hashable simultaneous-pair solver topology."""

    num_agents: int
    state_dim: int
    solver_iterations: int = 64
    multi_relaxation: float = 1.0

    def __post_init__(self) -> None:
        if self.num_agents < 1:
            raise ValueError(f"num_agents must be >= 1, got {self.num_agents}")
        if self.state_dim not in (5, 7):
            raise ValueError(
                f"state_dim must be 5 (KS) or 7 (ST), got {self.state_dim}"
            )
        iterations = int(self.solver_iterations)
        if iterations < 1:
            raise ValueError(
                f"solver_iterations must be >= 1, got {iterations}"
            )
        object.__setattr__(self, "solver_iterations", iterations)
        relaxation = float(self.multi_relaxation)
        if not 0.0 < relaxation <= 1.0:
            raise ValueError(
                "multi_relaxation must be in (0, 1], got "
                f"{relaxation}"
            )
        object.__setattr__(self, "multi_relaxation", relaxation)


def make_pair_table(num_agents: int, capacity: int | None = None) -> PairTable:
    """Build every unordered agent pair, padded with masked self-pairs."""
    if num_agents < 1:
        raise ValueError(f"num_agents must be >= 1, got {num_agents}")
    pairs = tuple(
        (left, right)
        for left in range(num_agents)
        for right in range(left + 1, num_agents)
    )
    minimum = len(pairs)
    if capacity is None:
        capacity = max(minimum, 1)
    capacity = int(capacity)
    if capacity < max(minimum, 1):
        raise ValueError(
            f"capacity must be >= {max(minimum, 1)}, got {capacity}"
        )
    padded = pairs + ((0, 0),) * (capacity - minimum)
    mask = (True,) * minimum + (False,) * (capacity - minimum)
    return PairTable(
        indices=jnp.asarray(padded, dtype=jnp.int32),
        mask=jnp.asarray(mask, dtype=jnp.bool_),
        num_agents=num_agents,
    )


def _cross(offset: jax.Array, vector: jax.Array) -> jax.Array:
    return offset[..., 0] * vector[..., 1] - offset[..., 1] * vector[..., 0]


def _contact_velocity(
    velocity: jax.Array,
    omega: jax.Array,
    offset: jax.Array,
) -> jax.Array:
    rotation = omega[..., None, None] * jnp.stack(
        (-offset[..., 1], offset[..., 0]), axis=-1
    )
    return velocity[..., None, :] + rotation


def solve_pair_impulses(
    velocities: jax.Array,
    omegas: jax.Array,
    centres: jax.Array,
    table: PairTable,
    points: jax.Array,
    depths: jax.Array,
    normals: jax.Array,
    mass: jax.Array,
    inertia: jax.Array,
    params: ContactParams,
    iterations: int,
    multi_relaxation: float = 1.0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Solve all pair manifolds from shared body velocities each Jacobi sweep.

    Every pair proposes equal-and-opposite impulses from the same global body
    state. Proposals are scatter-added by body and applied once per sweep. A
    pair uses one relaxation factor based on the larger endpoint degree, so
    both ends receive exactly opposite impulses and momentum is conserved.
    """
    if table.num_agents != velocities.shape[0]:
        raise ValueError(
            "pair table and velocity agent counts must match, got "
            f"{table.num_agents} and {velocities.shape[0]}"
        )
    left = table.indices[:, 0]
    right = table.indices[:, 1]
    centre_left = centres[left]
    centre_right = centres[right]
    offset_left = points - centre_left[:, None, :]
    offset_right = points - centre_right[:, None, :]
    pair_normals = jnp.broadcast_to(normals[:, None, :], points.shape)
    live = (depths > 0.0) & table.mask[:, None]
    pair_events = jnp.any(live, axis=1)
    pair_weight = pair_events.astype(velocities.dtype)
    degree = jnp.zeros((velocities.shape[0],), dtype=velocities.dtype)
    degree = degree.at[left].add(pair_weight)
    degree = degree.at[right].add(pair_weight)
    pair_degree = jnp.maximum(jnp.maximum(degree[left], degree[right]), 1.0)
    relaxation = jnp.where(
        pair_degree > 1.0,
        jnp.asarray(multi_relaxation, dtype=velocities.dtype) / pair_degree,
        1.0,
    )
    live_count = jnp.maximum(jnp.sum(live, axis=1), 1.0)
    inverse_mass = 1.0 / mass
    inverse_inertia = 1.0 / inertia

    def relative_velocity(current_v, current_w):
        velocity_left = _contact_velocity(
            current_v[left], current_w[left], offset_left
        )
        velocity_right = _contact_velocity(
            current_v[right], current_w[right], offset_right
        )
        return velocity_right - velocity_left

    initial_relative = relative_velocity(velocities, omegas)
    approach = jnp.sum(initial_relative * pair_normals, axis=-1)
    bounce = jnp.where(
        -approach > params.restitution_threshold,
        params.restitution * approach,
        0.0,
    )
    normal_arm_left = _cross(offset_left, pair_normals)
    normal_arm_right = _cross(offset_right, pair_normals)
    effective_normal_mass = jnp.maximum(
        2.0 * inverse_mass
        + (normal_arm_left**2 + normal_arm_right**2) * inverse_inertia,
        1.0e-12,
    )

    def sweep(_index, carry):
        current_v, current_w, accumulated_normal = carry
        relative = relative_velocity(current_v, current_w)
        normal_speed = jnp.sum(relative * pair_normals, axis=-1)
        proposed = -(normal_speed + bounce) / effective_normal_mass
        clamped = jnp.maximum(accumulated_normal + proposed, 0.0)
        normal_impulse = jnp.where(
            live,
            (clamped - accumulated_normal)
            * relaxation[:, None]
            / live_count[:, None],
            0.0,
        )
        accumulated_normal = accumulated_normal + normal_impulse

        tangent_velocity = relative - normal_speed[..., None] * pair_normals
        tangent_speed = jnp.linalg.norm(tangent_velocity, axis=-1, keepdims=True)
        tangent = jnp.where(
            tangent_speed > 1.0e-12,
            tangent_velocity / (tangent_speed + 1.0e-12),
            jnp.stack((-pair_normals[..., 1], pair_normals[..., 0]), axis=-1),
        )
        tangent_arm_left = _cross(offset_left, tangent)
        tangent_arm_right = _cross(offset_right, tangent)
        effective_tangent_mass = jnp.maximum(
            2.0 * inverse_mass
            + (tangent_arm_left**2 + tangent_arm_right**2) * inverse_inertia,
            1.0e-12,
        )
        friction_bound = params.friction * accumulated_normal
        tangent_impulse = jnp.clip(
            -jnp.sum(relative * tangent, axis=-1) / effective_tangent_mass,
            -friction_bound,
            friction_bound,
        )
        tangent_impulse = jnp.where(
            live,
            tangent_impulse * relaxation[:, None] / live_count[:, None],
            0.0,
        )

        impulse = (
            normal_impulse[..., None] * pair_normals
            + tangent_impulse[..., None] * tangent
        )
        total = jnp.sum(impulse, axis=1)
        velocity_delta = jnp.zeros_like(current_v)
        velocity_delta = velocity_delta.at[left].add(-total * inverse_mass)
        velocity_delta = velocity_delta.at[right].add(total * inverse_mass)
        omega_delta = jnp.zeros_like(current_w)
        omega_delta = omega_delta.at[left].add(
            -jnp.sum(_cross(offset_left, impulse), axis=1) * inverse_inertia
        )
        omega_delta = omega_delta.at[right].add(
            jnp.sum(_cross(offset_right, impulse), axis=1) * inverse_inertia
        )
        return (
            current_v + velocity_delta,
            current_w + omega_delta,
            accumulated_normal,
        )

    velocities, omegas, _accumulated = jax.lax.fori_loop(
        0,
        iterations,
        sweep,
        (velocities, omegas, jnp.zeros_like(depths)),
    )

    excess = jnp.maximum(depths - params.slop, 0.0) * live
    separation = (
        0.5
        * params.baumgarte
        * jnp.sum(excess, axis=1)
        / live_count
        * relaxation
    )[:, None] * normals
    separation = jnp.where(pair_events[:, None], separation, 0.0)
    correction = jnp.zeros_like(centres)
    correction = correction.at[left].add(-separation)
    correction = correction.at[right].add(separation)
    return velocities, omegas, correction, pair_events


def resolve_pair_contacts(
    model_state: jax.Array,
    table: PairTable,
    body: BodyParams,
    dynamics: DynamicsParams,
    params: ContactParams,
    config: PairContactConfig,
) -> tuple[jax.Array, jax.Array]:
    """Resolve every vehicle pair simultaneously and return agent events."""
    expected = (config.num_agents, config.state_dim)
    if model_state.shape != expected:
        raise ValueError(
            f"model_state must have shape {expected}, got {model_state.shape}"
        )
    if table.num_agents != config.num_agents:
        raise ValueError(
            "pair table and contact config agent counts must match, got "
            f"{table.num_agents} and {config.num_agents}"
        )
    if table.indices.ndim != 2 or table.indices.shape[1] != 2:
        raise ValueError(
            f"pair indices must have shape (pairs, 2), got {table.indices.shape}"
        )
    if table.mask.shape != (table.indices.shape[0],):
        raise ValueError(
            f"pair mask must have shape ({table.indices.shape[0]},), "
            f"got {table.mask.shape}"
        )

    poses = model_state[:, jnp.asarray((0, 1, 4))]
    vertices = jax.vmap(lambda pose: body_vertices(pose, body))(poses)
    left = table.indices[:, 0]
    right = table.indices[:, 1]
    lower = jnp.min(vertices, axis=1)
    upper = jnp.max(vertices, axis=1)
    broad = jnp.all(upper[left] >= lower[right], axis=1) & jnp.all(
        upper[right] >= lower[left], axis=1
    )
    valid = table.mask & broad
    manifolds = jax.vmap(
        lambda left_vertices, right_vertices, valid: body_contact(
            left_vertices, right_vertices, valid
        )
    )(vertices[left], vertices[right], valid)
    velocities, omegas = jax.vmap(lambda state: world_velocity(state, dynamics))(
        model_state
    )
    centres = model_state[:, :2]
    velocities, omegas, correction, pair_events = solve_pair_impulses(
        velocities,
        omegas,
        centres,
        table,
        manifolds.points,
        manifolds.depths,
        manifolds.normal,
        dynamics.m,
        dynamics.I,
        params,
        config.solver_iterations,
        config.multi_relaxation,
    )
    incident = jnp.zeros((config.num_agents,), dtype=jnp.int32)
    incident = incident.at[left].add(pair_events.astype(jnp.int32))
    incident = incident.at[right].add(pair_events.astype(jnp.int32))
    events = incident > 0
    corrected = jax.vmap(
        lambda state, velocity, omega, shift: apply_contact_response(
            state, velocity, omega, shift, dynamics
        )
    )(model_state, velocities, omegas, correction)
    return jnp.where(events[:, None], corrected, model_state), events


def resolve_contacts(
    model_state: jax.Array,
    track: TrackTable,
    table: PairTable,
    body: BodyParams,
    dynamics: DynamicsParams,
    params: ContactParams,
    timestep: jax.Array,
    wall_config: WallContactConfig,
    pair_config: PairContactConfig,
) -> tuple[jax.Array, jax.Array]:
    """Apply wall response, then simultaneous pairs, and OR fresh events."""
    if (
        wall_config.num_agents != pair_config.num_agents
        or wall_config.state_dim != pair_config.state_dim
    ):
        raise ValueError("wall and pair contact topology must match")
    wall_state, wall_events = resolve_wall_contacts(
        model_state,
        track,
        body,
        dynamics,
        params,
        timestep,
        wall_config,
    )
    pair_state, pair_events = resolve_pair_contacts(
        wall_state, table, body, dynamics, params, pair_config
    )
    return pair_state, wall_events | pair_events


__all__ = [
    "PairContactConfig",
    "PairTable",
    "make_pair_table",
    "resolve_contacts",
    "resolve_pair_contacts",
    "solve_pair_impulses",
]
