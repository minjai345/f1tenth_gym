"""Pure functional orchestration for fixed-shape wall contact."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from f1tenth_gym.envs.contact.kernels import segment_contact, speculative_gap
from f1tenth_gym.envs.contact.solver import (
    ContactParams,
    resolve,
    speculative_clamp,
)

from ..dynamic_models.jax import DynamicsParams
from ..track.functional import TrackTable, tile_candidates
from .geometry import BodyParams, body_vertices


@dataclass(frozen=True)
class WallContactConfig:
    """Hashable wall-solver topology for one compiled environment."""

    num_agents: int
    state_dim: int
    solver_iterations: int = 64

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
                "solver_iterations must be >= 1, got "
                f"{iterations}"
            )
        object.__setattr__(self, "solver_iterations", iterations)


def world_velocity(
    state: jax.Array,
    dynamics: DynamicsParams,
) -> tuple[jax.Array, jax.Array]:
    """Convert one supported native state to world linear/angular velocity."""
    yaw = state[4]
    speed = state[3]
    if state.shape[0] == 7:
        course = yaw + state[6]
        omega = state[5]
    elif state.shape[0] == 5:
        wheelbase = dynamics.lf + dynamics.lr
        beta = jnp.arctan(jnp.tan(state[2]) * dynamics.lr / wheelbase)
        course = yaw + beta
        omega = speed * jnp.cos(beta) * jnp.tan(state[2]) / wheelbase
    else:
        raise ValueError(
            f"state must have length 5 (KS) or 7 (ST), got {state.shape[0]}"
        )
    velocity = speed * jnp.stack((jnp.cos(course), jnp.sin(course)))
    return velocity, omega


def apply_contact_response(
    state: jax.Array,
    velocity: jax.Array,
    omega: jax.Array,
    correction: jax.Array,
    dynamics: DynamicsParams,
) -> jax.Array:
    """Project a rigid-body contact result back into a native KS or ST state."""
    corrected = state.at[:2].add(correction)
    yaw = corrected[4]
    if state.shape[0] == 7:
        speed = jnp.linalg.norm(velocity)
        course = jnp.arctan2(velocity[1], velocity[0])
        beta = jnp.arctan2(jnp.sin(course - yaw), jnp.cos(course - yaw))
        reverse = jnp.abs(beta) > jnp.pi / 2.0
        speed = jnp.where(reverse, -speed, speed)
        beta = jnp.where(
            reverse,
            jnp.arctan2(
                jnp.sin(beta - jnp.copysign(jnp.pi, beta)),
                jnp.cos(beta - jnp.copysign(jnp.pi, beta)),
            ),
            beta,
        )
        corrected = corrected.at[3].set(speed)
        corrected = corrected.at[5].set(omega)
        corrected = corrected.at[6].set(beta)
    elif state.shape[0] == 5:
        wheelbase = dynamics.lf + dynamics.lr
        beta = jnp.arctan(jnp.tan(corrected[2]) * dynamics.lr / wheelbase)
        course = yaw + beta
        speed = jnp.dot(velocity, jnp.stack((jnp.cos(course), jnp.sin(course))))
        corrected = corrected.at[3].set(speed)
    else:
        raise ValueError(
            f"state must have length 5 (KS) or 7 (ST), got {state.shape[0]}"
        )
    wrapped_yaw = jnp.arctan2(jnp.sin(corrected[4]), jnp.cos(corrected[4]))
    return corrected.at[4].set(wrapped_yaw)


def _resolve_one_wall_contact(
    state: jax.Array,
    track: TrackTable,
    body: BodyParams,
    dynamics: DynamicsParams,
    params: ContactParams,
    timestep: jax.Array,
    iterations: int,
) -> tuple[jax.Array, jax.Array]:
    pose = state[jnp.asarray((0, 1, 4))]
    centre = state[:2]
    vertices = body_vertices(pose, body)

    # Preserve the mutable simulator's current broad-phase lookup point. The
    # collision body may have an offset, but the tile query is anchored at CoG.
    candidates, tile_mask = tile_candidates(track.contact_tiles, centre)
    valid = tile_mask & track.walls.mask[candidates]
    seg_a = track.walls.a[candidates]
    seg_b = track.walls.b[candidates]
    normals = track.walls.normals[candidates]

    manifolds = jax.vmap(
        lambda a, b, normal, live: segment_contact(
            vertices, a, b, normal, live
        )
    )(seg_a, seg_b, normals, valid)
    gaps = jax.vmap(
        lambda a, b, normal, live: speculative_gap(
            vertices, a, b, normal, live
        )
    )(seg_a, seg_b, normals, valid)

    original_velocity, original_omega = world_velocity(state, dynamics)
    velocity = speculative_clamp(original_velocity, gaps, normals, timestep)
    points = manifolds.points.reshape((-1, 2))
    depths = manifolds.depths.reshape((-1,))
    contact_normals = jnp.repeat(normals, 2, axis=0)
    velocity, omega, correction = resolve(
        velocity,
        original_omega,
        dynamics.m,
        dynamics.I,
        points,
        depths,
        contact_normals,
        centre,
        params,
        iterations,
    )
    hit = jnp.any(depths > 0.0)
    corrected = apply_contact_response(
        state, velocity, omega, correction, dynamics
    )
    # Current host parity: a speculative clamp without a penetrating manifold
    # is calculated inside WallContact but discarded by F110Simulator.
    return jnp.where(hit, corrected, state), hit


def resolve_wall_contacts(
    model_state: jax.Array,
    track: TrackTable,
    body: BodyParams,
    dynamics: DynamicsParams,
    params: ContactParams,
    timestep: jax.Array,
    config: WallContactConfig,
) -> tuple[jax.Array, jax.Array]:
    """Resolve all agents against walls and return fresh per-step events."""
    expected = (config.num_agents, config.state_dim)
    if model_state.shape != expected:
        raise ValueError(
            f"model_state must have shape {expected}, got {model_state.shape}"
        )
    return jax.vmap(
        lambda state: _resolve_one_wall_contact(
            state,
            track,
            body,
            dynamics,
            params,
            timestep,
            config.solver_iterations,
        )
    )(model_state)


__all__ = [
    "WallContactConfig",
    "apply_contact_response",
    "resolve_wall_contacts",
    "world_velocity",
]
