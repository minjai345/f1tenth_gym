"""Functional JAX integration kernels."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp


DynamicsFn = Callable[[jax.Array, jax.Array, Any], jax.Array]
StepFn = Callable[[DynamicsFn, jax.Array, jax.Array, Any, Any], jax.Array]


def _wrap_yaw(state: jax.Array) -> jax.Array:
    yaw = jnp.arctan2(jnp.sin(state[4]), jnp.cos(state[4]))
    return state.at[4].set(yaw)


def euler_step(
    dynamics: DynamicsFn,
    state: jax.Array,
    control: jax.Array,
    dt: Any,
    params: Any,
) -> jax.Array:
    """Advance one Euler step and wrap yaw to ``[-pi, pi]``."""
    return _wrap_yaw(state + dt * dynamics(state, control, params))


def rk4_step(
    dynamics: DynamicsFn,
    state: jax.Array,
    control: jax.Array,
    dt: Any,
    params: Any,
) -> jax.Array:
    """Advance one classical fourth-order Runge-Kutta step."""
    k1 = dynamics(state, control, params)
    k2 = dynamics(state + 0.5 * dt * k1, control, params)
    k3 = dynamics(state + 0.5 * dt * k2, control, params)
    k4 = dynamics(state + dt * k3, control, params)
    return _wrap_yaw(state + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0)


def integrate_substeps(
    step_fn: StepFn,
    dynamics: DynamicsFn,
    state: jax.Array,
    control: jax.Array,
    timestep: Any,
    params: Any,
    *,
    num_substeps: int,
) -> jax.Array:
    """Integrate a fixed number of equal substeps with ``lax.fori_loop``."""
    if num_substeps < 1:
        raise ValueError(f"num_substeps must be >= 1, got {num_substeps}")
    dt = timestep / num_substeps

    def body(_index: int, current: jax.Array) -> jax.Array:
        return step_fn(dynamics, current, control, dt, params)

    return jax.lax.fori_loop(0, num_substeps, body, state)
