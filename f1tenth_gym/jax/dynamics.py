"""Pure JAX kernels for the supported CoG-referenced vehicle models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DynamicsParams:
    """Traced physical parameters used by KS and ST dynamics.

    Every field is a pytree leaf. A batch may therefore carry different
    vehicle parameters per environment without recompilation.
    """

    mu: Any
    C_Sf: Any
    C_Sr: Any
    lf: Any
    lr: Any
    h: Any
    m: Any
    I: Any
    s_min: Any
    s_max: Any
    sv_min: Any
    sv_max: Any
    v_switch: Any
    a_max: Any
    v_min: Any
    v_max: Any

    @classmethod
    def from_vehicle_parameters(cls, params: Any) -> "DynamicsParams":
        """Copy the active fields from the host ``VehicleParameters`` object."""
        return cls(
            **{name: getattr(params, name) for name in cls.__dataclass_fields__}
        )


def _upper_acceleration_limit(
    speed: jax.Array, params: DynamicsParams
) -> jax.Array:
    # Both branches of jnp.where are evaluated. Keep the inactive division
    # regular at zero so autodiff cannot inherit a NaN from it.
    safe_speed = jnp.maximum(speed, jnp.asarray(params.v_switch, dtype=speed.dtype))
    reduced = params.a_max * params.v_switch / safe_speed
    return jnp.where(speed > params.v_switch, reduced, params.a_max)


def _constrained_control(
    state: jax.Array, control: jax.Array, params: DynamicsParams
) -> tuple[jax.Array, jax.Array]:
    steering = state[2]
    speed = state[3]

    steering_rate = jnp.clip(control[0], params.sv_min, params.sv_max)
    steering_blocked = jnp.logical_or(
        jnp.logical_and(steering <= params.s_min, steering_rate <= 0.0),
        jnp.logical_and(steering >= params.s_max, steering_rate >= 0.0),
    )
    steering_rate = jnp.where(steering_blocked, 0.0, steering_rate)

    acceleration = jnp.clip(
        control[1], -params.a_max, _upper_acceleration_limit(speed, params)
    )
    acceleration_blocked = jnp.logical_or(
        jnp.logical_and(speed <= params.v_min, acceleration <= 0.0),
        jnp.logical_and(speed >= params.v_max, acceleration >= 0.0),
    )
    acceleration = jnp.where(acceleration_blocked, 0.0, acceleration)
    return steering_rate, acceleration


def kinematic_single_track(
    state: jax.Array, control: jax.Array, params: DynamicsParams
) -> jax.Array:
    """CoG-referenced kinematic single-track derivative.

    State is ``[x, y, steering, speed, yaw]`` and control is
    ``[steering_rate, acceleration]``.
    """
    state = jnp.asarray(state)
    control = jnp.asarray(control)
    steering = state[2]
    speed = state[3]
    yaw = state[4]
    steering_rate, acceleration = _constrained_control(state, control, params)
    wheelbase = params.lf + params.lr
    beta = jnp.arctan(jnp.tan(steering) * params.lr / wheelbase)
    yaw_rate = speed * jnp.cos(beta) * jnp.tan(steering) / wheelbase
    return jnp.stack(
        (
            speed * jnp.cos(yaw + beta),
            speed * jnp.sin(yaw + beta),
            steering_rate,
            acceleration,
            yaw_rate,
        )
    )


def _single_track_kinematic_branch(
    state: jax.Array,
    steering_rate: jax.Array,
    acceleration: jax.Array,
    params: DynamicsParams,
) -> jax.Array:
    steering = state[2]
    speed = state[3]
    yaw = state[4]
    beta_state = state[6]
    wheelbase = params.lf + params.lr
    beta = jnp.arctan(jnp.tan(steering) * params.lr / wheelbase)
    beta_rate = (
        1.0
        / (1.0 + (jnp.tan(steering) * params.lr / wheelbase) ** 2)
        * params.lr
        / (wheelbase * jnp.cos(steering) ** 2)
        * steering_rate
    )
    yaw_rate = speed * jnp.cos(beta) * jnp.tan(steering) / wheelbase
    yaw_acceleration = (
        acceleration * jnp.cos(beta_state) * jnp.tan(steering)
        - speed * jnp.sin(beta_state) * jnp.tan(steering) * beta_rate
        + speed
        * jnp.cos(beta_state)
        * steering_rate
        / jnp.cos(steering) ** 2
    ) / wheelbase
    return jnp.stack(
        (
            speed * jnp.cos(yaw + beta),
            speed * jnp.sin(yaw + beta),
            steering_rate,
            acceleration,
            yaw_rate,
            yaw_acceleration,
            beta_rate,
        )
    )


def _single_track_dynamic_branch(
    state: jax.Array,
    steering_rate: jax.Array,
    acceleration: jax.Array,
    params: DynamicsParams,
) -> jax.Array:
    steering = state[2]
    speed = state[3]
    yaw = state[4]
    yaw_rate = state[5]
    beta = state[6]
    gravity = jnp.asarray(9.81, dtype=state.dtype)
    rear_load_term = gravity * params.lr - acceleration * params.h
    front_load_term = gravity * params.lf + acceleration * params.h
    wheelbase = params.lf + params.lr

    yaw_acceleration = (params.mu * params.m / (params.I * wheelbase)) * (
        params.lf * params.C_Sf * rear_load_term * steering
        + (
            params.lr * params.C_Sr * front_load_term
            - params.lf * params.C_Sf * rear_load_term
        )
        * beta
        - (
            params.lf**2 * params.C_Sf * rear_load_term
            + params.lr**2 * params.C_Sr * front_load_term
        )
        * yaw_rate
        / speed
    )
    beta_rate = (params.mu / (speed * wheelbase)) * (
        params.C_Sf * rear_load_term * steering
        - (
            params.C_Sr * front_load_term
            + params.C_Sf * rear_load_term
        )
        * beta
        + (
            params.C_Sr * front_load_term * params.lr
            - params.C_Sf * rear_load_term * params.lf
        )
        * yaw_rate
        / speed
    ) - yaw_rate
    return jnp.stack(
        (
            speed * jnp.cos(yaw + beta),
            speed * jnp.sin(yaw + beta),
            steering_rate,
            acceleration,
            yaw_rate,
            yaw_acceleration,
            beta_rate,
        )
    )


def single_track(
    state: jax.Array, control: jax.Array, params: DynamicsParams
) -> jax.Array:
    """Current discontinuous ST derivative with a safe low-speed branch.

    The reference model switches when signed speed is below ``0.5 m/s``.
    Consequently all reverse motion uses the kinematic branch. ``lax.cond`` is
    deliberate: it prevents the inactive dynamic branch's divisions by speed
    from poisoning zero-speed gradients.
    """
    state = jnp.asarray(state)
    control = jnp.asarray(control)
    steering_rate, acceleration = _constrained_control(state, control, params)
    operands = (state, steering_rate, acceleration, params)
    return jax.lax.cond(
        state[3] < 0.5,
        lambda values: _single_track_kinematic_branch(*values),
        lambda values: _single_track_dynamic_branch(*values),
        operands,
    )
