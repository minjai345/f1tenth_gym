"""Pure JAX action adapters matching the environment control semantics."""

from __future__ import annotations

from enum import IntEnum
from typing import Any

import jax
import jax.numpy as jnp


class LongitudinalControlMode(IntEnum):
    """Interpret the second action column as acceleration or target speed."""

    ACCELERATION = 1
    TARGET_SPEED = 2


class SteeringControlMode(IntEnum):
    """Interpret the first action column as target angle or steering rate."""

    TARGET_ANGLE = 1
    STEERING_RATE = 2


def speed_control(target: jax.Array, state: jax.Array, params: Any) -> jax.Array:
    """Current four-quadrant target-speed controller.

    ``state`` may be one vehicle or a leading batch of vehicles. The final
    state axis follows the native KS/ST layout, where speed is column three.
    """
    current_speed = state[..., 3]
    error = target - current_speed
    forward_accel = 10.0 * params.a_max / params.v_max
    forward_brake = 10.0 * params.a_max / (-params.v_min)
    reverse_brake = 2.0 * params.a_max / params.v_max
    reverse_accel = 2.0 * params.a_max / (-params.v_min)
    gain = jnp.where(
        current_speed > 0.0,
        jnp.where(error > 0.0, forward_accel, forward_brake),
        jnp.where(error > 0.0, reverse_brake, reverse_accel),
    )
    return gain * error


def steering_angle_control(
    target: jax.Array,
    state: jax.Array,
    params: Any,
    gain: jax.Array,
) -> jax.Array:
    """Saturated steering P controller with the legacy relay escape hatch."""
    error = target - state[..., 2]
    relay = jnp.where(
        jnp.abs(error) > 1.0e-4,
        jnp.sign(error) * params.sv_max,
        0.0,
    )
    proportional = jnp.clip(gain * error, -params.sv_max, params.sv_max)
    return jnp.where(gain <= 0.0, relay, proportional)


def adapt_actions(
    actions: jax.Array,
    state: jax.Array,
    params: Any,
    *,
    longitudinal_mode: LongitudinalControlMode,
    steering_mode: SteeringControlMode,
    steer_kp: jax.Array,
) -> jax.Array:
    """Convert native ``[steering, longitudinal]`` commands to model efforts."""
    if steering_mode is SteeringControlMode.TARGET_ANGLE:
        steering = steering_angle_control(actions[..., 0], state, params, steer_kp)
    elif steering_mode is SteeringControlMode.STEERING_RATE:
        steering = actions[..., 0]
    else:
        raise ValueError(f"unsupported steering control mode: {steering_mode!r}")

    if longitudinal_mode is LongitudinalControlMode.TARGET_SPEED:
        longitudinal = speed_control(actions[..., 1], state, params)
    elif longitudinal_mode is LongitudinalControlMode.ACCELERATION:
        longitudinal = actions[..., 1]
    else:
        raise ValueError(f"unsupported longitudinal control mode: {longitudinal_mode!r}")

    return jnp.stack((steering, longitudinal), axis=-1)


__all__ = [
    "LongitudinalControlMode",
    "SteeringControlMode",
    "adapt_actions",
    "speed_control",
    "steering_angle_control",
]
