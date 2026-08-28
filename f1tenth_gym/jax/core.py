"""Device-resident state and free-flight transition for the functional core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp

from .controls import (
    LongitudinalControlMode,
    SteeringControlMode,
    adapt_actions,
)
from .dynamics import DynamicsParams
from .integrators import StepFn, integrate_substeps


DynamicsFn = Callable[[jax.Array, jax.Array, DynamicsParams], jax.Array]


@dataclass(frozen=True)
class DynamicsConfig:
    """Hashable structural choices for a compiled free-flight transition."""

    num_agents: int
    state_dim: int
    dynamics_fn: DynamicsFn
    integrator_fn: StepFn
    num_substeps: int = 1
    longitudinal_mode: LongitudinalControlMode = LongitudinalControlMode.TARGET_SPEED
    steering_mode: SteeringControlMode = SteeringControlMode.TARGET_ANGLE
    steer_delay_steps: int = 0
    throttle_delay_steps: int = 0
    derive_steer_kp: bool = True

    def __post_init__(self) -> None:
        if self.num_agents < 1:
            raise ValueError(f"num_agents must be >= 1, got {self.num_agents}")
        if self.state_dim not in (5, 7):
            raise ValueError(f"state_dim must be 5 (KS) or 7 (ST), got {self.state_dim}")
        if self.num_substeps < 1:
            raise ValueError(f"num_substeps must be >= 1, got {self.num_substeps}")
        if self.steer_delay_steps < 0:
            raise ValueError(
                f"steer_delay_steps must be >= 0, got {self.steer_delay_steps}"
            )
        if self.throttle_delay_steps < 0:
            raise ValueError(
                "throttle_delay_steps must be >= 0, got "
                f"{self.throttle_delay_steps}"
            )
        if not isinstance(self.longitudinal_mode, LongitudinalControlMode):
            raise TypeError("longitudinal_mode must be a LongitudinalControlMode")
        if not isinstance(self.steering_mode, SteeringControlMode):
            raise TypeError("steering_mode must be a SteeringControlMode")
        if not callable(self.dynamics_fn) or not callable(self.integrator_fn):
            raise TypeError("dynamics_fn and integrator_fn must be callable")


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EpisodeParams:
    """Traced values that may vary between environments without recompiling."""

    dynamics: DynamicsParams
    timestep: Any
    steer_kp: Any = 0.0
    steer_noise_std: Any = 0.0
    accel_noise_std: Any = 0.0


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DynamicsState:
    """Immutable, fixed-shape state for free-flight dynamics and actuation."""

    model: jax.Array
    control_input: jax.Array
    steer_delay_buffer: jax.Array
    steer_delay_head: jax.Array
    throttle_delay_buffer: jax.Array
    throttle_delay_head: jax.Array
    sim_time: jax.Array


def make_dynamics_state(
    model_state: jax.Array,
    config: DynamicsConfig,
) -> DynamicsState:
    """Construct zeroed control/FIFO state around native KS or ST state."""
    model = jnp.asarray(model_state)
    expected = (config.num_agents, config.state_dim)
    if model.shape != expected:
        raise ValueError(f"model_state must have shape {expected}, got {model.shape}")
    dtype = model.dtype
    return DynamicsState(
        model=model,
        control_input=jnp.zeros((config.num_agents, 2), dtype=dtype),
        steer_delay_buffer=jnp.zeros(
            (config.num_agents, config.steer_delay_steps), dtype=dtype
        ),
        steer_delay_head=jnp.zeros((config.num_agents,), dtype=jnp.int32),
        throttle_delay_buffer=jnp.zeros(
            (config.num_agents, config.throttle_delay_steps), dtype=dtype
        ),
        throttle_delay_head=jnp.zeros((config.num_agents,), dtype=jnp.int32),
        sim_time=jnp.asarray(0.0, dtype=dtype),
    )


def _push_fifo(
    values: jax.Array,
    buffer: jax.Array,
    head: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    rows = jnp.arange(values.shape[0], dtype=jnp.int32)
    delayed = buffer[rows, head]
    next_buffer = buffer.at[rows, head].set(values)
    next_head = (head + 1) % buffer.shape[1]
    return delayed, next_buffer, next_head


def step_dynamics(
    key: jax.Array,
    state: DynamicsState,
    actions: jax.Array,
    config: DynamicsConfig,
    episode: EpisodeParams,
) -> DynamicsState:
    """Advance control adapters, FIFO delays, and all agents by one timestep."""
    expected_state = (config.num_agents, config.state_dim)
    if state.model.shape != expected_state:
        raise ValueError(f"state.model must have shape {expected_state}, got {state.model.shape}")
    expected_action = (config.num_agents, 2)
    if actions.shape != expected_action:
        raise ValueError(f"actions must have shape {expected_action}, got {actions.shape}")

    actions = jnp.asarray(actions, dtype=state.model.dtype)
    steer_key, accel_key = jax.random.split(key)
    steer_commands = actions[:, 0] + episode.steer_noise_std * jax.random.normal(
        steer_key, (config.num_agents,), dtype=state.model.dtype
    )
    accel_commands = actions[:, 1] + episode.accel_noise_std * jax.random.normal(
        accel_key, (config.num_agents,), dtype=state.model.dtype
    )

    steer_buffer = state.steer_delay_buffer
    steer_head = state.steer_delay_head
    if config.steer_delay_steps:
        steer_commands, steer_buffer, steer_head = _push_fifo(
            steer_commands, steer_buffer, steer_head
        )

    throttle_buffer = state.throttle_delay_buffer
    throttle_head = state.throttle_delay_head
    if config.throttle_delay_steps:
        accel_commands, throttle_buffer, throttle_head = _push_fifo(
            accel_commands, throttle_buffer, throttle_head
        )

    delayed_actions = jnp.stack((steer_commands, accel_commands), axis=-1)
    if config.derive_steer_kp:
        steer_kp = (
            10.0
            * episode.dynamics.sv_max
            / (episode.dynamics.s_max - episode.dynamics.s_min)
        )
    else:
        steer_kp = episode.steer_kp
    efforts = adapt_actions(
        delayed_actions,
        state.model,
        episode.dynamics,
        longitudinal_mode=config.longitudinal_mode,
        steering_mode=config.steering_mode,
        steer_kp=steer_kp,
    )

    def advance(one_state: jax.Array, one_effort: jax.Array) -> jax.Array:
        return integrate_substeps(
            config.integrator_fn,
            config.dynamics_fn,
            one_state,
            one_effort,
            episode.timestep,
            episode.dynamics,
            num_substeps=config.num_substeps,
        )

    model = jax.vmap(advance)(state.model, efforts)
    return DynamicsState(
        model=model,
        control_input=delayed_actions,
        steer_delay_buffer=steer_buffer,
        steer_delay_head=steer_head,
        throttle_delay_buffer=throttle_buffer,
        throttle_delay_head=throttle_head,
        sim_time=state.sim_time + episode.timestep,
    )


def rollout_dynamics(
    key: jax.Array,
    state: DynamicsState,
    actions: jax.Array,
    config: DynamicsConfig,
    episode: EpisodeParams,
) -> tuple[DynamicsState, DynamicsState]:
    """Run a time-major action sequence with one compiled ``lax.scan``."""
    if actions.ndim != 3 or actions.shape[1:] != (config.num_agents, 2):
        raise ValueError(
            "actions must have shape (steps, num_agents, 2), got "
            f"{actions.shape}"
        )
    keys = jax.random.split(key, actions.shape[0])

    def body(carry: DynamicsState, inputs: tuple[jax.Array, jax.Array]):
        step_key, step_actions = inputs
        next_state = step_dynamics(step_key, carry, step_actions, config, episode)
        return next_state, next_state

    return jax.lax.scan(body, state, (keys, actions))


__all__ = [
    "DynamicsConfig",
    "DynamicsState",
    "EpisodeParams",
    "make_dynamics_state",
    "rollout_dynamics",
    "step_dynamics",
]
