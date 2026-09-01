"""Pure fixed-shape Frenet episode bookkeeping and termination policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import jax
import jax.numpy as jnp


class TerminationMode(IntEnum):
    """Reduce latched per-agent terminal status to one environment result."""

    EGO = 0
    ANY = 1
    ALL = 2


class BuiltinRewardMode(IntEnum):
    """Compiled reward choices shared by device-native adapters."""

    SURVIVAL = 0
    PROGRESS = 1


@dataclass(frozen=True)
class EpisodeConfig:
    """Static episode choices that affect compiled dispatch."""

    num_agents: int
    ego_index: int = 0
    count_partial_first_lap: bool = True
    termination_mode: TerminationMode = TerminationMode.EGO
    reward_mode: BuiltinRewardMode = BuiltinRewardMode.SURVIVAL

    def __post_init__(self) -> None:
        if self.num_agents < 1:
            raise ValueError(f"num_agents must be >= 1, got {self.num_agents}")
        if not 0 <= self.ego_index < self.num_agents:
            raise ValueError(
                f"ego_index must be in [0, {self.num_agents}), got {self.ego_index}"
            )
        if not isinstance(self.termination_mode, TerminationMode):
            raise TypeError("termination_mode must be a TerminationMode")
        if not isinstance(self.reward_mode, BuiltinRewardMode):
            raise TypeError("reward_mode must be a BuiltinRewardMode")


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EpisodeParams:
    """Traced episode limits, enables, and built-in reward weights."""

    terminate_on_collision: Any = True
    lap_limit_enabled: Any = True
    max_laps: Any = 1
    step_limit_enabled: Any = False
    max_episode_steps: Any = 0
    progress_weight: Any = 1.0
    velocity_weight: Any = 0.0
    timestep_weight: Any = 0.0
    collision_penalty: Any = 0.0


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EpisodeState:
    """Immutable per-environment state for progress, laps, and done latches."""

    frenet: jax.Array
    progress_previous_s: jax.Array
    lap_previous_s: jax.Array
    cumulative_s: jax.Array
    finish_crossings: jax.Array
    lap_counts: jax.Array
    lap_times: jax.Array
    lap_times_finish: jax.Array
    terminated_agents: jax.Array
    elapsed_steps: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EpisodeEvents:
    """Fresh per-agent facts from one transition."""

    collisions: jax.Array
    finish_crossed: jax.Array
    lap_completed: jax.Array
    newly_terminated: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EpisodeMetrics:
    """Adapter-neutral values produced by one transition."""

    progress: jax.Array
    lap_counts: jax.Array
    lap_times: jax.Array
    terminated_agents: jax.Array
    sim_time: jax.Array
    elapsed_steps: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EpisodeStatus:
    """Global Gymnasium-style episode-end results without auto-reset."""

    terminated: jax.Array
    truncated: jax.Array


def reset_episode_state(
    frenet: jax.Array,
    config: EpisodeConfig,
) -> EpisodeState:
    """Seed progress and finish-line references from reset Frenet poses."""
    expected = (config.num_agents, 3)
    if frenet.shape != expected:
        raise ValueError(f"frenet must have shape {expected}, got {frenet.shape}")
    frenet = jnp.asarray(frenet)
    initial_s = frenet[:, 0]
    return EpisodeState(
        frenet=frenet,
        progress_previous_s=initial_s,
        lap_previous_s=initial_s,
        cumulative_s=initial_s,
        finish_crossings=jnp.zeros((config.num_agents,), dtype=jnp.int32),
        lap_counts=jnp.zeros((config.num_agents,), dtype=jnp.int32),
        lap_times=jnp.zeros((config.num_agents,), dtype=frenet.dtype),
        lap_times_finish=jnp.zeros((config.num_agents,), dtype=frenet.dtype),
        terminated_agents=jnp.zeros((config.num_agents,), dtype=jnp.bool_),
        elapsed_steps=jnp.asarray(0, dtype=jnp.int32),
    )


def wrap_progress_delta(
    current_s: jax.Array,
    previous_s: jax.Array,
    track_length: jax.Array,
) -> jax.Array:
    """Return signed arclength progress corrected across the finish-line wrap."""
    delta = current_s - previous_s
    half_length = 0.5 * track_length
    return jnp.where(
        delta < -half_length,
        delta + track_length,
        jnp.where(delta > half_length, delta - track_length, delta),
    )


def _reduce_termination(
    terminated_agents: jax.Array,
    config: EpisodeConfig,
) -> jax.Array:
    if config.termination_mode is TerminationMode.EGO:
        return terminated_agents[config.ego_index]
    if config.termination_mode is TerminationMode.ANY:
        return jnp.any(terminated_agents)
    if config.termination_mode is TerminationMode.ALL:
        return jnp.all(terminated_agents)
    raise ValueError(f"unsupported termination mode: {config.termination_mode!r}")


def _builtin_rewards(
    progress: jax.Array,
    speeds: jax.Array,
    collisions: jax.Array,
    timestep: jax.Array,
    config: EpisodeConfig,
    params: EpisodeParams,
) -> jax.Array:
    if config.reward_mode is BuiltinRewardMode.SURVIVAL:
        return jnp.full(progress.shape, timestep, dtype=progress.dtype)
    if config.reward_mode is BuiltinRewardMode.PROGRESS:
        return (
            jnp.asarray(params.progress_weight, dtype=progress.dtype) * progress
            + jnp.asarray(params.velocity_weight, dtype=progress.dtype) * speeds
            + jnp.asarray(params.timestep_weight, dtype=progress.dtype) * timestep
            - jnp.asarray(params.collision_penalty, dtype=progress.dtype)
            * collisions.astype(progress.dtype)
        )
    raise ValueError(f"unsupported reward mode: {config.reward_mode!r}")


def advance_episode(
    state: EpisodeState,
    frenet: jax.Array,
    collisions: jax.Array,
    speeds: jax.Array,
    track_length: jax.Array,
    sim_time_before: jax.Array,
    sim_time_after: jax.Array,
    timestep: jax.Array,
    config: EpisodeConfig,
    params: EpisodeParams,
) -> tuple[
    EpisodeState,
    jax.Array,
    EpisodeEvents,
    EpisodeMetrics,
    EpisodeStatus,
]:
    """Advance progress, lap timing, reward, and episode status once.

    The two clocks are explicit because the mutable environment checks laps and
    builds observations before refreshing its environment-level clock, while
    step info exposes the post-transition simulator time.
    """
    expected_frenet = (config.num_agents, 3)
    if frenet.shape != expected_frenet:
        raise ValueError(
            f"frenet must have shape {expected_frenet}, got {frenet.shape}"
        )
    expected_agents = (config.num_agents,)
    if collisions.shape != expected_agents:
        raise ValueError(
            f"collisions must have shape {expected_agents}, got {collisions.shape}"
        )
    if speeds.shape != expected_agents:
        raise ValueError(
            f"speeds must have shape {expected_agents}, got {speeds.shape}"
        )
    if state.frenet.shape != expected_frenet:
        raise ValueError(
            "state.frenet must have shape "
            f"{expected_frenet}, got {state.frenet.shape}"
        )

    frenet = jnp.asarray(frenet, dtype=state.frenet.dtype)
    collisions = jnp.asarray(collisions, dtype=jnp.bool_)
    dtype = state.frenet.dtype
    speeds = jnp.asarray(speeds, dtype=dtype)
    track_length = jnp.asarray(track_length, dtype=dtype)
    sim_time_before = jnp.asarray(sim_time_before, dtype=dtype)
    sim_time_after = jnp.asarray(sim_time_after, dtype=dtype)
    timestep = jnp.asarray(timestep, dtype=dtype)

    progress = wrap_progress_delta(
        frenet[:, 0], state.progress_previous_s, track_length
    )
    lap_delta = wrap_progress_delta(
        frenet[:, 0], state.lap_previous_s, track_length
    )
    cumulative_s = state.cumulative_s + lap_delta
    crossings = (cumulative_s / track_length).astype(jnp.int32)
    finish_crossed = (
        (crossings > state.finish_crossings) & (sim_time_before > timestep)
    )
    finish_crossings = jnp.where(
        finish_crossed, crossings, state.finish_crossings
    )
    split = sim_time_before - state.lap_times_finish
    lap_times_finish = jnp.where(
        finish_crossed, sim_time_before, state.lap_times_finish
    )
    counted_laps = (
        crossings
        if config.count_partial_first_lap
        else crossings - jnp.asarray(1, dtype=jnp.int32)
    )
    lap_completed = finish_crossed & (counted_laps > state.lap_counts)
    lap_counts = jnp.where(lap_completed, counted_laps, state.lap_counts)
    lap_times = jnp.where(lap_completed, split, state.lap_times)

    terminal_now = (
        jnp.asarray(params.terminate_on_collision, dtype=jnp.bool_) & collisions
    ) | (
        jnp.asarray(params.lap_limit_enabled, dtype=jnp.bool_)
        & (lap_counts >= jnp.asarray(params.max_laps, dtype=jnp.int32))
    )
    terminated_agents = state.terminated_agents | terminal_now
    newly_terminated = terminated_agents & ~state.terminated_agents
    terminated = _reduce_termination(terminated_agents, config)

    elapsed_steps = state.elapsed_steps + jnp.asarray(1, dtype=jnp.int32)
    truncated = (
        jnp.asarray(params.step_limit_enabled, dtype=jnp.bool_)
        & (
            elapsed_steps
            >= jnp.asarray(params.max_episode_steps, dtype=jnp.int32)
        )
    )
    rewards = _builtin_rewards(
        progress, speeds, collisions, timestep, config, params
    )

    next_state = EpisodeState(
        frenet=frenet,
        progress_previous_s=frenet[:, 0],
        lap_previous_s=frenet[:, 0],
        cumulative_s=cumulative_s,
        finish_crossings=finish_crossings,
        lap_counts=lap_counts,
        lap_times=lap_times,
        lap_times_finish=lap_times_finish,
        terminated_agents=terminated_agents,
        elapsed_steps=elapsed_steps,
    )
    events = EpisodeEvents(
        collisions=collisions,
        finish_crossed=finish_crossed,
        lap_completed=lap_completed,
        newly_terminated=newly_terminated,
    )
    metrics = EpisodeMetrics(
        progress=progress,
        lap_counts=lap_counts,
        lap_times=lap_times,
        terminated_agents=terminated_agents,
        sim_time=sim_time_after,
        elapsed_steps=elapsed_steps,
    )
    status = EpisodeStatus(terminated=terminated, truncated=truncated)
    return next_state, rewards, events, metrics, status


__all__ = [
    "EpisodeParams",
    "BuiltinRewardMode",
    "EpisodeConfig",
    "EpisodeEvents",
    "EpisodeMetrics",
    "EpisodeState",
    "EpisodeStatus",
    "TerminationMode",
    "advance_episode",
    "reset_episode_state",
    "wrap_progress_delta",
]
