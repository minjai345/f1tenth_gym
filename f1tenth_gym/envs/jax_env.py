"""Gymnasium lifecycle adapter for the functional JAX environment core.

This module is intentionally a deep import.  It owns host-side configuration,
Gymnasium seeding, device transfers, mutable lifecycle state, observation
packaging, and rendering.  The transition itself remains in
``f1tenth_gym.envs.jax_core``.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
from pathlib import Path

import gymnasium as gym
import jax
import numpy as np

from f1tenth_gym.envs.action import (
    from_single_to_multi_action_space,
    get_action_space,
)
from f1tenth_gym.envs.dynamic_models import (
    PARAMETER_ORDER,
    VehicleParameters,
)
from f1tenth_gym.envs.env_config import (
    EnvConfig,
    ObservationConfig,
    RewardMode,
)
from f1tenth_gym.envs.f110_env import RenderClock
from f1tenth_gym.envs.observation import ObservationType
from f1tenth_gym.envs.rendering import make_renderer
from f1tenth_gym.envs.track import Track

from .jax_simulator import JaxSimulator
from .jax_core import CoreObservation, CoreState, observe_core
from .episode import BuiltinRewardMode
from .observation.jax_adapter import GymObservationAdapter


class JaxF110Env(gym.Env):
    """Gymnasium environment backed by the immutable functional JAX core.

    The native public action and observation layouts match :class:`F110Env`.
    A normal Gym call still crosses the host/device boundary; device-batched
    training should call the functional core directly instead.
    """

    metadata = {
        "render_modes": ["human", "human_fast", "rgb_array", "unlimited"],
        "render_fps": 100,
    }

    def __init__(
        self,
        config: EnvConfig = EnvConfig(),
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(config, EnvConfig):
            raise TypeError("config must be an EnvConfig instance")
        self.metadata = dict(type(self).metadata)
        self.render_mode = render_mode
        self.renderer = None
        self.render_config = None
        self.render_obs = None

        self._initialize_components(config, rearm_seed=True)

    @staticmethod
    def _resolve_track(config: EnvConfig) -> Track:
        """Resolve a prebuilt, path-backed, or named track."""
        source = config.map_name
        if isinstance(source, Track):
            return source
        if isinstance(source, (str, Path)):
            text = str(source)
            path = Path(source)
            if "/" in text or "\\" in text or path.suffix:
                return Track.from_track_path(path, track_scale=config.map_scale)
            return Track.from_track_name(text, track_scale=config.map_scale)
        raise TypeError("map must be a Track instance or a path/name string")

    @staticmethod
    def _initial_vehicle(config: EnvConfig) -> VehicleParameters:
        """Choose a valid placeholder before the first randomized reset."""
        randomization = config.domain_randomization_config
        if not randomization.randomized_fields():
            return config.params
        if randomization.low is None:  # guarded by EnvConfig validation
            raise RuntimeError("domain randomization is missing its low bound")
        return randomization.low

    def _initialize_components(
        self,
        config: EnvConfig,
        *,
        track: Track | None = None,
        vehicle_params: VehicleParameters | None = None,
        rearm_seed: bool,
        preserve_episode: bool = False,
    ) -> None:
        """Stage and atomically install all topology-dependent components."""
        if not isinstance(config, EnvConfig):
            raise TypeError("config must be an EnvConfig instance")

        resolved_track = self._resolve_track(config) if track is None else track
        effective = (
            self._initial_vehicle(config)
            if vehicle_params is None
            else vehicle_params
        )
        randomized = bool(
            config.domain_randomization_config.randomized_fields()
        )
        fallback = (
            BuiltinRewardMode.SURVIVAL
            if config.reward_config.mode is RewardMode.CUSTOM
            else None
        )
        simulator = JaxSimulator(
            config,
            resolved_track,
            vehicle_params=effective if randomized else None,
            _custom_reward_fallback=fallback,
        )
        observation = GymObservationAdapter.from_simulator(simulator)
        render_observation = (
            observation
            if config.observation_config.type is ObservationType.DEFAULT
            else GymObservationAdapter.from_simulator(
                simulator,
                ObservationConfig(type=ObservationType.DEFAULT),
            )
        )
        widest = config.domain_randomization_config.widest_params(config.params)
        single_action_space = get_action_space(
            config.control_config.longitudinal_mode,
            config.control_config.steering_mode,
            widest,
        )
        action_space = from_single_to_multi_action_space(
            single_action_space,
            config.num_agents,
        )
        agent_ids = [f"agent_{index}" for index in range(config.num_agents)]

        old_renderer = self.renderer
        if preserve_episode:
            renderer = old_renderer
            render_config = self.render_config
            render_clock = self._render_clock
            if renderer is not None:
                renderer.update_params(effective)
        else:
            renderer = None
            render_config = None
            if config.render_enabled:
                renderer, render_config = make_renderer(
                    params=config.params,
                    track=resolved_track,
                    agent_ids=agent_ids,
                    render_mode=self.render_mode,
                    render_config=config.render_config,
                )
            if self.render_mode == "human_fast":
                real_time_factor = 10.0
            elif self.render_mode == "unlimited":
                real_time_factor = float("inf")
            else:
                real_time_factor = config.render_config.real_time_factor
            render_clock = RenderClock(
                render_fps=config.render_config.render_fps,
                real_time_factor=real_time_factor,
                timestep=config.simulation_config.timestep,
            )

        self._apply_env_config(config, rearm_seed=rearm_seed)
        self.track = resolved_track
        self.sim = simulator
        self._episode_params = simulator.params
        self._observation_adapter = observation
        self._render_observation_adapter = render_observation
        self.observation_space = observation.observation_space
        self.action_space = action_space
        self.agent_ids = agent_ids
        self.space_vehicle_params = widest
        self._episode_vehicle_params = effective
        self.renderer = renderer
        self.render_config = render_config
        self._render_clock = render_clock
        self.metadata["render_fps"] = int(round(1.0 / self.timestep))

        if preserve_episode:
            if self._state is not None:
                current = observe_core(self._state)
                self.render_obs = self._package_render_observation(current)
        else:
            self._state: CoreState | None = None
            self._transition_key = None
            self.render_obs = None
            self._last_frame = None
            self._reset_public_episode_state()

        if (
            not preserve_episode
            and old_renderer is not None
            and old_renderer is not renderer
        ):
            old_renderer.close()

    def _apply_env_config(
        self, config: EnvConfig, *, rearm_seed: bool
    ) -> None:
        """Keep the established unwrapped attributes used by wrappers/users."""
        self.env_config = config
        self.seed = config.seed
        if rearm_seed:
            self._config_seed_used = False
        self.map = config.map_name
        self.map_scale = config.map_scale
        self.vehicle_params = config.params
        self.num_agents = config.num_agents
        self.ego_idx = config.ego_index

        self.control_cfg = config.control_config
        self.simulation_cfg = config.simulation_config
        self.observation_cfg = config.observation_config
        self.reset_cfg = config.reset_config
        self.lidar_cfg = config.lidar_config
        self.contact_cfg = config.contact_config
        self.render_cfg = config.render_config
        self.termination_cfg = config.termination_config
        self.reward_cfg = config.reward_config
        self.dr_cfg = config.domain_randomization_config

        self.max_episode_steps = self.termination_cfg.max_episode_steps
        self.terminate_on_collision = (
            self.termination_cfg.terminate_on_collision
        )
        self.agent_termination_mode = self.termination_cfg.agent_mode
        self.longitudinal_action_type = self.control_cfg.longitudinal_mode
        self.steer_action_type = self.control_cfg.steering_mode
        self.steer_delay_steps = self.control_cfg.steer_delay_steps
        self.timestep = self.simulation_cfg.timestep
        self.model = self.simulation_cfg.dynamics_model
        self.loop_counter_mode = self.simulation_cfg.loop_counter
        self.compute_frenet = self.simulation_cfg.compute_frenet_frame
        self.max_laps = self.simulation_cfg.max_laps
        self.count_partial_first_lap = (
            self.simulation_cfg.count_partial_first_lap
        )
        self.collision_check_mode = config.collision_check
        self.render_enabled = config.render_enabled

    def _reset_public_episode_state(self) -> None:
        # Match F110Env's host bookkeeping dtypes.  Device counters remain
        # int32; the public Gym boundary historically exposes float64 arrays.
        self.lap_times = np.zeros((self.num_agents,), dtype=np.float64)
        self.lap_counts = np.zeros((self.num_agents,), dtype=np.float64)
        self.terminated_agents = np.zeros(
            (self.num_agents,), dtype=np.bool_
        )
        self.collisions = np.zeros((self.num_agents,), dtype=np.float32)
        self.sim_time = 0.0
        self._elapsed_steps = 0

    @property
    def core_state(self) -> CoreState | None:
        """Current immutable device state, or ``None`` before reset."""
        return self._state

    @property
    def episode_vehicle_params(self) -> VehicleParameters:
        """Shared vehicle parameters active in the current device episode."""
        return self._episode_vehicle_params

    def configure(self, config: EnvConfig | None) -> None:
        """Transactionally rebuild every topology-dependent component."""
        if config is None:
            return
        if not isinstance(config, EnvConfig):
            raise TypeError("config must be an EnvConfig or None")
        self._initialize_components(config, rearm_seed=True)

    def update_map(self, map_name: Track | str) -> None:
        """Reconfigure with another named, path-backed, or prebuilt track."""
        self.configure(self.env_config.with_updates(map_name=map_name))

    def update_params(
        self, params: VehicleParameters, index: int = -1
    ) -> None:
        """Update shared vehicle values and every dependent table/space."""
        if index >= 0:
            raise NotImplementedError(
                "Per-agent parameter updates are not supported in the "
                "simplified simulator"
            )
        if not isinstance(params, VehicleParameters):
            raise TypeError("params must be a VehicleParameters instance")

        new_config = self.env_config.with_updates(params=params)
        # Body dimensions affect contact-table reach, so rebuild the simulator
        # while retaining the resolved track and current rollout carry.
        self._initialize_components(
            new_config,
            track=self.track,
            vehicle_params=params,
            rearm_seed=False,
            preserve_episode=True,
        )

    def _sample_vehicle_params(self) -> VehicleParameters:
        """Draw the mutable Gym environment's shared host DR vector."""
        low, high = self.dr_cfg.bounds_arrays()
        drawn = low.copy()
        finite = np.isfinite(low) & np.isfinite(high)
        drawn[finite] = self.np_random.uniform(low[finite], high[finite])
        return VehicleParameters(
            **{
                name: float(value)
                for name, value in zip(PARAMETER_ORDER, drawn, strict=True)
            }
        )

    def _episode_root_key(self):
        seed = int(
            self.np_random.integers(
                0, 2**32, dtype=np.uint64
            )
        )
        return jax.device_put(jax.random.key(seed), self.sim.device)

    def _params_for_reset(self):
        if not self.dr_cfg.randomized_fields():
            return self.sim.params, self.vehicle_params
        sampled = self._sample_vehicle_params()
        return self.sim.params_for_vehicle(sampled), sampled

    def _validate_reset_options(self, options):
        if options is not None and not isinstance(options, Mapping):
            raise TypeError("options must be a mapping or None")
        if options is not None and "poses" in options:
            name = "poses"
            value = options[name]
            expected = (self.num_agents, 3)
        elif options is not None and "states" in options:
            name = "states"
            value = options[name]
            expected = (self.num_agents, self.sim.config.dynamics.state_dim)
        else:
            return None, None
        if not isinstance(value, np.ndarray):
            raise TypeError(f"reset option {name!r} must be a numpy array")
        if value.shape != expected:
            raise ValueError(
                f"reset option {name!r} must have shape {expected}, "
                f"got {value.shape}"
            )
        return name, value

    def _package_render_observation(self, observation: CoreObservation):
        if self.renderer is None:
            return None
        if self.observation_cfg.type is ObservationType.DEFAULT:
            public = self._observation_adapter.package(observation)
        else:
            public = self._render_observation_adapter.package(observation)
        return copy.deepcopy(public)

    def reset(self, *, seed=None, options=None):
        """Start an episode and return ``(observation, reset_info)``."""
        if seed is None and self.seed is not None and not self._config_seed_used:
            seed = int(self.seed)
        self._config_seed_used = True
        super().reset(seed=seed)

        option, override = self._validate_reset_options(options)
        episode_params, effective_params = self._params_for_reset()
        root_key = self._episode_root_key()
        reset_key, transition_key = jax.random.split(root_key)
        if option == "poses":
            core_observation, state = self.sim.reset_from_poses(
                reset_key,
                override,
                params=episode_params,
            )
        elif option == "states":
            core_observation, state = self.sim.reset_from_state(
                reset_key,
                override,
                params=episode_params,
            )
        else:
            core_observation, state = self.sim.reset(
                reset_key,
                params=episode_params,
            )

        observation = self._observation_adapter.package(core_observation)
        render_observation = self._package_render_observation(core_observation)
        episode = jax.device_get(state.episode)

        self._episode_params = episode_params
        self._state = state
        self._transition_key = transition_key
        self._episode_vehicle_params = effective_params
        if self.renderer is not None:
            self.renderer.update_params(effective_params)
        self.render_obs = render_observation
        self._render_clock.reset()
        self._last_frame = None
        self.lap_times = np.array(
            episode.lap_times, dtype=np.float64, copy=True
        )
        self.lap_counts = np.array(
            episode.lap_counts, dtype=np.float64, copy=True
        )
        self.terminated_agents = np.array(
            episode.terminated_agents, dtype=np.bool_, copy=True
        )
        self.collisions = np.zeros((self.num_agents,), dtype=np.float32)
        self.sim_time = 0.0
        self._elapsed_steps = 0
        info = {
            "lap_times": self.lap_times.copy(),
            "lap_counts": self.lap_counts.copy(),
            "sim_time": self.sim_time,
            "terminated_agents": self.terminated_agents.copy(),
        }
        return observation, info

    def step(self, action):
        """Advance once without clipping the native ``(num_agents, 2)`` action."""
        if self._state is None or self._transition_key is None:
            raise gym.error.ResetNeeded("call reset() before step()")
        host_action = np.asarray(action)
        expected = (self.num_agents, 2)
        if host_action.shape != expected:
            raise ValueError(
                f"Control input has incorrect shape: expected {expected}, "
                f"got {host_action.shape}"
            )
        device_action = jax.device_put(
            host_action.astype(np.float32), self.sim.device
        )
        next_key, step_key = jax.random.split(self._transition_key)
        result = self.sim.step(
            step_key,
            self._state,
            device_action,
            params=self._episode_params,
        )
        core_observation, state, rewards, events, metrics = result
        observation = self._observation_adapter.package(core_observation)
        render_observation = self._package_render_observation(core_observation)
        host_rewards, host_events, host_metrics = jax.device_get(
            (rewards, events, metrics)
        )
        terminated = bool(host_metrics.status.terminated)
        truncated = bool(host_metrics.status.truncated)
        episode = host_metrics.episode

        self._state = state
        self._transition_key = next_key
        self.render_obs = render_observation
        self._render_clock.advance()
        self.lap_times = np.array(
            episode.lap_times, dtype=np.float64, copy=True
        )
        self.lap_counts = np.array(
            episode.lap_counts, dtype=np.float64, copy=True
        )
        self.terminated_agents = np.array(
            episode.terminated_agents, dtype=np.bool_, copy=True
        )
        self.collisions = np.array(
            host_events.collisions, dtype=np.float32, copy=True
        )
        self.sim_time = float(episode.sim_time)
        self._elapsed_steps = int(episode.elapsed_steps)
        info = {
            "lap_times": self.lap_times.copy(),
            "lap_counts": self.lap_counts.copy(),
            "sim_time": self.sim_time,
            "collisions": self.collisions.copy(),
            "terminated_agents": self.terminated_agents.copy(),
            "progress": np.array(
                episode.progress, dtype=np.float64, copy=True
            ),
        }
        if self.reward_cfg.mode is RewardMode.CUSTOM:
            reward = float(
                self.reward_cfg.reward_fn(
                    observation,
                    action,
                    info,
                    terminated,
                    truncated,
                )
            )
        else:
            reward = float(host_rewards[self.ego_idx])
        return observation, reward, terminated, truncated, info

    def add_render_callback(self, callback_func) -> None:
        if self.render_enabled and self.renderer is not None:
            self.renderer.add_renderer_callback(callback_func)

    def set_real_time_factor(self, real_time_factor: float) -> None:
        if not (real_time_factor > 0):
            raise ValueError(
                "real_time_factor must be > 0 (or float('inf')), got "
                f"{real_time_factor}"
            )
        self._render_clock.set_rtf(real_time_factor)

    @property
    def real_time_factor(self) -> float:
        return self._render_clock.rtf

    @property
    def render_fps(self) -> float:
        return self._render_clock.render_fps

    @property
    def frame_is_new(self) -> bool:
        return self._render_clock.frame_is_new

    def render(self, mode=None):
        selected = mode or self.render_mode
        if (
            selected not in self.metadata["render_modes"]
            or not self.render_enabled
            or self.renderer is None
        ):
            return None
        if selected in ("human", "human_fast", "unlimited"):
            if self._render_clock.display_due():
                self.renderer.update(obs=self.render_obs)
                self.renderer.render()
            self._render_clock.pace(self.sim_time)
            return None
        if self._render_clock.frame_is_new or self._last_frame is None:
            self.renderer.update(obs=self.render_obs)
            self._last_frame = self.renderer.render()
        return self._last_frame

    def close(self) -> None:
        renderer = self.renderer
        self.renderer = None
        if renderer is not None:
            renderer.close()
        super().close()


__all__ = ["JaxF110Env"]
