"""Optional SBX adapter for one device-vmapped F1TENTH simulator batch.

The physics, domain randomization, normalized-action scaling, and selective
auto-reset all remain in :mod:`f1tenth_gym.envs.batching`.  This module only
translates that functional batch contract to the NumPy-based Stable-Baselines3
``VecEnv`` protocol consumed by SBX.

Importing the module does not require SBX or Stable-Baselines3.  Constructing
an environment does, and reports the optional installation command when the
dependency is absent.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from .batching import (
    BatchState,
    PolicyField,
    PolicyLayout,
    policy_observation,
    scale_normalized_actions,
    select_ego_rewards,
)
from .env_config import EnvConfig, ObservationConfig
from .jax_simulator import JaxSimulator
from .observation import ObservationType
from .observation.jax_adapter import GymObservationAdapter
from .track import Track

try:
    from stable_baselines3.common.vec_env import VecEnv as _VecEnv
except ModuleNotFoundError as error:  # pragma: no cover - depends on extras
    if error.name != "stable_baselines3":
        raise
    _VecEnv = object
    _SB3_IMPORT_ERROR: ModuleNotFoundError | None = error
else:
    _SB3_IMPORT_ERROR = None


DEFAULT_POLICY_LAYOUT = PolicyLayout(
    (PolicyField.KINEMATIC_STATE, PolicyField.FRENET)
)

_POLICY_SPACE_FIELDS: dict[PolicyField, tuple[str, slice | None]] = {
    PolicyField.KINEMATIC_STATE: ("std_state", slice(0, 5)),
    PolicyField.DYNAMIC_STATE: ("std_state", None),
    PolicyField.NATIVE_STATE: ("state", None),
    PolicyField.SCAN: ("scan", None),
    PolicyField.COLLISION: ("collision", None),
    PolicyField.FRENET: ("frenet_pose", None),
    PolicyField.LAP_TIME: ("lap_time", None),
    PolicyField.LAP_COUNT: ("lap_count", None),
    PolicyField.SIM_TIME: ("sim_time", None),
}


def _policy_space(
    simulator: JaxSimulator,
    layout: PolicyLayout,
) -> gym.spaces.Box:
    """Build finite bounds in the same channel order as ``layout``."""
    fields = tuple(
        dict.fromkeys(_POLICY_SPACE_FIELDS[field][0] for field in layout.fields)
    )
    adapter = GymObservationAdapter.from_simulator(
        simulator,
        ObservationConfig(type=ObservationType.FEATURES, features=fields),
    )
    agent_space = adapter.observation_space["agent_0"]
    lows: list[np.ndarray] = []
    highs: list[np.ndarray] = []
    for field in layout.fields:
        observation_field, selection = _POLICY_SPACE_FIELDS[field]
        field_space = agent_space[observation_field]
        if not isinstance(field_space, gym.spaces.Box):
            raise TypeError(
                f"policy field {field.name} does not have a Box space"
            )
        low = np.asarray(field_space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(field_space.high, dtype=np.float32).reshape(-1)
        if selection is not None:
            low = low[selection]
            high = high[selection]
        lows.append(low)
        highs.append(high)
    return gym.spaces.Box(
        low=np.concatenate(lows),
        high=np.concatenate(highs),
        dtype=np.float32,
    )


class F110SBXVecEnv(_VecEnv):
    """SBX ``VecEnv`` backed by one vmapped simulator transition.

    The adapter currently represents one learning-controlled car per logical
    environment.  A single :class:`JaxSimulator` supplies the shared topology
    while every batch row keeps independent state, PRNG streams, and
    domain-randomized parameters on ``simulator.device``.

    Args:
        simulator: Fully constructed functional simulator.
        num_envs: Number of independent environment rows in the device batch.
        seed: Seed for rows not explicitly seeded through ``VecEnv.seed``.
        policy_layout: Ordered device observation channels exposed to SBX.
    """

    def __init__(
        self,
        simulator: JaxSimulator,
        num_envs: int,
        *,
        seed: int | None = None,
        policy_layout: PolicyLayout = DEFAULT_POLICY_LAYOUT,
    ) -> None:
        if _SB3_IMPORT_ERROR is not None:
            raise ImportError(
                "SBX support is optional; install it with "
                "`pip install f1tenth_gym[sbx]` or "
                "`uv sync --extra sbx`"
            ) from _SB3_IMPORT_ERROR
        if not isinstance(simulator, JaxSimulator):
            raise TypeError("simulator must be a JaxSimulator instance")
        if not isinstance(policy_layout, PolicyLayout):
            raise TypeError("policy_layout must be a PolicyLayout instance")
        if simulator.config.dynamics.num_agents != 1:
            raise ValueError(
                "F110SBXVecEnv currently requires exactly one agent"
            )
        if isinstance(num_envs, bool) or not isinstance(num_envs, int):
            raise TypeError("num_envs must be an integer")
        if num_envs < 1:
            raise ValueError("num_envs must be at least one")

        self.simulator = simulator
        self.policy_layout = policy_layout
        self.render_mode = None
        self._seed_generator = np.random.default_rng(
            simulator.env_config.seed if seed is None else seed
        )
        self._keys = None
        self._state: BatchState | None = None
        self._pending_step = None
        self._actions = None

        observation_space = _policy_space(simulator, policy_layout)
        action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        super().__init__(num_envs, observation_space, action_space)

        config = simulator.config

        @jax.jit
        def reset_program(keys):
            split_keys = jax.vmap(lambda key: jax.random.split(key, 2))(keys)
            next_keys = split_keys[:, 0]
            reset_keys = split_keys[:, 1]
            observation, state = simulator.reset_batch(reset_keys)
            policy_input = policy_observation(
                observation, config, policy_layout
            )[:, 0, :]
            return next_keys, state, policy_input

        @jax.jit
        def step_program(keys, state, normalized_actions):
            split_keys = jax.vmap(lambda key: jax.random.split(key, 3))(keys)
            next_keys = split_keys[:, 0]
            step_keys = split_keys[:, 1]
            reset_keys = split_keys[:, 2]
            physical_actions = scale_normalized_actions(
                normalized_actions[:, None, :], state, config
            )
            transition = simulator.step_batch_autoreset(
                step_keys,
                reset_keys,
                state,
                physical_actions,
            )
            next_observation = policy_observation(
                transition.next_observation, config, policy_layout
            )[:, 0, :]
            terminal_observation = policy_observation(
                transition.transition_observation, config, policy_layout
            )[:, 0, :]
            return (
                next_keys,
                transition.state,
                next_observation,
                terminal_observation,
                select_ego_rewards(transition.rewards, config),
                transition.metrics.status.terminated,
                transition.metrics.status.truncated,
            )

        self._reset_program = reset_program
        self._step_program = step_program

    @classmethod
    def from_config(
        cls,
        config: EnvConfig,
        track: Track,
        num_envs: int,
        *,
        device: str | jax.Device | None = None,
        seed: int | None = None,
        policy_layout: PolicyLayout = DEFAULT_POLICY_LAYOUT,
    ) -> "F110SBXVecEnv":
        """Construct the simulator and SBX adapter from host configuration."""
        if _SB3_IMPORT_ERROR is not None:
            raise ImportError(
                "SBX support is optional; install it with "
                "`pip install f1tenth_gym[sbx]` or "
                "`uv sync --extra sbx`"
            ) from _SB3_IMPORT_ERROR
        simulator = JaxSimulator(config, track, device=device)
        return cls(
            simulator,
            num_envs,
            seed=seed,
            policy_layout=policy_layout,
        )

    @property
    def batch_state(self) -> BatchState | None:
        """Current device state, or ``None`` before reset and after close."""
        return self._state

    def reset(self) -> np.ndarray:
        if any(self._options):
            raise ValueError("reset options are not supported")
        seeds = [
            int(
                self._seed_generator.integers(
                    0, 2**32, dtype=np.uint32
                )
            )
            if seed is None
            else int(seed) % 2**32
            for seed in self._seeds
        ]
        seed_values = jax.device_put(
            jnp.asarray(seeds, dtype=jnp.uint32), self.simulator.device
        )
        keys = jax.vmap(jax.random.key)(seed_values)
        self._keys, self._state, observation = self._reset_program(keys)
        self._pending_step = None
        self.reset_infos = [{"seed": seed} for seed in seeds]
        self._reset_seeds()
        self._reset_options()
        return np.array(
            jax.device_get(observation), dtype=np.float32, copy=True
        )

    def step_async(self, actions: np.ndarray) -> None:
        if self._state is None or self._keys is None:
            raise RuntimeError("reset() must be called before step()")
        if self._pending_step is not None:
            raise RuntimeError("step_wait() is required before another step")
        actions = np.asarray(actions, dtype=np.float32)
        expected = (self.num_envs, 2)
        if actions.shape != expected:
            raise ValueError(
                f"actions must have shape {expected}, got {actions.shape}"
            )
        if not np.all(np.isfinite(actions)):
            raise ValueError("actions must be finite")
        if np.any(actions < -1.0) or np.any(actions > 1.0):
            raise ValueError("normalized actions must stay within [-1, 1]")
        self._actions = jax.device_put(actions, self.simulator.device)
        self._pending_step = self._step_program(
            self._keys, self._state, self._actions
        )

    def step_wait(self):
        if self._pending_step is None:
            raise RuntimeError("step_async() must be called before step_wait()")
        (
            self._keys,
            self._state,
            next_observation,
            terminal_observation,
            rewards,
            terminated,
            truncated,
        ) = self._pending_step
        self._pending_step = None
        self._actions = None

        host = jax.device_get(
            (
                next_observation,
                terminal_observation,
                rewards,
                terminated,
                truncated,
            )
        )
        # SBX mutates timeout rewards when adding its value bootstrap.  Public
        # results therefore own writable memory rather than exposing a possibly
        # read-only NumPy view returned by device_get.
        observations = np.array(host[0], dtype=np.float32, copy=True)
        terminal_observations = np.array(
            host[1], dtype=np.float32, copy=True
        )
        rewards = np.array(host[2], dtype=np.float32, copy=True)
        terminated = np.array(host[3], dtype=np.bool_, copy=True)
        truncated = np.array(host[4], dtype=np.bool_, copy=True)
        dones = np.logical_or(terminated, truncated)

        infos = []
        for index in range(self.num_envs):
            info = {
                "TimeLimit.truncated": bool(
                    truncated[index] and not terminated[index]
                )
            }
            if dones[index]:
                info["terminal_observation"] = terminal_observations[
                    index
                ].copy()
            infos.append(info)
        return observations, rewards, dones, infos

    def close(self) -> None:
        self._pending_step = None
        self._actions = None
        self._keys = None
        self._state = None

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        if not hasattr(self, attr_name):
            raise AttributeError(attr_name)
        value = getattr(self, attr_name)
        return [value for _ in self._get_indices(indices)]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        raise NotImplementedError(
            "F110SBXVecEnv owns one shared device batch and cannot set "
            "attributes on individual logical environments"
        )

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices=None,
        **method_kwargs,
    ) -> list[Any]:
        raise NotImplementedError(
            "F110SBXVecEnv owns one shared device batch and cannot call "
            "methods on individual logical environments"
        )

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        return [False for _ in self._get_indices(indices)]

    def get_images(self):
        return [None for _ in range(self.num_envs)]


__all__ = [
    "DEFAULT_POLICY_LAYOUT",
    "F110SBXVecEnv",
]
