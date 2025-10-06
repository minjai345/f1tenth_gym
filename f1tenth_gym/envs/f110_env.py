from __future__ import annotations

import copy
from typing import Any

import gymnasium as gym
import numpy as np

from .dynamic_models import (
    VehicleParameters,
)
from .simulator import F110Simulator
from .env_config import (
    EnvConfig,
    LoopCounterMode,
)
from .integrators import integrator_from_type
from .action import (
    get_action_space,
    from_single_to_multi_action_space,
)
from .observation import ObservationType, observation_factory
from .reset import make_reset_fn
from .rendering import make_renderer
from .track import Track


class F110Env(gym.Env):
    """
    OpenAI gym environment for F1TENTH.

    Args:
        config: EnvConfig | None: Optional environment configuration.
        render_mode: Rendering mode requested by Gymnasium.
    """

    metadata = {"render_modes": ["human", "human_fast", "rgb_array", "unlimited"], "render_fps": 100}

    def __init__(
        self,
        config: EnvConfig = EnvConfig(),
        render_mode=None,
    ):
        super().__init__()
        if isinstance(config, EnvConfig):
            resolved_config = config
        else:
            raise TypeError("config must be an EnvConfig instance")

        self.env_config = resolved_config
        self._apply_env_config()

        self.near_start = True
        self.num_toggles = 0

        self.agents_prev_s = [None] * self.num_agents
        self.lap_times = np.zeros((self.num_agents,))
        self.lap_times_finish = np.zeros((self.num_agents,))
        self.lap_counts = np.zeros((self.num_agents,))
        self.sim_time = 0.0

        if isinstance(self.map, Track):
            self.track = self.map
        else:
            if "/" in self.map or "\\" in self.map:
                self.track = Track.from_track_path(self.map, track_scale=self.map_scale)
            else:
                self.track = Track.from_track_name(self.map, track_scale=self.map_scale)

        self.sim = F110Simulator(
            env_config=self.env_config,
            vehicle_params=self.vehicle_params,
            model=self.model,
            dynamics_fn=self.model.f_dynamics,
            integrator_fn=self.integrator_fn,
            longitudinal_type=self.longitudinal_action_type,
            steering_type=self.steer_action_type,
            track=self.track,
            seed=self.seed,
        )
        if isinstance(self.track, Track):
            self.sim.set_map(self.track, self.map_scale)

        self.agent_ids = [f"agent_{i}" for i in range(self.num_agents)]

        obs_kwargs: dict[str, Any] = {"type": self.observation_cfg.type}
        if self.observation_cfg.features is not None:
            obs_kwargs["features"] = self.observation_cfg.features
        self.observation_type = observation_factory(env=self, **obs_kwargs)
        self.observation_space = self.observation_type.space()
        self.render_obs_type = observation_factory(env=self, type=ObservationType.DIRECT)
        self.render_obs = None

        single_action_space = get_action_space(
            self.longitudinal_action_type,
            self.steer_action_type,
            self.vehicle_params,
        )
        self.action_space = from_single_to_multi_action_space(single_action_space, self.num_agents)

        self.reset_fn = make_reset_fn(
            track=self.track,
            num_agents=self.num_agents,
            type=self.reset_cfg.strategy,
        )
        self.render_mode = render_mode
        self.metadata["render_fps"] = int(1.0 / self.timestep)
        if self.render_mode == "human_fast":
            self.metadata["render_fps"] *= 10
        elif self.render_mode == "unlimited":
            self.metadata["render_fps"] = float("inf")

        self.renderer = None
        self.render_spec = None
        if self.render_enabled:
            self.renderer, self.render_spec = make_renderer(
                params=self.vehicle_params,
                track=self.track,
                agent_ids=self.agent_ids,
                render_mode=render_mode,
                render_fps=self.metadata["render_fps"],
            )

    def configure(self, config: EnvConfig | None) -> None:
        if config is None:
            return
        if isinstance(config, EnvConfig):
            new_config = config
        else:
            raise TypeError("config must be an EnvConfig or None")

        self.env_config = new_config
        self._apply_env_config()

        if hasattr(self, "sim"):
            self.sim.update_params(self.vehicle_params)
        if hasattr(self, "renderer") and self.renderer is not None:
            self.renderer.update_params(self.vehicle_params)
        if hasattr(self, "action_space"):
            single_action_space = get_action_space(
                self.longitudinal_action_type,
                self.steer_action_type,
                self.vehicle_params,
            )
            self.action_space = from_single_to_multi_action_space(single_action_space, self.num_agents)
        if hasattr(self, "reset_fn") and hasattr(self, "track"):
            self.reset_fn = make_reset_fn(
                track=self.track,
                num_agents=self.num_agents,
                type=self.reset_cfg.strategy,
            )

    def _apply_env_config(self) -> None:
        cfg = self.env_config

        self.seed = cfg.seed
        self.map = cfg.map_name
        self.map_scale = cfg.map_scale

        self.vehicle_params = cfg.params

        self.num_agents = cfg.num_agents
        self.ego_idx = cfg.ego_index

        self.control_cfg = cfg.control
        self.simulation_cfg = cfg.simulation
        self.observation_cfg = cfg.observation
        self.reset_cfg = cfg.reset
        self.lidar_cfg = cfg.lidar

        self.longitudinal_action_type = self.control_cfg.longitudinal_mode
        self.steer_action_type = self.control_cfg.steering_mode
        self.steer_delay_steps = self.control_cfg.steer_delay_steps

        self.timestep = self.simulation_cfg.timestep
        self.integrator_fn = integrator_from_type(self.simulation_cfg.integrator)
        self.model = self.simulation_cfg.dynamics_model
        self.loop_counter_mode = self.simulation_cfg.loop_counter
        self.compute_frenet = self.simulation_cfg.compute_frenet_frame
        self.max_laps = self.simulation_cfg.max_laps

        self.collision_check_mode = cfg.collision_check
        self.render_enabled = cfg.render_enabled
        self.collision_check_mode = cfg.collision_check
        self.render_enabled = cfg.render_enabled

    def _check_done(self):
        """
        Check if the current rollout is done
        """
        if (
            self.loop_counter_mode is LoopCounterMode.FRENET_BASED
            and self.compute_frenet
            and self.track is not None
        ):
            s_frame_max = self.track.centerline.spline.s_frame_max
            for ind in range(self.num_agents):
                current_s = float(self.sim.state.frenet[ind, 0])
                if self.agents_prev_s[ind] is None:
                    self.agents_prev_s[ind] = current_s
                    continue
                if (
                    self.agents_prev_s[ind] - current_s > s_frame_max * 0.85
                    and self.sim_time > self.timestep
                ):
                    self.lap_counts[ind] += 1
                    self.lap_times[ind] = self.sim_time - self.lap_times_finish[ind]
                    self.lap_times_finish[ind] = self.sim_time
                self.agents_prev_s[ind] = current_s

        done = bool(self.sim.collisions[self.ego_idx])
        if self.max_laps is not None:
            done = done or (self.lap_counts[self.ego_idx] >= self.max_laps)
        return done

    def step(self, action):
        """
        Step function for the gym env

        Args:
            action (np.ndarray(num_agents, 2))

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """

        # call simulation step
        self.sim.step(action)

        # check done
        done = self._check_done()

        # observation
        obs = self.observation_type.observe()
        if self.render_enabled:
            if self.observation_cfg.type is ObservationType.DIRECT:
                # for direct observation, also update the render_obs
                self.render_obs = copy.deepcopy(obs)
            else:
                # for other observation types, update the render_obs
                self.render_obs = copy.deepcopy(self.render_obs_type.observe())

        # times
        reward = self.timestep
        self.sim_time = self.sim.state.sim_time

        truncated = False
        info = {"lap_times": self.lap_times, "lap_counts": self.lap_counts, "sim_time": self.sim_time}

        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the gym environment by given poses

        Args:
            seed: random seed for the reset
            options: dictionary of options for the reset containing initial poses of the agents

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """
        if seed is not None:
            np.random.seed(seed=seed)
        super().reset(seed=seed)

        # reset counters and data members
        self.sim_time = 0.0
        self.agents_prev_s = [None] * self.num_agents
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True] * self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))
        # states after reset
        if options is not None and "poses" in options:
            poses = options["poses"]
            option = "pose"
        elif options is not None and "states" in options:
            poses = options["states"]
            option = "state"
        else:
            poses = self.reset_fn.sample()
            option = "pose"

        if option == 'pose':
            assert isinstance(poses, np.ndarray) and poses.shape == (
                self.num_agents,
                3,
            ), "Initial poses must be a numpy array of shape (num_agents, 3)"
        elif option == 'state':
            assert isinstance(poses, np.ndarray) and poses.shape == (
                self.num_agents,
                self.model.state_dim,
            ), f"Initial full state must be a numpy array of shape (num_agents, {self.model.state_dim})"
        else:
            raise ValueError(
                "Invalid reset option."
            )

        # call reset to simulator
        self.sim.reset(poses, option=option)

        self.start_xs = poses[:, 0]
        self.start_ys = poses[:, 1]
        self.start_thetas = poses[:, 2]
        self.start_rot = np.array(
            [
                [
                    np.cos(-self.start_thetas[self.ego_idx]),
                    -np.sin(-self.start_thetas[self.ego_idx]),
                ],
                [
                    np.sin(-self.start_thetas[self.ego_idx]),
                    np.cos(-self.start_thetas[self.ego_idx]),
                ],
            ]
        )

        # get no input observations
        action = np.zeros((self.num_agents, self.model.control_dim))
        obs, _, _, _, info = self.step(action)

        return obs, info

    def update_map(self, map_name: str):
        """
        Updates the map used by simulation

        Args:
            map_name (str): name of the map

        Returns:
            None
        """
        if "/" in map_name or "\\" in map_name:
            track = Track.from_track_path(map_name, track_scale=self.map_scale)
        else:
            track = Track.from_track_name(map_name, track_scale=self.map_scale)
        self.map = map_name
        self.track = track
        self.env_config = self.env_config.with_updates(map_name=map_name)
        self.sim.set_map(track, self.map_scale)
        self.reset_fn = make_reset_fn(
            track=self.track,
            num_agents=self.num_agents,
            type=self.reset_cfg.strategy,
        )

    def update_params(self, params, index=-1):
        """
        Update the shared vehicle parameters used by the simulator and renderers.

        Args:
            params (VehicleParameters): new vehicle parameters.
            index (int, default=-1): if >= 0 then only update a specific agent's params

        Returns:
            None
        """
        if index >= 0:
            raise NotImplementedError(
                "Per-agent parameter updates are not supported in the simplified simulator"
            )
        if isinstance(params, VehicleParameters):
            vehicle_params = params
        else:
            raise TypeError("params must be a VehicleParameters instance")

        self.vehicle_params = vehicle_params
        self.env_config = self.env_config.with_updates(params=vehicle_params)

        self.sim.update_params(self.vehicle_params)
        if hasattr(self, "renderer") and self.renderer is not None:
            self.renderer.update_params(self.vehicle_params)
        if hasattr(self, "action_space"):
            single_action_space = get_action_space(
                self.longitudinal_action_type,
                self.steer_action_type,
                self.vehicle_params,
            )
            self.action_space = from_single_to_multi_action_space(
                single_action_space, self.num_agents
            )            

    def add_render_callback(self, callback_func):
        """
        Add extra drawing function to call during rendering.

        Args:
            callback_func (function (EnvRenderer) -> None): custom function to called during render()
        """
        if self.render_enabled and self.renderer is not None:
            self.renderer.add_renderer_callback(callback_func)

    def render(self, mode="human"):
        """
        Renders the environment with pyglet. Use mouse scroll in the window to zoom in/out, use mouse click drag to pan.
        Shows the agents, the map, current fps (bottom left corner), and the race information near as text.

        Args:
            mode (str, default='human'): rendering mode, currently supports:
                'human': slowed down rendering such that the env is rendered in a way that sim time elapsed is close to real time elapsed
                'human_fast': render as fast as possible

        Returns:
            None
        """
        # NOTE: separate render (manage render-mode) from render_frame (actual rendering with pyglet)

        if self.render_mode not in self.metadata["render_modes"] or not self.render_enabled:
            return

        self.renderer.update(obs=self.render_obs)
        return self.renderer.render()

    def close(self):
        """
        Ensure renderer is closed upon deletion
        """
        if self.renderer is not None:
            self.renderer.close()
        super().close()
