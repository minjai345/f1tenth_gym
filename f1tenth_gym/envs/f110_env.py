import copy
import gymnasium as gym
import numpy as np

from .dynamic_models import (
    DynamicModel,
    F1TENTH_VEHICLE_PARAMETERS,
)
from .simulator import SimpleSimulator
from .env_config import (
    EnvConfig,
    LoopCounterMode,
    build_env_config,
)
from .integrators import IntegratorType, integrator_from_type
from .action import (
    get_action_space,
    from_single_to_multi_action_space,
    LongitudinalActionType,
    SteerActionType,
)
from .observation.observation import ObservationType
from .reset import ResetStrategy
from .collision_models import CollisionCheckMode
from .observation.observation import observation_factory
from .rendering import make_renderer
from .reset import make_reset_fn
from .track import Track
from .utils import deep_update

class F110Env(gym.Env):
    """
    OpenAI gym environment for F1TENTH

    Env should be initialized by calling gym.make('f110_gym:f110-v0', **kwargs)

    Args:
        kwargs:
            seed (int, default=12345): seed for random state and reproducibility
            map (str, default='vegas'): name of the map used for the environment.

            params (dict, default={'mu': 1.0489, 'C_Sf':, 'C_Sr':, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch':7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}): dictionary of vehicle parameters.
            mu: surface friction coefficient
            C_Sf: Cornering stiffness coefficient, front
            C_Sr: Cornering stiffness coefficient, rear
            lf: Distance from center of gravity to front axle
            lr: Distance from center of gravity to rear axle
            h: Height of center of gravity
            m: Total mass of the vehicle
            I: Moment of inertial of the entire vehicle about the z axis
            s_min: Minimum steering angle constraint
            s_max: Maximum steering angle constraint
            sv_min: Minimum steering velocity constraint
            sv_max: Maximum steering velocity constraint
            v_switch: Switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
            a_max: Maximum longitudinal acceleration
            v_min: Minimum longitudinal velocity
            v_max: Maximum longitudinal velocity
            width: width of the vehicle in meters
            length: length of the vehicle in meters

            num_agents (int, default=2): number of agents in the environment

            timestep (float, default=0.01): physics timestep

            ego_idx (int, default=0): ego's index in list of agents
    """

    # NOTE: change matadata with default rendering-modes, add definition of render_fps
    metadata = {"render_modes": ["human", "human_fast", "rgb_array", "unlimited"], "render_fps": 100}

    def __init__(self, config: dict = None, render_mode=None, **kwargs):
        super().__init__()

        # Configuration
        self.config = self.default_config()
        self.configure(config)

        # env states
        self.poses_x = []
        self.poses_y = []
        self.poses_theta = []
        

        # loop completion
        self.near_start = True
        self.num_toggles = 0

        # finish line info
        if self.loop_counter_mode is LoopCounterMode.FRENET_BASED:
            self.config["loop_counting_method"] = LoopCounterMode.FRENET_BASED
            self.agents_prev_s = [None] * self.num_agents
            self.lap_times = np.zeros((self.num_agents, ))
            self.lap_times_finish = np.zeros((self.num_agents, ))
            self.lap_counts = np.zeros((self.num_agents, ))
            self.sim_time = 0.0
        if type(self.map) is not Track:
            if '/' in self.map or '\\' in self.map:
                self.track = Track.from_track_path(
                    self.map,
                    track_scale=self.config["map_scale"],
                )
            else:
                self.track = Track.from_track_name(
                    self.map,
                    track_scale=self.config["map_scale"],
                )  # load track in gym env for convenience
        else:
            self.track = self.map  # use the track directly
        
        # initiate simulator
        self.sim = SimpleSimulator(
            env_config=self.env_config,
            params=self.params,
            model=self.model,
            dynamics_fn=self.model.f_dynamics,
            integrator_fn=self.integrator_fn,
            longitudinal_type=self.longitudinal_action_type,
            steering_type=self.steer_action_type,
            track=self.track,
            seed=self.seed,
        )
        if isinstance(self.track, Track):
            self.sim.set_map(self.track, self.config["map_scale"])

        # observations
        self.agent_ids = [f"agent_{i}" for i in range(self.num_agents)]

        assert (
            "type" in self.observation_config
        ), "observation_config must contain 'type' key"
        self.observation_type = observation_factory(env=self, **self.observation_config)
        self.observation_space = self.observation_type.space()
        self.render_obs_type = observation_factory(env=self, type=ObservationType.DIRECT)
        self.render_obs = None

        # action space
        single_action_space = get_action_space(
            self.longitudinal_action_type,
            self.steer_action_type,
            self.params
        )
        self.action_space = from_single_to_multi_action_space(
            single_action_space, self.num_agents
        )

        # reset modes
        self.reset_fn = make_reset_fn(
            **self.config["reset_config"], track=self.track, num_agents=self.num_agents
        )

        # stateful observations for rendering
        # add choice of colors (same, random, ...)
        self.render_mode = render_mode

        # match render_fps to integration timestep
        self.metadata["render_fps"] = int(1.0 / self.timestep)
        if self.render_mode == "human_fast":
            self.metadata["render_fps"] *= 10  # boost fps by 10x
        elif self.render_mode == "unlimited":
            self.metadata["render_fps"] = float('inf')
        if self.config["enable_rendering"]:
            self.renderer, self.render_spec = make_renderer(
                params=self.params,
                track=self.track,
                agent_ids=self.agent_ids,
                render_mode=render_mode,
                render_fps=self.metadata["render_fps"],
            )
            
    @classmethod
    def default_config(cls) -> dict:
        """
        Default environment configuration.

        Can be overloaded in environment implementations, or by calling configure().

        Args:
            None

        Returns:
            a configuration dict
        """
        return {
            "seed": 12345,
            "map": "Spielberg",
            "map_scale": 1.0,
            "params": F1TENTH_VEHICLE_PARAMETERS.as_mapping(),
            "num_agents": 1,
            "timestep": 0.01,
            "integrator_timestep": 0.01,
            "ego_idx": 0,
            "max_laps": 1,  # 'inf' for infinite laps, or a positive integer
            "integrator": IntegratorType.RK4,
            "model": DynamicModel.ST,  # DynamicModel.KS, DynamicModel.ST, DynamicModel.MB
            "control_input": [
                LongitudinalActionType.SPEED,
                SteerActionType.STEERING_ANGLE,
            ],  # default speed + steering angle control
            "observation_config": {"type": ObservationType.DIRECT},
            "reset_config": {"type": ResetStrategy.RL_GRID_STATIC},
            "enable_rendering": True,
            "enable_scan": True, # NOTE no lidar scan and collision if False
            "lidar_fov": 4.712389,
            "lidar_num_beams": 1080,
            "lidar_range": 30.0,
            "lidar_noise_std": 0.01,
            "steer_delay_buffer_size": 0,
            "compute_frenet": True,
            "collision_check_method": CollisionCheckMode.LIDAR_SCAN,  # CollisionCheckMode.LIDAR_SCAN or CollisionCheckMode.BOUNDING_BOX
            "loop_counting_method": LoopCounterMode.FRENET_BASED, # "toggle", "frenet_based", "winding_angle"
        }
    
    def configure(self, config: dict) -> None:
        if config:
            self.config = deep_update(self.config, config)

        self._sync_env_config()

        if hasattr(self, "sim"):
            self.sim.update_params(self.params)

        if hasattr(self, "renderer") and self.renderer is not None:
            # if renderer exists, update the params
            self.renderer.update_params(self.params)

        if hasattr(self, "action_space"):
            # if some parameters changed, recompute action space
            single_action_space = get_action_space(
                self.longitudinal_action_type,
                self.steer_action_type,
                self.params
            )
            self.action_space = from_single_to_multi_action_space(
                single_action_space, self.num_agents
            )

    def _sync_env_config(self) -> None:
        "Rebuild typed config view from the legacy config dictionary."
        self.env_config: EnvConfig = build_env_config(self.config)
        self._apply_env_config()

    def _apply_env_config(self) -> None:
        "Apply values from env_config to legacy attributes."
        self.config["compute_frenet"] = self.env_config.simulation.compute_frenet_frame

        self.seed = self.env_config.seed
        self.map = self.env_config.map_name
        param_mapping = self.env_config.params.as_mapping()
        self.config["params"] = param_mapping
        self.params = param_mapping
        self.vehicle_params = self.env_config.params
        self.num_agents = self.env_config.num_agents
        self.timestep = self.env_config.simulation.timestep
        self.ego_idx = self.env_config.ego_index
        self.max_laps = self.env_config.simulation.max_laps

        self.integrator_fn = integrator_from_type(
            self.env_config.simulation.integrator
        )

        self.model = self.env_config.simulation.dynamics_model

        observation_dict = dict(self.config.get("observation_config", {"type": ObservationType.ORIGINAL}))
        observation_dict["type"] = self.env_config.observation.type
        if self.env_config.observation.features is not None:
            observation_dict["features"] = tuple(self.env_config.observation.features)
        else:
            observation_dict.pop("features", None)
        self.config["observation_config"] = observation_dict
        self.observation_config = dict(observation_dict)
        self.observation_cfg = self.env_config.observation
        self.reset_cfg = self.env_config.reset
        self.lidar_cfg = self.env_config.lidar
        self.control_cfg = self.env_config.control
        self.simulation_cfg = self.env_config.simulation
        self.loop_counter_mode = self.env_config.simulation.loop_counter
        self.collision_check_mode = self.env_config.collision_check
        self.render_enabled = self.env_config.render_enabled

        self.longitudinal_action_type = self.control_cfg.longitudinal_mode
        self.steer_action_type = self.control_cfg.steering_mode
        self.steer_delay_steps = self.control_cfg.steer_delay_steps

        self.config["integrator"] = self.env_config.simulation.integrator
        self.config["model"] = self.env_config.simulation.dynamics_model
        self.config["control_input"] = [
            self.control_cfg.longitudinal_mode,
            self.control_cfg.steering_mode,
        ]
        self.config["enable_rendering"] = self.render_enabled
        self.config["enable_scan"] = self.lidar_cfg.enabled
        self.config["lidar_num_beams"] = self.lidar_cfg.num_beams
        self.config["lidar_fov"] = self.lidar_cfg.field_of_view
        self.config["lidar_range"] = self.lidar_cfg.maximum_range
        self.config["lidar_noise_std"] = self.lidar_cfg.noise_std
        self.config["steer_delay_buffer_size"] = self.steer_delay_steps

        reset_config = dict(self.config.get("reset_config", {}))
        reset_config["type"] = self.reset_cfg.strategy
        self.config["reset_config"] = reset_config
        self.config["collision_check_method"] = self.collision_check_mode
        self.config["loop_counting_method"] = self.loop_counter_mode
        self.config["max_laps"] = "inf" if self.max_laps is None else self.max_laps

    def _check_done(self):
        """
        Check if the current rollout is done
        """
        if (
            self.loop_counter_mode is LoopCounterMode.FRENET_BASED
            and self.config["compute_frenet"]
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
        if self.config["enable_rendering"]:
            if self.observation_config["type"] == "direct":
                # for direct observation, also update the render_obs
                self.render_obs = copy.deepcopy(obs)
            else:
                # for other observation types, update the render_obs
                self.render_obs = copy.deepcopy(self.render_obs_type.observe())

        # times
        reward = self.timestep
        self.sim_time = self.sim.state.sim_time

        truncated = False
        info = {"lap_times": self.lap_times, 
                "lap_counts": self.lap_counts,
                "sim_time": self.sim_time}

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
            track = Track.from_track_path(map_name, track_scale=self.config["map_scale"])
        else:
            track = Track.from_track_name(map_name, track_scale=self.config["map_scale"])
        self.map = map_name
        self.track = track
        self.sim.set_map(track, self.config["map_scale"])


    def update_params(self, params, index=-1):
        """
        Updates the parameters used by simulation for vehicles

        Args:
            params (dict): dictionary of parameters
            index (int, default=-1): if >= 0 then only update a specific agent's params

        Returns:
            None
        """
        if index >= 0:
            raise NotImplementedError(
                "Per-agent parameter updates are not supported in the simplified simulator"
            )
        self.params.update(params)
        self.sim.update_params(self.params)

    def add_render_callback(self, callback_func):
        """
        Add extra drawing function to call during rendering.

        Args:
            callback_func (function (EnvRenderer) -> None): custom function to called during render()
        """
        if self.config["enable_rendering"]:
            self.renderer.add_renderer_callback(callback_func)

    def render(self, mode="human"):
        """
        Renders the environment with pyglet. Use mouse scroll in the window to zoom in/out, use mouse click drag to pan. Shows the agents, the map, current fps (bottom left corner), and the race information near as text.

        Args:
            mode (str, default='human'): rendering mode, currently supports:
                'human': slowed down rendering such that the env is rendered in a way that sim time elapsed is close to real time elapsed
                'human_fast': render as fast as possible

        Returns:
            None
        """
        # NOTE: separate render (manage render-mode) from render_frame (actual rendering with pyglet)

        if self.render_mode not in self.metadata["render_modes"] or not self.config["enable_rendering"]:
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






