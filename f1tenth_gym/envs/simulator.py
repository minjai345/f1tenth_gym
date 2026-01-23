"""Core F110 simulator handling multi-agent dynamics, LiDAR, and collision logic."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Optional

import numpy as np

from .action import (
    LongitudinalActionType,
    SteerActionType,
    longitudinal_action_from_type,
    steer_action_from_type,
)
from .collision_models import collision_multiple, get_vertices
from .dynamic_models import DynamicModel, VehicleParameters
from .env_config import EnvConfig
from .lidar import ScanSimulator2D, check_ttc_jit, ray_cast
from .state import SimulationState
from .track import Track

DynamicsFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
IntegratorFn = Callable[[DynamicsFn, np.ndarray, np.ndarray, float, np.ndarray], np.ndarray]
AccelerationFn = Callable[[float, np.ndarray, VehicleParameters], float]
SteeringFn = Callable[[float, np.ndarray, VehicleParameters], float]

@dataclass
class ScanCache:
    angles: np.ndarray
    cosines: np.ndarray
    side_distances: np.ndarray


class F110Simulator:
    """State-driven simulator that steps all agents without per-agent objects."""

    ttc_threshold: float = 0.005

    def __init__(
        self,
        *,
        env_config: EnvConfig,
        vehicle_params: VehicleParameters,
        model: DynamicModel,
        dynamics_fn: DynamicsFn,
        integrator_fn: IntegratorFn,
        longitudinal_type: LongitudinalActionType,
        steering_type: SteerActionType,
        track: Optional[Track],
        seed: int,
    ) -> None:
        self.config = env_config
        self.vehicle_params = vehicle_params
        self.model = model
        self.dynamics_fn = dynamics_fn
        self.integrator_fn = integrator_fn
        self.track = track
        self.seed = seed

        self.num_agents = env_config.num_agents
        self.ego_idx = env_config.ego_index
        self.time_step = env_config.simulation_config.timestep
        self.integrator_dt = env_config.simulation_config.integrator_timestep
        if not np.isclose(self.time_step % self.integrator_dt, 0.0):
            raise ValueError("time_step must be an integer multiple of integrator_timestep")
        self.substeps = max(1, int(round(self.time_step / self.integrator_dt)))

        self.longitudinal_fn: AccelerationFn = longitudinal_action_from_type(longitudinal_type)
        self.steering_fn: SteeringFn = steer_action_from_type(steering_type)

        self.longitudinal_type = longitudinal_type
        self.steering_type = steering_type

        self.params_array = self.vehicle_params.to_array(self.model)

        # Allocate simulation state buffers
        initial_state = self.model.get_initial_state(params=self.params_array)
        self.state_dim = initial_state.shape[0]
        self.control_dim = self.model.control_dim
        scan_size = env_config.lidar_config.num_beams if env_config.lidar_config.enabled else 1
        self.state = SimulationState.allocate(
            num_agents=self.num_agents,
            state_dim=self.state_dim,
            scan_size=scan_size,
            control_dim=self.control_dim,
            delay_steps=env_config.control_config.steer_delay_steps,
        )

        # Static helpers
        self.standard_state_fn = self.model.get_standardized_state_fn()
        self.scan_enabled = env_config.lidar_config.enabled
        self.scan_max_range = env_config.lidar_config.maximum_range

        self.scan_sims: list[ScanSimulator2D] = []
        self.scan_rngs: list[np.random.Generator] = []
        self.scan_cache: list[ScanCache] = []
        if self.scan_enabled:
            for agent_index in range(self.num_agents):
                rng = np.random.default_rng(seed + agent_index)
                simulator = ScanSimulator2D(
                    env_config.lidar_config.num_beams,
                    env_config.lidar_config.field_of_view,
                    std_dev=env_config.lidar_config.noise_std,
                    max_range=env_config.lidar_config.maximum_range,
                )
                if self.track is not None:
                    simulator.set_map(self.track, env_config.map_scale)
                cache = self._build_scan_cache(simulator, self.vehicle_params)
                self.scan_sims.append(simulator)
                self.scan_rngs.append(rng)
                self.scan_cache.append(cache)

        # Geometry buffers for collision checks
        self.agent_vertices = np.zeros((self.num_agents, 4, 2), dtype=np.float64)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def set_map(self, track: Track, map_scale: float = 1.0) -> None:
        self.track = track
        if not self.scan_enabled:
            return
        for simulator in self.scan_sims:
            simulator.set_map(track, map_scale)

    def update_params(self, vehicle_params: VehicleParameters, agent_idx: int = -1) -> None:
        if agent_idx >= 0:
            raise NotImplementedError("Per-agent parameter updates are not supported")
        self.vehicle_params = vehicle_params
        self.params_array = vehicle_params.to_array(self.model)
        if self.scan_enabled:
            for i, simulator in enumerate(self.scan_sims):
                self.scan_cache[i] = self._build_scan_cache(simulator, vehicle_params)

    def reset(self, poses: np.ndarray, *, option: str = "pose") -> None:
        if poses.shape[0] != self.num_agents:
            raise ValueError("Number of poses does not match number of agents")

        self.state.reset()
        if self.scan_enabled:
            for idx in range(self.num_agents):
                self.scan_rngs[idx] = np.random.default_rng(self.seed + idx)
        for i in range(self.num_agents):
            if option == "pose":
                self.state.state[i] = self.model.get_initial_state(
                    pose=poses[i], params=self.params_array
                ).astype(np.float32)
            elif option == "state":
                if poses.shape[1] != self.state_dim:
                    raise ValueError("State reset has incorrect dimension")
                self.state.state[i] = poses[i].astype(np.float32)
            else:
                raise ValueError("Unsupported reset option")
            self.state.standard_state[i] = self.standard_state_fn(self.state.state[i]).astype(
                np.float32
            )
            self.state.poses[i] = np.array(
                [self.state.state[i, 0], self.state.state[i, 1], self.state.state[i, 4]],
                dtype=np.float32,
            )
            if self.config.simulation_config.compute_frenet_frame and self.track is not None:
                self.state.frenet[i] = np.array(
                    self.track.cartesian_to_frenet(
                        float(self.state.state[i, 0]),
                        float(self.state.state[i, 1]),
                        float(self.state.state[i, 4]),
                        use_s_guess=False,
                    ),
                    dtype=np.float32,
                )
    def step(self, control_inputs: np.ndarray) -> None:
        if control_inputs.shape != (self.num_agents, self.control_dim):
            raise ValueError("Control input has incorrect shape")

        steer_commands = control_inputs[:, 0].astype(np.float32)
        accel_commands = control_inputs[:, 1].astype(np.float32)
        self.state.control_input[:, 1] = accel_commands

        if self.state.delay_buffer is not None:
            delayed_steer = self.state.push_delay(steer_commands)
        else:
            delayed_steer = steer_commands
        self.state.control_input[:, 0] = delayed_steer

        for agent_idx in range(self.num_agents):
            state = self.state.state[agent_idx]
            steer_effort = self.steering_fn(delayed_steer[agent_idx], state, self.vehicle_params)
            accel_effort = self.longitudinal_fn(accel_commands[agent_idx], state, self.vehicle_params)
            control_vector = np.array([steer_effort, accel_effort], dtype=np.float32)

            for _ in range(self.substeps):
                state = self.integrator_fn(
                    self.dynamics_fn,
                    state,
                    control_vector,
                    self.integrator_dt,
                    self.params_array,
                )
            state[4] = (state[4] + np.pi) % (2 * np.pi) - np.pi
            self.state.state[agent_idx] = state.astype(np.float32)
            self.state.standard_state[agent_idx] = self.standard_state_fn(state).astype(
                np.float32
            )
            self.state.poses[agent_idx] = np.array(
                [state[0], state[1], state[4]], dtype=np.float32
            )

        if self.config.simulation_config.compute_frenet_frame and self.track is not None:
            for agent_idx in range(self.num_agents):
                pose = self.state.poses[agent_idx]
                self.state.frenet[agent_idx] = np.array(
                    self.track.cartesian_to_frenet(
                        float(pose[0]), float(pose[1]), float(pose[2])
                    ),
                    dtype=np.float32,
                )

        if self.scan_enabled:
            self._update_scans()
        else:
            self.state.scans.fill(0.0)
            self.state.collisions.fill(0.0)

        self._update_agent_collisions()
        self.state.sim_time += self.time_step

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def agent_scans(self) -> np.ndarray:
        return self.state.scans

    @property
    def collisions(self) -> np.ndarray:
        return self.state.collisions

    @property
    def scan_num_beams(self) -> int:
        return self.config.lidar_config.num_beams if self.scan_enabled else 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_scan_cache(self, simulator: ScanSimulator2D, vehicle_params: VehicleParameters) -> ScanCache:
        num_beams = simulator.num_beams
        angles = np.zeros(num_beams, dtype=np.float32)
        cosines = np.zeros(num_beams, dtype=np.float32)
        side_distances = np.zeros(num_beams, dtype=np.float32)

        half_length = float(vehicle_params.length) / 2.0
        half_width = float(vehicle_params.width) / 2.0
        if not np.isfinite(half_length) or not np.isfinite(half_width):
            raise ValueError("Vehicle length and width must be finite to build LiDAR cache")

        increment = simulator.get_increment()
        fov = simulator.fov
        for idx in range(num_beams):
            angle = -fov / 2.0 + idx * increment
            angles[idx] = angle
            cosines[idx] = math.cos(angle)
            sin_angle = math.sin(angle)
            cos_angle = cosines[idx]
            if math.isclose(angle, 0.0, abs_tol=1e-9):
                side_distances[idx] = half_length
            elif 0.0 < angle < math.pi / 2:
                side_distances[idx] = min(
                    half_width / max(sin_angle, 1e-9),
                    half_length / max(cos_angle, 1e-9),
                )
            elif math.pi / 2 <= angle <= math.pi:
                side_distances[idx] = min(
                    half_width / max(math.cos(angle - math.pi / 2), 1e-9),
                    half_length / max(math.sin(angle - math.pi / 2), 1e-9),
                )
            elif -math.pi / 2 < angle < 0.0:
                side_distances[idx] = min(
                    half_width / max(math.sin(-angle), 1e-9),
                    half_length / max(math.cos(-angle), 1e-9),
                )
            else:
                side_distances[idx] = min(
                    half_width / max(math.cos(-angle - math.pi / 2), 1e-9),
                    half_length / max(math.sin(-angle - math.pi / 2), 1e-9),
                )
        return ScanCache(angles=angles, cosines=cosines, side_distances=side_distances)


    def _lidar_pose_from_base(self, pose: np.ndarray) -> np.ndarray:
        tf = self.config.lidar_config.base_link_to_lidar_tf
        dx, dy, dtheta = tf
        if dx == 0.0 and dy == 0.0 and dtheta == 0.0:
            return pose
        cos_yaw = math.cos(pose[2])
        sin_yaw = math.sin(pose[2])
        scan_x = pose[0] + dx * cos_yaw - dy * sin_yaw
        scan_y = pose[1] + dx * sin_yaw + dy * cos_yaw
        scan_theta = pose[2] + dtheta
        return np.array([scan_x, scan_y, scan_theta], dtype=pose.dtype)


    def _update_scans(self) -> None:
        for agent_idx, simulator in enumerate(self.scan_sims):
            pose = self.state.poses[agent_idx]
            scan_pose = self._lidar_pose_from_base(pose)
            scan = simulator.scan(scan_pose, self.scan_rngs[agent_idx])
            cache = self.scan_cache[agent_idx]
            in_collision = check_ttc_jit(
                scan,
                self.state.standard_state[agent_idx, 3],
                cache.angles,
                cache.cosines,
                cache.side_distances,
                self.ttc_threshold,
            )
            if in_collision:
                self.state.state[agent_idx, 3:] = 0.0
                self.state.collisions[agent_idx] = 1.0
            else:
                self.state.collisions[agent_idx] = 0.0

            origin = scan_pose.astype(np.float64)
            adjusted_scan = scan
            for opp_idx in range(self.num_agents):
                if opp_idx == agent_idx:
                    continue
                opp_pose = self.state.poses[opp_idx]
                opp_vertices = get_vertices(
                    np.array([opp_pose[0], opp_pose[1], opp_pose[2]], dtype=np.float64),
                    self.vehicle_params.length,
                    self.vehicle_params.width,
                )
                adjusted_scan = ray_cast(origin, adjusted_scan, cache.angles, opp_vertices)
            self.state.scans[agent_idx] = adjusted_scan.astype(np.float32)

    def _update_agent_collisions(self) -> None:
        for agent_idx in range(self.num_agents):
            pose = self.state.poses[agent_idx]
            self.agent_vertices[agent_idx] = get_vertices(
                np.array([pose[0], pose[1], pose[2]], dtype=np.float64),
                self.vehicle_params.length,
                self.vehicle_params.width,
            )
        collisions, _ = collision_multiple(self.agent_vertices)
        self.state.collisions = np.maximum(self.state.collisions, collisions.astype(np.float32))
