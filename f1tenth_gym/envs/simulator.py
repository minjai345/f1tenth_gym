"""Core F110 simulator handling multi-agent dynamics, LiDAR, and collision logic."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Optional
import warnings

import numpy as np

from .action import (
    LongitudinalActionType,
    SteerActionType,
    longitudinal_action_from_type,
    steer_action_from_type,
)
from .collision_models import CollisionCheckMode, collision_multiple, get_vertices
from .dynamic_models import DynamicModel, PoseReference, VehicleParameters
from .env_config import EnvConfig
from .lidar import ScanSimulator2D, check_collision, ray_cast
from .state import SimulationState
from .track import Track

DynamicsFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
IntegratorFn = Callable[[DynamicsFn, np.ndarray, np.ndarray, float, np.ndarray], np.ndarray]
AccelerationFn = Callable[[float, np.ndarray, VehicleParameters], float]
SteeringFn = Callable[[float, np.ndarray, VehicleParameters], float]

@dataclass
class ScanCache:
    """Precomputed LiDAR geometry for collision detection.

    Attributes:
        angles: Beam angles relative to vehicle heading (used by ray_cast).
        side_distances: Distance from the LiDAR to the vehicle body edge per
            beam (used by the collision check).
    """

    angles: np.ndarray
    side_distances: np.ndarray


class F110Simulator:
    """Core simulator for F1TENTH multi-agent racing.

    Handles vehicle dynamics integration, LiDAR simulation, and collision detection
    for all agents in a single state-driven update loop.

    Attributes:
        state: Current simulation state containing poses, velocities, scans.
        track: The racing track being simulated.
        vehicle_params: Physical parameters of the vehicle.
        num_agents: Number of agents in the simulation.
    """

    collision_margin: float = 0.005  # metres; a beam collides when scan - side_distance <= this

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
        # Validate the way we compute. Testing `time_step % integrator_dt` against 0
        # rejects exact multiples that IEEE754 cannot represent: 0.03 % 0.01 is
        # 0.00999999999999999847, not 0. Compare the ratio to the substep count that
        # is actually used instead, which accepts every pair that divides evenly in
        # real arithmetic.
        ratio = self.time_step / self.integrator_dt
        self.substeps = max(1, int(round(ratio)))
        if not np.isclose(ratio, self.substeps, rtol=0.0, atol=1e-9):
            raise ValueError(
                f"timestep ({self.time_step}) must be an integer multiple of "
                f"integrator_timestep ({self.integrator_dt}), got a ratio of {ratio}"
            )

        self.collision_check_mode: CollisionCheckMode = env_config.collision_check
        self.longitudinal_fn: AccelerationFn = longitudinal_action_from_type(longitudinal_type)
        self.steering_fn: SteeringFn = steer_action_from_type(
            steering_type, steer_kp=env_config.control_config.steer_kp
        )

        self.longitudinal_type = longitudinal_type
        self.steering_type = steering_type

        self.params_array = self.vehicle_params.to_array()

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

        # actuation realism: command noise + longitudinal (throttle) delay
        cc = env_config.control_config
        self._steer_noise_std = float(cc.steer_noise_std)
        self._accl_noise_std = float(cc.accl_noise_std)
        self._throttle_delay_steps = int(cc.throttle_delay_steps)
        # A None seed (EnvConfig.seed unset) still needs a concrete int here:
        # the per-agent scan RNGs below are derived as seed + agent_index, and
        # reset(noise_seed=None) falls back to self.seed the same way.
        if self.seed is None:
            self.seed = int(np.random.SeedSequence().generate_state(1)[0])
            seed = self.seed
        self.control_rng = np.random.default_rng(seed)  # command-noise stream
        if self._throttle_delay_steps > 0:
            self._throttle_buffer = np.zeros((self.num_agents, self._throttle_delay_steps), dtype=np.float32)
            self._throttle_head = np.zeros((self.num_agents,), dtype=np.int32)
        else:
            self._throttle_buffer = None
            self._throttle_head = None

        # Static helpers
        self.standard_state_fn = self.model.get_standardized_state_fn()
        self.scan_enabled = env_config.lidar_config.enabled
        self.scan_max_range = env_config.lidar_config.maximum_range

        self.scan_sims: list[ScanSimulator2D] = []
        self.scan_rngs: list[np.random.Generator] = []
        self.scan_cache: list[ScanCache] = []
        # per-agent, per-beam systematic range bias (drawn per episode at reset)
        self.scan_bias = np.zeros((self.num_agents, env_config.lidar_config.num_beams), dtype=np.float32)
        if self.scan_enabled:
            lidar_cfg = env_config.lidar_config
            for agent_index in range(self.num_agents):
                rng = np.random.default_rng(seed + agent_index)
                simulator = ScanSimulator2D(
                    lidar_cfg.num_beams,
                    lidar_cfg.field_of_view,
                    angle_min=lidar_cfg.angle_min,
                    angle_max=lidar_cfg.angle_max,
                    std_dev=lidar_cfg.noise_std,
                    min_range=lidar_cfg.range_min,
                    max_range=lidar_cfg.range_max,
                )
                if self.track is not None:
                    simulator.set_map(self.track, env_config.map_scale)
                cache = self._build_scan_cache(simulator, self.vehicle_params)
                self.scan_sims.append(simulator)
                self.scan_rngs.append(rng)
                self.scan_cache.append(cache)

        # Geometry buffers for collision checks
        self.agent_vertices = np.zeros((self.num_agents, 4, 2), dtype=np.float64)
        self._adjusted_scans = np.zeros((self.num_agents, scan_size), dtype=np.float32)

        self._collision_body_dx, self._collision_body_dy = self._compute_collision_body_offset(
            self.vehicle_params, self.model
        )
        self._cog_offset = self._compute_cog_offset(self.vehicle_params, self.model)
        self.contact = None
        self._build_contact()
        # pose at the top of the current step, used to undo a move that ends in
        # contact (see _halt_on_collision). None => nothing to restore yet.
        self._pre_pose: np.ndarray | None = None

    @staticmethod
    def _compute_cog_offset(vehicle_params: VehicleParameters, model: DynamicModel) -> float:
        """Forward shift from the model's native x/y anchor to the CoG.

        ``standard_state`` (and every observation derived from it) is
        CoG-anchored for all models; rear-axle models need +lr along the yaw.
        """
        if model.pose_reference is PoseReference.REAR_AXLE:
            lr = float(vehicle_params.lr)
            return lr if math.isfinite(lr) else 0.0
        return 0.0

    def _standardize(self, state: np.ndarray) -> np.ndarray:
        """A ``standard_state`` row from a native state row, CoG-normalised."""
        std = self.standard_state_fn(state).astype(np.float32)
        if self._cog_offset != 0.0:
            std[0] += self._cog_offset * math.cos(std[4])
            std[1] += self._cog_offset * math.sin(std[4])
        return std

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def set_map(self, track: Track, map_scale: float = 1.0) -> None:
        """Set or update the track used for simulation.

        Args:
            track: Track object with occupancy map and reference lines.
            map_scale: Scale factor applied to the map.
        """
        self.track = track
        if not self.scan_enabled:
            return
        for simulator in self.scan_sims:
            simulator.set_map(track, map_scale)

    def update_params(self, vehicle_params: VehicleParameters, agent_idx: int = -1) -> None:
        """Update vehicle parameters for all agents.

        Args:
            vehicle_params: New vehicle physical parameters.
            agent_idx: Agent index (-1 for all agents, per-agent not supported).
        """
        if agent_idx >= 0:
            raise NotImplementedError("Per-agent parameter updates are not supported")
        self.vehicle_params = vehicle_params
        self.params_array = vehicle_params.to_array()
        self._collision_body_dx, self._collision_body_dy = self._compute_collision_body_offset(
            self.vehicle_params, self.model
        )
        self._cog_offset = self._compute_cog_offset(self.vehicle_params, self.model)
        if self.scan_enabled:
            for i, simulator in enumerate(self.scan_sims):
                self.scan_cache[i] = self._build_scan_cache(simulator, vehicle_params)
        self._build_contact()

    def reset(self, poses: np.ndarray, *, option: str = "pose", noise_seed: int | None = None) -> None:
        """Reset all agents to initial positions.

        Args:
            poses: Initial positions, shape (num_agents, 3) for poses or
                   (num_agents, state_dim) for full state.
            option: Reset mode - "pose" for (x, y, theta) or "state" for full state.
            noise_seed: Seed for the per-agent LiDAR-noise RNGs. ``None`` falls
                back to ``self.seed``, which is the simulator's *construction*
                seed — and that is drawn from OS entropy when ``EnvConfig.seed``
                is ``None`` (the default), since the RNGs below need a concrete
                int. So ``noise_seed=None`` is only reproducible when the config
                carries a seed. ``F110Env.reset`` always passes a value derived
                from ``reset(seed=...)``, so the env path is unaffected and the
                noise stream is controlled by the reset seed per the gymnasium
                contract; this fallback matters only when driving
                ``F110Simulator`` directly.
        """
        if poses.shape[0] != self.num_agents:
            raise ValueError("Number of poses does not match number of agents")

        self.state.reset()
        base_seed = self.seed if noise_seed is None else int(noise_seed)
        # command-noise stream, deterministic in the reset seed (offset well away
        # from the per-agent scan seeds base_seed+idx)
        self.control_rng = np.random.default_rng(base_seed + 2 ** 20)
        if self._throttle_buffer is not None:
            self._throttle_buffer.fill(0.0)
            self._throttle_head.fill(0)
        if self.scan_enabled:
            bias_std = self.config.lidar_config.range_bias_std
            for idx in range(self.num_agents):
                self.scan_rngs[idx] = np.random.default_rng(base_seed + idx)
                # per-episode per-beam systematic bias (constant across the rollout)
                if bias_std > 0.0:
                    self.scan_bias[idx] = self.scan_rngs[idx].normal(
                        0.0, bias_std, size=self.scan_bias.shape[1]
                    ).astype(np.float32)
                else:
                    self.scan_bias[idx] = 0.0
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
            self.state.standard_state[i] = self._standardize(self.state.state[i])
            self.state.poses[i] = np.array(
                [self.state.state[i, 0], self.state.state[i, 1], self.state.state[i, 4]],
                dtype=np.float32,
            )
            if self.config.simulation_config.compute_frenet_frame and self.track is not None:
                # anchor Frenet at the CoG (standard_state), matching the
                # observed pose_x/pose_y whatever the model's native frame
                self.state.frenet[i] = np.array(
                    self.track.cartesian_to_frenet(
                        float(self.state.standard_state[i, 0]),
                        float(self.state.standard_state[i, 1]),
                        float(self.state.standard_state[i, 4]),
                        use_s_guess=False,
                    ),
                    dtype=np.float32,
                )

        # A restore must never reach back into the previous episode.
        self._pre_pose = None

        # One LiDAR sweep so the first observation is not all zeros (#15).
        if self.scan_enabled:
            self._spawned_in_contact = []
            self._update_scans(flag_collisions=False)
            if self._spawned_in_contact:
                # The halt rejects the move that causes contact, which keeps a
                # car out of geometry by induction from a clear pre-step pose.
                # A spawn that is ALREADY inside the margin breaks that base
                # case: every later move is rejected too, so the car cannot
                # drive or reverse out. Only reachable by supplying the pose --
                # every shipped reset strategy spawns well clear.
                warnings.warn(
                    f"agents {self._spawned_in_contact} spawned inside the collision "
                    f"margin ({self.collision_margin} m). A collision halt rejects "
                    f"the move that causes contact, so a car that starts in contact "
                    f"cannot move until it is reset to a clear pose.",
                    RuntimeWarning,
                    stacklevel=3,
                )

    def step(self, control_inputs: np.ndarray) -> None:
        """Advance simulation by one timestep.

        Args:
            control_inputs: Control commands, shape (num_agents, 2) with
                            [steering, acceleration/speed] per agent.
        """
        if control_inputs.shape != (self.num_agents, self.control_dim):
            raise ValueError("Control input has incorrect shape")

        # Snapshot before the dynamics move anything: _halt_on_collision
        # restores from this to reject a move that ends inside geometry.
        self._pre_pose = self.state.state.copy()

        steer_commands = control_inputs[:, 0].astype(np.float32)
        accel_commands = control_inputs[:, 1].astype(np.float32)

        # actuator command noise (added at command time, before the lag buffers)
        if self._steer_noise_std > 0.0:
            steer_commands = steer_commands + self.control_rng.normal(
                0.0, self._steer_noise_std, self.num_agents
            ).astype(np.float32)
        if self._accl_noise_std > 0.0:
            accel_commands = accel_commands + self.control_rng.normal(
                0.0, self._accl_noise_std, self.num_agents
            ).astype(np.float32)

        # longitudinal (throttle) actuator delay
        if self._throttle_buffer is not None:
            accel_commands = self._push_throttle_delay(accel_commands)
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
            self.state.standard_state[agent_idx] = self._standardize(state)
            self.state.poses[agent_idx] = np.array(
                [state[0], state[1], state[4]], dtype=np.float32
            )

        # Before the Frenet block, so the corrected pose is what the Frenet frame,
        # the scan and the observation all see, with nothing to re-derive.
        if self.contact is not None:
            self._resolve_contacts()

        if self.config.simulation_config.compute_frenet_frame and self.track is not None:
            for agent_idx in range(self.num_agents):
                # CoG-anchored (standard_state), matching the observed pose
                std = self.state.standard_state[agent_idx]
                # Anchor the local arclength search to THIS agent's own previous s.
                # Falling back to the shared Track.s_guess windows each agent around
                # the *previous* agent's position, corrupting multi-agent Frenet.
                prev_s = float(self.state.frenet[agent_idx, 0])
                self.state.frenet[agent_idx] = np.array(
                    self.track.cartesian_to_frenet(
                        float(std[0]), float(std[1]), float(std[4]),
                        s_guess=prev_s, use_s_guess=True,
                    ),
                    dtype=np.float32,
                )

        if self.scan_enabled:
            self._update_scans()
        else:
            self.state.scans.fill(0.0)
            # SEGMENT_CONTACT already wrote the flag and needs no scan; the other
            # modes detect walls inside _update_scans, so they genuinely have none.
            if self.contact is None:
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
        """Build precomputed scan geometry for collision detection.

        Computes side_distances as the distance from the LiDAR position to the
        vehicle body edge for each beam angle. This correctly accounts for the
        LiDAR offset (base_link_to_lidar_tf) and collision body center offset
        (collision_body_center_x/y) from the vehicle parameters, with an
        adjustment for collision checks when the pose is at the center of gravity.
        """
        num_beams = simulator.num_beams

        half_length = float(vehicle_params.length) / 2.0
        half_width = float(vehicle_params.width) / 2.0
        if not np.isfinite(half_length) or not np.isfinite(half_width):
            raise ValueError("Vehicle length and width must be finite to build LiDAR cache")

        # Get LiDAR offset from base_link
        lidar_tf = self.config.lidar_config.base_link_to_lidar_tf
        lidar_dx, lidar_dy, lidar_dtheta = lidar_tf

        # Get collision body center offset from base_link
        body_dx = vehicle_params.collision_body_center_x
        body_dy = vehicle_params.collision_body_center_y

        # If state is referenced at the center of gravity, base_link is behind it by lr.
        # Leave LiDAR in the state frame to avoid shifting the scan origin.
        base_dx = 0.0
        if self.model.pose_reference is PoseReference.COG:
            base_dx = -float(vehicle_params.lr)
            if not math.isfinite(base_dx):
                base_dx = 0.0

        # Compute LiDAR position relative to collision body center
        # (the collision body is centered at (body_dx, body_dy) from base_link)
        body_x = base_dx + body_dx
        lidar_x_in_body = lidar_dx - body_x
        lidar_y_in_body = lidar_dy - body_dy

        increment = simulator.get_increment()
        angle_min = simulator.angle_min

        beam_angles = (angle_min + np.arange(num_beams, dtype=np.float64) * increment).astype(np.float32)
        ray_angles = beam_angles + lidar_dtheta
        dir_cos = np.cos(ray_angles)
        dir_sin = np.sin(ray_angles)
        side_distances = self._ray_to_rect_distance_vec(
            lidar_x_in_body, lidar_y_in_body,
            dir_cos, dir_sin,
            half_length, half_width,
        ).astype(np.float32)

        return ScanCache(angles=beam_angles, side_distances=side_distances)


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

    @staticmethod
    def _compute_collision_body_offset(
        vehicle_params: VehicleParameters, model: DynamicModel
    ) -> tuple[float, float]:
        base_dx = 0.0
        if model.pose_reference is PoseReference.COG:
            base_dx = -float(vehicle_params.lr)
            if not math.isfinite(base_dx):
                base_dx = 0.0
        dx = base_dx + float(vehicle_params.collision_body_center_x)
        dy = float(vehicle_params.collision_body_center_y)
        return dx, dy

    def _collision_pose_from_base(self, pose: np.ndarray) -> np.ndarray:
        dx, dy = self._collision_body_dx, self._collision_body_dy
        if dx == 0.0 and dy == 0.0:
            return pose
        cos_yaw = math.cos(pose[2])
        sin_yaw = math.sin(pose[2])
        body_x = pose[0] + dx * cos_yaw - dy * sin_yaw
        body_y = pose[1] + dx * sin_yaw + dy * cos_yaw
        return np.array([body_x, body_y, pose[2]], dtype=pose.dtype)

    @staticmethod
    def _ray_to_rect_distance(
        origin_x: float,
        origin_y: float,
        dir_cos: float,
        dir_sin: float,
        half_length: float,
        half_width: float,
    ) -> float:
        """Compute distance from a point to the rectangle boundary along a ray.

        Args:
            origin_x, origin_y: Ray origin point (e.g., LiDAR position in body frame).
            dir_cos, dir_sin: Ray direction as (cos(angle), sin(angle)).
            half_length: Half of vehicle length (x extent from center).
            half_width: Half of vehicle width (y extent from center).

        Returns:
            Distance from origin to rectangle edge along the ray direction.
            Returns 0.0 if origin is outside the rectangle.
        """
        x_min, x_max = -half_length, half_length
        y_min, y_max = -half_width, half_width

        eps = 1e-9
        if not (x_min - eps <= origin_x <= x_max + eps and
                y_min - eps <= origin_y <= y_max + eps):
            return 0.0

        min_t = float('inf')

        if abs(dir_cos) > eps:
            t = (x_max - origin_x) / dir_cos
            if t > eps:
                y_intersect = origin_y + t * dir_sin
                if y_min - eps <= y_intersect <= y_max + eps:
                    min_t = min(min_t, t)

            t = (x_min - origin_x) / dir_cos
            if t > eps:
                y_intersect = origin_y + t * dir_sin
                if y_min - eps <= y_intersect <= y_max + eps:
                    min_t = min(min_t, t)

        if abs(dir_sin) > eps:
            t = (y_max - origin_y) / dir_sin
            if t > eps:
                x_intersect = origin_x + t * dir_cos
                if x_min - eps <= x_intersect <= x_max + eps:
                    min_t = min(min_t, t)

            t = (y_min - origin_y) / dir_sin
            if t > eps:
                x_intersect = origin_x + t * dir_cos
                if x_min - eps <= x_intersect <= x_max + eps:
                    min_t = min(min_t, t)

        if min_t == float('inf'):
            return 0.0

        return float(min_t)

    @staticmethod
    def _ray_to_rect_distance_vec(
        origin_x: float,
        origin_y: float,
        dir_cos: np.ndarray,
        dir_sin: np.ndarray,
        half_length: float,
        half_width: float,
    ) -> np.ndarray:
        """Vectorised version of _ray_to_rect_distance for all beams at once."""
        eps = 1e-9
        inside = (
            (-half_length - eps) <= origin_x <= (half_length + eps)
            and (-half_width - eps) <= origin_y <= (half_width + eps)
        )
        if not inside:
            return np.zeros(len(dir_cos), dtype=np.float64)

        min_t = np.full(len(dir_cos), np.inf, dtype=np.float64)

        mask_cx = np.abs(dir_cos) > eps
        safe_dc = np.where(mask_cx, dir_cos, 1.0)

        t = np.where(mask_cx, (half_length - origin_x) / safe_dc, np.inf)
        yi = origin_y + t * dir_sin
        valid = mask_cx & (t > eps) & (yi >= -half_width - eps) & (yi <= half_width + eps)
        min_t = np.where(valid, np.minimum(min_t, t), min_t)

        t = np.where(mask_cx, (-half_length - origin_x) / safe_dc, np.inf)
        yi = origin_y + t * dir_sin
        valid = mask_cx & (t > eps) & (yi >= -half_width - eps) & (yi <= half_width + eps)
        min_t = np.where(valid, np.minimum(min_t, t), min_t)

        mask_sy = np.abs(dir_sin) > eps
        safe_ds = np.where(mask_sy, dir_sin, 1.0)

        t = np.where(mask_sy, (half_width - origin_y) / safe_ds, np.inf)
        xi = origin_x + t * dir_cos
        valid = mask_sy & (t > eps) & (xi >= -half_length - eps) & (xi <= half_length + eps)
        min_t = np.where(valid, np.minimum(min_t, t), min_t)

        t = np.where(mask_sy, (-half_width - origin_y) / safe_ds, np.inf)
        xi = origin_x + t * dir_cos
        valid = mask_sy & (t > eps) & (xi >= -half_length - eps) & (xi <= half_length + eps)
        min_t = np.where(valid, np.minimum(min_t, t), min_t)

        return np.where(min_t == np.inf, 0.0, min_t)

    def _halt_on_collision(self, agent_idx: int) -> None:
        """Stop a car that has collided, undo the move that caused it, and flag it.

        Two things happen, and both are needed:

        1. Zero the velocities and rates, but KEEP every other pose-like state.
           The indices come from ``model.velocity_indices()`` — a blanket
           ``[5:] = 0`` would wipe MB's roll/pitch angles and ride heights
           (indices 5-28) and divide-by-zero the suspension math next step.
        2. Restore ``(x, y, yaw)`` to the pose captured at the top of ``step``,
           rejecting the move that produced the contact. Zeroing the velocity
           alone is not enough: the dynamics integrate BEFORE this check runs,
           so each step a car held against a wall re-accelerates from ``v=0``,
           gains a few hundred micrometres inside the wall, and keeps only the
           position. That penetration is monotonic — it accumulated past
           Spielberg's 23 cm walls in ~1800 steps and the car drove out the far
           side. Undoing the move restores the invariant "a halted car is never
           inside geometry", by induction on the pre-step pose being clear.

        A move *away* from the wall does not trigger a contact, so it is never
        rejected — a car can always reverse off a wall it is pinned against.
        That guarantee holds by induction from a collision-free pre-step pose;
        a pose handed in through ``options={"poses"/"states": ...}`` that
        already overlaps geometry breaks the base case and cannot recover while
        ``terminate_on_collision=False``.

        ``standard_state`` is always ``[X, Y, steer, speed, yaw, yaw_rate,
        beta]``, so its speed (3) and rates (5:) zero unconditionally; yaw at
        [4] is preserved in both buffers (the old ``state[3:] = 0`` snapped the
        heading to east).

        Rewinding the pose invalidates everything derived from it earlier in the
        step, so all four mirrors are re-derived here: ``standard_state``,
        ``poses``, the Frenet frame and this agent's collision vertices. The
        published scan is re-traced by the caller, which owns the scan
        simulator.
        """
        for i in self.model.velocity_indices():
            self.state.state[agent_idx, i] = 0.0

        if self._pre_pose is not None:
            pose_indices = self.model.pose_indices()
            for i in pose_indices:
                self.state.state[agent_idx, i] = self._pre_pose[agent_idx, i]
            # re-derive the mirrors so state / standard_state / poses agree.
            # `poses` is built from pose_indices() too, not from hardcoded
            # columns: a model that laid its state out differently would
            # otherwise have the halt restore the right cells and then publish
            # the wrong ones.
            self.state.standard_state[agent_idx] = self._standardize(
                self.state.state[agent_idx]
            )
            self.state.poses[agent_idx] = np.array(
                [self.state.state[agent_idx, i] for i in pose_indices],
                dtype=np.float32,
            )
            self._refresh_derived_from_pose(agent_idx)

        self.state.standard_state[agent_idx, 3] = 0.0
        self.state.standard_state[agent_idx, 5:] = 0.0
        self.state.collisions[agent_idx] = 1.0

    def _refresh_derived_from_pose(self, agent_idx: int) -> None:
        """Re-derive the Frenet frame and collision vertices after a pose rewind.

        ``step`` computes the Frenet frame before the collision pass, and
        ``_update_scans`` snapshots every agent's vertices before the per-agent
        loop, so both describe the pose the halt has just undone. Without this,
        ``obs['frenet_pose']`` disagrees with the ``pose_x``/``pose_y`` beside
        it, and a later agent's ``ray_cast`` (and the ``BOUNDING_BOX`` GJK pass)
        occludes against a body the simulator has already moved back.
        """
        if self.config.simulation_config.compute_frenet_frame and self.track is not None:
            std = self.state.standard_state[agent_idx]
            prev_s = float(self.state.frenet[agent_idx, 0])
            self.state.frenet[agent_idx] = np.array(
                self.track.cartesian_to_frenet(
                    float(std[0]), float(std[1]), float(std[4]),
                    s_guess=prev_s, use_s_guess=True,
                ),
                dtype=np.float32,
            )

        # _update_agent_collisions can reach the halt before any vertex snapshot
        # exists, so this is guarded rather than assumed.
        if getattr(self, "_all_vertices", None) is not None:
            cp = self._collision_pose_from_base(self.state.poses[agent_idx])
            self._all_vertices[agent_idx] = get_vertices(
                np.array([cp[0], cp[1], cp[2]], dtype=np.float64),
                self.vehicle_params.length,
                self.vehicle_params.width,
            )

    def _push_throttle_delay(self, values: np.ndarray) -> np.ndarray:
        """Ring-buffer delay for the longitudinal command (mirrors the steering
        delay in SimulationState.push_delay)."""
        buf = self._throttle_buffer
        head = self._throttle_head
        n = buf.shape[1]
        delayed = np.empty_like(values)
        for i in range(values.shape[0]):
            idx = head[i]
            delayed[i] = buf[i, idx]
            buf[i, idx] = values[i]
            head[i] = (idx + 1) % n
        return delayed

    def _compute_all_vertices(self) -> list:
        """Bounding-box corner vertices for every agent from their current poses.

        Independent of the LiDAR scan, so BOUNDING_BOX collision checking works
        even when the LiDAR is disabled.
        """
        verts = []
        for i in range(self.num_agents):
            cp = self._collision_pose_from_base(self.state.poses[i])
            verts.append(get_vertices(
                np.array([cp[0], cp[1], cp[2]], dtype=np.float64),
                self.vehicle_params.length,
                self.vehicle_params.width,
            ))
        return verts

    def _update_scans(self, *, flag_collisions: bool = True) -> None:
        # Precompute collision vertices once (reused by the ray_cast loop and _update_agent_collisions)
        self._all_vertices = self._compute_all_vertices()
        all_vertices = self._all_vertices
        # flag_collisions=False fills scans without adjudicating: a spawn is not a
        # crash. Enumerated so SEGMENT_CONTACT, which owns walls, is not halted here.
        wall_collision_enabled = flag_collisions and self.collision_check_mode in (
            CollisionCheckMode.LIDAR_SCAN,
            CollisionCheckMode.BOUNDING_BOX,
        )
        spawn_sweep = not flag_collisions and getattr(self, "_spawned_in_contact", None) is not None

        for agent_idx, simulator in enumerate(self.scan_sims):
            pose = self.state.poses[agent_idx]
            scan_pose = self._lidar_pose_from_base(pose)

            # Get noise-free scan for collision detection
            scan_clean = simulator.scan(scan_pose, rng=None)
            cache = self.scan_cache[agent_idx]

            # The spawn sweep records contact instead of adjudicating it, so
            # reset() can warn about a pose that starts inside the margin. This
            # branch only runs at reset, never on the stepping path.
            if spawn_sweep and check_collision(
                scan_clean, cache.side_distances, self.collision_margin
            ):
                self._spawned_in_contact.append(agent_idx)

            # Wall collision: contact check on the wall-only scan (skipped when collisions disabled)
            if wall_collision_enabled and check_collision(
                scan_clean,
                cache.side_distances,
                self.collision_margin,
            ):
                self._halt_on_collision(agent_idx)
                # The halt rewound the pose, so `scan_clean` describes a place
                # the car no longer is. Re-trace from the restored pose: this is
                # both what gets published and what the opponent ray_cast below
                # shortens, and it costs one extra sweep on contact steps only.
                scan_pose = self._lidar_pose_from_base(self.state.poses[agent_idx])
                scan_clean = simulator.scan(scan_pose, rng=None)
            elif wall_collision_enabled:
                # Only clear the flag this branch owns: SEGMENT_CONTACT has already
                # written it, and clearing here would erase it.
                self.state.collisions[agent_idx] = 0.0

            # Ray cast against other agents (also noise-free)
            origin = scan_pose.astype(np.float64)
            adjusted_scan = scan_clean
            for opp_idx in range(self.num_agents):
                if opp_idx == agent_idx:
                    continue
                adjusted_scan = ray_cast(origin, adjusted_scan, cache.angles, all_vertices[opp_idx])
            self._adjusted_scans[agent_idx] = adjusted_scan

            # Sensor noise, applied to the OBSERVED scan only (collisions use the
            # clean scan): Gaussian range noise + per-beam systematic bias, then
            # random per-beam dropout (no-return -> max_range).
            rng = self.scan_rngs[agent_idx]
            lidar_cfg = self.config.lidar_config
            noisy_scan = adjusted_scan + rng.normal(0.0, simulator.std_dev, size=simulator.num_beams)
            if lidar_cfg.range_bias_std > 0.0:
                noisy_scan = noisy_scan + self.scan_bias[agent_idx]
            noisy_scan = np.clip(noisy_scan, simulator.min_range, simulator.max_range)
            if lidar_cfg.dropout_prob > 0.0:
                dropped = rng.random(simulator.num_beams) < lidar_cfg.dropout_prob
                noisy_scan[dropped] = simulator.max_range
            self.state.scans[agent_idx] = noisy_scan.astype(np.float32)

    def _build_contact(self) -> None:
        """Compile the wall-contact kernels for the current track and body."""
        if self.collision_check_mode is not CollisionCheckMode.SEGMENT_CONTACT:
            self.contact = None
            return
        if self.track is None:
            raise ValueError("SEGMENT_CONTACT needs a track to extract walls from")
        from .contact.adapter import build

        self.contact = build(
            self.track,
            self.vehicle_params,
            self.config.contact_config,
            self.time_step,
            self.config.domain_randomization_config,
        )

    def _world_velocity(self, state):
        """(vx, vy, omega) in world axes for the model's native state."""
        yaw = float(state[4])
        if self.model is DynamicModel.ST:
            speed, beta, omega = float(state[3]), float(state[6]), float(state[5])
            course = yaw + beta
            return np.array([speed * math.cos(course), speed * math.sin(course)]), omega
        speed = float(state[3])
        wheelbase = float(self.vehicle_params.lf) + float(self.vehicle_params.lr)
        omega = speed * math.tan(float(state[2])) / wheelbase
        return np.array([speed * math.cos(yaw), speed * math.sin(yaw)]), omega

    def _apply_contact(self, state, velocity, omega, correction):
        """Write a corrected world velocity back into the model's native state.

        ST carries slip and yaw rate, so the projection is exact. KS carries neither:
        it can only absorb the longitudinal component at the rear axle, and any
        lateral or angular impulse is discarded.
        """
        state[0] += correction[0]
        state[1] += correction[1]
        yaw = float(state[4])
        if self.model is DynamicModel.ST:
            speed = float(np.hypot(velocity[0], velocity[1]))
            course = math.atan2(velocity[1], velocity[0])
            beta = (course - yaw + math.pi) % (2 * math.pi) - math.pi
            # The tyre model is linear in beta, so the wrong branch of this
            # two-fold ambiguity produces enormous restoring moments.
            if abs(beta) > math.pi / 2:
                speed = -speed
                beta = (beta - math.copysign(math.pi, beta) + math.pi) % (2 * math.pi) - math.pi
            state[3] = speed
            state[5] = omega
            state[6] = beta
            return state
        # Transporting to the rear axle adds omega x (-lr * heading), which is
        # purely lateral, so the longitudinal projection is the same at either point.
        state[3] = velocity[0] * math.cos(yaw) + velocity[1] * math.sin(yaw)
        return state

    def _resolve_contacts(self) -> None:
        """Resolve wall contact for every agent and refresh the SoA mirrors."""
        lr = float(self.vehicle_params.lr)
        rear_axle = self.model.pose_reference is not PoseReference.COG
        for agent_idx in range(self.num_agents):
            state = self.state.state[agent_idx].astype(np.float64)
            pose = self.state.poses[agent_idx]
            verts = get_vertices(
                np.asarray(self._collision_pose_from_base(pose), dtype=np.float64),
                self.vehicle_params.length,
                self.vehicle_params.width,
            )
            yaw = float(state[4])
            centre = np.array([float(state[0]), float(state[1])])
            if rear_axle:
                centre = centre + lr * np.array([math.cos(yaw), math.sin(yaw)])

            velocity, omega = self._world_velocity(state)
            velocity, omega, correction, hit = self.contact(
                verts, centre, velocity, omega,
                float(self.vehicle_params.m), float(self.vehicle_params.I),
            )
            self.state.collisions[agent_idx] = 1.0 if hit else 0.0
            if not hit:
                continue
            state = self._apply_contact(state, velocity, omega, correction)
            state[4] = (state[4] + np.pi) % (2 * np.pi) - np.pi
            self.state.state[agent_idx] = state.astype(np.float32)
            self.state.standard_state[agent_idx] = self._standardize(state)
            self.state.poses[agent_idx] = np.array(
                [state[0], state[1], state[4]], dtype=np.float32
            )

    def _update_agent_collisions(self) -> None:
        """Detect agent-vs-agent collisions using the configured mode."""
        mode = self.collision_check_mode
        if mode is CollisionCheckMode.NONE:
            return
        if mode is CollisionCheckMode.LIDAR_SCAN:
            # Agent-vs-agent via TTC on opponent-shortened scans. This needs the
            # LiDAR scan; with the LiDAR disabled there is no signal, so skip
            # (rather than indexing the empty scan cache).
            if not self.scan_enabled:
                return
            for agent_idx in range(self.num_agents):
                if self.state.collisions[agent_idx]:
                    continue  # already in wall collision
                cache = self.scan_cache[agent_idx]
                if check_collision(
                    self._adjusted_scans[agent_idx],
                    cache.side_distances,
                    self.collision_margin,
                ):
                    self._halt_on_collision(agent_idx)
        elif mode in (CollisionCheckMode.BOUNDING_BOX, CollisionCheckMode.SEGMENT_CONTACT):
            # Agent-vs-agent via GJK on pose-derived vertices, so it works with the
            # LiDAR off. SEGMENT_CONTACT borrows it; walls it handles itself.
            vertices = self._all_vertices if self.scan_enabled else self._compute_all_vertices()
            for agent_idx in range(self.num_agents):
                self.agent_vertices[agent_idx] = vertices[agent_idx]
            collisions, _ = collision_multiple(self.agent_vertices)
            # in place: rebinding self.state.collisions would leave any handle
            # captured from sim.state.collisions stale (LIDAR_SCAN writes in place).
            self.state.collisions[:] = np.maximum(self.state.collisions, collisions.astype(np.float32))
        else:
            raise ValueError(f"unhandled collision check mode: {mode!r}")
