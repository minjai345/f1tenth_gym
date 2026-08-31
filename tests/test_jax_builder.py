"""Host ``EnvConfig`` conversion contracts for the functional JAX core."""

import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from f1tenth_gym.envs.action import (
    LongitudinalActionType,
    SteerActionType,
)
from f1tenth_gym.envs.collision_models import CollisionCheckMode
from f1tenth_gym.envs.contact import ContactConfig
from f1tenth_gym.envs.dynamic_models import (
    DynamicModel,
    F1TENTH_VEHICLE_PARAMETERS,
)
from f1tenth_gym.envs.env_config import (
    ControlConfig,
    DomainRandomizationConfig,
    EnvConfig,
    LoopCounterMode,
    ResetConfig as HostResetConfig,
    RewardConfig,
    RewardMode,
    SimulationConfig,
    TerminationConfig,
)
from f1tenth_gym.envs.integrators import IntegratorType
from f1tenth_gym.envs.lidar import LiDARConfig
from f1tenth_gym.envs.reset import ReferenceLine, ResetStrategy
from f1tenth_gym.envs.termination import AgentTerminationMode
from f1tenth_gym.envs.track import Track
from f1tenth_gym.envs.track.budget import widest_query_half_extent
from f1tenth_gym.jax.builder import (
    build_core,
    build_core_config,
    build_core_params,
    build_core_tables,
)
from f1tenth_gym.jax.controls import (
    LongitudinalControlMode,
    SteeringControlMode,
)
from f1tenth_gym.jax.dynamics import kinematic_single_track, single_track
from f1tenth_gym.jax.environment import reset_core, step_core
from f1tenth_gym.jax.episode import BuiltinRewardMode, TerminationMode
from f1tenth_gym.jax.integrators import euler_step, rk4_step


VEHICLE = F1TENTH_VEHICLE_PARAMETERS


def circle_track(count: int = 64, radius: float = 5.0) -> Track:
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return Track.from_refline(
        x=radius * np.cos(theta),
        y=radius * np.sin(theta),
        velx=np.full(count, 4.0),
    )


def lightweight_config(**changes) -> EnvConfig:
    """Return an offline config that does not preprocess unused geometry."""
    defaults = {
        "collision_check": CollisionCheckMode.NONE,
        "lidar_config": LiDARConfig(enabled=False, num_beams=7),
        "render_enabled": False,
    }
    defaults.update(changes)
    return EnvConfig(**defaults)


class TestCoreConfigMapping(unittest.TestCase):
    def test_default_topology_maps_without_loading_a_track(self):
        config = build_core_config(EnvConfig(render_enabled=False))

        self.assertEqual(config.dynamics.state_dim, 7)
        self.assertIs(config.dynamics.dynamics_fn, single_track)
        self.assertIs(config.dynamics.integrator_fn, rk4_step)
        self.assertEqual(config.dynamics.num_substeps, 1)
        self.assertIs(
            config.dynamics.longitudinal_mode,
            LongitudinalControlMode.TARGET_SPEED,
        )
        self.assertIs(
            config.dynamics.steering_mode,
            SteeringControlMode.TARGET_ANGLE,
        )
        self.assertTrue(config.dynamics.derive_steer_kp)
        self.assertEqual(config.scan.num_beams, 1080)
        self.assertTrue(config.scan_enabled)
        self.assertTrue(config.contact_enabled)
        self.assertTrue(config.frenet_enabled)
        self.assertEqual(config.wall_contact.solver_iterations, 64)
        self.assertEqual(config.pair_contact.solver_iterations, 64)
        self.assertIs(config.episode.termination_mode, TerminationMode.EGO)
        self.assertIs(config.episode.reward_mode, BuiltinRewardMode.SURVIVAL)

    def test_nondefault_static_and_traced_values_map_exactly(self):
        track = circle_track()
        host = lightweight_config(
            num_agents=2,
            ego_index=1,
            control_config=ControlConfig(
                longitudinal_mode=LongitudinalActionType.ACCL,
                steering_mode=SteerActionType.STEERING_SPEED,
                steer_delay_steps=2,
                throttle_delay_steps=3,
                steer_noise_std=0.12,
                accl_noise_std=0.34,
                steer_kp=2.75,
            ),
            simulation_config=SimulationConfig(
                timestep=0.03,
                integrator_timestep=0.01,
                integrator=IntegratorType.EULER,
                dynamics_model=DynamicModel.KS,
                max_laps=None,
                count_partial_first_lap=False,
            ),
            reset_config=HostResetConfig(
                strategy=ResetStrategy.RL_GRID_RANDOM,
                min_dist=0.7,
                max_dist=1.8,
                shuffle=False,
                move_laterally=True,
                reference_line=ReferenceLine.CENTERLINE,
                start_width=0.8,
            ),
            contact_config=ContactConfig(
                restitution=0.2,
                friction=0.7,
                restitution_threshold=0.4,
                baumgarte=0.3,
                slop=0.004,
                solver_iterations=9,
            ),
            termination_config=TerminationConfig(
                max_episode_steps=11,
                terminate_on_collision=False,
                agent_mode=AgentTerminationMode.ALL,
            ),
            reward_config=RewardConfig(
                mode=RewardMode.PROGRESS,
                progress_weight=2.0,
                velocity_weight=0.5,
                timestep_weight=0.25,
                collision_penalty=3.0,
            ),
        )

        config = build_core_config(host)
        tables = build_core_tables(host, track)
        params = build_core_params(host, tables.track)

        self.assertEqual(config.dynamics.state_dim, 5)
        self.assertIs(config.dynamics.dynamics_fn, kinematic_single_track)
        self.assertIs(config.dynamics.integrator_fn, euler_step)
        self.assertEqual(config.dynamics.num_substeps, 3)
        self.assertIs(
            config.dynamics.longitudinal_mode,
            LongitudinalControlMode.ACCELERATION,
        )
        self.assertIs(
            config.dynamics.steering_mode,
            SteeringControlMode.STEERING_RATE,
        )
        self.assertEqual(config.dynamics.steer_delay_steps, 2)
        self.assertEqual(config.dynamics.throttle_delay_steps, 3)
        self.assertFalse(config.dynamics.derive_steer_kp)
        self.assertEqual(config.scan.num_beams, 1)
        self.assertFalse(config.scan_enabled)
        self.assertFalse(config.contact_enabled)
        self.assertTrue(config.reset.move_laterally)
        self.assertFalse(config.reset.shuffle)
        self.assertEqual(config.episode.ego_index, 1)
        self.assertFalse(config.episode.count_partial_first_lap)
        self.assertIs(config.episode.termination_mode, TerminationMode.ALL)
        self.assertIs(config.episode.reward_mode, BuiltinRewardMode.PROGRESS)
        self.assertEqual(config.wall_contact.solver_iterations, 9)
        self.assertEqual(config.pair_contact.solver_iterations, 9)

        expected_waypoints = np.stack(
            (track.centerline.xs, track.centerline.ys), axis=1
        ).astype(np.float32)
        np.testing.assert_array_equal(tables.reset.waypoints, expected_waypoints)
        self.assertLess(
            tables.reset.start_indices.shape[0], tables.reset.waypoints.shape[0]
        )
        np.testing.assert_array_equal(tables.pairs.indices, [[0, 0]])
        np.testing.assert_array_equal(tables.pairs.mask, [False])
        self.assertFalse(bool(np.asarray(tables.track.contact_tiles.mask).any()))
        self.assertFalse(bool(np.asarray(tables.track.ray_tiles.mask).any()))
        self.assertEqual(float(tables.track.contact_tiles.reach), 0.0)
        self.assertEqual(float(tables.track.ray_tiles.reach), 0.0)

        self.assertAlmostEqual(float(params.transition.timestep), 0.03)
        self.assertAlmostEqual(float(params.transition.steer_kp), 2.75)
        self.assertAlmostEqual(float(params.transition.steer_noise_std), 0.12)
        self.assertAlmostEqual(float(params.transition.accel_noise_std), 0.34)
        self.assertAlmostEqual(float(params.contact.restitution), 0.2)
        self.assertAlmostEqual(float(params.contact.friction), 0.7)
        self.assertAlmostEqual(float(params.contact.slop), 0.004)
        self.assertFalse(bool(params.bookkeeping.terminate_on_collision))
        self.assertFalse(bool(params.bookkeeping.lap_limit_enabled))
        self.assertTrue(bool(params.bookkeeping.step_limit_enabled))
        self.assertEqual(int(params.bookkeeping.max_episode_steps), 11)
        self.assertAlmostEqual(float(params.bookkeeping.progress_weight), 2.0)
        self.assertAlmostEqual(float(params.bookkeeping.velocity_weight), 0.5)
        self.assertAlmostEqual(float(params.bookkeeping.timestep_weight), 0.25)
        self.assertAlmostEqual(float(params.bookkeeping.collision_penalty), 3.0)

    def test_disabled_geometry_skips_both_acceleration_indexes(self):
        host = lightweight_config()
        with (
            mock.patch(
                "f1tenth_gym.jax.preprocess.build_contact_tiles",
                side_effect=AssertionError("contact index should be skipped"),
            ),
            mock.patch(
                "f1tenth_gym.jax.preprocess.build_ray_tiles",
                side_effect=AssertionError("ray index should be skipped"),
            ),
            mock.patch(
                "f1tenth_gym.jax.preprocess.wall_segments",
                side_effect=AssertionError("wall extraction should be skipped"),
            ),
        ):
            tables = build_core_tables(host, circle_track())
        self.assertFalse(bool(np.asarray(tables.track.contact_tiles.mask).any()))
        self.assertFalse(bool(np.asarray(tables.track.ray_tiles.mask).any()))
        self.assertEqual(tables.pairs.indices.shape, (1, 2))
        self.assertFalse(bool(np.asarray(tables.pairs.mask).any()))


class TestUnsupportedHostSurface(unittest.TestCase):
    def test_invalid_substep_ratio_is_rejected(self):
        host = lightweight_config(
            simulation_config=SimulationConfig(
                timestep=0.01,
                integrator_timestep=0.006,
            )
        )
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            build_core_config(host)

    def test_map_reset_and_winding_laps_are_explicitly_rejected(self):
        map_reset = lightweight_config(
            reset_config=HostResetConfig(
                strategy=ResetStrategy.MAP_RANDOM_STATIC
            )
        )
        with self.assertRaisesRegex(ValueError, "MAP_RANDOM_STATIC"):
            build_core_config(map_reset)

        winding = lightweight_config(
            simulation_config=SimulationConfig(
                loop_counter=LoopCounterMode.WINDING_ANGLE,
                compute_frenet_frame=False,
            )
        )
        with self.assertRaisesRegex(ValueError, "FRENET_BASED"):
            build_core_config(winding)

    def test_custom_reward_and_nondefault_wall_tolerance_are_rejected(self):
        custom = lightweight_config(
            reward_config=RewardConfig(
                mode=RewardMode.CUSTOM,
                reward_fn=lambda *_args: 0.0,
            )
        )
        with self.assertRaisesRegex(ValueError, "adapter-only"):
            build_core_config(custom)

        tolerance = EnvConfig(
            contact_config=ContactConfig(wall_tolerance_px=0.5),
            render_enabled=False,
        )
        with self.assertRaisesRegex(ValueError, "wall_tolerance_px"):
            build_core_config(tolerance)

    def test_active_contact_and_scan_must_request_one_device(self):
        host = EnvConfig(
            contact_config=ContactConfig(device="gpu"),
            lidar_config=LiDARConfig(
                num_beams=3,
                range_max=2.0,
                scan_device="cpu",
            ),
            render_enabled=False,
        )
        with self.assertRaisesRegex(ValueError, "different devices"):
            build_core(host, circle_track())

    def test_public_builders_require_an_env_config(self):
        with self.assertRaisesRegex(TypeError, "EnvConfig"):
            build_core_config(object())


class TestDomainRandomizationMapping(unittest.TestCase):
    def setUp(self):
        self.track = circle_track()
        low = VEHICLE.with_updates(m=3.0)
        high = VEHICLE.with_updates(m=4.0)
        self.host = lightweight_config(
            domain_randomization_config=DomainRandomizationConfig(
                enabled=True,
                low=low,
                high=high,
            )
        )
        self.tables = build_core_tables(self.host, self.track)

    def test_full_builder_requires_an_explicit_episode_draw(self):
        with self.assertRaisesRegex(ValueError, "explicit sampled"):
            build_core(self.host, self.track)

    def test_sampled_draw_is_copied_and_validated_against_bounds(self):
        sampled = VEHICLE.with_updates(m=3.5)
        params = build_core_params(
            self.host,
            self.tables.track,
            vehicle_params=sampled,
        )
        self.assertAlmostEqual(float(params.transition.dynamics.m), 3.5)

        bundle = build_core(self.host, self.track, vehicle_params=sampled)
        self.assertIs(bundle.env_config, self.host)
        self.assertIs(bundle.track, self.track)
        self.assertAlmostEqual(float(bundle.params.transition.dynamics.m), 3.5)
        for leaf in jax.tree.leaves((bundle.tables, bundle.params)):
            self.assertEqual(leaf.device, bundle.device)

        outside = VEHICLE.with_updates(m=4.1)
        with self.assertRaisesRegex(ValueError, "outside DR bounds"):
            build_core_params(
                self.host,
                self.tables.track,
                vehicle_params=outside,
            )

        fixed_outside = sampled.with_updates(width=VEHICLE.width + 0.01)
        with self.assertRaisesRegex(ValueError, "vehicle_params.width"):
            build_core_params(
                self.host,
                self.tables.track,
                vehicle_params=fixed_outside,
            )

    def test_contact_table_uses_the_widest_randomized_body(self):
        low = VEHICLE.with_updates(length=0.60, width=0.30)
        high = VEHICLE.with_updates(length=0.80, width=0.50)
        host = lightweight_config(
            collision_check=CollisionCheckMode.SEGMENT_CONTACT,
            domain_randomization_config=DomainRandomizationConfig(
                enabled=True,
                low=low,
                high=high,
            ),
        )
        tables = build_core_tables(host, self.track)
        self.assertAlmostEqual(
            float(tables.track.contact_tiles.reach),
            widest_query_half_extent(VEHICLE, host.domain_randomization_config),
            places=6,
        )

    def test_enabled_but_constant_bounds_need_no_sample(self):
        host = lightweight_config(
            domain_randomization_config=DomainRandomizationConfig(
                enabled=True,
                low=VEHICLE,
                high=VEHICLE,
            )
        )
        bundle = build_core(host, self.track)
        self.assertAlmostEqual(float(bundle.params.transition.dynamics.m), VEHICLE.m)

    def test_production_parameter_dtypes_do_not_follow_python_values(self):
        host = lightweight_config(params=VEHICLE.with_updates(m=4))
        bundle = build_core(host, self.track)
        params = bundle.params

        for leaf in jax.tree.leaves(
            (
                params.transition,
                params.body,
                params.contact,
                params.scan,
            )
        ):
            self.assertEqual(leaf.dtype, jnp.dtype(jnp.float32))
        self.assertEqual(
            params.bookkeeping.max_laps.dtype,
            jnp.dtype(jnp.int32),
        )
        self.assertEqual(
            params.bookkeeping.terminate_on_collision.dtype,
            jnp.dtype(jnp.bool_),
        )


class TestBuiltCoreSmoke(unittest.TestCase):
    def test_reset_and_jitted_step_run_from_one_host_bundle(self):
        host = lightweight_config(
            lidar_config=LiDARConfig(
                num_beams=5,
                range_max=3.0,
                noise_std=0.0,
            ),
            simulation_config=SimulationConfig(
                dynamics_model=DynamicModel.KS,
                max_laps=None,
            ),
            control_config=ControlConfig(
                longitudinal_mode=LongitudinalActionType.ACCL,
                steering_mode=SteerActionType.STEERING_SPEED,
            ),
            reset_config=HostResetConfig(
                strategy=ResetStrategy.RL_RANDOM_STATIC
            ),
        )
        bundle = build_core(host, circle_track())
        reset = jax.jit(reset_core, static_argnums=2)
        step = jax.jit(step_core, static_argnums=4)

        observation, state = reset(
            jax.random.key(1),
            bundle.tables,
            bundle.config,
            bundle.params,
        )
        self.assertEqual(observation.state.shape, (1, 5))
        self.assertEqual(observation.scans.shape, (1, 5))
        observation, state, rewards, events, metrics = step(
            jax.random.key(2),
            state,
            jnp.asarray([[0.0, 1.0]], dtype=jnp.float32),
            bundle.tables,
            bundle.config,
            bundle.params,
        )
        self.assertEqual(observation.standard_state.shape, (1, 7))
        self.assertAlmostEqual(float(state.dynamics.sim_time), 0.01, places=7)
        np.testing.assert_allclose(rewards, [0.01], atol=1.0e-7)
        np.testing.assert_array_equal(events.collisions, [False])
        self.assertFalse(bool(metrics.status.terminated))
        self.assertFalse(bool(metrics.status.truncated))


if __name__ == "__main__":
    unittest.main()
