"""Public ``JaxSimulator`` construction and conversion contracts."""

import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from f1tenth_gym.envs.batching import reset_batch
from f1tenth_gym.envs.action import (
    LongitudinalActionType,
    SteerActionType,
)
from f1tenth_gym.envs.action_jax import (
    LongitudinalControlMode,
    SteeringControlMode,
)
from f1tenth_gym.envs.collision_models import CollisionCheckMode
from f1tenth_gym.envs.contact import ContactConfig
from f1tenth_gym.envs.dynamic_models import (
    PARAMETER_ORDER,
    DynamicModel,
    F1TENTH_VEHICLE_PARAMETERS,
)
from f1tenth_gym.envs.dynamic_models.jax import (
    kinematic_single_track,
    single_track,
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
from f1tenth_gym.envs.episode import BuiltinRewardMode, TerminationMode
from f1tenth_gym.envs.integrators import IntegratorType
from f1tenth_gym.envs.integrators_jax import euler_step, rk4_step
from f1tenth_gym.envs.jax_simulator import JaxSimulator
from f1tenth_gym.envs.lidar import LiDARConfig
from f1tenth_gym.envs.reset import ReferenceLine, ResetStrategy
from f1tenth_gym.envs.termination import AgentTerminationMode
from f1tenth_gym.envs.track import Track
from f1tenth_gym.envs.track.budget import widest_query_half_extent


VEHICLE = F1TENTH_VEHICLE_PARAMETERS


def circle_track(count: int = 64, radius: float = 5.0) -> Track:
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return Track.from_refline(
        x=radius * np.cos(theta),
        y=radius * np.sin(theta),
        velx=np.full(count, 4.0),
    )


def lightweight_config(**changes) -> EnvConfig:
    """Return an offline config that skips unused geometric indexes."""
    defaults = {
        "collision_check": CollisionCheckMode.NONE,
        "lidar_config": LiDARConfig(enabled=False, num_beams=7),
        "render_enabled": False,
    }
    defaults.update(changes)
    return EnvConfig(**defaults)


class TestJaxSimulatorMapping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.track = circle_track()

    def test_constructor_exposes_one_complete_device_simulator(self):
        host = lightweight_config()
        simulator = JaxSimulator(host, self.track, device="cpu")

        self.assertIs(simulator.env_config, host)
        self.assertIs(simulator.track, self.track)
        self.assertEqual(simulator.device.platform, "cpu")
        self.assertEqual(simulator.config.dynamics.state_dim, 7)
        self.assertIs(simulator.config.dynamics.dynamics_fn, single_track)
        self.assertIs(simulator.config.dynamics.integrator_fn, rk4_step)
        self.assertEqual(simulator.config.dynamics.num_substeps, 1)
        self.assertIs(
            simulator.config.dynamics.longitudinal_mode,
            LongitudinalControlMode.TARGET_SPEED,
        )
        self.assertIs(
            simulator.config.dynamics.steering_mode,
            SteeringControlMode.TARGET_ANGLE,
        )
        self.assertTrue(simulator.config.dynamics.derive_steer_kp)
        self.assertFalse(simulator.config.scan_enabled)
        self.assertFalse(simulator.config.contact_enabled)
        self.assertTrue(simulator.config.frenet_enabled)
        self.assertIs(
            simulator.config.episode.termination_mode,
            TerminationMode.EGO,
        )
        self.assertIs(
            simulator.config.episode.reward_mode,
            BuiltinRewardMode.SURVIVAL,
        )
        for leaf in jax.tree.leaves(
            (simulator.tables, simulator.params, simulator.randomization)
        ):
            self.assertEqual(leaf.device, simulator.device)

    def test_nondefault_topology_and_values_map_exactly(self):
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
        simulator = JaxSimulator(host, self.track)
        config = simulator.config
        tables = simulator.tables
        params = simulator.params

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
            (self.track.centerline.xs, self.track.centerline.ys), axis=1
        ).astype(np.float32)
        np.testing.assert_array_equal(tables.reset.waypoints, expected_waypoints)
        self.assertLess(
            tables.reset.start_indices.shape[0],
            tables.reset.waypoints.shape[0],
        )
        np.testing.assert_array_equal(tables.pairs.indices, [[0, 0]])
        np.testing.assert_array_equal(tables.pairs.mask, [False])
        self.assertFalse(bool(np.asarray(tables.track.contact_tiles.mask).any()))
        self.assertFalse(bool(np.asarray(tables.track.ray_tiles.mask).any()))

        self.assertAlmostEqual(float(params.dynamics.timestep), 0.03)
        self.assertAlmostEqual(float(params.dynamics.steer_kp), 2.75)
        self.assertAlmostEqual(float(params.dynamics.steer_noise_std), 0.12)
        self.assertAlmostEqual(float(params.dynamics.accel_noise_std), 0.34)
        self.assertAlmostEqual(float(params.contact.restitution), 0.2)
        self.assertAlmostEqual(float(params.contact.friction), 0.7)
        self.assertAlmostEqual(float(params.contact.slop), 0.004)
        self.assertFalse(bool(params.episode.terminate_on_collision))
        self.assertFalse(bool(params.episode.lap_limit_enabled))
        self.assertTrue(bool(params.episode.step_limit_enabled))
        self.assertEqual(int(params.episode.max_episode_steps), 11)
        self.assertAlmostEqual(float(params.episode.progress_weight), 2.0)
        self.assertAlmostEqual(float(params.episode.velocity_weight), 0.5)
        self.assertAlmostEqual(float(params.episode.timestep_weight), 0.25)
        self.assertAlmostEqual(float(params.episode.collision_penalty), 3.0)

    def test_disabled_geometry_skips_acceleration_indexes(self):
        with (
            mock.patch(
                "f1tenth_gym.envs.track.preprocessing.build_contact_tiles",
                side_effect=AssertionError("contact index should be skipped"),
            ),
            mock.patch(
                "f1tenth_gym.envs.track.preprocessing.build_ray_tiles",
                side_effect=AssertionError("ray index should be skipped"),
            ),
            mock.patch(
                "f1tenth_gym.envs.track.preprocessing.wall_segments",
                side_effect=AssertionError("wall extraction should be skipped"),
            ),
        ):
            simulator = JaxSimulator(lightweight_config(), self.track)
        self.assertFalse(
            bool(np.asarray(simulator.tables.track.contact_tiles.mask).any())
        )
        self.assertFalse(
            bool(np.asarray(simulator.tables.track.ray_tiles.mask).any())
        )
        self.assertEqual(simulator.tables.pairs.indices.shape, (1, 2))
        self.assertFalse(bool(np.asarray(simulator.tables.pairs.mask).any()))


class TestUnsupportedHostSurface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.track = circle_track()

    def test_invalid_substep_ratio_is_rejected(self):
        host = lightweight_config(
            simulation_config=SimulationConfig(
                timestep=0.01,
                integrator_timestep=0.006,
            )
        )
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            JaxSimulator(host, self.track)

    def test_map_reset_and_winding_laps_are_rejected(self):
        map_reset = lightweight_config(
            reset_config=HostResetConfig(
                strategy=ResetStrategy.MAP_RANDOM_STATIC
            )
        )
        with self.assertRaisesRegex(ValueError, "MAP_RANDOM_STATIC"):
            JaxSimulator(map_reset, self.track)

        winding = lightweight_config(
            simulation_config=SimulationConfig(
                loop_counter=LoopCounterMode.WINDING_ANGLE,
                compute_frenet_frame=False,
            )
        )
        with self.assertRaisesRegex(ValueError, "FRENET_BASED"):
            JaxSimulator(winding, self.track)

    def test_custom_reward_and_wall_tolerance_are_rejected(self):
        custom = lightweight_config(
            reward_config=RewardConfig(
                mode=RewardMode.CUSTOM,
                reward_fn=lambda *_args: 0.0,
            )
        )
        with self.assertRaisesRegex(ValueError, "adapter-only"):
            JaxSimulator(custom, self.track)

        tolerance = EnvConfig(
            contact_config=ContactConfig(wall_tolerance_px=0.5),
            render_enabled=False,
        )
        with self.assertRaisesRegex(ValueError, "wall_tolerance_px"):
            JaxSimulator(tolerance, self.track)

    def test_active_contact_and_scan_must_share_one_device(self):
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
            JaxSimulator(host, self.track)

        overridden = JaxSimulator(host, self.track, device="cpu")
        self.assertEqual(overridden.device.platform, "cpu")

    def test_device_accepts_platform_name_or_jax_device(self):
        host = lightweight_config()
        cpu = jax.devices("cpu")[0]

        by_name = JaxSimulator(host, self.track, device="cpu")
        by_object = JaxSimulator(host, self.track, device=cpu)

        self.assertEqual(by_name.device.platform, "cpu")
        self.assertIs(by_object.device, cpu)
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            JaxSimulator(host, self.track, device="not-a-jax-platform")
        with self.assertRaisesRegex(TypeError, "device"):
            JaxSimulator(host, self.track, device=object())

    def test_constructor_requires_host_types(self):
        with self.assertRaisesRegex(TypeError, "EnvConfig"):
            JaxSimulator(object(), self.track)
        with self.assertRaisesRegex(TypeError, "resolved Track"):
            JaxSimulator(lightweight_config(), object())


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
        self.simulator = JaxSimulator(self.host, self.track)

    def test_constructor_keeps_nominal_params_and_device_bounds(self):
        simulator = self.simulator
        self.assertAlmostEqual(
            float(simulator.params.dynamics.vehicle.m), VEHICLE.m
        )
        self.assertTrue(bool(simulator.randomization.enabled))
        self.assertAlmostEqual(float(simulator.randomization.low.m), 3.0)
        self.assertAlmostEqual(float(simulator.randomization.high.m), 4.0)

        nominal = np.asarray(simulator.randomization.nominal.as_array())
        low = np.asarray(simulator.randomization.low.as_array())
        high = np.asarray(simulator.randomization.high.as_array())
        expected_nominal = np.asarray(
            [getattr(VEHICLE, name) for name in PARAMETER_ORDER[:20]],
            dtype=np.float32,
        )
        expected_low = np.asarray(
            [
                getattr(self.host.domain_randomization_config.low, name)
                for name in PARAMETER_ORDER[:20]
            ],
            dtype=np.float32,
        )
        expected_high = np.asarray(
            [
                getattr(self.host.domain_randomization_config.high, name)
                for name in PARAMETER_ORDER[:20]
            ],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(nominal, expected_nominal)
        np.testing.assert_array_equal(low, expected_low)
        np.testing.assert_array_equal(high, expected_high)
        self.assertEqual(nominal.dtype, np.dtype(np.float32))
        self.assertTrue(np.isfinite(nominal).all())
        self.assertTrue(np.isfinite(low).all())
        self.assertTrue(np.isfinite(high).all())

    def test_params_for_vehicle_validates_and_places_one_episode_draw(self):
        sampled = VEHICLE.with_updates(m=3.5)
        params = self.simulator.params_for_vehicle(sampled)
        self.assertAlmostEqual(float(params.dynamics.vehicle.m), 3.5)
        for leaf in jax.tree.leaves(params):
            self.assertEqual(leaf.device, self.simulator.device)

        explicit = JaxSimulator(
            self.host,
            self.track,
            vehicle_params=sampled,
        )
        self.assertAlmostEqual(float(explicit.params.dynamics.vehicle.m), 3.5)
        self.assertAlmostEqual(float(explicit.randomization.nominal.m), 3.5)
        self.assertAlmostEqual(float(explicit.randomization.low.m), 3.0)
        self.assertAlmostEqual(float(explicit.randomization.high.m), 4.0)

        with self.assertRaisesRegex(TypeError, "VehicleParameters"):
            self.simulator.params_for_vehicle(object())
        with self.assertRaisesRegex(ValueError, "outside the runtime envelope"):
            self.simulator.params_for_vehicle(VEHICLE.with_updates(m=4.1))
        with self.assertRaisesRegex(ValueError, "vehicle_params.width"):
            self.simulator.params_for_vehicle(
                sampled.with_updates(width=VEHICLE.width + 0.01)
            )
        with self.assertRaisesRegex(ValueError, "vehicle_params.I.*finite"):
            self.simulator.params_for_vehicle(sampled.with_updates(I=np.nan))

    def test_explicit_vehicle_is_preserved_by_disabled_dr_batch_reset(self):
        explicit_vehicle = VEHICLE.with_updates(m=4.25)
        simulator = JaxSimulator(
            lightweight_config(),
            self.track,
            vehicle_params=explicit_vehicle,
            device="cpu",
        )
        keys = jax.random.split(jax.random.key(17), 2)

        _observation, batch_state = reset_batch(
            keys,
            simulator.tables,
            simulator.config,
            simulator.params,
            simulator.randomization,
        )

        self.assertAlmostEqual(float(simulator.randomization.nominal.m), 4.25)
        np.testing.assert_allclose(
            batch_state.params.dynamics.vehicle.m,
            np.full((2,), 4.25, dtype=np.float32),
        )

    def test_effective_vehicle_extends_the_varying_runtime_envelope(self):
        effective = VEHICLE.with_updates(m=4.25)
        simulator = JaxSimulator(
            self.host,
            self.track,
            vehicle_params=effective,
        )

        inside = simulator.params_for_vehicle(VEHICLE.with_updates(m=4.20))
        self.assertAlmostEqual(float(inside.dynamics.vehicle.m), 4.20, places=6)
        self.assertAlmostEqual(float(simulator.randomization.high.m), 4.0)
        with self.assertRaisesRegex(ValueError, "runtime envelope"):
            simulator.params_for_vehicle(VEHICLE.with_updates(m=4.26))

    def test_fixed_vehicle_changes_require_rebuilding_the_simulator(self):
        simulator = JaxSimulator(
            lightweight_config(
                collision_check=CollisionCheckMode.SEGMENT_CONTACT,
            ),
            self.track,
            vehicle_params=VEHICLE,
        )

        with self.assertRaisesRegex(
            ValueError,
            "fixed vehicle envelope.*new JaxSimulator",
        ):
            simulator.params_for_vehicle(
                VEHICLE.with_updates(width=10.0, length=10.0)
            )

    def test_active_bounds_must_be_finite_and_safe_for_every_draw(self):
        low = VEHICLE.with_updates(s_min=-0.2, s_max=0.1)
        high = VEHICLE.with_updates(s_min=0.2, s_max=0.3)
        crossing = lightweight_config(
            domain_randomization_config=DomainRandomizationConfig(
                enabled=True,
                low=low,
                high=high,
            )
        )
        with self.assertRaisesRegex(ValueError, "intervals.*s_min <= s_max"):
            JaxSimulator(crossing, self.track)

        nonfinite_low = VEHICLE.with_updates(mu=np.nan, m=3.0)
        nonfinite_high = VEHICLE.with_updates(mu=np.nan, m=4.0)
        nonfinite = lightweight_config(
            domain_randomization_config=DomainRandomizationConfig(
                enabled=True,
                low=nonfinite_low,
                high=nonfinite_high,
            )
        )
        with self.assertRaisesRegex(ValueError, "low.mu.*finite"):
            JaxSimulator(nonfinite, self.track)

    def test_contact_tables_cover_the_widest_randomized_body(self):
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
        simulator = JaxSimulator(host, self.track)
        self.assertAlmostEqual(
            float(simulator.tables.track.contact_tiles.reach),
            widest_query_half_extent(
                VEHICLE,
                host.domain_randomization_config,
            ),
            places=6,
        )

    def test_constant_bounds_collapse_to_nominal(self):
        host = lightweight_config(
            domain_randomization_config=DomainRandomizationConfig(
                enabled=True,
                low=VEHICLE,
                high=VEHICLE,
            )
        )
        simulator = JaxSimulator(host, self.track)
        self.assertFalse(bool(simulator.randomization.enabled))
        np.testing.assert_array_equal(
            simulator.randomization.low.as_array(),
            simulator.randomization.nominal.as_array(),
        )
        np.testing.assert_array_equal(
            simulator.randomization.high.as_array(),
            simulator.randomization.nominal.as_array(),
        )

    def test_production_dtypes_ignore_python_scalar_types(self):
        simulator = JaxSimulator(
            lightweight_config(params=VEHICLE.with_updates(m=4)),
            self.track,
        )
        params = simulator.params
        for leaf in jax.tree.leaves(
            (params.dynamics, params.body, params.contact, params.scan)
        ):
            self.assertEqual(leaf.dtype, jnp.dtype(jnp.float32))
        self.assertEqual(
            params.episode.max_laps.dtype,
            jnp.dtype(jnp.int32),
        )
        self.assertEqual(
            params.episode.terminate_on_collision.dtype,
            jnp.dtype(jnp.bool_),
        )


class TestJaxSimulatorSmoke(unittest.TestCase):
    def test_reset_and_jitted_step_use_the_public_simulator(self):
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
        simulator = JaxSimulator(host, circle_track())
        observation, state = simulator.reset(jax.random.key(1))
        self.assertEqual(observation.state.shape, (1, 5))
        self.assertEqual(observation.scans.shape, (1, 5))
        observation, state, rewards, events, metrics = simulator.step(
            jax.random.key(2),
            state,
            jnp.asarray([[0.0, 1.0]], dtype=jnp.float32),
        )
        self.assertEqual(observation.standard_state.shape, (1, 7))
        self.assertAlmostEqual(float(state.dynamics.sim_time), 0.01, places=7)
        np.testing.assert_allclose(rewards, [0.01], atol=1.0e-7)
        np.testing.assert_array_equal(events.collisions, [False])
        self.assertFalse(bool(metrics.status.terminated))
        self.assertFalse(bool(metrics.status.truncated))


if __name__ == "__main__":
    unittest.main()
