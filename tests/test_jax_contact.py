"""Differential gates for functional fixed-shape wall contact."""

import ast
from dataclasses import replace
import math
import pathlib
import unittest
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from f1tenth_gym.envs.action import LongitudinalActionType, SteerActionType
from f1tenth_gym.envs.action_jax import (
    LongitudinalControlMode,
    SteeringControlMode,
)
from f1tenth_gym.envs.collision_models import CollisionCheckMode
from f1tenth_gym.envs.contact import ContactParams
from f1tenth_gym.envs.contact.functional import (
    WallContactConfig,
    apply_contact_response,
    resolve_wall_contacts,
    world_velocity,
)
from f1tenth_gym.envs.contact.geometry import BodyParams
from f1tenth_gym.envs.dynamic_models import (
    DynamicModel,
    F1TENTH_VEHICLE_PARAMETERS,
)
from f1tenth_gym.envs.dynamic_models.jax import (
    DynamicsParams,
    kinematic_single_track,
    single_track,
)
from f1tenth_gym.envs.dynamic_models.jax_core import (
    DynamicsConfig,
    DynamicsRuntimeParams,
    make_dynamics_state,
    step_dynamics,
)
from f1tenth_gym.envs.env_config import ControlConfig, EnvConfig, SimulationConfig
from f1tenth_gym.envs.integrators import IntegratorType, rk4_integration
from f1tenth_gym.envs.integrators_jax import rk4_step
from f1tenth_gym.envs.lidar import LiDARConfig
from f1tenth_gym.envs.simulator import F110Simulator
from f1tenth_gym.envs.track import Track
from f1tenth_gym.envs.track.functional import TileTable, WallTable
from f1tenth_gym.envs.track.preprocessing import preprocess_track
from f1tenth_gym.envs.track.walls import wall_segments


DT = 0.01
VEHICLE = F1TENTH_VEHICLE_PARAMETERS
CONTACT = ContactParams(0.0, 0.6, 0.6, 0.4, 0.002)


def longest_wall(track):
    walls = wall_segments(track)
    index = int(np.argmax(walls.length))
    return (
        walls.n[index].astype(np.float64),
        0.5 * (walls.a[index] + walls.b[index]).astype(np.float64),
    )


def contact_state(track, state_dim, speed=3.0):
    normal, midpoint = longest_wall(track)
    centre = midpoint - normal * (VEHICLE.width / 2.0 - 0.03)
    yaw = math.atan2(-normal[1], -normal[0])
    state = np.zeros((1, state_dim), dtype=np.float32)
    state[0, :5] = [centre[0], centre[1], 0.0, speed, yaw]
    return state


def synthetic_contact_table(
    base,
    *,
    wall_a,
    wall_b,
    normal,
    indices,
    mask,
    origin=(-10.0, -10.0),
    tile_size=20.0,
):
    wall_a = np.asarray(wall_a, dtype=np.float32).reshape((-1, 2))
    wall_b = np.asarray(wall_b, dtype=np.float32).reshape((-1, 2))
    normal = np.asarray(normal, dtype=np.float32).reshape((-1, 2))
    walls = WallTable(
        a=jnp.asarray(wall_a),
        b=jnp.asarray(wall_b),
        normals=jnp.asarray(normal),
        adjacency=jnp.zeros((len(wall_a), 2), dtype=jnp.int32),
        adjacency_mask=jnp.zeros((len(wall_a), 2), dtype=jnp.bool_),
        lengths=jnp.linalg.norm(jnp.asarray(wall_b - wall_a), axis=1),
        mask=jnp.ones((len(wall_a),), dtype=jnp.bool_),
    )
    tiles = TileTable(
        indices=jnp.asarray(indices, dtype=jnp.int32),
        mask=jnp.asarray(mask, dtype=jnp.bool_),
        origin=jnp.asarray(origin, dtype=jnp.float32),
        tile_size=jnp.asarray(tile_size, dtype=jnp.float32),
        reach=jnp.asarray(2.0, dtype=jnp.float32),
    )
    return replace(base, walls=walls, contact_tiles=tiles)


class TestNativeRigidBodyMapping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dynamics = DynamicsParams.from_vehicle_parameters(VEHICLE)

    def test_world_velocity_matches_the_mutable_simulator(self):
        for model, state in (
            (DynamicModel.KS, np.array([1.0, 2.0, 0.23, -2.4, 0.7])),
            (DynamicModel.ST,
             np.array([1.0, 2.0, -0.1, 3.4, -1.2, 0.31, -0.18])),
        ):
            with self.subTest(model=model.name):
                fake = type("HostSimulator", (), {
                    "model": model,
                    "vehicle_params": VEHICLE,
                })()
                expected_v, expected_w = F110Simulator._world_velocity(fake, state)
                actual_v, actual_w = world_velocity(jnp.asarray(state), self.dynamics)
                np.testing.assert_allclose(actual_v, expected_v, atol=1.0e-6)
                self.assertAlmostEqual(float(actual_w), expected_w, places=6)

    def test_response_writeback_matches_ks_and_st_host_projection(self):
        velocity = np.array([-1.4, 0.8])
        correction = np.array([0.03, -0.02])
        for model, state in (
            (DynamicModel.KS, np.array([1.0, 2.0, 0.2, 3.0, -0.4])),
            (DynamicModel.ST,
             np.array([1.0, 2.0, 0.1, 3.0, 2.8, -0.2, 0.05])),
        ):
            with self.subTest(model=model.name):
                fake = type("HostSimulator", (), {
                    "model": model,
                    "vehicle_params": VEHICLE,
                })()
                expected = F110Simulator._apply_contact(
                    fake, state.copy(), velocity, 0.37, correction
                )
                actual = apply_contact_response(
                    jnp.asarray(state), jnp.asarray(velocity), jnp.asarray(0.37),
                    jnp.asarray(correction), self.dynamics,
                )
                np.testing.assert_allclose(actual, expected, atol=2.0e-7)

    def test_st_dead_stop_response_has_finite_forward_and_reverse_jacobians(self):
        state = jnp.asarray([1.0, 2.0, 0.1, 3.0, 0.4, -0.2, 0.05])
        correction = jnp.asarray([0.03, -0.02])

        def respond(velocity):
            return apply_contact_response(
                state, velocity, jnp.asarray(0.37), correction, self.dynamics
            )

        expected = np.asarray(state).copy()
        expected[:2] += np.asarray(correction)
        expected[3] = 0.0
        expected[5] = 0.37
        expected[6] = -0.4
        tiny = jnp.finfo(jnp.float32).tiny
        for velocity in (
            jnp.zeros((2,), dtype=jnp.float32),
            jnp.asarray((tiny, tiny), dtype=jnp.float32),
        ):
            with self.subTest(velocity=np.asarray(velocity)):
                actual = respond(velocity)
                np.testing.assert_allclose(actual, expected, atol=1.0e-7)

                forward = jax.jacfwd(respond)(velocity)
                reverse = jax.jacrev(respond)(velocity)
                self.assertTrue(bool(jnp.all(jnp.isfinite(forward))))
                self.assertTrue(bool(jnp.all(jnp.isfinite(reverse))))
                np.testing.assert_allclose(forward, reverse, atol=1.0e-7)

    def test_st_nonzero_response_preserves_speed_and_course_jacobian(self):
        state = jnp.asarray([1.0, 2.0, 0.1, 3.0, 0.4, -0.2, 0.05])

        def respond(velocity):
            return apply_contact_response(
                state,
                velocity,
                jnp.asarray(0.37),
                jnp.zeros((2,), dtype=velocity.dtype),
                self.dynamics,
            )

        velocity = jnp.asarray([2.0, 1.0])
        jacobian = jax.jacrev(respond)(velocity)
        speed = np.sqrt(5.0)
        np.testing.assert_allclose(
            jacobian[3], np.asarray([2.0 / speed, 1.0 / speed]), atol=1.0e-7
        )
        np.testing.assert_allclose(
            jacobian[6], np.asarray([-1.0 / 5.0, 2.0 / 5.0]), atol=1.0e-7
        )
        self.assertTrue(bool(jnp.all(jnp.isfinite(jacobian))))


class TestWallContactParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.track = Track.from_track_name("Spielberg", 1.0)
        cls.table = preprocess_track(cls.track, VEHICLE)
        cls.body = BodyParams.from_vehicle_parameters(VEHICLE)
        cls.dynamics = DynamicsParams.from_vehicle_parameters(VEHICLE)

    def test_complete_step_matches_the_live_simulator_for_ks_and_st(self):
        for model, state_dim, dynamics_fn in (
            (DynamicModel.KS, 5, kinematic_single_track),
            (DynamicModel.ST, 7, single_track),
        ):
            with self.subTest(model=model.name), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                host_config = EnvConfig(
                    num_agents=1,
                    simulation_config=SimulationConfig(
                        dynamics_model=model,
                        integrator=IntegratorType.RK4,
                        compute_frenet_frame=False,
                        max_laps=None,
                    ),
                    control_config=ControlConfig(
                        longitudinal_mode=LongitudinalActionType.ACCL,
                        steering_mode=SteerActionType.STEERING_SPEED,
                    ),
                    lidar_config=LiDARConfig(enabled=False),
                    collision_check=CollisionCheckMode.SEGMENT_CONTACT,
                    render_enabled=False,
                )
                simulator = F110Simulator(
                    env_config=host_config,
                    vehicle_params=VEHICLE,
                    model=model,
                    dynamics_fn=model.f_dynamics,
                    integrator_fn=rk4_integration,
                    longitudinal_type=LongitudinalActionType.ACCL,
                    steering_type=SteerActionType.STEERING_SPEED,
                    track=self.track,
                    seed=4,
                )
                initial = contact_state(self.track, state_dim)
                simulator.reset(initial, option="state", noise_seed=4)
                actions = np.zeros((1, 2), dtype=np.float32)
                simulator.step(actions)

                dynamics_config = DynamicsConfig(
                    num_agents=1,
                    state_dim=state_dim,
                    dynamics_fn=dynamics_fn,
                    integrator_fn=rk4_step,
                    longitudinal_mode=LongitudinalControlMode.ACCELERATION,
                    steering_mode=SteeringControlMode.STEERING_RATE,
                )
                device_state = step_dynamics(
                    jax.random.key(4),
                    make_dynamics_state(initial, dynamics_config),
                    jnp.asarray(actions),
                    dynamics_config,
                    DynamicsRuntimeParams(self.dynamics, DT),
                )
                model_state, events = resolve_wall_contacts(
                    device_state.model,
                    self.table,
                    self.body,
                    self.dynamics,
                    CONTACT,
                    jnp.asarray(DT, jnp.float32),
                    WallContactConfig(1, state_dim),
                )
                np.testing.assert_allclose(
                    model_state, simulator.state.state, rtol=2.0e-4,
                    atol=2.0e-4,
                )
                np.testing.assert_array_equal(
                    np.asarray(events), simulator.state.collisions.astype(bool)
                )

    def test_multiple_agents_resolve_in_one_jitted_call(self):
        hit = contact_state(self.track, 7)[0]
        clear = hit.copy()
        clear[:2] += 5.0
        states = jnp.asarray(np.stack((hit, clear)))
        config = WallContactConfig(2, 7, 16)
        run = jax.jit(
            lambda value: resolve_wall_contacts(
                value, self.table, self.body, self.dynamics, CONTACT, DT, config
            )
        )
        corrected, events = run(states)
        self.assertEqual(corrected.shape, (2, 7))
        np.testing.assert_array_equal(np.asarray(events), [True, False])


class TestWallContactContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = preprocess_track(
            Track.from_track_name("Spielberg", 1.0), VEHICLE
        )
        cls.dynamics = DynamicsParams.from_vehicle_parameters(VEHICLE)
        cls.body = BodyParams(VEHICLE.length, VEHICLE.width, 0.0, 0.0)
        cls.config = WallContactConfig(1, 7, 16)

    def _horizontal_table(self, live=True):
        return synthetic_contact_table(
            self.base,
            wall_a=[[-5.0, 0.0]],
            wall_b=[[5.0, 0.0]],
            normal=[[0.0, 1.0]],
            indices=np.zeros((1, 1, 1), dtype=np.int32),
            mask=np.full((1, 1, 1), live),
        )

    def test_empty_candidates_leave_state_and_event_clear(self):
        state = jnp.asarray(
            [[0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=jnp.float32
        )
        table = self._horizontal_table(False)

        def run(value):
            return resolve_wall_contacts(
                value,
                table,
                self.body,
                self.dynamics,
                CONTACT,
                DT,
                self.config,
            )

        result, events = run(state)
        np.testing.assert_array_equal(np.asarray(result), np.asarray(state))
        np.testing.assert_array_equal(np.asarray(events), [False])

        jacobian = jax.jacrev(lambda value: run(value)[0])(state)
        self.assertTrue(bool(jnp.all(jnp.isfinite(jacobian))))
        np.testing.assert_array_equal(
            jacobian[0, :, 0, :], np.eye(7, dtype=np.float32)
        )

    def test_speculative_only_clamp_is_discarded_for_host_parity(self):
        state = jnp.asarray(
            [[0.0, VEHICLE.length / 2 + 0.05, 0.0, 20.0,
              -np.pi / 2, 0.0, 0.0]],
            dtype=jnp.float32,
        )
        result, events = resolve_wall_contacts(
            state, self._horizontal_table(), self.body, self.dynamics,
            CONTACT, DT, self.config,
        )
        np.testing.assert_array_equal(np.asarray(events), [False])
        np.testing.assert_array_equal(np.asarray(result), np.asarray(state))

    def test_events_are_recomputed_and_never_freeze_or_latch(self):
        colliding = jnp.asarray([[0.0, 0.14, 0.0, 0.0, 0.0, 0.0, 0.0]])
        corrected, hit = resolve_wall_contacts(
            colliding, self._horizontal_table(), self.body, self.dynamics,
            CONTACT, DT, self.config,
        )
        clear = corrected.at[0, 1].set(2.0).at[0, 3].set(1.0)
        continued, next_hit = resolve_wall_contacts(
            clear, self._horizontal_table(), self.body, self.dynamics,
            CONTACT, DT, self.config,
        )
        np.testing.assert_array_equal(np.asarray(hit), [True])
        np.testing.assert_array_equal(np.asarray(next_hit), [False])
        self.assertEqual(float(continued[0, 3]), 1.0)

    def test_tile_lookup_remains_cog_anchored_for_oracle_parity(self):
        table = synthetic_contact_table(
            self.base,
            wall_a=[[0.8, 0.0]],
            wall_b=[[1.7, 0.0]],
            normal=[[0.0, 1.0]],
            indices=np.zeros((1, 2, 1), dtype=np.int32),
            mask=np.array([[[True], [False]]]),
            origin=(0.0, -1.0),
            tile_size=1.0,
        )
        offset_body = BodyParams(VEHICLE.length, VEHICLE.width, 1.0, 0.0)
        state = jnp.asarray([[0.25, 0.14, 0.0, 0.0, 0.0, 0.0, 0.0]])
        _corrected, event = resolve_wall_contacts(
            state, table, offset_body, self.dynamics, CONTACT, DT, self.config
        )
        np.testing.assert_array_equal(np.asarray(event), [True])


class TestTransformability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.track = Track.from_track_name("Spielberg", 1.0)
        cls.table = preprocess_track(cls.track, VEHICLE)
        cls.dynamics = DynamicsParams.from_vehicle_parameters(VEHICLE)
        cls.body = BodyParams.from_vehicle_parameters(VEHICLE)
        cls.state = jnp.asarray(contact_state(cls.track, 7))
        cls.config = WallContactConfig(1, 7, 8)

    def test_jit_eval_shape_and_free_space_gradient(self):
        run = jax.jit(
            lambda value: resolve_wall_contacts(
                value, self.table, self.body, self.dynamics, CONTACT, DT,
                self.config,
            )
        )
        corrected, events = run(self.state)
        shaped = jax.eval_shape(run, self.state)
        self.assertEqual(corrected.shape, shaped[0].shape)
        self.assertEqual(events.shape, shaped[1].shape)
        free = self.state.at[0, :2].add(5.0)
        gradient = jax.grad(lambda value: jnp.sum(run(value)[0]))(free)
        self.assertTrue(bool(jnp.all(jnp.isfinite(gradient))))

    def test_environment_vmap_accepts_traced_body_mass_and_solver_values(self):
        states = jnp.stack((self.state, self.state))
        bodies = jax.tree.map(
            lambda value: jnp.asarray([value, value * 1.03]), self.body
        )
        dynamics = jax.tree.map(
            lambda value: jnp.asarray([value, value]), self.dynamics
        )
        dynamics = replace(
            dynamics, m=jnp.asarray([VEHICLE.m, VEHICLE.m * 1.2])
        )
        first = ContactParams(0.0, 0.2, 0.6, 0.4, 0.002)
        second = ContactParams(0.1, 0.8, 0.6, 0.3, 0.003)
        params = jax.tree.map(
            lambda left, right: jnp.asarray([left, right]), first, second
        )
        run = jax.jit(
            jax.vmap(
                lambda state, body, dynamic, contact: resolve_wall_contacts(
                    state, self.table, body, dynamic, contact, DT, self.config
                )
            )
        )
        corrected, events = run(states, bodies, dynamics, params)
        self.assertEqual(corrected.shape, (2, 1, 7))
        self.assertEqual(events.shape, (2, 1))
        self.assertTrue(bool(jnp.all(jnp.isfinite(corrected))))

    def test_functional_orchestration_contains_no_numpy_or_host_callback(self):
        path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "f1tenth_gym" / "envs" / "contact" / "functional.py"
        )
        source = path.read_text()
        tree = ast.parse(source)
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(("." * node.level) + (node.module or ""))
        self.assertFalse(any(value.startswith("numpy") for value in imported))
        self.assertNotIn("gymnasium", source)
        self.assertNotIn("pure_callback", source)


class TestValidation(unittest.TestCase):
    def test_static_shape_errors_are_named(self):
        for args in ((0, 7, 8), (1, 6, 8), (1, 7, 0)):
            with self.assertRaises(ValueError):
                WallContactConfig(*args)
        self.assertEqual(WallContactConfig(1, 7, 3.9).solver_iterations, 3)
        table = preprocess_track(
            Track.from_track_name("Spielberg", 1.0), VEHICLE
        )
        body = BodyParams.from_vehicle_parameters(VEHICLE)
        dynamics = DynamicsParams.from_vehicle_parameters(VEHICLE)
        with self.assertRaisesRegex(ValueError, "model_state"):
            resolve_wall_contacts(
                jnp.zeros((2, 7)), table, body, dynamics, CONTACT, DT,
                WallContactConfig(1, 7),
            )


if __name__ == "__main__":
    unittest.main()
