"""Global Jacobi gates for simultaneous functional vehicle-pair contact."""

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
from f1tenth_gym.envs.contact import ContactParams, body_contact, resolve_pair
from f1tenth_gym.envs.contact.functional import (
    WallContactConfig,
    apply_contact_response,
    resolve_wall_contacts,
    world_velocity,
)
from f1tenth_gym.envs.contact.geometry import BodyParams, body_vertices
from f1tenth_gym.envs.contact.pairs import (
    PairContactConfig,
    make_pair_table,
    resolve_contacts,
    resolve_pair_contacts,
    solve_pair_impulses,
)
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
from f1tenth_gym.envs.track.walls import wall_segments
from f1tenth_gym.envs.track.preprocessing import preprocess_track


VEHICLE = F1TENTH_VEHICLE_PARAMETERS
DYNAMICS = DynamicsParams.from_vehicle_parameters(VEHICLE)
BODY = BodyParams(VEHICLE.length, VEHICLE.width, 0.0, 0.0)


def states_at(
    xs,
    speeds,
    *,
    ys=None,
    yaws=None,
    yaw_rates=None,
    betas=None,
):
    """Build ST states for simple multi-body contact scenarios."""
    count = len(xs)
    states = np.zeros((count, 7), dtype=np.float32)
    states[:, 0] = xs
    states[:, 1] = np.zeros(count) if ys is None else ys
    states[:, 3] = speeds
    states[:, 4] = np.zeros(count) if yaws is None else yaws
    states[:, 5] = np.zeros(count) if yaw_rates is None else yaw_rates
    states[:, 6] = np.zeros(count) if betas is None else betas
    return jnp.asarray(states)


def rigid_inputs(states, body=BODY, dynamics=DYNAMICS):
    poses = states[:, jnp.asarray((0, 1, 4))]
    vertices = jax.vmap(lambda pose: body_vertices(pose, body))(poses)
    velocities, omegas = jax.vmap(lambda state: world_velocity(state, dynamics))(
        states
    )
    return vertices, velocities, omegas


def manifolds_for(states, table, body=BODY):
    vertices, velocities, omegas = rigid_inputs(states, body)
    left, right = table.indices[:, 0], table.indices[:, 1]
    manifolds = jax.vmap(
        lambda a, b, valid: body_contact(a, b, valid)
    )(vertices[left], vertices[right], table.mask)
    return manifolds, velocities, omegas


def low_level_solve(states, table, params, iterations=64, relaxation=1.0):
    manifolds, velocities, omegas = manifolds_for(states, table)
    return solve_pair_impulses(
        velocities,
        omegas,
        states[:, :2],
        table,
        manifolds.points,
        manifolds.depths,
        manifolds.normal,
        DYNAMICS.m,
        DYNAMICS.I,
        params,
        iterations,
        relaxation,
    )


def momentum(velocities):
    return VEHICLE.m * np.asarray(velocities).sum(axis=0)


def angular_momentum(centres, velocities, omegas):
    centres = np.asarray(centres)
    velocities = np.asarray(velocities)
    orbital = VEHICLE.m * (
        centres[:, 0] * velocities[:, 1]
        - centres[:, 1] * velocities[:, 0]
    )
    return float(np.sum(VEHICLE.I * np.asarray(omegas) + orbital))


class TestPairTable(unittest.TestCase):
    def test_upper_triangle_and_padding_are_fixed_shape(self):
        table = make_pair_table(4, capacity=8)
        np.testing.assert_array_equal(
            np.asarray(table.indices[:6]),
            [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]],
        )
        np.testing.assert_array_equal(
            np.asarray(table.mask), [True] * 6 + [False] * 2
        )
        np.testing.assert_array_equal(np.asarray(table.indices[6:]), 0)

    def test_one_agent_has_one_safe_masked_dummy(self):
        table = make_pair_table(1)
        self.assertEqual(table.indices.shape, (1, 2))
        np.testing.assert_array_equal(np.asarray(table.indices), [[0, 0]])
        np.testing.assert_array_equal(np.asarray(table.mask), [False])

    def test_invalid_topology_arguments_are_named(self):
        with self.assertRaises(ValueError):
            make_pair_table(0)
        with self.assertRaises(ValueError):
            make_pair_table(3, capacity=2)
        for args in ((0, 7, 8), (2, 6, 8), (2, 7, 0)):
            with self.assertRaises(ValueError):
                PairContactConfig(*args)
        with self.assertRaises(ValueError):
            PairContactConfig(2, 7, multi_relaxation=0.0)

        with self.assertRaisesRegex(ValueError, "agent counts"):
            resolve_pair_contacts(
                states_at([-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
                make_pair_table(4),
                BODY,
                DYNAMICS,
                ContactParams(),
                PairContactConfig(3, 7),
            )


class TestIsolatedParity(unittest.TestCase):
    def test_one_pair_matches_the_existing_solver(self):
        table = make_pair_table(2)
        scenarios = (
            (
                states_at([-0.28, 0.28], [3.0, 2.0],
                          yaws=[0.0, np.pi]),
                ContactParams(0.0, 0.0, 0.6, 0.4, 0.002),
                table,
                1.0,
            ),
            (
                states_at([-0.28, 0.28], [3.0, 3.0],
                          yaws=[0.0, np.pi]),
                ContactParams(0.5, 0.0, 0.6, 0.4, 0.002),
                table,
                1.0,
            ),
            (
                states_at([-0.27, 0.27], [3.0, 2.0],
                          yaws=[0.0, np.pi], betas=[0.18, -0.11]),
                ContactParams(0.0, 0.7, 0.6, 0.4, 0.002),
                table,
                1.0,
            ),
            (
                states_at(
                    [0.0, 0.45],
                    [2.0, -1.0],
                    ys=[0.0, 0.13],
                    yaws=[0.0, 0.7],
                    yaw_rates=[0.4, -0.3],
                    betas=[0.12, -0.08],
                ),
                ContactParams(0.2, 0.5, 0.4, 0.4, 0.002),
                make_pair_table(2, capacity=4),
                0.25,
            ),
        )
        for states, params, scenario_table, relaxation in scenarios:
            with self.subTest(params=params):
                manifolds, velocities, omegas = manifolds_for(
                    states, scenario_table
                )
                expected = resolve_pair(
                    velocities[0],
                    omegas[0],
                    velocities[1],
                    omegas[1],
                    DYNAMICS.m,
                    DYNAMICS.I,
                    manifolds.points[0],
                    manifolds.depths[0],
                    manifolds.normal[0],
                    states[0, :2],
                    states[1, :2],
                    params,
                    64,
                )
                expected_states = jnp.stack(
                    (
                        apply_contact_response(
                            states[0], expected[0], expected[1], -expected[4],
                            DYNAMICS,
                        ),
                        apply_contact_response(
                            states[1], expected[2], expected[3], expected[4],
                            DYNAMICS,
                        ),
                    )
                )
                actual, events = resolve_pair_contacts(
                    states,
                    scenario_table,
                    BODY,
                    DYNAMICS,
                    params,
                    PairContactConfig(2, 7, multi_relaxation=relaxation),
                )
                np.testing.assert_allclose(
                    actual, expected_states, rtol=2.0e-6, atol=2.0e-6
                )
                np.testing.assert_array_equal(np.asarray(events), [True, True])

    def test_disjoint_pairs_equal_two_independent_solves(self):
        states = states_at(
            [-0.28, 0.28, -0.28, 0.28],
            [3.0, 3.0, 2.0, 2.0],
            ys=[0.0, 0.0, 3.0, 3.0],
            yaws=[0.0, np.pi, 0.0, np.pi],
        )
        params = ContactParams(0.2, 0.3, 0.6, 0.4, 0.002)
        together, events = resolve_pair_contacts(
            states, make_pair_table(4), BODY, DYNAMICS, params,
            PairContactConfig(4, 7),
        )
        first, _ = resolve_pair_contacts(
            states[:2], make_pair_table(2), BODY, DYNAMICS, params,
            PairContactConfig(2, 7),
        )
        second, _ = resolve_pair_contacts(
            states[2:], make_pair_table(2), BODY, DYNAMICS, params,
            PairContactConfig(2, 7),
        )
        np.testing.assert_allclose(together, jnp.concatenate((first, second)),
                                   atol=2.0e-6)
        np.testing.assert_array_equal(np.asarray(events), [True] * 4)


class TestSimultaneousInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.states = states_at(
            [-0.72, -0.24, 0.25, 0.74],
            [2.4, 0.8, -1.3, -2.1],
            ys=[0.02, -0.01, 0.03, -0.02],
            yaws=[0.04, -0.03, 0.02, -0.05],
            betas=[0.05, -0.02, 0.04, -0.01],
        )
        cls.params = ContactParams(0.0, 0.35, 0.6, 0.4, 0.002)
        cls.config = PairContactConfig(4, 7)

    def test_pair_row_order_does_not_change_body_results(self):
        table = make_pair_table(4)
        expected, expected_events = resolve_pair_contacts(
            self.states, table, BODY, DYNAMICS, self.params, self.config
        )
        for order in (
            np.arange(len(table.mask))[::-1],
            np.random.default_rng(3).permutation(len(table.mask)),
        ):
            permuted = replace(
                table, indices=table.indices[order], mask=table.mask[order]
            )
            actual, events = resolve_pair_contacts(
                self.states, permuted, BODY, DYNAMICS, self.params, self.config
            )
            np.testing.assert_allclose(actual, expected, rtol=2.0e-5,
                                       atol=2.0e-5)
            np.testing.assert_array_equal(events, expected_events)

    def test_agent_labels_are_equivariant(self):
        expected, expected_events = resolve_pair_contacts(
            self.states, make_pair_table(4), BODY, DYNAMICS, self.params,
            self.config,
        )
        permutation = np.array([2, 0, 3, 1])
        inverse = np.argsort(permutation)
        actual, events = resolve_pair_contacts(
            self.states[permutation], make_pair_table(4), BODY, DYNAMICS,
            self.params, self.config,
        )
        np.testing.assert_allclose(actual[inverse], expected, rtol=2.0e-5,
                                   atol=2.0e-5)
        np.testing.assert_array_equal(events[inverse], expected_events)

    def test_linear_and_angular_momentum_are_conserved(self):
        table = make_pair_table(4)
        _manifolds, before_v, before_w = manifolds_for(self.states, table)
        after_v, after_w, correction, events = low_level_solve(
            self.states, table, self.params
        )
        self.assertTrue(bool(jnp.any(events)))
        np.testing.assert_allclose(momentum(after_v), momentum(before_v),
                                   rtol=1.0e-5, atol=1.0e-5)
        self.assertAlmostEqual(
            angular_momentum(self.states[:, :2], after_v, after_w),
            angular_momentum(self.states[:, :2], before_v, before_w),
            delta=2.0e-4,
        )
        np.testing.assert_allclose(np.asarray(correction).sum(axis=0), 0.0,
                                   atol=1.0e-7)

    def test_zero_restitution_does_not_add_kinetic_energy(self):
        table = make_pair_table(4)
        params = self.params._replace(restitution=0.0)
        _manifolds, before_v, before_w = manifolds_for(self.states, table)
        before = 0.5 * VEHICLE.m * jnp.sum(before_v**2) + 0.5 * VEHICLE.I * jnp.sum(
            before_w**2
        )
        for iterations in (1, 8, 64, 128):
            for relaxation in (1.0, 0.65):
                with self.subTest(
                    iterations=iterations, relaxation=relaxation
                ):
                    after_v, after_w, correction, _events = low_level_solve(
                        self.states,
                        table,
                        params,
                        iterations=iterations,
                        relaxation=relaxation,
                    )
                    after = (
                        0.5 * VEHICLE.m * jnp.sum(after_v**2)
                        + 0.5 * VEHICLE.I * jnp.sum(after_w**2)
                    )
                    self.assertTrue(bool(jnp.all(jnp.isfinite(after_v))))
                    self.assertTrue(bool(jnp.all(jnp.isfinite(after_w))))
                    self.assertTrue(bool(jnp.all(jnp.isfinite(correction))))
                    self.assertLessEqual(float(after), float(before) + 2.0e-5)

    def test_symmetric_three_car_squeeze_has_no_middle_drift(self):
        states = states_at([-0.54, 0.0, 0.54], [2.0, 0.0, -2.0])
        result, events = resolve_pair_contacts(
            states,
            make_pair_table(3),
            BODY,
            DYNAMICS,
            ContactParams(0.0, 0.0, 0.6, 0.4, 0.002),
            PairContactConfig(3, 7),
        )
        np.testing.assert_array_equal(np.asarray(events), [True, True, True])
        self.assertAlmostEqual(float(result[1, 3]), 0.0, places=6)
        self.assertAlmostEqual(float(result[1, 5]), 0.0, places=6)
        self.assertAlmostEqual(float(result[1, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(result[0, 3]), -float(result[2, 3]),
                               places=5)
        self.assertAlmostEqual(float(result[0, 0]), -float(result[2, 0]),
                               places=5)


class TestMasksAndEvents(unittest.TestCase):
    def test_one_agent_and_masked_overlap_are_identity(self):
        single = states_at([0.0], [2.0])
        result, event = resolve_pair_contacts(
            single, make_pair_table(1), BODY, DYNAMICS, ContactParams(),
            PairContactConfig(1, 7),
        )
        np.testing.assert_array_equal(result, single)
        np.testing.assert_array_equal(np.asarray(event), [False])

        overlap = states_at([-0.28, 0.28], [3.0, 3.0], yaws=[0.0, np.pi])
        masked = replace(make_pair_table(2), mask=jnp.asarray([False]))
        result, event = resolve_pair_contacts(
            overlap, masked, BODY, DYNAMICS, ContactParams(),
            PairContactConfig(2, 7),
        )
        np.testing.assert_array_equal(result, overlap)
        np.testing.assert_array_equal(np.asarray(event), [False, False])

    def test_events_clear_on_the_next_noncontact_call(self):
        overlap = states_at([-0.28, 0.28], [0.0, 0.0])
        corrected, first = resolve_pair_contacts(
            overlap, make_pair_table(2), BODY, DYNAMICS, ContactParams(),
            PairContactConfig(2, 7),
        )
        clear = corrected.at[0, 0].set(-2.0).at[1, 0].set(2.0)
        continued, second = resolve_pair_contacts(
            clear, make_pair_table(2), BODY, DYNAMICS, ContactParams(),
            PairContactConfig(2, 7),
        )
        np.testing.assert_array_equal(np.asarray(first), [True, True])
        np.testing.assert_array_equal(np.asarray(second), [False, False])
        np.testing.assert_array_equal(continued, clear)


class TestLiveSimulatorParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.track = Track.from_track_name("Spielberg_blank", 1.0)
        cls.table = preprocess_track(cls.track, VEHICLE)
        cls.body = BodyParams.from_vehicle_parameters(VEHICLE)

    def test_two_car_step_matches_the_mutable_pair_path_for_ks_and_st(self):
        for model, state_dim, dynamics_fn in (
            (DynamicModel.KS, 5, kinematic_single_track),
            (DynamicModel.ST, 7, single_track),
        ):
            with self.subTest(model=model.name), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                host_config = EnvConfig(
                    num_agents=2,
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
                    seed=8,
                )
                initial = np.zeros((2, state_dim), dtype=np.float32)
                initial[:, :5] = [
                    [-0.28, 0.0, 0.0, 3.0, 0.0],
                    [0.28, 0.0, 0.0, 3.0, np.pi],
                ]
                simulator.reset(initial, option="state", noise_seed=8)
                actions = np.zeros((2, 2), dtype=np.float32)
                simulator.step(actions)

                dynamics_config = DynamicsConfig(
                    num_agents=2,
                    state_dim=state_dim,
                    dynamics_fn=dynamics_fn,
                    integrator_fn=rk4_step,
                    longitudinal_mode=LongitudinalControlMode.ACCELERATION,
                    steering_mode=SteeringControlMode.STEERING_RATE,
                )
                free = step_dynamics(
                    jax.random.key(8),
                    make_dynamics_state(initial, dynamics_config),
                    jnp.asarray(actions),
                    dynamics_config,
                    DynamicsRuntimeParams(DYNAMICS, 0.01),
                )
                actual, events = resolve_contacts(
                    free.model,
                    self.table,
                    make_pair_table(2),
                    self.body,
                    DYNAMICS,
                    ContactParams(),
                    0.01,
                    WallContactConfig(2, state_dim),
                    PairContactConfig(2, state_dim),
                )
                expected = simulator.state.state
                columns = [0, 1, 2, 3, 4] if state_dim == 5 else [0, 1, 2, 3, 4, 5]
                np.testing.assert_allclose(
                    actual[:, columns], expected[:, columns],
                    rtol=3.0e-4, atol=3.0e-4,
                )
                if state_dim == 7:
                    actual_velocity = jax.vmap(
                        lambda state: world_velocity(state, DYNAMICS)[0]
                    )(actual)
                    expected_velocity = jax.vmap(
                        lambda state: world_velocity(state, DYNAMICS)[0]
                    )(jnp.asarray(expected))
                    np.testing.assert_allclose(
                        actual_velocity, expected_velocity,
                        rtol=3.0e-4, atol=3.0e-4,
                    )
                np.testing.assert_array_equal(
                    np.asarray(events), simulator.state.collisions.astype(bool)
                )


class TestContactComposition(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.track = Track.from_track_name("Spielberg", 1.0)
        cls.table = preprocess_track(cls.track, VEHICLE)
        cls.body = BodyParams.from_vehicle_parameters(VEHICLE)

    def test_wrapper_is_wall_then_pairs_with_event_union(self):
        walls = wall_segments(self.track)
        wall_index = int(np.argmax(walls.length))
        normal = np.asarray(walls.n[wall_index], dtype=np.float32)
        midpoint = np.asarray(
            0.5 * (walls.a[wall_index] + walls.b[wall_index]),
            dtype=np.float32,
        )
        centre = midpoint - normal * (VEHICLE.width / 2.0 - 0.03)
        tangent = np.asarray((-normal[1], normal[0]), dtype=np.float32)
        states = np.zeros((2, 7), dtype=np.float32)
        states[:, :2] = np.stack((centre - 0.08 * tangent,
                                  centre + 0.08 * tangent))
        states[:, 4] = math.atan2(-normal[1], -normal[0])
        states[:, 3] = 1.0
        states = jnp.asarray(states)
        params = ContactParams()
        wall_config = WallContactConfig(2, 7)
        pair_config = PairContactConfig(2, 7)
        pair_table = make_pair_table(2)

        wall_state, wall_events = resolve_wall_contacts(
            states, self.table, self.body, DYNAMICS, params, 0.01, wall_config
        )
        expected, pair_events = resolve_pair_contacts(
            wall_state, pair_table, self.body, DYNAMICS, params, pair_config
        )
        actual, events = resolve_contacts(
            states,
            self.table,
            pair_table,
            self.body,
            DYNAMICS,
            params,
            0.01,
            wall_config,
            pair_config,
        )

        self.assertTrue(bool(jnp.any(wall_events)))
        self.assertTrue(bool(jnp.any(pair_events)))
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(events, wall_events | pair_events)


class TestTransformability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.states = states_at([-0.54, 0.0, 0.54], [2.0, 0.0, -2.0])
        cls.table = make_pair_table(3)
        cls.config = PairContactConfig(3, 7, solver_iterations=16)

    def test_jit_eval_shape_vmap_and_scan(self):
        transition = jax.jit(
            lambda state, dynamics, params: resolve_pair_contacts(
                state, self.table, BODY, dynamics, params, self.config
            )
        )
        params = ContactParams()
        result, events = transition(self.states, DYNAMICS, params)
        shaped = jax.eval_shape(transition, self.states, DYNAMICS, params)
        self.assertEqual(result.shape, shaped[0].shape)
        self.assertEqual(events.shape, shaped[1].shape)

        dynamics = jax.tree.map(
            lambda value: jnp.asarray([value, value]), DYNAMICS
        )
        dynamics = replace(
            dynamics, m=jnp.asarray([VEHICLE.m, VEHICLE.m * 1.2])
        )
        params_batch = jax.tree.map(
            lambda left, right: jnp.asarray([left, right]),
            ContactParams(friction=0.2),
            ContactParams(friction=0.8),
        )
        vmapped = jax.jit(jax.vmap(transition))(
            jnp.stack((self.states, self.states)), dynamics, params_batch
        )
        self.assertEqual(vmapped[0].shape, (2, 3, 7))
        self.assertFalse(
            bool(jnp.allclose(vmapped[0][0], vmapped[0][1], atol=1.0e-7))
        )

        def body(state, _unused):
            next_state, collision = transition(state, DYNAMICS, params)
            return next_state, collision

        final, history = jax.jit(
            lambda state: jax.lax.scan(body, state, None, length=3)
        )(self.states)
        self.assertEqual(final.shape, self.states.shape)
        self.assertEqual(history.shape, (3, 3))

    def test_free_space_gradient_is_finite(self):
        clear = states_at([-3.0, 0.0, 3.0], [1.0, 0.5, -1.0])

        def run(state):
            return resolve_pair_contacts(
                state, self.table, BODY, DYNAMICS, ContactParams(), self.config
            )[0]

        gradient = jax.grad(lambda state: jnp.sum(run(state)))(clear)
        self.assertTrue(bool(jnp.all(jnp.isfinite(gradient))))

    def test_zero_tangent_simultaneous_solver_has_a_finite_jacobian(self):
        table = make_pair_table(3)
        manifolds, velocities, omegas = manifolds_for(self.states, table)

        def run(current_velocities):
            solved, _omegas, _correction, _events = solve_pair_impulses(
                current_velocities,
                omegas,
                self.states[:, :2],
                table,
                manifolds.points,
                manifolds.depths,
                manifolds.normal,
                DYNAMICS.m,
                DYNAMICS.I,
                ContactParams(),
                self.config.solver_iterations,
                self.config.multi_relaxation,
            )
            return solved

        jacobian = jax.jacrev(run)(velocities)

        self.assertTrue(bool(jnp.all(jnp.isfinite(jacobian))))

    def test_pair_module_contains_no_numpy_or_host_callback(self):
        path = (
            pathlib.Path(__file__).resolve().parents[1]
            / "f1tenth_gym" / "envs" / "contact" / "pairs.py"
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


if __name__ == "__main__":
    unittest.main()
