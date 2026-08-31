"""End-to-end contracts for the composed functional JAX environment core."""

from dataclasses import replace
import unittest
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from f1tenth_gym.envs.action import (
    LongitudinalActionType,
    SteerActionType,
)
from f1tenth_gym.envs.collision_models import CollisionCheckMode
from f1tenth_gym.envs.dynamic_models import (
    DynamicModel,
    F1TENTH_VEHICLE_PARAMETERS,
)
from f1tenth_gym.envs.env_config import (
    ControlConfig,
    EnvConfig,
    ObservationConfig,
    SimulationConfig,
)
from f1tenth_gym.envs.f110_env import F110Env
from f1tenth_gym.envs.integrators import IntegratorType
from f1tenth_gym.envs.lidar import LiDARConfig
from f1tenth_gym.envs.observation import ObservationType
from f1tenth_gym.envs.track import Track
from f1tenth_gym.jax import (
    BodyParams,
    BookkeepingParams,
    BuiltinRewardMode,
    ContactParams,
    CoreConfig,
    CoreParams,
    CoreTables,
    DynamicsConfig,
    DynamicsParams,
    EpisodeConfig,
    EpisodeParams,
    FrenetProjectionConfig,
    LongitudinalControlMode,
    PairContactConfig,
    ResetConfig,
    ScanConfig,
    SteeringControlMode,
    TerminationMode,
    WallContactConfig,
    cartesian_to_frenet,
    kinematic_single_track,
    make_dynamics_state,
    observe_core,
    reset_core,
    reset_episode_state,
    rk4_step,
    single_track,
    standardize_state,
    step_core,
)
from f1tenth_gym.jax.preprocess import (
    build_pair_table,
    build_reset_table,
    build_scan_params,
    build_track_table,
)


VEHICLE = F1TENTH_VEHICLE_PARAMETERS


def circle_track(count=80, radius=8.0):
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return Track.from_refline(
        x=radius * np.cos(theta),
        y=radius * np.sin(theta),
        velx=np.full(count, 4.0),
    )


def core_fixture(
    *,
    num_agents=1,
    state_dim=5,
    contact_enabled=False,
    scan_enabled=True,
    frenet_enabled=True,
    reward_mode=BuiltinRewardMode.SURVIVAL,
    termination_mode=TerminationMode.EGO,
    noise_std=0.0,
    range_bias_std=0.0,
    dropout_prob=0.0,
):
    track = circle_track()
    lidar = LiDARConfig(
        num_beams=9,
        range_max=5.0,
        noise_std=noise_std,
        range_bias_std=range_bias_std,
        dropout_prob=dropout_prob,
    )
    track_table = build_track_table(
        track,
        VEHICLE,
        ray_max_range=lidar.range_max,
    )
    dynamics_fn = kinematic_single_track if state_dim == 5 else single_track
    config = CoreConfig(
        dynamics=DynamicsConfig(
            num_agents=num_agents,
            state_dim=state_dim,
            dynamics_fn=dynamics_fn,
            integrator_fn=rk4_step,
            longitudinal_mode=LongitudinalControlMode.ACCELERATION,
            steering_mode=SteeringControlMode.STEERING_RATE,
        ),
        reset=ResetConfig(num_agents=num_agents),
        scan=ScanConfig(
            num_agents,
            lidar.num_beams if scan_enabled else 1,
            lidar.angle_min,
            lidar.angle_max,
        ),
        wall_contact=WallContactConfig(
            num_agents,
            state_dim,
            solver_iterations=8,
        ),
        pair_contact=PairContactConfig(
            num_agents,
            state_dim,
            solver_iterations=8,
        ),
        episode=EpisodeConfig(
            num_agents,
            termination_mode=termination_mode,
            reward_mode=reward_mode,
        ),
        frenet=FrenetProjectionConfig(),
        contact_enabled=contact_enabled,
        scan_enabled=scan_enabled,
        frenet_enabled=frenet_enabled,
    )
    tables = CoreTables(
        reset=build_reset_table(
            track.raceline,
            min_dist=1.0,
            max_dist=2.0,
        ),
        track=track_table,
        pairs=build_pair_table(num_agents),
    )
    params = CoreParams(
        transition=EpisodeParams(
            dynamics=DynamicsParams.from_vehicle_parameters(VEHICLE),
            timestep=0.01,
        ),
        body=BodyParams.from_vehicle_parameters(VEHICLE),
        contact=ContactParams(),
        scan=build_scan_params(lidar, track_table),
        bookkeeping=BookkeepingParams(),
    )
    return track, lidar, config, tables, params


class TestCoreConstruction(unittest.TestCase):
    def test_standard_state_preserves_st_and_pads_ks_like_the_host(self):
        ks = jnp.arange(5, dtype=jnp.float32)
        st = jnp.arange(7, dtype=jnp.float32)
        np.testing.assert_array_equal(
            jax.jit(standardize_state)(ks),
            np.asarray([0, 1, 2, 3, 4, 0, 0], dtype=np.float32),
        )
        np.testing.assert_array_equal(jax.jit(standardize_state)(st), st)
        for bad in (jnp.zeros((4,)), jnp.zeros((1, 5))):
            with self.assertRaisesRegex(ValueError, "shape"):
                standardize_state(bad)

    def test_cross_topology_and_progress_requirements_fail_early(self):
        _track, _lidar, config, _tables, _params = core_fixture()
        with self.assertRaisesRegex(ValueError, "same num_agents"):
            replace(
                config,
                scan=ScanConfig(
                    2,
                    config.scan.num_beams,
                    config.scan.angle_min,
                    config.scan.angle_max,
                ),
            )
        with self.assertRaisesRegex(ValueError, "same state_dim"):
            replace(config, pair_contact=PairContactConfig(1, 7))
        with self.assertRaisesRegex(ValueError, "PROGRESS"):
            replace(
                config,
                episode=replace(
                    config.episode,
                    reward_mode=BuiltinRewardMode.PROGRESS,
                ),
                frenet_enabled=False,
            )
        with self.assertRaisesRegex(ValueError, "one-beam"):
            replace(config, scan_enabled=False)


class TestCoreResetAndStep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.track, cls.lidar, cls.config, cls.tables, cls.params = core_fixture()
        cls.reset = staticmethod(jax.jit(reset_core, static_argnums=2))
        cls.step = staticmethod(jax.jit(step_core, static_argnums=4))

    def test_reset_builds_every_canonical_field_and_a_real_scan(self):
        observation, state = self.reset(
            jax.random.key(3), self.tables, self.config, self.params
        )
        self.assertEqual(observation.state.shape, (1, 5))
        self.assertEqual(observation.standard_state.shape, (1, 7))
        self.assertEqual(observation.scans.shape, (1, 9))
        self.assertEqual(observation.frenet.shape, (1, 3))
        self.assertEqual(observation.collisions.dtype, jnp.float32)
        self.assertEqual(state.collisions.dtype, jnp.bool_)
        np.testing.assert_array_equal(observation.scans, self.lidar.range_max)
        np.testing.assert_array_equal(observation.collisions, 0.0)
        self.assertEqual(float(observation.sim_time), 0.0)
        expected_frenet = jax.vmap(
            lambda pose: cartesian_to_frenet(
                self.tables.track.centerline, pose
            )
        )(observation.state[:, jnp.asarray((0, 1, 4))])
        np.testing.assert_allclose(observation.frenet, expected_frenet, atol=1e-6)
        np.testing.assert_array_equal(observe_core(state).state, observation.state)

    def test_step_preserves_the_observation_info_clock_split(self):
        _observation, state = self.reset(
            jax.random.key(4), self.tables, self.config, self.params
        )
        actions = jnp.asarray([[0.05, 1.0]], dtype=jnp.float32)
        first = self.step(
            jax.random.key(5),
            state,
            actions,
            self.tables,
            self.config,
            self.params,
        )
        observation, state, rewards, events, metrics = first
        self.assertEqual(float(observation.sim_time), 0.0)
        self.assertAlmostEqual(float(state.dynamics.sim_time), 0.01, places=7)
        self.assertAlmostEqual(float(metrics.episode.sim_time), 0.01, places=7)
        np.testing.assert_allclose(rewards, [0.01], atol=1e-7)
        np.testing.assert_array_equal(events.collisions, [False])
        self.assertFalse(bool(metrics.status.terminated))
        self.assertFalse(bool(metrics.status.truncated))

        observation, state, *_tail = self.step(
            jax.random.key(6),
            state,
            actions,
            self.tables,
            self.config,
            self.params,
        )
        self.assertAlmostEqual(float(observation.sim_time), 0.01, places=7)
        self.assertAlmostEqual(float(state.dynamics.sim_time), 0.02, places=7)

    def test_explicit_keys_replay_the_whole_stochastic_transition(self):
        _track, _lidar, config, tables, params = core_fixture(
            noise_std=0.1,
            range_bias_std=0.2,
        )
        params = replace(
            params,
            transition=replace(
                params.transition,
                steer_noise_std=0.1,
                accel_noise_std=0.2,
            ),
        )
        reset = jax.jit(reset_core, static_argnums=2)
        step = jax.jit(step_core, static_argnums=4)
        first_obs, first_state = reset(jax.random.key(10), tables, config, params)
        replay_obs, replay_state = reset(jax.random.key(10), tables, config, params)
        np.testing.assert_array_equal(first_obs.scans, replay_obs.scans)
        np.testing.assert_array_equal(
            first_state.scan.range_bias, replay_state.scan.range_bias
        )
        actions = jnp.zeros((1, 2), dtype=jnp.float32)
        first = step(
            jax.random.key(11), first_state, actions, tables, config, params
        )
        replay = step(
            jax.random.key(11), first_state, actions, tables, config, params
        )
        other = step(
            jax.random.key(12), first_state, actions, tables, config, params
        )
        np.testing.assert_array_equal(first[0].scans, replay[0].scans)
        np.testing.assert_array_equal(first[1].dynamics.model, replay[1].dynamics.model)
        self.assertFalse(np.array_equal(first[0].scans, other[0].scans))
        self.assertFalse(
            np.array_equal(first[1].dynamics.model, other[1].dynamics.model)
        )

    def test_disabled_sensing_and_frenet_have_fixed_zero_shapes(self):
        _track, _lidar, config, tables, params = core_fixture(
            scan_enabled=False,
            frenet_enabled=False,
        )
        observation, state = reset_core(
            jax.random.key(13), tables, config, params
        )
        np.testing.assert_array_equal(observation.scans, 0.0)
        self.assertEqual(observation.scans.shape, (1, 1))
        np.testing.assert_array_equal(observation.frenet, 0.0)
        observation, _state, _rewards, events, metrics = step_core(
            jax.random.key(14),
            state,
            jnp.zeros((1, 2), dtype=jnp.float32),
            tables,
            config,
            params,
        )
        np.testing.assert_array_equal(observation.frenet, 0.0)
        np.testing.assert_array_equal(metrics.episode.progress, 0.0)
        np.testing.assert_array_equal(events.finish_crossed, [False])
        self.assertFalse(bool(metrics.status.terminated))


class TestCoreContactComposition(unittest.TestCase):
    def test_contact_events_terminate_but_never_freeze_vehicle_state(self):
        _track, _lidar, config, tables, params = core_fixture(
            num_agents=2,
            state_dim=7,
            contact_enabled=True,
            scan_enabled=False,
            termination_mode=TerminationMode.ANY,
        )
        params = replace(
            params,
            bookkeeping=replace(
                params.bookkeeping,
                lap_limit_enabled=False,
            ),
        )
        _observation, state = reset_core(
            jax.random.key(20), tables, config, params
        )
        model = jnp.zeros((2, 7), dtype=jnp.float32)
        model = model.at[:, 0].set(jnp.asarray([-0.28, 0.28]))
        model = model.at[:, 4].set(jnp.asarray([0.0, jnp.pi]))
        poses = model[:, jnp.asarray((0, 1, 4))]
        frenet = jax.vmap(
            lambda pose: cartesian_to_frenet(tables.track.centerline, pose)
        )(poses)
        state = replace(
            state,
            dynamics=make_dynamics_state(model, config.dynamics),
            episode=reset_episode_state(frenet, config.episode),
        )
        observation, state, _rewards, events, metrics = step_core(
            jax.random.key(21),
            state,
            jnp.zeros((2, 2), dtype=jnp.float32),
            tables,
            config,
            params,
        )
        np.testing.assert_array_equal(events.collisions, [True, True])
        np.testing.assert_array_equal(events.newly_terminated, [True, True])
        np.testing.assert_array_equal(observation.collisions, [1.0, 1.0])
        self.assertTrue(bool(metrics.status.terminated))

        clear_model = state.dynamics.model.at[:, 0].set(
            jnp.asarray([-3.0, 3.0])
        )
        state = replace(
            state,
            dynamics=replace(state.dynamics, model=clear_model),
        )
        _observation, state, _rewards, events, metrics = step_core(
            jax.random.key(22),
            state,
            jnp.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=jnp.float32),
            tables,
            config,
            params,
        )
        np.testing.assert_array_equal(events.collisions, [False, False])
        self.assertTrue(bool(jnp.all(state.dynamics.model[:, 3] > 0.0)))
        np.testing.assert_array_equal(
            state.episode.terminated_agents, [True, True]
        )
        self.assertTrue(bool(metrics.status.terminated))


class TestCoreTransforms(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _track, _lidar, cls.config, cls.tables, cls.params = core_fixture()

    def test_eval_shape_jit_and_lax_scan_cover_the_composed_core(self):
        reset = jax.jit(reset_core, static_argnums=2)
        step = jax.jit(step_core, static_argnums=4)
        observation, state = reset(
            jax.random.key(30), self.tables, self.config, self.params
        )
        shaped = jax.eval_shape(
            lambda key: reset_core(key, self.tables, self.config, self.params),
            jax.random.key(30),
        )
        self.assertEqual(observation.scans.shape, shaped[0].scans.shape)

        keys = jax.random.split(jax.random.key(31), 4)
        actions = jnp.zeros((4, 1, 2), dtype=jnp.float32)

        def body(current, inputs):
            key, action = inputs
            obs, next_state, rewards, events, metrics = step(
                key,
                current,
                action,
                self.tables,
                self.config,
                self.params,
            )
            output = (obs.sim_time, rewards, events.collisions, metrics.episode.sim_time)
            return next_state, output

        final, history = jax.jit(
            lambda initial: jax.lax.scan(body, initial, (keys, actions))
        )(state)
        self.assertAlmostEqual(float(final.dynamics.sim_time), 0.04, places=7)
        self.assertEqual(history[0].shape, (4,))
        self.assertEqual(history[1].shape, (4, 1))
        np.testing.assert_allclose(history[0], [0.0, 0.01, 0.02, 0.03], atol=1e-7)
        np.testing.assert_allclose(history[3], [0.01, 0.02, 0.03, 0.04], atol=1e-7)

    def test_shared_tables_vmap_over_heterogeneous_traced_params(self):
        batched_params = jax.tree.map(
            lambda value: jnp.stack((jnp.asarray(value), jnp.asarray(value))),
            self.params,
        )
        batched_params = replace(
            batched_params,
            transition=replace(
                batched_params.transition,
                timestep=jnp.asarray([0.01, 0.02], dtype=jnp.float32),
            ),
        )
        keys = jax.random.split(jax.random.key(32), 2)
        observations, states = jax.jit(
            jax.vmap(
                lambda key, params: reset_core(
                    key, self.tables, self.config, params
                )
            )
        )(keys, batched_params)
        self.assertEqual(observations.state.shape, (2, 1, 5))
        actions = jnp.zeros((2, 1, 2), dtype=jnp.float32)
        result = jax.jit(
            jax.vmap(
                lambda key, state, action, params: step_core(
                    key,
                    state,
                    action,
                    self.tables,
                    self.config,
                    params,
                )
            )
        )(keys, states, actions, batched_params)
        np.testing.assert_allclose(
            result[1].dynamics.sim_time,
            [0.01, 0.02],
            atol=1e-7,
        )
        np.testing.assert_allclose(result[2], [[0.01], [0.02]], atol=1e-7)


class TestMutableEnvironmentParity(unittest.TestCase):
    def test_noise_free_ks_rollout_matches_the_current_environment_order(self):
        track, lidar, config, tables, params = core_fixture()
        core_observation, core_state = reset_core(
            jax.random.key(40), tables, config, params
        )
        poses = np.asarray(core_observation.state[:, (0, 1, 4)])
        host_config = EnvConfig(
            map_name=track,
            num_agents=1,
            control_config=ControlConfig(
                longitudinal_mode=LongitudinalActionType.ACCL,
                steering_mode=SteerActionType.STEERING_SPEED,
            ),
            simulation_config=SimulationConfig(
                dynamics_model=DynamicModel.KS,
                integrator=IntegratorType.RK4,
                timestep=0.01,
                integrator_timestep=0.01,
            ),
            observation_config=ObservationConfig(type=ObservationType.DIRECT),
            lidar_config=lidar,
            collision_check=CollisionCheckMode.NONE,
            render_enabled=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            env = F110Env(host_config)
            try:
                host_observation, _info = env.reset(
                    seed=40,
                    options={"poses": poses},
                )
                np.testing.assert_allclose(
                    core_observation.frenet,
                    host_observation["frenet"],
                    atol=2e-3,
                )
                np.testing.assert_allclose(
                    core_observation.scans,
                    host_observation["scans"],
                    atol=2e-3,
                )
                for index in range(6):
                    action = np.asarray(
                        [[0.03 * np.sin(index), 0.8]], dtype=np.float32
                    )
                    host = env.step(action)
                    core = step_core(
                        jax.random.fold_in(jax.random.key(41), index),
                        core_state,
                        jnp.asarray(action),
                        tables,
                        config,
                        params,
                    )
                    core_observation, core_state, rewards, events, metrics = core
                    host_observation, reward, terminated, truncated, info = host
                    np.testing.assert_allclose(
                        core_observation.state,
                        host_observation["state"],
                        rtol=2e-4,
                        atol=2e-4,
                    )
                    np.testing.assert_allclose(
                        core_observation.frenet,
                        host_observation["frenet"],
                        atol=3e-3,
                    )
                    np.testing.assert_allclose(
                        core_observation.scans,
                        host_observation["scans"],
                        atol=2e-3,
                    )
                    np.testing.assert_allclose(rewards[0], reward, atol=1e-7)
                    np.testing.assert_array_equal(
                        events.collisions,
                        info["collisions"].astype(bool),
                    )
                    self.assertEqual(bool(metrics.status.terminated), terminated)
                    self.assertEqual(bool(metrics.status.truncated), truncated)
                    self.assertAlmostEqual(
                        float(core_observation.sim_time),
                        float(host_observation["sim_time"]),
                        places=7,
                    )
                    self.assertAlmostEqual(
                        float(metrics.episode.sim_time),
                        float(info["sim_time"]),
                        places=7,
                    )
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
