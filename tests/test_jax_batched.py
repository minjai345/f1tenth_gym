"""Device-native shared-map batching, auto-reset, and policy layout gates."""

from dataclasses import replace
import ast
import pathlib
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from f1tenth_gym.envs.dynamic_models import F1TENTH_VEHICLE_PARAMETERS
from f1tenth_gym.envs.lidar import LiDARConfig
from f1tenth_gym.envs.track import Track
from f1tenth_gym.jax.batched import (
    AutoResetBatchStep,
    BatchState,
    BatchStep,
    PolicyField,
    PolicyLayout,
    flatten_joint_observation,
    policy_observation,
    reset_batch,
    reset_batch_from_poses,
    reset_batch_from_state,
    select_ego_rewards,
    step_batch,
    step_batch_autoreset,
)
from f1tenth_gym.jax.contact import ContactParams, WallContactConfig
from f1tenth_gym.jax.core import (
    DynamicsConfig,
    EpisodeParams,
    LongitudinalControlMode,
    SteeringControlMode,
)
from f1tenth_gym.jax.dynamics import DynamicsParams, kinematic_single_track
from f1tenth_gym.jax.environment import (
    CoreConfig,
    CoreObservation,
    CoreParams,
    CoreTables,
    reset_core,
    step_core,
)
from f1tenth_gym.jax.episode import (
    BookkeepingParams,
    BuiltinRewardMode,
    EpisodeConfig,
    TerminationMode,
)
from f1tenth_gym.jax.geometry import BodyParams
from f1tenth_gym.jax.integrators import rk4_step
from f1tenth_gym.jax.lidar import ScanConfig
from f1tenth_gym.jax.pairs import PairContactConfig
from f1tenth_gym.jax.preprocess import (
    build_pair_table,
    build_reset_table,
    build_scan_params,
    build_track_table,
)
from f1tenth_gym.jax.randomization import (
    ActiveVehicleParams,
    VehicleRandomizationParams,
)
from f1tenth_gym.jax.reset import ResetConfig
from f1tenth_gym.jax.track import FrenetProjectionConfig


VEHICLE = F1TENTH_VEHICLE_PARAMETERS
ROOT = pathlib.Path(__file__).resolve().parents[1]


def _circle_track(count=64, radius=8.0):
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return Track.from_refline(
        x=radius * np.cos(theta),
        y=radius * np.sin(theta),
        velx=np.full(count, 3.0),
    )


def _fixture(num_agents=2):
    track = _circle_track()
    lidar = LiDARConfig(num_beams=5, range_max=5.0)
    track_table = build_track_table(
        track,
        VEHICLE,
        ray_max_range=lidar.range_max,
    )
    config = CoreConfig(
        dynamics=DynamicsConfig(
            num_agents=num_agents,
            state_dim=5,
            dynamics_fn=kinematic_single_track,
            integrator_fn=rk4_step,
            longitudinal_mode=LongitudinalControlMode.ACCELERATION,
            steering_mode=SteeringControlMode.STEERING_RATE,
        ),
        reset=ResetConfig(num_agents=num_agents),
        scan=ScanConfig(num_agents, 1, lidar.angle_min, lidar.angle_max),
        wall_contact=WallContactConfig(num_agents, 5),
        pair_contact=PairContactConfig(num_agents, 5),
        episode=EpisodeConfig(
            num_agents,
            ego_index=1 if num_agents > 1 else 0,
            termination_mode=TerminationMode.EGO,
            reward_mode=BuiltinRewardMode.SURVIVAL,
        ),
        frenet=FrenetProjectionConfig(),
        contact_enabled=False,
        scan_enabled=False,
        frenet_enabled=True,
    )
    tables = CoreTables(
        reset=build_reset_table(track.raceline, min_dist=1.0, max_dist=2.0),
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
        bookkeeping=BookkeepingParams(
            terminate_on_collision=False,
            lap_limit_enabled=False,
        ),
    )
    return config, tables, params


def _active_vehicle(params):
    dynamics = params.transition.dynamics
    body = params.body
    return ActiveVehicleParams(
        mu=dynamics.mu,
        C_Sf=dynamics.C_Sf,
        C_Sr=dynamics.C_Sr,
        lf=dynamics.lf,
        lr=dynamics.lr,
        h=dynamics.h,
        m=dynamics.m,
        I=dynamics.I,
        s_min=dynamics.s_min,
        s_max=dynamics.s_max,
        sv_min=dynamics.sv_min,
        sv_max=dynamics.sv_max,
        v_switch=dynamics.v_switch,
        a_max=dynamics.a_max,
        v_min=dynamics.v_min,
        v_max=dynamics.v_max,
        width=body.width,
        length=body.length,
        collision_body_center_x=body.centre_x + dynamics.lr,
        collision_body_center_y=body.centre_y,
    )


def _randomization(params, enabled=False):
    nominal = _active_vehicle(params)
    low = replace(
        nominal,
        m=jnp.asarray(nominal.m) * 0.7,
        length=jnp.asarray(nominal.length) * 0.8,
    )
    high = replace(
        nominal,
        m=jnp.asarray(nominal.m) * 1.3,
        length=jnp.asarray(nominal.length) * 1.2,
    )
    return VehicleRandomizationParams(
        nominal=nominal,
        low=low,
        high=high,
        enabled=enabled,
    )


def _assert_tree_equal(test, actual, expected):
    actual_leaves = jax.tree.leaves(actual)
    expected_leaves = jax.tree.leaves(expected)
    test.assertEqual(len(actual_leaves), len(expected_leaves))
    for actual_leaf, expected_leaf in zip(
        actual_leaves, expected_leaves, strict=True
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


class TestBatchedResetAndStep(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.tables, cls.params = _fixture()
        cls.disabled = _randomization(cls.params)
        cls.batch_size = 3
        cls.reset_keys = jax.random.split(jax.random.key(100), cls.batch_size)

    def test_reset_is_jittable_and_preserves_scalar_reset_streams(self):
        reset = jax.jit(reset_batch, static_argnums=2)
        observation, state = reset(
            self.reset_keys,
            self.tables,
            self.config,
            self.params,
            self.disabled,
        )
        self.assertIsInstance(state, BatchState)
        self.assertEqual(observation.state.shape, (3, 2, 5))
        self.assertEqual(state.params.transition.timestep.shape, (3,))
        for index in range(self.batch_size):
            params = jax.tree.map(lambda value: value[index], state.params)
            scalar = reset_core(
                self.reset_keys[index], self.tables, self.config, params
            )
            _assert_tree_equal(
                self,
                jax.tree.map(lambda value: value[index], observation),
                scalar[0],
            )
            _assert_tree_equal(
                self,
                jax.tree.map(lambda value: value[index], state.core),
                scalar[1],
            )

        replay = reset(
            self.reset_keys,
            self.tables,
            self.config,
            self.params,
            self.disabled,
        )
        _assert_tree_equal(self, (observation, state), replay)

    def test_pose_and_native_state_overrides_are_batched(self):
        poses = jnp.zeros((3, 2, 3), dtype=jnp.float32)
        poses = poses.at[:, :, 0].set(jnp.arange(6).reshape(3, 2))
        observation, state = jax.jit(
            reset_batch_from_poses, static_argnums=3
        )(
            self.reset_keys,
            poses,
            self.tables,
            self.config,
            self.params,
            self.disabled,
        )
        np.testing.assert_array_equal(observation.state[..., 0], poses[..., 0])
        np.testing.assert_array_equal(observation.state[..., 3], 0.0)

        model = observation.state.at[..., 3].set(2.0)
        observation, state = jax.jit(
            reset_batch_from_state, static_argnums=3
        )(
            self.reset_keys,
            model,
            self.tables,
            self.config,
            self.params,
            self.disabled,
        )
        np.testing.assert_array_equal(observation.state, model)
        np.testing.assert_array_equal(state.core.dynamics.control_input, 0.0)

    def test_step_jit_scalar_parity_custom_reward_and_ego_selection(self):
        _observation, state = reset_batch(
            self.reset_keys,
            self.tables,
            self.config,
            self.params,
            self.disabled,
        )
        step_keys = jax.random.split(jax.random.key(101), self.batch_size)
        actions = jnp.arange(12, dtype=jnp.float32).reshape(3, 2, 2) * 0.01
        step = jax.jit(step_batch, static_argnums=(4, 5))
        result = step(
            step_keys, state, actions, self.tables, self.config, None
        )
        self.assertIsInstance(result, BatchStep)
        for index in range(self.batch_size):
            expected = step_core(
                step_keys[index],
                jax.tree.map(lambda value: value[index], state.core),
                actions[index],
                self.tables,
                self.config,
                jax.tree.map(lambda value: value[index], state.params),
            )
            _assert_tree_equal(
                self,
                jax.tree.map(lambda value: value[index], result.observation),
                expected[0],
            )
            np.testing.assert_array_equal(result.rewards[index], expected[2])

        def reward_fn(observation, action, events, metrics, params):
            del events
            return (
                observation.standard_state[:, 3]
                + action[:, 1]
                + metrics.episode.progress
                + params.transition.timestep
            )

        custom = step(
            step_keys, state, actions, self.tables, self.config, reward_fn
        )
        self.assertEqual(custom.rewards.shape, (3, 2))
        np.testing.assert_array_equal(
            select_ego_rewards(custom.rewards, self.config),
            custom.rewards[:, 1],
        )

        def scalar_reward(*_args):
            return jnp.asarray(1.0, dtype=jnp.float32)

        with self.assertRaisesRegex(ValueError, "reward_fn must return shape"):
            step(
                step_keys,
                state,
                actions,
                self.tables,
                self.config,
                scalar_reward,
            )

    def test_scan_of_batch_matches_batch_of_scans(self):
        _observation, state = reset_batch(
            self.reset_keys,
            self.tables,
            self.config,
            self.params,
            self.disabled,
        )
        time_steps = 3
        keys = jax.random.split(
            jax.random.key(102), time_steps * self.batch_size
        ).reshape(time_steps, self.batch_size)
        actions = jnp.zeros((time_steps, self.batch_size, 2, 2), jnp.float32)

        def batch_body(carry, inputs):
            result = step_batch(
                inputs[0], carry, inputs[1], self.tables, self.config
            )
            return result.state, result.observation.state

        batch_final, batch_history = jax.jit(
            lambda initial: jax.lax.scan(batch_body, initial, (keys, actions))
        )(state)

        def environment_rollout(core, params, environment_keys, environment_actions):
            def body(carry, inputs):
                observation, next_core, *_tail = step_core(
                    inputs[0],
                    carry,
                    inputs[1],
                    self.tables,
                    self.config,
                    params,
                )
                return next_core, observation.state

            return jax.lax.scan(
                body, core, (environment_keys, environment_actions)
            )

        scalar_final, scalar_history = jax.jit(jax.vmap(environment_rollout))(
            state.core,
            state.params,
            jnp.swapaxes(keys, 0, 1),
            jnp.swapaxes(actions, 0, 1),
        )
        np.testing.assert_array_equal(
            batch_history, jnp.swapaxes(scalar_history, 0, 1)
        )
        _assert_tree_equal(self, batch_final.core, scalar_final)

    def test_raw_step_reports_done_without_reset_or_freeze(self):
        _observation, state = reset_batch(
            self.reset_keys,
            self.tables,
            self.config,
            self.params,
            self.disabled,
        )
        state = replace(
            state,
            params=replace(
                state.params,
                bookkeeping=replace(
                    state.params.bookkeeping,
                    step_limit_enabled=jnp.ones((3,), dtype=jnp.bool_),
                    max_episode_steps=jnp.ones((3,), dtype=jnp.int32),
                ),
            ),
        )
        actions = jnp.zeros((3, 2, 2), dtype=jnp.float32)
        actions = actions.at[..., 1].set(1.0)
        first = step_batch(
            jax.random.split(jax.random.key(103), 3),
            state,
            actions,
            self.tables,
            self.config,
        )
        np.testing.assert_array_equal(first.metrics.status.truncated, True)
        np.testing.assert_array_equal(first.state.core.episode.elapsed_steps, 1)
        second = step_batch(
            jax.random.split(jax.random.key(104), 3),
            first.state,
            actions,
            self.tables,
            self.config,
        )
        np.testing.assert_array_equal(second.state.core.episode.elapsed_steps, 2)
        self.assertTrue(
            bool(jnp.all(second.observation.state[..., 3] > first.observation.state[..., 3]))
        )


class TestSelectiveAutoReset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.tables, cls.params = _fixture()
        cls.randomization = _randomization(cls.params, enabled=True)

    def test_terminal_outputs_are_preserved_and_only_done_rows_reset(self):
        keys = jax.random.split(jax.random.key(200), 3)
        _observation, state = reset_batch(
            keys,
            self.tables,
            self.config,
            self.params,
            self.randomization,
        )
        self.assertEqual(state.params.transition.dynamics.m.shape, (3,))
        self.assertEqual(state.params.body.length.shape, (3,))
        self.assertEqual(
            np.unique(np.asarray(state.params.transition.dynamics.m)).size,
            3,
        )
        terminated_agents = state.core.episode.terminated_agents.at[0, 1].set(
            True
        )
        state = replace(
            state,
            core=replace(
                state.core,
                episode=replace(
                    state.core.episode,
                    terminated_agents=terminated_agents,
                ),
            ),
            params=replace(
                state.params,
                bookkeeping=replace(
                    state.params.bookkeeping,
                    step_limit_enabled=jnp.ones((3,), dtype=jnp.bool_),
                    max_episode_steps=jnp.asarray([3, 3, 1], dtype=jnp.int32),
                ),
            ),
        )
        step_keys = jax.random.split(jax.random.key(201), 3)
        reset_keys = jax.random.split(jax.random.key(202), 3)
        actions = jnp.zeros((3, 2, 2), dtype=jnp.float32)
        actions = actions.at[..., 1].set(1.0)
        raw = step_batch(
            step_keys, state, actions, self.tables, self.config
        )
        candidate_observation, candidate_state = reset_batch(
            reset_keys,
            self.tables,
            self.config,
            self.params,
            self.randomization,
        )
        run = jax.jit(step_batch_autoreset, static_argnums=(5, 8))
        result = run(
            step_keys,
            reset_keys,
            state,
            actions,
            self.tables,
            self.config,
            self.params,
            self.randomization,
            None,
        )
        self.assertIsInstance(result, AutoResetBatchStep)
        np.testing.assert_array_equal(result.reset, [True, False, True])
        np.testing.assert_array_equal(
            result.metrics.status.terminated, [True, False, False]
        )
        np.testing.assert_array_equal(
            result.metrics.status.truncated, [False, False, True]
        )
        _assert_tree_equal(self, result.transition_observation, raw.observation)
        _assert_tree_equal(self, result.rewards, raw.rewards)
        _assert_tree_equal(self, result.events, raw.events)
        _assert_tree_equal(self, result.metrics, raw.metrics)

        for index, reset in enumerate((True, False, True)):
            expected_observation = (
                candidate_observation if reset else raw.observation
            )
            expected_state = candidate_state if reset else raw.state
            _assert_tree_equal(
                self,
                jax.tree.map(
                    lambda value: value[index], result.next_observation
                ),
                jax.tree.map(
                    lambda value: value[index], expected_observation
                ),
            )
            _assert_tree_equal(
                self,
                jax.tree.map(lambda value: value[index], result.state),
                jax.tree.map(lambda value: value[index], expected_state),
            )

        self.assertFalse(
            np.array_equal(
                result.transition_observation.state[0],
                result.next_observation.state[0],
            )
        )
        np.testing.assert_array_equal(
            result.state.core.episode.elapsed_steps, [0, 1, 0]
        )
        replay = run(
            step_keys,
            reset_keys,
            state,
            actions,
            self.tables,
            self.config,
            self.params,
            self.randomization,
            None,
        )
        _assert_tree_equal(self, result, replay)


class TestPolicyLayouts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        disabled_config, _tables, _params = _fixture()
        cls.disabled_config = disabled_config
        cls.config = replace(
            disabled_config,
            scan=ScanConfig(2, 3, -1.0, 1.0),
            scan_enabled=True,
        )

    def setUp(self):
        self.observation = CoreObservation(
            scans=jnp.arange(12, dtype=jnp.float32).reshape(2, 2, 3),
            state=(100 + jnp.arange(20, dtype=jnp.float32)).reshape(2, 2, 5),
            standard_state=(200 + jnp.arange(28, dtype=jnp.float32)).reshape(2, 2, 7),
            collisions=jnp.asarray([[0.0, 1.0], [1.0, 0.0]], jnp.float32),
            frenet=(300 + jnp.arange(12, dtype=jnp.float32)).reshape(2, 2, 3),
            lap_times=jnp.asarray([[1.0, 2.0], [3.0, 4.0]], jnp.float32),
            lap_counts=jnp.asarray([[5.0, 6.0], [7.0, 8.0]], jnp.float32),
            sim_time=jnp.asarray([9.0, 10.0], jnp.float32),
        )

    def test_explicit_field_order_and_joint_flattening(self):
        layout = PolicyLayout(
            (
                PolicyField.COLLISION,
                PolicyField.KINEMATIC_STATE,
                PolicyField.DYNAMIC_STATE,
                PolicyField.NATIVE_STATE,
                PolicyField.SCAN,
                PolicyField.FRENET,
                PolicyField.LAP_TIME,
                PolicyField.LAP_COUNT,
                PolicyField.SIM_TIME,
            )
        )
        actual = jax.jit(policy_observation, static_argnums=(1, 2))(
            self.observation, self.config, layout
        )
        expected = jnp.concatenate(
            (
                self.observation.collisions[..., None],
                self.observation.standard_state[..., :5],
                self.observation.standard_state,
                self.observation.state,
                self.observation.scans,
                self.observation.frenet,
                self.observation.lap_times[..., None],
                self.observation.lap_counts[..., None],
                jnp.broadcast_to(
                    self.observation.sim_time[:, None, None], (2, 2, 1)
                ),
            ),
            axis=-1,
        )
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape, (2, 2, 27))
        joint = jax.jit(flatten_joint_observation)(actual)
        np.testing.assert_array_equal(joint, expected.reshape(2, 54))

    def test_invalid_layouts_and_shapes_fail_early(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            PolicyLayout(())
        with self.assertRaisesRegex(TypeError, "PolicyField"):
            PolicyLayout(("state",))
        with self.assertRaisesRegex(ValueError, "leading batch"):
            policy_observation(
                replace(self.observation, state=self.observation.state[0]),
                self.config,
                PolicyLayout((PolicyField.NATIVE_STATE,)),
            )
        with self.assertRaisesRegex(ValueError, "sensing is disabled"):
            policy_observation(
                self.observation,
                self.disabled_config,
                PolicyLayout((PolicyField.SCAN,)),
            )
        with self.assertRaisesRegex(ValueError, "Frenet frame is disabled"):
            policy_observation(
                self.observation,
                replace(self.config, frenet_enabled=False),
                PolicyLayout((PolicyField.FRENET,)),
            )
        with self.assertRaisesRegex(ValueError, "batch, agents, features"):
            flatten_joint_observation(jnp.zeros((2, 3)))


class TestValidationAndPurity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.tables, cls.params = _fixture()
        cls.randomization = _randomization(cls.params)

    def test_public_shape_errors_are_specific(self):
        scalar_key = jax.random.key(300)
        with self.assertRaisesRegex(ValueError, "batch axis"):
            reset_batch(
                scalar_key,
                self.tables,
                self.config,
                self.params,
                self.randomization,
            )
        keys = jax.random.split(scalar_key, 2)
        with self.assertRaisesRegex(ValueError, "poses must have shape"):
            reset_batch_from_poses(
                keys,
                jnp.zeros((2, 1, 3)),
                self.tables,
                self.config,
                self.params,
                self.randomization,
            )
        with self.assertRaisesRegex(ValueError, "model_state must have shape"):
            reset_batch_from_state(
                keys,
                jnp.zeros((2, 2, 7)),
                self.tables,
                self.config,
                self.params,
                self.randomization,
            )
        _observation, state = reset_batch(
            keys,
            self.tables,
            self.config,
            self.params,
            self.randomization,
        )
        with self.assertRaisesRegex(ValueError, "actions must have shape"):
            step_batch(
                keys,
                state,
                jnp.zeros((2, 2)),
                self.tables,
                self.config,
            )
        with self.assertRaisesRegex(ValueError, "same number"):
            step_batch_autoreset(
                keys,
                jax.random.split(jax.random.key(301), 3),
                state,
                jnp.zeros((2, 2, 2)),
                self.tables,
                self.config,
                self.params,
                self.randomization,
            )

    def test_compiled_batch_needs_no_host_transfer(self):
        keys = jax.random.split(jax.random.key(302), 2)
        reset = jax.jit(reset_batch, static_argnums=2)
        step = jax.jit(step_batch, static_argnums=(4, 5))
        _observation, state = reset(
            keys,
            self.tables,
            self.config,
            self.params,
            self.randomization,
        )
        actions = jnp.zeros((2, 2, 2), dtype=jnp.float32)
        step(keys, state, actions, self.tables, self.config, None)
        with jax.transfer_guard("disallow"):
            result = step(
                keys, state, actions, self.tables, self.config, None
            )
            jax.block_until_ready(result.state.core.dynamics.model)

    def test_module_has_no_host_or_framework_imports(self):
        path = ROOT / "f1tenth_gym" / "jax" / "batched.py"
        tree = ast.parse(path.read_text())
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                imports.add(node.module or "")
        self.assertFalse(any(name.startswith("numpy") for name in imports))
        self.assertFalse(any(name.startswith("gymnasium") for name in imports))
        self.assertFalse(any(name.startswith("f1tenth_gym.envs") for name in imports))
        source = path.read_text()
        self.assertNotIn("device_get", source)


if __name__ == "__main__":
    unittest.main()
