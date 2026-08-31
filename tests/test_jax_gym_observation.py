"""Gymnasium packaging contracts for functional JAX observations."""

from dataclasses import replace
import unittest
import warnings

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from f1tenth_gym.envs.collision_models import CollisionCheckMode
from f1tenth_gym.envs.dynamic_models import DynamicModel
from f1tenth_gym.envs.env_config import (
    DomainRandomizationConfig,
    EnvConfig,
    LoopCounterMode,
    ObservationConfig,
    SimulationConfig,
)
from f1tenth_gym.envs.f110_env import F110Env
from f1tenth_gym.envs.lidar import LiDARConfig
from f1tenth_gym.envs.observation import (
    FEATURE_PRESETS,
    ObservationType,
)
from f1tenth_gym.envs.observation.fields import BASE_FIELDS, DERIVED_FIELDS
from f1tenth_gym.envs.track import Track
from f1tenth_gym.jax.builder import build_core
from f1tenth_gym.jax.environment import (
    reset_core_from_state,
    step_core,
)
from f1tenth_gym.jax.gym_observation import GymObservationAdapter


def circle_track(count: int = 64, radius: float = 5.0) -> Track:
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return Track.from_refline(
        x=radius * np.cos(theta),
        y=radius * np.sin(theta),
        velx=np.full(count, 4.0),
    )


def observation_config(
    selected: ObservationType,
    features: tuple[str, ...] | None = None,
) -> ObservationConfig:
    return ObservationConfig(type=selected, features=features)


class TestGymObservationAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.track = circle_track()
        cls.config = EnvConfig(
            map_name=cls.track,
            num_agents=2,
            simulation_config=SimulationConfig(
                dynamics_model=DynamicModel.ST,
                max_laps=None,
            ),
            lidar_config=LiDARConfig(
                num_beams=7,
                range_max=4.0,
                noise_std=0.0,
                range_bias_std=0.0,
                dropout_prob=0.0,
            ),
            collision_check=CollisionCheckMode.NONE,
            render_enabled=False,
        )
        cls.bundle = build_core(cls.config, cls.track)
        # Both rows exercise every ST-derived field.  The first vehicle is in
        # reverse with a non-zero slip angle; the vehicles are farther apart
        # than LiDAR range so opponent occlusion does not complicate parity.
        cls.model_state = np.asarray(
            [
                [5.0, 0.0, 0.10, -2.0, np.pi / 2.0, 0.15, 0.35],
                [-5.0, 0.0, -0.08, 3.0, -np.pi / 2.0, -0.12, -0.25],
            ],
            dtype=np.float32,
        )
        reset = jax.jit(reset_core_from_state, static_argnums=3)
        cls.core_observation, cls.core_state = reset(
            jax.random.key(101),
            cls.model_state,
            cls.bundle.tables,
            cls.bundle.config,
            cls.bundle.params,
        )

    def _config_for(
        self,
        selected: ObservationType,
        features: tuple[str, ...] | None = None,
    ) -> EnvConfig:
        return self.config.with_updates(
            observation_config=observation_config(selected, features)
        )

    def _adapter(
        self,
        selected: ObservationType,
        features: tuple[str, ...] | None = None,
    ) -> GymObservationAdapter:
        config = self._config_for(selected, features)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return GymObservationAdapter.from_bundle(
                self.bundle, config.observation_config
            )

    def _assert_float32_tree(self, space: gym.Space, value) -> None:
        self.assertTrue(space.contains(value))
        if isinstance(space, gym.spaces.Dict):
            self.assertEqual(set(value), set(space.spaces))
            for key, child in value.items():
                self._assert_float32_tree(space.spaces[key], child)
            return
        self.assertIsInstance(value, np.ndarray)
        self.assertEqual(value.dtype, np.dtype(np.float32))
        self.assertEqual(value.shape, space.shape)

    def _assert_nested_close(self, actual, expected, atol: float = 0.0) -> None:
        self.assertEqual(set(actual), set(expected))
        for key in actual:
            if isinstance(actual[key], dict):
                self._assert_nested_close(actual[key], expected[key], atol=atol)
            else:
                np.testing.assert_allclose(
                    actual[key], expected[key], rtol=0.0, atol=atol
                )

    def test_every_observation_type_has_exact_keys_shapes_and_spaces(self):
        custom = ("state", "linear_vel_y", "scan", "sim_time")
        raw_keys = (
            "scans",
            "state",
            "standard_state",
            "collisions",
            "lap_times",
            "lap_counts",
            "sim_time",
            "frenet",
        )
        cases = {
            ObservationType.DIRECT: (None, None, raw_keys),
            ObservationType.DEFAULT: (None, BASE_FIELDS, None),
            ObservationType.FEATURES: (custom, custom, None),
            ObservationType.KINEMATIC_STATE: (
                None,
                FEATURE_PRESETS[ObservationType.KINEMATIC_STATE],
                None,
            ),
            ObservationType.DYNAMIC_STATE: (
                None,
                FEATURE_PRESETS[ObservationType.DYNAMIC_STATE],
                None,
            ),
            ObservationType.FRENET_DYNAMIC_STATE: (
                None,
                FEATURE_PRESETS[ObservationType.FRENET_DYNAMIC_STATE],
                None,
            ),
        }

        for selected, (features, expected_fields, expected_raw) in cases.items():
            with self.subTest(observation_type=selected.name):
                adapter = self._adapter(selected, features)
                packaged = adapter.package(self.core_observation)
                self.assertEqual(adapter.fields, expected_fields)
                if selected is ObservationType.DIRECT:
                    self.assertEqual(adapter.raw_keys, expected_raw)
                    self.assertEqual(tuple(packaged), expected_raw)
                else:
                    self.assertEqual(tuple(packaged), ("agent_0", "agent_1"))
                    for agent in packaged.values():
                        self.assertEqual(tuple(agent), expected_fields)
                self._assert_float32_tree(adapter.observation_space, packaged)

    def test_direct_preserves_the_public_meaning_change_warning(self):
        direct = observation_config(ObservationType.DIRECT)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            GymObservationAdapter.from_bundle(self.bundle, direct)
        self.assertTrue(
            any("changed meaning" in str(item.message) for item in caught)
        )

    def test_default_and_direct_omit_disabled_features(self):
        lidar_off = self.config.with_updates(
            lidar_config=replace(self.config.lidar_config, enabled=False)
        )
        lidar_bundle = build_core(lidar_off, self.track)
        lidar_observation, _state = reset_core_from_state(
            jax.random.key(105),
            self.model_state,
            lidar_bundle.tables,
            lidar_bundle.config,
            lidar_bundle.params,
        )
        for selected in (ObservationType.DEFAULT, ObservationType.DIRECT):
            config = lidar_off.with_updates(
                observation_config=observation_config(selected)
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                adapter = GymObservationAdapter.from_bundle(
                    lidar_bundle, config.observation_config
                )
            packaged = adapter.package(lidar_observation)
            keys = (
                tuple(packaged)
                if adapter.fields is None
                else tuple(packaged["agent_0"])
            )
            self.assertNotIn("scans", keys)
            self.assertNotIn("scan", keys)
            self._assert_float32_tree(adapter.observation_space, packaged)

        no_frenet = self.config.with_updates(
            simulation_config=replace(
                self.config.simulation_config,
                loop_counter=LoopCounterMode.WINDING_ANGLE,
                compute_frenet_frame=False,
            )
        )
        no_frenet_core = replace(self.bundle.config, frenet_enabled=False)
        no_frenet_bundle = replace(
            self.bundle, env_config=no_frenet, config=no_frenet_core
        )
        for selected in (ObservationType.DEFAULT, ObservationType.DIRECT):
            config = no_frenet.with_updates(
                observation_config=observation_config(selected)
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                adapter = GymObservationAdapter.from_bundle(
                    no_frenet_bundle, config.observation_config
                )
            packaged = adapter.package(self.core_observation)
            keys = (
                tuple(packaged)
                if adapter.fields is None
                else tuple(packaged["agent_0"])
            )
            self.assertNotIn("frenet", keys)
            self.assertNotIn("frenet_pose", keys)
            self._assert_float32_tree(adapter.observation_space, packaged)

    def test_explicit_disabled_feature_requests_fail_loudly(self):
        lidar_off = self.config.with_updates(
            lidar_config=replace(self.config.lidar_config, enabled=False),
            observation_config=observation_config(
                ObservationType.FEATURES, ("scan", "state")
            ),
        )
        lidar_bundle = build_core(lidar_off, self.track)
        with self.assertRaisesRegex(ValueError, "LiDAR is disabled"):
            GymObservationAdapter.from_bundle(lidar_bundle)

        no_frenet = self.config.with_updates(
            simulation_config=replace(
                self.config.simulation_config,
                loop_counter=LoopCounterMode.WINDING_ANGLE,
                compute_frenet_frame=False,
            ),
            observation_config=observation_config(
                ObservationType.FEATURES, ("frenet_pose", "state")
            ),
        )
        no_frenet_core = replace(self.bundle.config, frenet_enabled=False)
        no_frenet_bundle = replace(
            self.bundle, env_config=no_frenet, config=no_frenet_core
        )
        with self.assertRaisesRegex(ValueError, "does not compute the Frenet"):
            GymObservationAdapter.from_bundle(no_frenet_bundle)

    def test_custom_feature_selection_is_validated(self):
        invalid = (
            (None, "features.*specified"),
            ((), "at least one feature"),
            (("not_a_field",), "Unknown observation feature"),
        )
        for features, message in invalid:
            with self.subTest(features=features):
                config = self._config_for(ObservationType.FEATURES, features)
                with self.assertRaisesRegex(ValueError, message):
                    GymObservationAdapter.from_bundle(
                        self.bundle, config.observation_config
                    )

    def test_derived_fields_use_velocity_components_and_true_magnitude(self):
        adapter = self._adapter(ObservationType.FEATURES, DERIVED_FIELDS)
        packaged = adapter.package(self.core_observation)

        for index, agent_id in enumerate(adapter.agent_ids):
            row = self.model_state[index]
            speed, beta = row[3], row[6]
            expected = {
                "pose_x": row[0],
                "pose_y": row[1],
                "pose_theta": row[4],
                "linear_vel_x": speed * np.cos(beta),
                "linear_vel_y": speed * np.sin(beta),
                "linear_vel_magnitude": np.hypot(
                    speed * np.cos(beta), speed * np.sin(beta)
                ),
                "ang_vel_z": row[5],
                "delta": row[2],
                "beta": beta,
            }
            for field, value in expected.items():
                np.testing.assert_allclose(
                    packaged[agent_id][field], value, rtol=1.0e-6
                )
        self.assertLess(float(packaged["agent_0"]["linear_vel_x"]), 0.0)
        self.assertAlmostEqual(
            float(packaged["agent_0"]["linear_vel_magnitude"]), 2.0
        )
        self._assert_float32_tree(adapter.observation_space, packaged)

    def test_every_public_leaf_is_an_independent_copy(self):
        for selected in (ObservationType.DEFAULT, ObservationType.DIRECT):
            with self.subTest(observation_type=selected.name):
                adapter = self._adapter(selected)
                first = adapter.package(self.core_observation)
                second = adapter.package(self.core_observation)
                first_leaves = jax.tree.leaves(first)
                second_leaves = jax.tree.leaves(second)
                self.assertEqual(len(first_leaves), len(second_leaves))
                for original, replay in zip(
                    first_leaves, second_leaves, strict=True
                ):
                    self.assertFalse(np.shares_memory(original, replay))
                    snapshot = replay.copy()
                    original[...] = -123.0
                    np.testing.assert_array_equal(replay, snapshot)

    def test_state_presets_transfer_only_standard_state(self):
        irrelevant = object()
        projected = replace(
            self.core_observation,
            scans=irrelevant,
            state=irrelevant,
            collisions=irrelevant,
            frenet=irrelevant,
            lap_times=irrelevant,
            lap_counts=irrelevant,
            sim_time=irrelevant,
        )
        for selected in (
            ObservationType.KINEMATIC_STATE,
            ObservationType.DYNAMIC_STATE,
            ObservationType.FRENET_DYNAMIC_STATE,
        ):
            with self.subTest(observation_type=selected.name):
                adapter = self._adapter(selected)
                self.assertEqual(adapter.dependencies, ("standard_state",))
                packaged = adapter.package(projected)
                self._assert_float32_tree(
                    adapter.observation_space, packaged
                )

    def test_config_and_core_topology_must_match(self):
        mismatches = (
            (self.config.with_updates(num_agents=1), "dynamics topology"),
            (
                self.config.with_updates(
                    simulation_config=replace(
                        self.config.simulation_config,
                        dynamics_model=DynamicModel.KS,
                    )
                ),
                "dynamics topology",
            ),
            (
                self.config.with_updates(
                    lidar_config=replace(
                        self.config.lidar_config, num_beams=11
                    )
                ),
                "LiDAR topology",
            ),
            (
                self.config.with_updates(
                    lidar_config=replace(
                        self.config.lidar_config, angle_min=-1.0
                    )
                ),
                "LiDAR topology",
            ),
            (
                self.config.with_updates(
                    simulation_config=replace(
                        self.config.simulation_config,
                        loop_counter=LoopCounterMode.WINDING_ANGLE,
                        compute_frenet_frame=False,
                    )
                ),
                "Frenet topology",
            ),
        )
        for config, message in mismatches:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    GymObservationAdapter.from_bundle(
                        replace(self.bundle, env_config=config)
                    )

        with self.assertRaisesRegex(TypeError, "bundle must be a CoreBundle"):
            GymObservationAdapter.from_bundle(object())
        with self.assertRaisesRegex(TypeError, "bundle.env_config must be"):
            GymObservationAdapter.from_bundle(
                replace(self.bundle, env_config=object())
            )
        with self.assertRaisesRegex(TypeError, "observation_config must be"):
            GymObservationAdapter.from_bundle(self.bundle, object())
        with self.assertRaisesRegex(TypeError, "bundle.track must be a resolved Track"):
            GymObservationAdapter.from_bundle(
                replace(self.bundle, track=object())
            )

        mismatched_range = self.config.with_updates(
            lidar_config=replace(self.config.lidar_config, range_max=0.5)
        )
        with self.assertRaisesRegex(ValueError, "LiDAR range do not match"):
            GymObservationAdapter.from_bundle(
                replace(self.bundle, env_config=mismatched_range)
            )

    def test_space_uses_widest_randomized_limits_and_integrator_overshoot(self):
        params = self.config.params
        low = params.with_updates(v_min=-8.0, s_min=-0.55)
        high = params.with_updates(
            v_max=24.0,
            s_max=0.60,
            sv_max=4.0,
            a_max=16.0,
        )
        config = self.config.with_updates(
            simulation_config=replace(
                self.config.simulation_config,
                timestep=0.01,
                integrator_timestep=0.002,
            ),
            domain_randomization_config=DomainRandomizationConfig(
                enabled=True,
                low=low,
                high=high,
            ),
        )
        bundle = build_core(
            config,
            self.track,
            vehicle_params=config.params,
        )
        adapter = GymObservationAdapter.from_bundle(bundle)
        standard = adapter.observation_space["agent_0"]["std_state"]
        self.assertAlmostEqual(
            float(standard.low[2]), -0.55 - 4.0 * 0.002, places=6
        )
        self.assertAlmostEqual(
            float(standard.high[2]), 0.60 + 4.0 * 0.002, places=6
        )
        self.assertAlmostEqual(
            float(standard.low[3]), -8.0 - 16.0 * 0.002, places=6
        )
        self.assertAlmostEqual(
            float(standard.high[3]), 24.0 + 16.0 * 0.002, places=6
        )

    def test_jitted_core_outputs_package_with_lagged_observation_clock(self):
        adapter = self._adapter(ObservationType.DEFAULT)
        step = jax.jit(step_core, static_argnums=4)
        action = jnp.zeros((2, 2), dtype=jnp.float32)

        observation, state, _rewards, _events, metrics = step(
            jax.random.key(102),
            self.core_state,
            action,
            self.bundle.tables,
            self.bundle.config,
            self.bundle.params,
        )
        packaged = adapter.package(observation)
        self.assertEqual(float(packaged["agent_0"]["sim_time"]), 0.0)
        self.assertAlmostEqual(float(state.dynamics.sim_time), 0.01, places=7)
        self.assertAlmostEqual(float(metrics.episode.sim_time), 0.01, places=7)
        self._assert_float32_tree(adapter.observation_space, packaged)

        observation, state, *_tail = step(
            jax.random.key(103),
            state,
            action,
            self.bundle.tables,
            self.bundle.config,
            self.bundle.params,
        )
        packaged = adapter.package(observation)
        self.assertAlmostEqual(
            float(packaged["agent_0"]["sim_time"]), 0.01, places=7
        )
        self.assertAlmostEqual(float(state.dynamics.sim_time), 0.02, places=7)

    def test_every_layout_space_and_values_match_mutable_environment(self):
        cases = (
            (ObservationType.DEFAULT, None),
            (ObservationType.DIRECT, None),
            (
                ObservationType.FEATURES,
                ("scan", "state", "collision", "linear_vel_y", "sim_time"),
            ),
            (ObservationType.KINEMATIC_STATE, None),
            (ObservationType.DYNAMIC_STATE, None),
            (ObservationType.FRENET_DYNAMIC_STATE, None),
        )
        for selected, features in cases:
            with self.subTest(observation_type=selected.name):
                adapter = self._adapter(selected, features)
                expected = adapter.package(self.core_observation)
                config = self._config_for(selected, features)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    env = F110Env(config)
                try:
                    actual, _info = env.reset(
                        seed=101,
                        options={"states": self.model_state.copy()},
                    )
                    self.assertEqual(
                        adapter.observation_space, env.observation_space
                    )
                    self._assert_nested_close(actual, expected, atol=3.0e-3)
                    self.assertTrue(env.observation_space.contains(expected))
                    self.assertTrue(adapter.observation_space.contains(actual))
                finally:
                    env.close()

    def test_ks_packaging_matches_padded_mutable_derived_state(self):
        config = EnvConfig(
            map_name=self.track,
            simulation_config=SimulationConfig(
                dynamics_model=DynamicModel.KS,
                max_laps=None,
            ),
            observation_config=observation_config(
                ObservationType.FRENET_DYNAMIC_STATE
            ),
            lidar_config=LiDARConfig(enabled=False),
            collision_check=CollisionCheckMode.NONE,
            render_enabled=False,
        )
        bundle = build_core(config, self.track)
        model = np.asarray([[1.5, -0.4, 0.2, -2.0, 0.7]], dtype=np.float32)
        core, _state = reset_core_from_state(
            jax.random.key(104),
            model,
            bundle.tables,
            bundle.config,
            bundle.params,
        )
        adapter = GymObservationAdapter.from_bundle(bundle)
        expected = adapter.package(core)
        env = F110Env(config)
        try:
            actual, _info = env.reset(
                seed=104, options={"states": model.copy()}
            )
            self.assertEqual(adapter.observation_space, env.observation_space)
            self._assert_nested_close(actual, expected, atol=3.0e-3)
            self.assertEqual(float(expected["agent_0"]["ang_vel_z"]), 0.0)
            self.assertEqual(float(expected["agent_0"]["beta"]), 0.0)
            self.assertEqual(float(expected["agent_0"]["linear_vel_y"]), 0.0)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
