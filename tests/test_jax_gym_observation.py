"""Gymnasium packaging contracts for functional JAX observations."""

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
from f1tenth_gym.envs.jax_simulator import JaxSimulator
from f1tenth_gym.envs.jax_core import (
    reset_core_from_state,
    step_core,
)
from f1tenth_gym.envs.observation.jax_adapter import GymObservationAdapter


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
        cls.simulator = JaxSimulator(cls.config, cls.track)
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
            cls.simulator.tables,
            cls.simulator.config,
            cls.simulator.params,
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
            return GymObservationAdapter.from_simulator(
                self.simulator, config.observation_config
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
            GymObservationAdapter.from_simulator(self.simulator, direct)
        self.assertTrue(
            any("changed meaning" in str(item.message) for item in caught)
        )

    def test_default_and_direct_omit_disabled_lidar(self):
        lidar_off = self.config.with_updates(
            lidar_config=self.config.lidar_config.with_updates(enabled=False)
        )
        lidar_simulator = JaxSimulator(lidar_off, self.track)
        lidar_observation, _state = reset_core_from_state(
            jax.random.key(105),
            self.model_state,
            lidar_simulator.tables,
            lidar_simulator.config,
            lidar_simulator.params,
        )
        for selected in (ObservationType.DEFAULT, ObservationType.DIRECT):
            config = lidar_off.with_updates(
                observation_config=observation_config(selected)
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                adapter = GymObservationAdapter.from_simulator(
                    lidar_simulator, config.observation_config
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

    def test_explicit_disabled_lidar_request_fails_loudly(self):
        lidar_off = self.config.with_updates(
            lidar_config=self.config.lidar_config.with_updates(enabled=False),
            observation_config=observation_config(
                ObservationType.FEATURES, ("scan", "state")
            ),
        )
        lidar_simulator = JaxSimulator(lidar_off, self.track)
        with self.assertRaisesRegex(ValueError, "LiDAR is disabled"):
            GymObservationAdapter.from_simulator(lidar_simulator)

    def test_public_frenet_lap_config_keeps_frenet_observations_enabled(self):
        config = self.config.with_updates(
            simulation_config=SimulationConfig(
                dynamics_model=DynamicModel.ST,
                compute_frenet_frame=False,
                max_laps=None,
            ),
        )
        simulator = JaxSimulator(config, self.track)
        adapter = GymObservationAdapter.from_simulator(simulator)

        self.assertTrue(config.simulation_config.compute_frenet_frame)
        self.assertTrue(simulator.config.frenet_enabled)
        self.assertIn("frenet_pose", adapter.fields)

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
                    GymObservationAdapter.from_simulator(
                        self.simulator, config.observation_config
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
        for selected in (
            ObservationType.KINEMATIC_STATE,
            ObservationType.DYNAMIC_STATE,
            ObservationType.FRENET_DYNAMIC_STATE,
        ):
            with self.subTest(observation_type=selected.name):
                adapter = self._adapter(selected)
                self.assertEqual(adapter.dependencies, ("standard_state",))
                packaged = adapter.package(self.core_observation)
                self._assert_float32_tree(
                    adapter.observation_space, packaged
                )

    def test_public_inputs_and_real_simulator_topology_are_respected(self):
        with self.assertRaisesRegex(TypeError, "JaxSimulator"):
            GymObservationAdapter.from_simulator(object())
        with self.assertRaisesRegex(TypeError, "observation_config must be"):
            GymObservationAdapter.from_simulator(self.simulator, object())

        config = EnvConfig(
            map_name=self.track,
            num_agents=1,
            simulation_config=SimulationConfig(
                dynamics_model=DynamicModel.KS,
                max_laps=None,
            ),
            observation_config=observation_config(
                ObservationType.KINEMATIC_STATE
            ),
            lidar_config=LiDARConfig(enabled=False),
            collision_check=CollisionCheckMode.NONE,
            render_enabled=False,
        )
        simulator = JaxSimulator(config, self.track)
        adapter = GymObservationAdapter.from_simulator(simulator)
        self.assertEqual(adapter.state_dim, 5)
        self.assertEqual(adapter.agent_ids, ("agent_0",))
        self.assertFalse(adapter.scan_enabled)

        override = observation_config(
            ObservationType.FEATURES,
            ("state", "sim_time"),
        )
        overridden = GymObservationAdapter.from_simulator(
            simulator,
            override,
        )
        self.assertEqual(overridden.fields, override.features)
        self.assertIs(
            simulator.env_config.observation_config.type,
            ObservationType.KINEMATIC_STATE,
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
            simulation_config=self.config.simulation_config.with_updates(
                timestep=0.01,
                integrator_timestep=0.002,
            ),
            domain_randomization_config=DomainRandomizationConfig(
                enabled=True,
                low=low,
                high=high,
            ),
        )
        simulator = JaxSimulator(
            config,
            self.track,
            vehicle_params=config.params,
        )
        adapter = GymObservationAdapter.from_simulator(simulator)
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

    def test_space_covers_explicit_constructor_vehicle(self):
        low = self.config.params.with_updates(v_min=-8.0)
        high = self.config.params.with_updates(v_max=24.0)
        config = self.config.with_updates(
            domain_randomization_config=DomainRandomizationConfig(
                enabled=True,
                low=low,
                high=high,
            )
        )
        explicit = config.params.with_updates(v_max=40.0)
        simulator = JaxSimulator(
            config,
            self.track,
            vehicle_params=explicit,
        )

        adapter = GymObservationAdapter.from_simulator(simulator)
        standard = adapter.observation_space["agent_0"]["std_state"]
        expected_max = (
            explicit.v_max
            + explicit.a_max
            * config.simulation_config.integrator_timestep
        )

        self.assertEqual(simulator.effective_vehicle_params, explicit)
        self.assertEqual(simulator.space_vehicle_params.v_min, -8.0)
        self.assertEqual(simulator.space_vehicle_params.v_max, 40.0)
        self.assertEqual(float(simulator.randomization.high.v_max), 24.0)
        self.assertAlmostEqual(float(standard.high[3]), expected_max, places=6)

    def test_jitted_core_outputs_package_with_lagged_observation_clock(self):
        adapter = self._adapter(ObservationType.DEFAULT)
        step = jax.jit(step_core, static_argnums=4)
        action = jnp.zeros((2, 2), dtype=jnp.float32)

        observation, state, _rewards, _events, metrics = step(
            jax.random.key(102),
            self.core_state,
            action,
            self.simulator.tables,
            self.simulator.config,
            self.simulator.params,
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
            self.simulator.tables,
            self.simulator.config,
            self.simulator.params,
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
        simulator = JaxSimulator(config, self.track)
        model = np.asarray([[1.5, -0.4, 0.2, -2.0, 0.7]], dtype=np.float32)
        core, _state = reset_core_from_state(
            jax.random.key(104),
            model,
            simulator.tables,
            simulator.config,
            simulator.params,
        )
        adapter = GymObservationAdapter.from_simulator(simulator)
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
