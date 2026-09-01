"""Host simulator construction and exact-shape indexed-map routing."""

from dataclasses import replace
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from f1tenth_gym.envs import jax_simulator as simulator_module
from f1tenth_gym.envs.collision_models import CollisionCheckMode
from f1tenth_gym.envs.dynamic_models import F1TENTH_VEHICLE_PARAMETERS
from f1tenth_gym.envs.env_config import EnvConfig, RewardConfig, RewardMode
from f1tenth_gym.envs.episode import BuiltinRewardMode
from f1tenth_gym.envs.indexed_batching import (
    reset_indexed_batch,
    step_indexed_batch,
)
from f1tenth_gym.envs.jax_simulator import (
    IndexedCoreBucket,
    IndexedJaxSimulator,
    JaxSimulator,
)
from f1tenth_gym.envs.lidar import LiDARConfig
from f1tenth_gym.envs.track import Track


VEHICLE = F1TENTH_VEHICLE_PARAMETERS


def _circle_track(
    *,
    count: int = 64,
    radius: float = 5.0,
    center_x: float = 0.0,
) -> Track:
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return Track.from_refline(
        x=center_x + radius * np.cos(theta),
        y=radius * np.sin(theta),
        velx=np.full(count, 4.0),
    )


def _lightweight_config(**changes) -> EnvConfig:
    """Return an offline config that skips unused geometry preprocessing."""
    defaults = {
        "collision_check": CollisionCheckMode.NONE,
        "lidar_config": LiDARConfig(enabled=False, num_beams=7),
        "render_enabled": False,
    }
    defaults.update(changes)
    return EnvConfig(**defaults)


class TestIndexedJaxSimulator(unittest.TestCase):
    def test_repeated_identities_are_preprocessed_once_and_routed_locally(self):
        config = _lightweight_config()
        first = _circle_track()
        translated = _circle_track(center_x=30.0)
        tracks = (first, first, translated, first, translated)

        with mock.patch.object(
            simulator_module,
            "_make_tables",
            wraps=simulator_module._make_tables,
        ) as preprocess:
            simulator = IndexedJaxSimulator(
                config,
                tracks,
                max_table_bytes=123456,
                device="cpu",
            )

        self.assertEqual(preprocess.call_count, 2)
        self.assertTrue(
            all(
                call.kwargs["max_bytes"] == 123456
                for call in preprocess.mock_calls
            )
        )
        self.assertEqual(simulator.num_environments, 5)
        self.assertEqual(simulator.num_unique_tracks, 2)
        self.assertEqual(simulator.tracks, (first, translated))
        self.assertEqual(len(simulator.buckets), 1)

        bucket = simulator.buckets[0]
        self.assertIsInstance(bucket, IndexedCoreBucket)
        self.assertEqual(bucket.tracks, (first, translated))
        self.assertEqual(bucket.source_indices, (0, 1, 2, 3, 4))
        np.testing.assert_array_equal(bucket.map_indices, (0, 0, 1, 0, 1))
        self.assertEqual(bucket.tables.num_maps, 2)

        self.assertEqual(simulator.device.platform, "cpu")
        for leaf in jax.tree.leaves(
            (bucket.tables, simulator.params, simulator.randomization)
        ):
            self.assertEqual(leaf.device.platform, "cpu")

    def test_indexed_reset_and_step_follow_source_routing(self):
        config = _lightweight_config()
        first = _circle_track()
        translated = _circle_track(center_x=30.0)
        simulator = IndexedJaxSimulator(
            config,
            (first, first, translated, first, translated),
            device="cpu",
        )
        bucket = simulator.buckets[0]
        batch_size = len(bucket.source_indices)
        one_key = jax.random.key(901)
        keys = jnp.stack([one_key] * batch_size)

        observation, state = jax.jit(
            reset_indexed_batch,
            static_argnums=3,
        )(
            keys,
            bucket.map_indices,
            bucket.tables,
            simulator.config,
            simulator.params,
            simulator.randomization,
        )
        transition = jax.jit(
            step_indexed_batch,
            static_argnums=5,
        )(
            keys,
            bucket.map_indices,
            state,
            jnp.zeros((batch_size, config.num_agents, 2), dtype=jnp.float32),
            bucket.tables,
            simulator.config,
        )

        reset_xy = self._stitch_xy(simulator, bucket, observation.state)
        step_xy = self._stitch_xy(
            simulator,
            bucket,
            transition.observation.state,
        )
        for xy in (reset_xy, step_xy):
            np.testing.assert_allclose(xy[0], xy[1])
            np.testing.assert_allclose(xy[0], xy[3])
            np.testing.assert_allclose(xy[2], xy[4])
            self.assertAlmostEqual(float(xy[2, 0] - xy[0, 0]), 30.0, places=4)
        np.testing.assert_array_equal(
            transition.state.core.episode.elapsed_steps,
            np.ones((batch_size,), dtype=np.int32),
        )

    @staticmethod
    def _stitch_xy(simulator, bucket, model_state):
        stitched = np.empty(
            (simulator.num_environments, 2),
            dtype=np.float32,
        )
        stitched[np.asarray(bucket.source_indices)] = np.asarray(
            model_state[:, 0, :2]
        )
        return stitched

    def test_heterogeneous_exact_shapes_form_separate_compile_buckets(self):
        config = _lightweight_config()
        small = _circle_track(radius=4.0)
        large = _circle_track(radius=7.0)
        translated_small = _circle_track(radius=4.0, center_x=20.0)
        simulator = IndexedJaxSimulator(
            config,
            (small, large, small, translated_small, large),
        )

        self.assertEqual(len(simulator.buckets), 2)
        first, second = simulator.buckets
        self.assertEqual(first.tracks, (small, translated_small))
        self.assertEqual(first.source_indices, (0, 2, 3))
        np.testing.assert_array_equal(first.map_indices, (0, 0, 1))
        self.assertEqual(second.tracks, (large,))
        self.assertEqual(second.source_indices, (1, 4))
        np.testing.assert_array_equal(second.map_indices, (0, 0))

    def test_bucket_signature_covers_every_core_table_leaf(self):
        config = _lightweight_config()
        first = _circle_track()
        second = _circle_track(center_x=10.0)
        base = JaxSimulator(config, first).tables
        different_reset = replace(
            base,
            reset=replace(
                base.reset,
                start_indices=jnp.concatenate(
                    (base.reset.start_indices, base.reset.start_indices[:1])
                ),
            ),
        )

        with mock.patch.object(
            simulator_module,
            "_make_tables",
            side_effect=(base, different_reset),
        ):
            simulator = IndexedJaxSimulator(config, (first, second))

        self.assertEqual(len(simulator.buckets), 2)
        self.assertEqual(simulator.buckets[0].source_indices, (0,))
        self.assertEqual(simulator.buckets[1].source_indices, (1,))

    def test_single_and_indexed_simulators_share_vehicle_translation(self):
        config = _lightweight_config()
        vehicle = VEHICLE.with_updates(m=4.25)
        first = _circle_track()
        second = _circle_track(center_x=15.0)
        single = JaxSimulator(
            config,
            first,
            vehicle_params=vehicle,
            device="cpu",
        )
        indexed = IndexedJaxSimulator(
            config,
            (first, second),
            vehicle_params=vehicle,
            device=jax.devices("cpu")[0],
        )

        for simulator in (single, indexed):
            self.assertAlmostEqual(
                float(simulator.params.dynamics.vehicle.m),
                4.25,
            )
            self.assertAlmostEqual(
                float(simulator.randomization.nominal.m),
                4.25,
            )
            self.assertFalse(bool(simulator.randomization.enabled))
            self.assertEqual(simulator.device.platform, "cpu")

        bucket = indexed.buckets[0]
        keys = jax.random.split(
            jax.random.key(44),
            len(bucket.source_indices),
        )
        _observation, batch_state = reset_indexed_batch(
            keys,
            bucket.map_indices,
            bucket.tables,
            indexed.config,
            indexed.params,
            indexed.randomization,
        )
        np.testing.assert_allclose(
            batch_state.params.dynamics.vehicle.m,
            np.full((len(bucket.source_indices),), 4.25, dtype=np.float32),
        )

        with self.assertRaisesRegex(
            ValueError,
            "fixed vehicle envelope.*new JaxSimulator",
        ):
            indexed.params_for_vehicle(vehicle.with_updates(width=10.0))

    def test_track_validation_is_shared_by_indexed_construction(self):
        config = _lightweight_config()
        with self.assertRaisesRegex(ValueError, "at least one"):
            IndexedJaxSimulator(config, ())
        with self.assertRaisesRegex(TypeError, r"tracks\[0\].*Track"):
            IndexedJaxSimulator(config, (object(),))
        with self.assertRaisesRegex(TypeError, "resolved Track"):
            JaxSimulator(config, object())

        invalid = _circle_track()
        invalid.raceline = None
        with self.assertRaisesRegex(ValueError, "centerline and raceline"):
            IndexedJaxSimulator(config, (invalid,))

    def test_custom_reward_requires_the_host_adapter_fallback(self):
        custom = _lightweight_config(
            reward_config=RewardConfig(
                mode=RewardMode.CUSTOM,
                reward_fn=lambda *_args: 0.0,
            )
        )
        track = _circle_track()
        constructors = (
            lambda **kwargs: JaxSimulator(custom, track, **kwargs),
            lambda **kwargs: IndexedJaxSimulator(custom, (track,), **kwargs),
        )
        for construct in constructors:
            with self.subTest(constructor=construct):
                with self.assertRaisesRegex(ValueError, "adapter-only"):
                    construct()
                simulator = construct(
                    _custom_reward_fallback=BuiltinRewardMode.SURVIVAL,
                    device="cpu",
                )
                self.assertIs(
                    simulator.config.episode.reward_mode,
                    BuiltinRewardMode.SURVIVAL,
                )
                self.assertEqual(simulator.device.platform, "cpu")


if __name__ == "__main__":
    unittest.main()
