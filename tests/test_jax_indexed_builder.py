"""Host exact-shape routing for indexed functional map batches."""

from dataclasses import replace
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from f1tenth_gym.envs.dynamic_models import F1TENTH_VEHICLE_PARAMETERS
from f1tenth_gym.envs.env_config import RewardConfig, RewardMode
from f1tenth_gym.envs.track import Track
from f1tenth_gym.jax.builder import (
    IndexedCoreBucket,
    IndexedCoreBundle,
    build_core_tables,
    build_indexed_core,
)
from f1tenth_gym.jax.episode import BuiltinRewardMode
from f1tenth_gym.jax.indexed import reset_indexed_batch
from tests.test_jax_builder import lightweight_config


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


class TestIndexedCoreBuilder(unittest.TestCase):
    def test_repeated_identities_are_preprocessed_once_and_routed_locally(self):
        config = lightweight_config()
        first = _circle_track()
        translated = _circle_track(center_x=30.0)
        tracks = (first, first, translated, first, translated)

        with mock.patch(
            "f1tenth_gym.jax.builder.build_core_tables",
            wraps=build_core_tables,
        ) as preprocess:
            bundle = build_indexed_core(
                config,
                tracks,
                max_bytes=123456,
                target_device="cpu",
            )

        self.assertIsInstance(bundle, IndexedCoreBundle)
        self.assertEqual(preprocess.call_count, 2)
        self.assertTrue(
            all(call.kwargs["max_bytes"] == 123456 for call in preprocess.mock_calls)
        )
        self.assertEqual(bundle.num_environments, 5)
        self.assertEqual(bundle.num_unique_tracks, 2)
        self.assertEqual(len(bundle.buckets), 1)
        bucket = bundle.buckets[0]
        self.assertIsInstance(bucket, IndexedCoreBucket)
        self.assertEqual(bucket.tracks, (first, translated))
        self.assertEqual(bucket.source_indices, (0, 1, 2, 3, 4))
        np.testing.assert_array_equal(bucket.map_indices, (0, 0, 1, 0, 1))
        self.assertEqual(bucket.tables.num_maps, 2)
        for leaf in jax.tree.leaves(
            (bucket.tables, bundle.params, bundle.randomization)
        ):
            self.assertEqual(leaf.device.platform, "cpu")

        # The route can directly drive one indexed executable and stitch rows
        # back into the caller's original ordering.
        one_key = jax.random.key(901)
        keys = jnp.stack([one_key] * len(bucket.source_indices))
        observation, _state = jax.jit(
            reset_indexed_batch,
            static_argnums=3,
        )(
            keys,
            bucket.map_indices,
            bucket.tables,
            bundle.config,
            bundle.params,
            bundle.randomization,
        )
        stitched = np.empty((bundle.num_environments, 2), dtype=np.float32)
        stitched[np.asarray(bucket.source_indices)] = np.asarray(
            observation.state[:, 0, :2]
        )
        np.testing.assert_allclose(stitched[0], stitched[1])
        np.testing.assert_allclose(stitched[0], stitched[3])
        np.testing.assert_allclose(stitched[2], stitched[4])
        self.assertAlmostEqual(
            float(stitched[2, 0] - stitched[0, 0]), 30.0, places=4
        )

    def test_heterogeneous_exact_shapes_form_separate_compile_buckets(self):
        config = lightweight_config()
        small = _circle_track(radius=4.0)
        large = _circle_track(radius=7.0)
        translated_small = _circle_track(radius=4.0, center_x=20.0)
        bundle = build_indexed_core(
            config,
            (small, large, small, translated_small, large),
        )

        self.assertEqual(len(bundle.buckets), 2)
        first, second = bundle.buckets
        self.assertEqual(first.tracks, (small, translated_small))
        self.assertEqual(first.source_indices, (0, 2, 3))
        np.testing.assert_array_equal(first.map_indices, (0, 0, 1))
        self.assertEqual(second.tracks, (large,))
        self.assertEqual(second.source_indices, (1, 4))
        np.testing.assert_array_equal(second.map_indices, (0, 0))

    def test_signature_covers_complete_core_tables_not_only_track_geometry(self):
        config = lightweight_config()
        first = _circle_track()
        second = _circle_track(center_x=10.0)
        base = build_core_tables(config, first)
        different_reset = replace(
            base,
            reset=replace(
                base.reset,
                start_indices=jnp.concatenate(
                    (
                        base.reset.start_indices,
                        base.reset.start_indices[:1],
                    )
                ),
            ),
        )

        with mock.patch(
            "f1tenth_gym.jax.builder.build_core_tables",
            side_effect=(base, different_reset),
        ):
            bundle = build_indexed_core(config, (first, second))

        self.assertEqual(len(bundle.buckets), 2)
        self.assertEqual(bundle.buckets[0].source_indices, (0,))
        self.assertEqual(bundle.buckets[1].source_indices, (1,))

    def test_shared_params_accept_an_explicit_vehicle_draw(self):
        config = lightweight_config()
        vehicle = F1TENTH_VEHICLE_PARAMETERS.with_updates(m=4.25)
        bundle = build_indexed_core(
            config,
            (_circle_track(), _circle_track(center_x=15.0)),
            vehicle_params=vehicle,
        )

        self.assertAlmostEqual(float(bundle.params.transition.dynamics.m), 4.25)
        self.assertFalse(bool(bundle.randomization.enabled))

    def test_validation_and_custom_reward_fallback_match_single_map_builder(self):
        config = lightweight_config()
        with self.assertRaisesRegex(ValueError, "at least one"):
            build_indexed_core(config, ())
        with self.assertRaisesRegex(TypeError, r"tracks\[0\].*Track"):
            build_indexed_core(config, (object(),))

        invalid = _circle_track()
        invalid.raceline = None
        with self.assertRaisesRegex(ValueError, "centerline and raceline"):
            build_indexed_core(config, (invalid,))

        custom = lightweight_config(
            reward_config=RewardConfig(
                mode=RewardMode.CUSTOM,
                reward_fn=lambda *_args: 0.0,
            )
        )
        track = _circle_track()
        with self.assertRaisesRegex(ValueError, "adapter-only"):
            build_indexed_core(custom, (track,))
        bundle = build_indexed_core(
            custom,
            (track,),
            custom_reward_fallback=BuiltinRewardMode.SURVIVAL,
            target_device=jax.devices("cpu")[0],
        )
        self.assertIs(
            bundle.config.episode.reward_mode,
            BuiltinRewardMode.SURVIVAL,
        )
        self.assertEqual(bundle.device.platform, "cpu")


if __name__ == "__main__":
    unittest.main()
