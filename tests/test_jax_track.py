"""Fixed-shape JAX track preprocessing, transforms, bucketing, and resets."""

import ast
from pathlib import Path
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from f1tenth_gym.envs.dynamic_models import F1TENTH_VEHICLE_PARAMETERS
from f1tenth_gym.envs.track import Track
from f1tenth_gym.jax import (
    DynamicsConfig,
    ResetConfig,
    cartesian_to_frenet,
    evaluate_spline,
    frenet_to_cartesian,
    kinematic_single_track,
    reset_dynamics_state,
    rk4_step,
    sample_reset_poses,
    tile_candidates,
)
from f1tenth_gym.jax.preprocess import (
    bucket_track_tables,
    build_reset_table,
    build_track_table,
    build_track_table_set,
    compare_batch_layout,
)


ROOT = Path(__file__).resolve().parents[1]


def circle_track(count=80, radius=8.0):
    theta = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return Track.from_refline(
        x=radius * np.cos(theta),
        y=radius * np.sin(theta),
        velx=np.full(count, 4.0),
    )


class TestTrackPreprocessing(unittest.TestCase):
    def test_spline_values_and_coordinate_transforms_match_the_host(self):
        track = circle_track()
        tables = build_track_table(
            track, F1TENTH_VEHICLE_PARAMETERS, ray_max_range=5.0
        )
        evaluate = jax.jit(evaluate_spline)
        for s in np.linspace(0.0, track.centerline.s_frame_max, 13, endpoint=False):
            actual = np.asarray(evaluate(tables.centerline, s))
            expected = np.asarray(track.centerline.spline.spline(s))
            np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

        to_cartesian = jax.jit(frenet_to_cartesian)
        to_frenet = jax.jit(cartesian_to_frenet)
        for frenet in (
            np.array([1.3, 0.15, 0.0]),
            np.array([8.2, 0.25, -0.1]),
            np.array([track.centerline.s_frame_max - 0.2, -0.3, 0.2]),
        ):
            expected_pose = track.frenet_to_cartesian(*frenet)
            actual_pose = np.asarray(to_cartesian(tables.centerline, frenet))
            np.testing.assert_allclose(actual_pose, expected_pose, atol=3e-5)

            expected_frenet = track.cartesian_to_frenet(
                *expected_pose, use_s_guess=False
            )
            actual_frenet = np.asarray(to_frenet(tables.centerline, actual_pose))
            np.testing.assert_allclose(actual_frenet, expected_frenet, atol=2e-3)

    def test_empty_and_nonempty_wall_tables_are_masked_and_gatherable(self):
        empty_track = circle_track()
        empty = build_track_table(
            empty_track, F1TENTH_VEHICLE_PARAMETERS, ray_max_range=5.0
        )
        self.assertEqual(empty.walls.a.shape, (1, 2))
        self.assertFalse(bool(empty.walls.mask[0]))
        self.assertFalse(bool(jnp.any(empty.contact_tiles.mask)))
        self.assertFalse(bool(jnp.any(empty.ray_tiles.mask)))

        occupied_track = circle_track()
        occupied_track.occupancy_map[80:120, 80:120] = 0.0
        occupied = build_track_table(
            occupied_track, F1TENTH_VEHICLE_PARAMETERS, ray_max_range=5.0
        )
        self.assertTrue(bool(jnp.any(occupied.walls.mask)))
        self.assertTrue(bool(jnp.any(occupied.contact_tiles.mask)))
        point = (0.5 * (occupied.walls.a[0] + occupied.walls.b[0]))[None]
        indices, mask = jax.jit(tile_candidates)(occupied.contact_tiles, point)
        self.assertEqual(indices.shape[0], 1)
        self.assertTrue(bool(jnp.any(mask)))
        self.assertTrue(bool(jnp.all(indices[mask] < occupied.walls.a.shape[0])))

    def test_exact_shape_bucketing_avoids_implicit_worst_map_padding(self):
        tables = [
            build_track_table(
                circle_track(40), F1TENTH_VEHICLE_PARAMETERS, ray_max_range=4.0
            ),
            build_track_table(
                circle_track(90), F1TENTH_VEHICLE_PARAMETERS, ray_max_range=4.0
            ),
        ]
        buckets = bucket_track_tables(tables)
        report = compare_batch_layout(tables)
        self.assertEqual(len(buckets), 2)
        self.assertEqual(report.bucket_count, 2)
        self.assertGreater(report.global_padded_bytes, report.exact_bytes)
        self.assertGreater(report.global_padding_ratio, 1.0)

        shared = bucket_track_tables((tables[0], tables[0]))
        self.assertEqual(len(shared), 1)
        self.assertEqual(shared[0].source_indices, (0, 1))
        self.assertEqual(shared[0].tables.centerline.knots.shape[0], 2)

        first, second = circle_track(40), circle_track(90)
        table_set = build_track_table_set(
            (first, first, second),
            F1TENTH_VEHICLE_PARAMETERS,
            ray_max_range=4.0,
        )
        self.assertEqual(table_set.unique_count, 2)
        np.testing.assert_array_equal(table_set.map_indices, [0, 0, 1])
        self.assertEqual(len(table_set.buckets), 2)

    def test_only_the_host_preprocessor_imports_numpy(self):
        for filename in ("controls.py", "core.py", "dynamics.py", "integrators.py", "reset.py", "track.py"):
            tree = ast.parse((ROOT / "f1tenth_gym" / "jax" / filename).read_text())
            imports = {
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            }
            self.assertNotIn("numpy", imports, filename)
            self.assertNotIn("np", imports, filename)


class TestJaxReset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.track = circle_track(100)
        cls.table = build_reset_table(
            cls.track.raceline,
            min_dist=1.5,
            max_dist=2.5,
            start_width=1.0,
        )

    def test_sampling_is_key_driven_fixed_shape_and_jittable(self):
        config = ResetConfig(num_agents=3, move_laterally=False, shuffle=False)
        sample = jax.jit(sample_reset_poses, static_argnums=2)
        first = sample(jax.random.key(11), self.table, config)
        replay = sample(jax.random.key(11), self.table, config)
        other = sample(jax.random.key(12), self.table, config)
        self.assertEqual(first.shape, (3, 3))
        np.testing.assert_array_equal(first, replay)
        self.assertFalse(np.array_equal(first, other))

        waypoint_xy = np.asarray(self.table.waypoints)
        for pose in np.asarray(first):
            self.assertLess(float(np.linalg.norm(waypoint_xy - pose[:2], axis=1).min()), 1e-6)

    def test_lateral_shuffle_vmap_and_native_state_initialization(self):
        reset_config = ResetConfig(num_agents=2, move_laterally=True, shuffle=True)
        keys = jax.random.split(jax.random.key(5), 4)
        batched = jax.jit(
            jax.vmap(lambda key: sample_reset_poses(key, self.table, reset_config))
        )(keys)
        self.assertEqual(batched.shape, (4, 2, 3))
        self.assertTrue(bool(jnp.all(jnp.isfinite(batched))))

        dynamics_config = DynamicsConfig(
            num_agents=2,
            state_dim=5,
            dynamics_fn=kinematic_single_track,
            integrator_fn=rk4_step,
        )
        poses, state = jax.jit(
            reset_dynamics_state,
            static_argnums=(2, 3),
        )(jax.random.key(8), self.table, reset_config, dynamics_config)
        np.testing.assert_array_equal(state.model[:, :2], poses[:, :2])
        np.testing.assert_array_equal(state.model[:, 4], poses[:, 2])
        np.testing.assert_array_equal(state.model[:, 2:4], 0.0)
        self.assertEqual(float(state.sim_time), 0.0)

    def test_reset_preprocessing_validates_ranges(self):
        with self.assertRaisesRegex(ValueError, "max_dist"):
            build_reset_table(self.track.raceline, min_dist=2.0, max_dist=1.0)
        with self.assertRaisesRegex(ValueError, "start_width"):
            build_reset_table(
                self.track.raceline,
                min_dist=1.0,
                max_dist=2.0,
                start_width=0.0,
            )


if __name__ == "__main__":
    unittest.main()
