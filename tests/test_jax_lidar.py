"""Differential gates for the fixed-shape functional exact LiDAR path."""

import ast
from dataclasses import replace
import pathlib
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from f1tenth_gym.envs.collision_models import get_vertices
from f1tenth_gym.envs.dynamic_models import F1TENTH_VEHICLE_PARAMETERS
from f1tenth_gym.envs.lidar.config import LiDARConfig
from f1tenth_gym.envs.lidar.laser_models import ray_cast
from f1tenth_gym.envs.lidar.segment_scan import SegmentScanSimulator2D
from f1tenth_gym.envs.simulator import F110Simulator
from f1tenth_gym.envs.track import Track
from f1tenth_gym.jax.geometry import BodyParams, body_vertices
from f1tenth_gym.jax.lidar import (
    ScanConfig,
    ScanParams,
    beam_angles,
    clean_scan,
    lidar_poses,
    opponent_ranges,
)
from f1tenth_gym.jax.preprocess import build_scan_params, build_track_table
from f1tenth_gym.jax.track import TileTable, WallTable


HALF_FOV = 2.3561945
MAX_RANGE = 30.0


def model_states(poses, state_dim=7):
    """Place world poses into supported native-state columns."""
    poses = np.asarray(poses, dtype=np.float32)
    states = np.zeros((len(poses), state_dim), dtype=np.float32)
    states[:, 0] = poses[:, 0]
    states[:, 1] = poses[:, 1]
    states[:, 4] = poses[:, 2]
    return jnp.asarray(states)


def scan_config(num_agents, num_beams=120, angle_min=-HALF_FOV,
                angle_max=HALF_FOV):
    return ScanConfig(num_agents, num_beams, angle_min, angle_max)


class TestRigidTransforms(unittest.TestCase):
    def test_lidar_mount_matches_the_current_host_calculation(self):
        lidar = LiDARConfig(
            num_beams=4,
            base_link_to_lidar_tf=(0.31, -0.07, 0.23),
            noise_std=0.0,
        )
        params = ScanParams.from_lidar_config(lidar)
        poses = np.array(
            [[1.2, -3.4, 0.8], [-0.2, 4.1, -2.7]], dtype=np.float32
        )
        got = np.asarray(lidar_poses(model_states(poses), params))
        fake = type("SimulatorConfig", (), {})()
        fake.config = type("EnvConfig", (), {"lidar_config": lidar})()
        expected = np.stack(
            [F110Simulator._lidar_pose_from_base(fake, pose) for pose in poses]
        )
        np.testing.assert_allclose(got, expected, atol=1.0e-6)

    def test_body_vertices_match_the_host_with_the_cog_offset(self):
        vehicle = F1TENTH_VEHICLE_PARAMETERS.with_updates(
            collision_body_center_y=0.04
        )
        body = BodyParams.from_vehicle_parameters(vehicle)
        pose = np.array([2.3, -1.7, 1.1], dtype=np.float32)
        got = np.asarray(body_vertices(jnp.asarray(pose), body))
        offset_pose = pose.copy()
        dx = -vehicle.lr + vehicle.collision_body_center_x
        dy = vehicle.collision_body_center_y
        cosine, sine = np.cos(pose[2]), np.sin(pose[2])
        offset_pose[0] += dx * cosine - dy * sine
        offset_pose[1] += dx * sine + dy * cosine
        expected = get_vertices(offset_pose.astype(np.float64), vehicle.length,
                                vehicle.width)
        np.testing.assert_allclose(got, expected, atol=2.0e-7)


class TestFunctionalWallScan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.track = Track.from_track_name("Spielberg", 1.0)
        cls.vehicle = F1TENTH_VEHICLE_PARAMETERS
        cls.body = BodyParams.from_vehicle_parameters(cls.vehicle)
        cls.lidar = LiDARConfig(num_beams=120, noise_std=0.0)
        cls.config = scan_config(1, cls.lidar.num_beams,
                                 cls.lidar.angle_min, cls.lidar.angle_max)
        cls.table = build_track_table(
            cls.track, cls.vehicle, ray_max_range=cls.lidar.range_max
        )
        cls.params = build_scan_params(cls.lidar, cls.table)
        cls.host = SegmentScanSimulator2D(
            cls.lidar.num_beams,
            cls.lidar.field_of_view,
            angle_min=cls.lidar.angle_min,
            angle_max=cls.lidar.angle_max,
            std_dev=0.0,
            min_range=cls.lidar.range_min,
            max_range=cls.lidar.range_max,
        )
        cls.host.set_map(cls.track)

    def test_wall_ranges_match_the_live_segment_backend(self):
        indexes = (25, 130, 310, 505)
        poses = np.stack(
            [
                self.track.raceline.xs[list(indexes)],
                self.track.raceline.ys[list(indexes)],
                self.track.raceline.yaws[list(indexes)],
            ],
            axis=1,
        ).astype(np.float32)
        fake = type("SimulatorConfig", (), {})()
        fake.config = type("EnvConfig", (), {"lidar_config": self.lidar})()
        run = jax.jit(
            lambda state: clean_scan(
                state, self.table, self.body, self.config, self.params
            )
        )
        for pose in poses:
            got = np.asarray(run(model_states([pose]))[0])
            sensor_pose = F110Simulator._lidar_pose_from_base(fake, pose)
            expected = self.host.scan(sensor_pose, rng=None)
            np.testing.assert_allclose(got, expected, atol=2.0e-3)

    def test_an_empty_map_returns_max_range(self):
        blank = Track.from_track_name("Spielberg_blank", 1.0)
        table = build_track_table(blank, self.vehicle, ray_max_range=MAX_RANGE)
        params = build_scan_params(self.lidar, table)
        got = clean_scan(
            model_states([[0.0, 0.0, 0.0]]),
            table,
            self.body,
            self.config,
            params,
        )
        np.testing.assert_array_equal(
            np.asarray(got), np.full(got.shape, MAX_RANGE, dtype=np.float32)
        )

    def test_a_masked_candidate_cannot_alias_real_wall_zero(self):
        wall = WallTable(
            a=jnp.array([[1.0, -1.0]], dtype=jnp.float32),
            b=jnp.array([[1.0, 1.0]], dtype=jnp.float32),
            normals=jnp.array([[-1.0, 0.0]], dtype=jnp.float32),
            adjacency=jnp.zeros((1, 2), dtype=jnp.int32),
            adjacency_mask=jnp.zeros((1, 2), dtype=jnp.bool_),
            lengths=jnp.array([2.0], dtype=jnp.float32),
            mask=jnp.ones((1,), dtype=jnp.bool_),
        )
        tiles = TileTable(
            indices=jnp.zeros((1, 1, 1), dtype=jnp.int32),
            mask=jnp.zeros((1, 1, 1), dtype=jnp.bool_),
            origin=jnp.array([-10.0, -10.0], dtype=jnp.float32),
            tile_size=jnp.asarray(20.0, dtype=jnp.float32),
            reach=jnp.asarray(MAX_RANGE, dtype=jnp.float32),
        )
        table = replace(self.table, walls=wall, ray_tiles=tiles)
        config = scan_config(1, 1, 0.0, 1.0)
        params = replace(self.params, offset_x=0.0, offset_y=0.0,
                         offset_yaw=0.0)
        got = clean_scan(
            model_states([[0.0, 0.0, 0.0]]), table, self.body, config, params
        )
        self.assertEqual(float(got[0, 0]), MAX_RANGE)

    def test_one_beam_uses_angle_min(self):
        config = scan_config(1, 1, -0.73, 1.8)
        np.testing.assert_array_equal(
            np.asarray(beam_angles(config)), np.asarray([-0.73], dtype=np.float32)
        )

    def test_ray_table_reach_is_validated_on_the_host(self):
        short = replace(
            self.table,
            ray_tiles=replace(
                self.table.ray_tiles, reach=jnp.asarray(10.0, jnp.float32)
            ),
        )
        with self.assertRaisesRegex(ValueError, "exceeds the ray-table reach"):
            build_scan_params(self.lidar, short)


class TestOpponentOcclusion(unittest.TestCase):
    def test_random_opponents_match_the_current_brute_force_result(self):
        rng = np.random.default_rng(9)
        pose = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self_vertices = get_vertices(pose.astype(np.float64), 0.58, 0.31)
        for beams, field_of_view in ((181, np.pi), (241, 2.0 * np.pi)):
            config = scan_config(2, beams, -field_of_view / 2, field_of_view / 2)
            angles = np.asarray(beam_angles(config))
            run = jax.jit(
                lambda vertices: opponent_ranges(
                    jnp.asarray(pose), vertices, jnp.int32(0),
                    beam_angles(config), jnp.float32(MAX_RANGE)
                )
            )
            for _ in range(24):
                while True:
                    opponent_pose = np.array(
                        [rng.uniform(-5, 5), rng.uniform(-5, 5),
                         rng.uniform(-np.pi, np.pi)]
                    )
                    if np.linalg.norm(opponent_pose[:2]) > 0.4:
                        break
                other = get_vertices(opponent_pose, 0.58, 0.31)
                vertices = jnp.asarray(np.stack((self_vertices, other)), jnp.float32)
                got = np.asarray(run(vertices))
                expected = ray_cast(
                    pose.astype(np.float64),
                    np.full(beams, MAX_RANGE),
                    angles.astype(np.float64),
                    other,
                )
                np.testing.assert_allclose(got, expected, atol=2.0e-5)

    def test_multiple_opponents_are_pair_order_invariant(self):
        config = scan_config(3, 256)
        angles = beam_angles(config)
        pose = jnp.zeros(3, dtype=jnp.float32)
        bodies = np.stack(
            (
                get_vertices(np.array([0.0, 0.0, 0.0]), 0.58, 0.31),
                get_vertices(np.array([2.0, 0.5, 0.3]), 0.58, 0.31),
                get_vertices(np.array([1.2, -1.4, -0.6]), 0.58, 0.31),
            )
        )
        first = opponent_ranges(
            pose, jnp.asarray(bodies), jnp.int32(0), angles, MAX_RANGE
        )
        second = opponent_ranges(
            pose, jnp.asarray(bodies[[0, 2, 1]]), jnp.int32(0), angles, MAX_RANGE
        )
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))

    def test_a_body_behind_a_partial_sweep_does_not_occlude(self):
        config = scan_config(2, 181, -np.pi / 2, np.pi / 2)
        bodies = jnp.asarray(
            np.stack(
                (
                    get_vertices(np.array([0.0, 0.0, 0.0]), 0.58, 0.31),
                    get_vertices(np.array([-2.0, 0.0, 0.0]), 0.58, 0.31),
                )
            ),
            jnp.float32,
        )
        got = opponent_ranges(
            jnp.zeros(3), bodies, jnp.int32(0), beam_angles(config), MAX_RANGE
        )
        np.testing.assert_array_equal(
            np.asarray(got), np.full(got.shape, MAX_RANGE, dtype=np.float32)
        )


class TestTransformability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.track = Track.from_track_name("Spielberg", 1.0)
        cls.vehicle = F1TENTH_VEHICLE_PARAMETERS
        cls.table = build_track_table(cls.track, cls.vehicle,
                                      ray_max_range=MAX_RANGE)
        cls.body = BodyParams.from_vehicle_parameters(cls.vehicle)
        cls.config = scan_config(1, 48)
        cls.params = ScanParams.from_lidar_config(
            LiDARConfig(num_beams=48, noise_std=0.0)
        )
        cls.pose = np.array(
            [cls.track.raceline.xs[200], cls.track.raceline.ys[200],
             cls.track.raceline.yaws[200]],
            dtype=np.float32,
        )

    def test_clean_scan_jits_evaluates_shapes_and_has_a_finite_pose_gradient(self):
        run = jax.jit(
            lambda state: clean_scan(
                state, self.table, self.body, self.config, self.params
            )
        )
        state = model_states([self.pose])
        result = run(state)
        shaped = jax.eval_shape(run, state)
        self.assertEqual(result.shape, (1, 48))
        self.assertEqual(shaped.shape, result.shape)
        gradient = jax.grad(lambda value: jnp.sum(run(value)))(state)
        self.assertTrue(bool(jnp.all(jnp.isfinite(gradient))))

    def test_environment_vmap_accepts_different_traced_sensor_and_body_values(self):
        state = model_states([self.pose])
        states = jnp.stack((state, state))
        bodies = jax.tree.map(
            lambda value: jnp.asarray([value, value * 1.02]), self.body
        )
        params = jax.tree.map(
            lambda value: jnp.asarray([value, value]), self.params
        )
        params = replace(
            params,
            offset_x=jnp.asarray([self.params.offset_x,
                                  self.params.offset_x + 0.08]),
        )
        run = jax.jit(
            jax.vmap(
                lambda one_state, one_body, one_params: clean_scan(
                    one_state, self.table, one_body, self.config, one_params
                )
            )
        )
        got = run(states, bodies, params)
        self.assertEqual(got.shape, (2, 1, 48))
        self.assertFalse(np.allclose(np.asarray(got[0]), np.asarray(got[1])))

    def test_pure_scan_modules_do_not_import_numpy_gym_or_envs(self):
        root = pathlib.Path(__file__).resolve().parents[1] / "f1tenth_gym" / "jax"
        for name in ("geometry.py", "lidar.py", "lidar_kernels.py"):
            source = (root / name).read_text()
            tree = ast.parse(source)
            imported = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imported.append(("." * node.level) + (node.module or ""))
            self.assertFalse(any(value.startswith("numpy") for value in imported))
            self.assertFalse(any("envs" in value for value in imported))
            self.assertNotIn("gymnasium", source)


class TestValidation(unittest.TestCase):
    def test_static_shape_errors_are_named(self):
        for args in ((0, 4, -1.0, 1.0), (1, 0, -1.0, 1.0),
                     (1, 4, 1.0, 1.0)):
            with self.assertRaises(ValueError):
                ScanConfig(*args)
        config = scan_config(2, 4)
        vehicle = F1TENTH_VEHICLE_PARAMETERS
        track = build_track_table(
            Track.from_track_name("Spielberg", 1.0), vehicle,
            ray_max_range=MAX_RANGE,
        )
        body = BodyParams.from_vehicle_parameters(vehicle)
        params = ScanParams(MAX_RANGE, 0.0, 0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "model_state"):
            clean_scan(jnp.zeros((1, 7)), track, body, config, params)


if __name__ == "__main__":
    unittest.main()
