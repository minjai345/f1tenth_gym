"""Configuration and public-environment gates for exact segment LiDAR."""

import math
import unittest

import gymnasium as gym

from f1tenth_gym.envs.env_config import EnvConfig
import f1tenth_gym.envs.lidar as lidar
from f1tenth_gym.envs.lidar import LiDARConfig
import f1tenth_gym.envs.lidar.config as lidar_config_module


class LiDARConfigTests(unittest.TestCase):
    """Tests for the immutable LiDAR configuration."""

    def test_from_fov_sets_a_symmetric_angle_pair(self):
        config = LiDARConfig.from_fov(2.0)
        self.assertAlmostEqual(config.angle_min, -1.0)
        self.assertAlmostEqual(config.angle_max, 1.0)
        self.assertAlmostEqual(config.field_of_view, 2.0)

    def test_from_fov_forwards_other_fields(self):
        config = LiDARConfig.from_fov(2.0, num_beams=540, noise_std=0.0)
        self.assertEqual(config.num_beams, 540)
        self.assertEqual(config.noise_std, 0.0)

    def test_explicit_angles(self):
        config = LiDARConfig(angle_min=-0.5, angle_max=1.5)
        self.assertAlmostEqual(config.angle_min, -0.5)
        self.assertAlmostEqual(config.angle_max, 1.5)

    def test_angle_increment(self):
        config = LiDARConfig(num_beams=101, angle_min=-1.0, angle_max=1.0)
        self.assertAlmostEqual(config.angle_increment, 0.02)

    def test_field_of_view_always_matches_the_angles(self):
        config = LiDARConfig(angle_min=-0.5, angle_max=1.5)
        self.assertAlmostEqual(config.field_of_view, 2.0)

    def test_field_of_view_is_read_only(self):
        with self.assertRaises(AttributeError):
            LiDARConfig().field_of_view = 1.0

    def test_field_of_view_is_not_a_constructor_argument(self):
        with self.assertRaises(TypeError):
            LiDARConfig(field_of_view=2.0)

    def test_backend_selector_is_removed(self):
        with self.assertRaises(TypeError):
            LiDARConfig(backend="raster")
        self.assertFalse(hasattr(lidar_config_module, "ScanBackend"))
        self.assertFalse(hasattr(lidar, "ScanSimulator2D"))

    def test_over_determined_geometry_raises(self):
        with self.assertRaises(ValueError):
            LiDARConfig.from_fov(5.0, angle_min=-0.5, angle_max=0.5)
        with self.assertRaises(ValueError):
            LiDARConfig.from_fov(3.0, angle_min=-1.0)

    def test_non_positive_fov_raises(self):
        for bad in (0.0, -1.0):
            with self.assertRaises(ValueError):
                LiDARConfig.from_fov(bad)

    def test_with_updates_of_an_angle_re_derives_the_fov(self):
        config = LiDARConfig().with_updates(angle_min=-1.0)
        self.assertAlmostEqual(config.angle_min, -1.0)
        self.assertAlmostEqual(config.angle_max, 2.3561945, places=7)
        self.assertAlmostEqual(config.field_of_view, 3.3561945, places=7)

    def test_with_updates_of_another_field_leaves_geometry_alone(self):
        base = LiDARConfig()
        config = base.with_updates(num_beams=540)
        self.assertEqual(config.num_beams, 540)
        self.assertAlmostEqual(config.angle_min, base.angle_min)
        self.assertAlmostEqual(config.angle_max, base.angle_max)
        self.assertAlmostEqual(config.field_of_view, base.field_of_view)

    def test_default_geometry_is_unchanged(self):
        config = LiDARConfig()
        self.assertAlmostEqual(config.field_of_view, 4.712389, places=6)
        self.assertAlmostEqual(config.angle_min, -2.3561945, places=7)
        self.assertAlmostEqual(config.angle_max, 2.3561945, places=7)
        self.assertAlmostEqual(
            config.angle_increment, 0.004367367006487488, places=15
        )

    def test_maximum_range_alias(self):
        config = LiDARConfig(range_max=15.0)
        self.assertEqual(config.maximum_range, 15.0)
        self.assertEqual(config.maximum_range, config.range_max)

    def test_validation_angle_min_greater_than_max(self):
        with self.assertRaises(ValueError):
            LiDARConfig(angle_min=1.0, angle_max=0.5)

    def test_validation_range_min_greater_than_max(self):
        with self.assertRaises(ValueError):
            LiDARConfig(range_min=10.0, range_max=5.0)

    def test_validation_negative_range_min(self):
        with self.assertRaises(ValueError):
            LiDARConfig(range_min=-1.0)

    def test_with_updates(self):
        config = LiDARConfig(num_beams=100, range_max=20.0)
        updated = config.with_updates(num_beams=200, range_max=15.0)
        self.assertEqual(config.num_beams, 100)
        self.assertEqual(config.range_max, 20.0)
        self.assertEqual(updated.num_beams, 200)
        self.assertEqual(updated.range_max, 15.0)
        self.assertEqual(updated.enabled, config.enabled)
        self.assertEqual(updated.noise_std, config.noise_std)


class LiDARIntegrationTests(unittest.TestCase):
    def test_custom_lidar_in_env(self):
        config = EnvConfig(
            map_name="Spielberg",
            num_agents=1,
            lidar_config=LiDARConfig(
                angle_min=-math.radians(90),
                angle_max=math.radians(90),
                num_beams=180,
                range_min=0.0,
                range_max=10.0,
            ),
            render_enabled=False,
        )
        env = gym.make("f1tenth_gym:f1tenth-v0", config=config)
        try:
            observation, _info = env.reset(seed=4)
            scan = observation["agent_0"]["scan"]
            self.assertEqual(scan.shape, (180,))
            self.assertLessEqual(float(scan.max()), 10.0 + 0.001)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
