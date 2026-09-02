import math
import unittest
import warnings
from unittest import mock

import gymnasium as gym
import numpy as np

from f1tenth_gym.envs.action import LongitudinalActionType, SteerActionType
from f1tenth_gym.envs.collision_models import CollisionCheckMode
from f1tenth_gym.envs.contact import ContactConfig
from f1tenth_gym.envs.dynamic_models import DynamicModel
from f1tenth_gym.envs.env_config import (
    ControlConfig,
    EnvConfig,
    SimulationConfig,
    TerminationConfig,
)
from f1tenth_gym.envs.lidar import LiDARConfig
from f1tenth_gym.envs.track.walls import wall_segments

COAST = np.array([[0.0, 0.0]], dtype=np.float32)


def _gpu_available():
    import jax

    try:
        return bool(jax.devices("gpu"))
    except RuntimeError:
        return False


_HAS_GPU = _gpu_available()


def make_env(mode, lidar=True, friction=0.0, terminate=False):
    return gym.make(
        "f1tenth_gym:f1tenth-v0",
        config=EnvConfig(
            simulation_config=SimulationConfig(max_laps=None),
            termination_config=TerminationConfig(terminate_on_collision=terminate),
            control_config=ControlConfig(
                longitudinal_mode=LongitudinalActionType.ACCL,
                steering_mode=SteerActionType.STEERING_SPEED,
            ),
            lidar_config=LiDARConfig(enabled=lidar),
            contact_config=ContactConfig(friction=friction),
            collision_check=mode,
            render_enabled=False,
        ),
    )


def longest_wall(env):
    walls = wall_segments(env.unwrapped.track)
    k = int(np.argmax(walls.length))
    normal = walls.n[k].astype(np.float64)
    return normal, 0.5 * (walls.a[k] + walls.b[k]).astype(np.float64)


def drive_into_wall(env, degrees, speed=8.0, standoff=1.2, limit=200):
    """Aim at the longest wall at `degrees` off its normal; return (v_in, v_out)."""
    normal, mid = longest_wall(env)
    tangent = np.array([-normal[1], normal[0]])
    theta = math.radians(degrees)
    heading = -normal * math.cos(theta) + tangent * math.sin(theta)
    start = mid + normal * standoff
    state = np.zeros((1, 7))
    state[0] = [start[0], start[1], 0.0, speed, math.atan2(heading[1], heading[0]), 0.0, 0.0]
    env.reset(seed=7, options={"states": state})
    sim = env.unwrapped.sim
    for _ in range(limit):
        before = float(sim.state.standard_state[0][3])
        _obs, _r, _term, _trunc, info = env.step(COAST)
        if info["collisions"][0]:
            return before, float(sim.state.standard_state[0][3])
    return None, None


class TestGlancingContact(unittest.TestCase):
    """Geometric contact preserves the tangential component of motion."""

    def test_segment_contact_keeps_the_tangential_component(self):
        env = make_env(CollisionCheckMode.SEGMENT_CONTACT, friction=0.0)
        try:
            for degrees in (30, 60, 75):
                v_in, v_out = drive_into_wall(env, degrees)
                self.assertIsNotNone(v_in, f"{degrees} deg never made contact")
                self.assertGreater(v_out / v_in, 0.8 * math.sin(math.radians(degrees)))
        finally:
            env.close()


class TestModeWiring(unittest.TestCase):
    def test_none_disables_detection_and_response(self):
        env = make_env(CollisionCheckMode.NONE, terminate=True)
        try:
            normal, mid = longest_wall(env)
            state = np.zeros((1, 7))
            state[0] = [
                mid[0],
                mid[1],
                0.0,
                0.0,
                math.atan2(-normal[1], -normal[0]),
                0.0,
                0.0,
            ]
            env.reset(seed=7, options={"states": state})
            _obs, _reward, terminated, _truncated, info = env.step(COAST)
            self.assertFalse(terminated)
            self.assertEqual(float(info["collisions"][0]), 0.0)
        finally:
            env.close()

    def test_walls_are_detected_with_the_lidar_off(self):
        env = make_env(CollisionCheckMode.SEGMENT_CONTACT, lidar=False)
        try:
            v_in, _v_out = drive_into_wall(env, 0, speed=5.0, standoff=1.0, limit=150)
            self.assertIsNotNone(v_in, "no wall contact with the LiDAR disabled")
        finally:
            env.close()

    def test_the_flag_is_set_and_cleared(self):
        env = make_env(CollisionCheckMode.SEGMENT_CONTACT)
        try:
            normal, mid = longest_wall(env)
            state = np.zeros((1, 7))
            centre = mid - normal * (0.31 / 2 - 0.03)
            yaw = math.atan2(-normal[1], -normal[0])
            state[0] = [centre[0], centre[1], 0.0, 0.0, yaw, 0.0, 0.0]
            env.reset(seed=7, options={"states": state})
            _o, _r, _t, _tr, info = env.step(COAST)
            self.assertEqual(float(info["collisions"][0]), 1.0)

            clear = np.zeros((1, 7))
            free = mid + normal * 3.0
            clear[0] = [free[0], free[1], 0.0, 0.0, yaw, 0.0, 0.0]
            env.reset(seed=7, options={"states": clear})
            _o, _r, _t, _tr, info = env.step(COAST)
            self.assertEqual(float(info["collisions"][0]), 0.0)
        finally:
            env.close()

    def test_termination_still_works(self):
        env = make_env(CollisionCheckMode.SEGMENT_CONTACT, terminate=True)
        try:
            normal, mid = longest_wall(env)
            state = np.zeros((1, 7))
            centre = mid - normal * (0.31 / 2 - 0.03)
            state[0] = [centre[0], centre[1], 0.0, 0.0,
                        math.atan2(-normal[1], -normal[0]), 0.0, 0.0]
            env.reset(seed=7, options={"states": state})
            _o, _r, terminated, _tr, _i = env.step(COAST)
            self.assertTrue(terminated)
        finally:
            env.close()

    def test_the_car_never_ends_up_inside_a_wall(self):
        from f1tenth_gym.envs.track.walls import _occupied_at

        env = make_env(CollisionCheckMode.SEGMENT_CONTACT, friction=0.6)
        try:
            track = env.unwrapped.track
            occupied = track.occupancy_map == 0.0
            res = float(track.spec.resolution)
            origin = tuple(float(v) for v in track.spec.origin[:3])
            normal, mid = longest_wall(env)
            theta = math.radians(50)
            tangent = np.array([-normal[1], normal[0]])
            heading = -normal * math.cos(theta) + tangent * math.sin(theta)
            start = mid + normal * 1.2
            state = np.zeros((1, 7))
            state[0] = [start[0], start[1], 0.0, 7.0,
                        math.atan2(heading[1], heading[0]), 0.0, 0.0]
            env.reset(seed=7, options={"states": state})
            sim = env.unwrapped.sim
            worst = False
            for _ in range(300):
                env.step(COAST)
                centre = sim.state.standard_state[0][:2].astype(np.float64)
                worst |= bool(_occupied_at(centre[None], occupied, res, origin)[0])
            self.assertFalse(worst, "the car's centre entered an occupied cell")
        finally:
            env.close()


class TestConfigGuards(unittest.TestCase):
    def test_multi_body_is_refused_at_config_build(self):
        with self.assertRaises(ValueError) as caught:
            EnvConfig(
                collision_check=CollisionCheckMode.SEGMENT_CONTACT,
                simulation_config=SimulationConfig(dynamics_model=DynamicModel.MB),
            )
        self.assertIn("MB", str(caught.exception))

    def test_kinematic_contact_warns_about_diagonals(self):
        """KS projects an angled response onto its course; say so, do not refuse."""
        for config in (
            dict(collision_check=CollisionCheckMode.SEGMENT_CONTACT),
            dict(),  # SEGMENT_CONTACT is the default, so plain KS warns too
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                EnvConfig(
                    simulation_config=SimulationConfig(dynamics_model=DynamicModel.KS),
                    **config,
                )
            messages = [str(w.message) for w in caught]
            self.assertTrue(any("diagonal" in m for m in messages), messages)

    def test_the_warning_is_specific_to_kinematic_segment_contact(self):
        for config in (
            dict(
                simulation_config=SimulationConfig(dynamics_model=DynamicModel.ST),
                collision_check=CollisionCheckMode.SEGMENT_CONTACT,
            ),
            dict(
                simulation_config=SimulationConfig(dynamics_model=DynamicModel.KS),
                collision_check=CollisionCheckMode.NONE,
            ),
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                EnvConfig(**config)
            self.assertEqual([str(w.message) for w in caught], [])

    def test_a_raw_int_is_coerced_not_silently_mishandled(self):
        for value, expected in (
            (0, CollisionCheckMode.NONE),
            (3, CollisionCheckMode.SEGMENT_CONTACT),
        ):
            self.assertIs(EnvConfig(collision_check=value).collision_check, expected)
        for invalid in (1, 2, 99):
            with self.assertRaises(ValueError):
                EnvConfig(collision_check=invalid)

    def test_contact_is_the_default_mode(self):
        self.assertIs(EnvConfig().collision_check, CollisionCheckMode.SEGMENT_CONTACT)

    def test_multi_body_is_named_as_unsupported(self):
        with self.assertRaises(ValueError) as caught:
            EnvConfig(
                simulation_config=SimulationConfig(dynamics_model=DynamicModel.MB),
            )
        message = str(caught.exception)
        self.assertIn("unsupported", message)
        self.assertIn("DynamicModel.ST", message)

    def test_the_supported_modes_are_reachable(self):
        for mode in (CollisionCheckMode.NONE, CollisionCheckMode.SEGMENT_CONTACT):
            self.assertIs(EnvConfig(collision_check=mode).collision_check, mode)

    def test_the_section_round_trips(self):
        cfg = EnvConfig(contact_config=ContactConfig(restitution=0.25))
        self.assertIsInstance(cfg.contact_config, ContactConfig)
        self.assertEqual(cfg.contact_config.restitution, 0.25)
        self.assertIsInstance(hash(cfg), int)

    def test_a_non_config_section_is_rejected(self):
        with self.assertRaises(TypeError):
            EnvConfig(contact_config={"restitution": 0.5})


class TestDevicePlacement(unittest.TestCase):
    """Where the kernels run, and that asking for the impossible says so."""

    def test_only_the_two_known_devices_are_accepted(self):
        for name in ("cpu", "gpu"):
            self.assertEqual(ContactConfig(device=name).device, name)
        for bad in ("default", "cuda", "tpu", "", None):
            with self.assertRaises(ValueError):
                ContactConfig(device=bad)

    def test_the_default_is_cpu(self):
        self.assertEqual(ContactConfig().device, "cpu")

    def test_cpu_placement_is_honoured(self):
        from f1tenth_gym.envs.contact import adapter

        cpu = adapter.resolve_device("cpu")
        self.assertEqual(cpu.platform, "cpu")

        # JAX may cache CPU and then raise while initializing an unrelated GPU
        # plugin. The cached requested backend remains valid on the retry.
        with mock.patch.object(
            adapter.jax,
            "devices",
            side_effect=[RuntimeError("CUDA plugin failed"), [cpu]],
        ):
            self.assertIs(adapter.resolve_device("cpu"), cpu)

    def test_an_absent_backend_is_named_not_silently_swapped(self):
        """A quiet fallback would cost 10x with nothing said."""
        from f1tenth_gym.envs.contact.adapter import resolve_device

        with self.assertRaises(ValueError) as caught:
            resolve_device("nonexistent-backend")
        message = str(caught.exception)
        self.assertIn("nonexistent-backend", message)
        self.assertIn("available", message)

    @unittest.skipUnless(_HAS_GPU, "no GPU backend visible to JAX")
    def test_the_device_does_not_change_the_physics(self):
        states = []
        for device in ("cpu", "gpu"):
            env = gym.make(
                "f1tenth_gym:f1tenth-v0",
                config=EnvConfig(
                    simulation_config=SimulationConfig(max_laps=None),
                    termination_config=TerminationConfig(terminate_on_collision=False),
                    contact_config=ContactConfig(device=device),
                    collision_check=CollisionCheckMode.SEGMENT_CONTACT,
                    render_enabled=False,
                ),
            )
            env.reset(seed=7)
            for _ in range(150):
                env.step(np.array([[0.28, 5.0]], dtype=np.float32))
            states.append(env.unwrapped.sim.state.state[0].copy())
            env.close()
        np.testing.assert_allclose(states[0], states[1], atol=1e-4)


if __name__ == "__main__":
    unittest.main()
