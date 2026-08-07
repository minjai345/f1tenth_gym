"""Tests for the action module."""
import unittest

import gymnasium as gym
import numpy as np

from f1tenth_gym.envs.action import (
    LongitudinalActionType,
    SteerActionType,
    accl_action,
    speed_action,
    steering_angle_action,
    steering_speed_action,
    longitudinal_action_from_type,
    steer_action_from_type,
    get_action_space,
    from_single_to_multi_action_space,
)
from f1tenth_gym.envs.dynamic_models import F1TENTH_VEHICLE_PARAMETERS


class TestActionTypes(unittest.TestCase):
    """Tests for action type enums."""

    def test_longitudinal_action_types(self):
        """Test LongitudinalActionType enum values."""
        self.assertEqual(LongitudinalActionType.ACCL, 1)
        self.assertEqual(LongitudinalActionType.SPEED, 2)

    def test_steer_action_types(self):
        """Test SteerActionType enum values."""
        self.assertEqual(SteerActionType.STEERING_ANGLE, 1)
        self.assertEqual(SteerActionType.STEERING_SPEED, 2)


class TestActionFunctions(unittest.TestCase):
    """Tests for action functions."""

    def setUp(self):
        self.params = F1TENTH_VEHICLE_PARAMETERS
        self.state = np.array([0.0, 0.0, 0.1, 5.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def test_accl_action(self):
        """Test direct acceleration action returns input unchanged."""
        action = 2.5
        result = accl_action(action, self.state, self.params)
        self.assertEqual(result, action)

    def test_speed_action(self):
        """Test speed action uses PID control."""
        target_speed = 10.0
        result = speed_action(target_speed, self.state, self.params)
        # Should return positive acceleration since target > current (5.0)
        self.assertGreater(result, 0)

    def test_steering_angle_action(self):
        """Test steering angle action uses PID control."""
        target_angle = 0.2
        result = steering_angle_action(target_angle, self.state, self.params)
        # Should return positive steering velocity since target > current (0.1)
        self.assertGreater(result, 0)

    def test_steering_speed_action(self):
        """Test direct steering speed action returns input unchanged."""
        action = 1.5
        result = steering_speed_action(action, self.state, self.params)
        self.assertEqual(result, action)


class TestActionFromType(unittest.TestCase):
    """Tests for action function lookup."""

    def test_longitudinal_action_from_type_accl(self):
        """Test getting acceleration action function."""
        fn = longitudinal_action_from_type(LongitudinalActionType.ACCL)
        self.assertEqual(fn, accl_action)

    def test_longitudinal_action_from_type_speed(self):
        """Test getting speed action function."""
        fn = longitudinal_action_from_type(LongitudinalActionType.SPEED)
        self.assertEqual(fn, speed_action)

    def test_longitudinal_action_from_type_invalid(self):
        """Test invalid longitudinal action type raises ValueError."""
        with self.assertRaises(ValueError):
            longitudinal_action_from_type(99)

    def test_steer_action_from_type_angle(self):
        """Test getting steering angle action function."""
        fn = steer_action_from_type(SteerActionType.STEERING_ANGLE)
        self.assertEqual(fn, steering_angle_action)

    def test_steer_action_from_type_speed(self):
        """Test getting steering speed action function."""
        fn = steer_action_from_type(SteerActionType.STEERING_SPEED)
        self.assertEqual(fn, steering_speed_action)

    def test_steer_action_from_type_invalid(self):
        """Test invalid steering action type raises ValueError."""
        with self.assertRaises(ValueError):
            steer_action_from_type(99)


class TestActionSpace(unittest.TestCase):
    """Tests for action space generation."""

    def setUp(self):
        self.params = F1TENTH_VEHICLE_PARAMETERS

    def test_action_space_accl_angle(self):
        """Test action space for acceleration + steering angle."""
        space = get_action_space(
            LongitudinalActionType.ACCL,
            SteerActionType.STEERING_ANGLE,
            self.params,
        )
        self.assertIsInstance(space, gym.spaces.Box)
        self.assertEqual(space.shape, (2,))
        # Steering limits
        self.assertAlmostEqual(space.low[0], self.params.s_min)
        self.assertAlmostEqual(space.high[0], self.params.s_max)
        # Acceleration limits
        self.assertAlmostEqual(space.low[1], -self.params.a_max)
        self.assertAlmostEqual(space.high[1], self.params.a_max)

    def test_action_space_speed_steer_speed(self):
        """Test action space for speed + steering speed."""
        space = get_action_space(
            LongitudinalActionType.SPEED,
            SteerActionType.STEERING_SPEED,
            self.params,
        )
        self.assertIsInstance(space, gym.spaces.Box)
        self.assertEqual(space.shape, (2,))
        # Steering speed limits
        self.assertAlmostEqual(space.low[0], self.params.sv_min)
        self.assertAlmostEqual(space.high[0], self.params.sv_max)
        # Velocity limits
        self.assertAlmostEqual(space.low[1], self.params.v_min)
        self.assertAlmostEqual(space.high[1], self.params.v_max)

    def test_action_space_invalid_longitudinal(self):
        """Test invalid longitudinal type raises ValueError."""
        with self.assertRaises(ValueError):
            get_action_space(99, SteerActionType.STEERING_ANGLE, self.params)

    def test_action_space_invalid_steer(self):
        """Test invalid steering type raises ValueError."""
        with self.assertRaises(ValueError):
            get_action_space(LongitudinalActionType.ACCL, 99, self.params)


class TestMultiAgentActionSpace(unittest.TestCase):
    """Tests for multi-agent action space conversion."""

    def test_single_to_multi_action_space(self):
        """Test converting single-agent to multi-agent action space."""
        single_space = gym.spaces.Box(
            low=np.array([-1.0, -2.0]),
            high=np.array([1.0, 2.0]),
            dtype=np.float32,
        )
        num_agents = 3
        multi_space = from_single_to_multi_action_space(single_space, num_agents)

        self.assertEqual(multi_space.shape, (num_agents, 2))
        for i in range(num_agents):
            np.testing.assert_array_almost_equal(multi_space.low[i], single_space.low)
            np.testing.assert_array_almost_equal(multi_space.high[i], single_space.high)


class TestSteeringPController(unittest.TestCase):
    """Pins ISSUES_PLAN.md #16: STEERING_ANGLE is a saturated P controller."""

    def test_proportional_saturation_and_relay_hatch(self):
        from f1tenth_gym.envs.dynamic_models.utils import pid_steer

        self.assertAlmostEqual(pid_steer(0.01, 0.0, 3.2, 10.0), 0.1, places=6)
        self.assertAlmostEqual(pid_steer(1.0, 0.0, 3.2, 10.0), 3.2, places=6)
        self.assertAlmostEqual(pid_steer(-1.0, 0.0, 3.2, 10.0), -3.2, places=6)
        # kp <= 0 is the legacy relay: full authority for any error > deadband
        self.assertAlmostEqual(pid_steer(0.01, 0.0, 3.2, -1.0), 3.2, places=6)
        self.assertAlmostEqual(pid_steer(0.0, 0.0, 3.2, -1.0), 0.0, places=6)

    def _tail_deltas(self, control_config):
        from f1tenth_gym.envs.env_config import ControlConfig, EnvConfig, SimulationConfig

        env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config=EnvConfig(
                control_config=control_config,
                simulation_config=SimulationConfig(max_laps=None),
                render_enabled=False,
            ),
        )
        env.reset(seed=42)
        deltas = []
        for _ in range(120):
            obs, *_ = env.step(np.array([[0.2, 1.0]], dtype=np.float32))
            deltas.append(float(obs["agent_0"]["std_state"][2]))
        env.close()
        return np.array(deltas[-30:])

    def test_default_settles_where_the_relay_limit_cycles(self):
        from f1tenth_gym.envs.env_config import ControlConfig

        settled = self._tail_deltas(ControlConfig())
        self.assertLess(float(np.abs(settled - 0.2).max()), 1e-3)
        relay = self._tail_deltas(ControlConfig(steer_kp=-1.0))
        self.assertGreater(float(np.ptp(relay)), 0.02, "relay hatch no longer limit-cycles")
