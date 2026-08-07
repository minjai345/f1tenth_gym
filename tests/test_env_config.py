"""Tests for env_config validation."""
import unittest

from f1tenth_gym.envs.env_config import (
    EnvConfig,
    ControlConfig,
    SimulationConfig,
    ObservationConfig,
    ResetConfig,
    LoopCounterMode,
)
from f1tenth_gym.envs.lidar import LiDARConfig
from f1tenth_gym.envs.dynamic_models import F1TENTH_VEHICLE_PARAMETERS
from f1tenth_gym.envs.integrators import IntegratorType
from f1tenth_gym.envs.observation import ObservationType


class TestControlConfigValidation(unittest.TestCase):
    """Tests for ControlConfig validation."""

    def test_valid_config(self):
        """Test creating valid ControlConfig."""
        cfg = ControlConfig(steer_delay_steps=5)
        self.assertEqual(cfg.steer_delay_steps, 5)

    def test_negative_delay_steps(self):
        """Test that negative steer_delay_steps raises ValueError."""
        with self.assertRaises(ValueError):
            ControlConfig(steer_delay_steps=-1)

    def test_with_updates(self):
        """Test with_updates method."""
        cfg = ControlConfig(steer_delay_steps=3)
        updated = cfg.with_updates(steer_delay_steps=5)
        self.assertEqual(cfg.steer_delay_steps, 3)
        self.assertEqual(updated.steer_delay_steps, 5)


class TestSimulationConfigValidation(unittest.TestCase):
    """Tests for SimulationConfig validation."""

    def test_valid_config(self):
        """Test creating valid SimulationConfig."""
        cfg = SimulationConfig(timestep=0.02, integrator_timestep=0.01)
        self.assertEqual(cfg.timestep, 0.02)

    def test_zero_timestep(self):
        """Test that zero timestep raises ValueError."""
        with self.assertRaises(ValueError):
            SimulationConfig(timestep=0)

    def test_negative_timestep(self):
        """Test that negative timestep raises ValueError."""
        with self.assertRaises(ValueError):
            SimulationConfig(timestep=-0.01)

    def test_zero_integrator_timestep(self):
        """Test that zero integrator_timestep raises ValueError."""
        with self.assertRaises(ValueError):
            SimulationConfig(integrator_timestep=0)

    def test_negative_integrator_timestep(self):
        """Test that negative integrator_timestep raises ValueError."""
        with self.assertRaises(ValueError):
            SimulationConfig(integrator_timestep=-0.01)

    def test_zero_max_laps(self):
        """Test that zero max_laps raises ValueError."""
        with self.assertRaises(ValueError):
            SimulationConfig(max_laps=0)

    def test_negative_max_laps(self):
        """Test that negative max_laps raises ValueError."""
        with self.assertRaises(ValueError):
            SimulationConfig(max_laps=-1)

    def test_none_max_laps(self):
        """Test that None max_laps is allowed (infinite laps)."""
        cfg = SimulationConfig(max_laps=None)
        self.assertIsNone(cfg.max_laps)

    def test_frenet_enabled_with_frenet_loop_counter(self):
        """Test that frenet is auto-enabled for FRENET_BASED loop counter."""
        cfg = SimulationConfig(
            loop_counter=LoopCounterMode.FRENET_BASED,
            compute_frenet_frame=False,
        )
        updated = cfg.with_updates()
        self.assertTrue(updated.compute_frenet_frame)


class TestEnvConfigValidation(unittest.TestCase):
    """Tests for EnvConfig validation."""

    def test_valid_config(self):
        """Test creating valid EnvConfig."""
        cfg = EnvConfig(num_agents=2, ego_index=1)
        self.assertEqual(cfg.num_agents, 2)
        self.assertEqual(cfg.ego_index, 1)

    def test_zero_num_agents(self):
        """Test that zero num_agents raises ValueError."""
        with self.assertRaises(ValueError):
            EnvConfig(num_agents=0)

    def test_negative_num_agents(self):
        """Test that negative num_agents raises ValueError."""
        with self.assertRaises(ValueError):
            EnvConfig(num_agents=-1)

    def test_ego_index_out_of_range_high(self):
        """Test that ego_index >= num_agents raises ValueError."""
        with self.assertRaises(ValueError):
            EnvConfig(num_agents=2, ego_index=2)

    def test_ego_index_out_of_range_negative(self):
        """Test that negative ego_index raises ValueError."""
        with self.assertRaises(ValueError):
            EnvConfig(num_agents=2, ego_index=-1)

    def test_zero_map_scale(self):
        """Test that zero map_scale raises ValueError."""
        with self.assertRaises(ValueError):
            EnvConfig(map_scale=0)

    def test_negative_map_scale(self):
        """Test that negative map_scale raises ValueError."""
        with self.assertRaises(ValueError):
            EnvConfig(map_scale=-1.0)

    def test_invalid_params_type(self):
        """Test that non-VehicleParameters params raises TypeError."""
        with self.assertRaises(TypeError):
            EnvConfig(params="invalid")

    def test_invalid_control_config_type(self):
        """Test that non-ControlConfig control_config raises TypeError."""
        with self.assertRaises(TypeError):
            EnvConfig(control_config="invalid")

    def test_invalid_simulation_config_type(self):
        """Test that non-SimulationConfig simulation_config raises TypeError."""
        with self.assertRaises(TypeError):
            EnvConfig(simulation_config="invalid")

    def test_invalid_observation_config_type(self):
        """Test that non-ObservationConfig observation_config raises TypeError."""
        with self.assertRaises(TypeError):
            EnvConfig(observation_config="invalid")

    def test_invalid_reset_config_type(self):
        """Test that non-ResetConfig reset_config raises TypeError."""
        with self.assertRaises(TypeError):
            EnvConfig(reset_config="invalid")

    def test_invalid_lidar_config_type(self):
        """Test that non-LiDARConfig lidar_config raises TypeError."""
        with self.assertRaises(TypeError):
            EnvConfig(lidar_config="invalid")

    def test_with_updates(self):
        """Test with_updates method."""
        cfg = EnvConfig(num_agents=1)
        updated = cfg.with_updates(num_agents=3, ego_index=2)
        self.assertEqual(cfg.num_agents, 1)
        self.assertEqual(updated.num_agents, 3)
        self.assertEqual(updated.ego_index, 2)

    def test_type_coercion(self):
        """Test that numeric types are coerced correctly."""
        cfg = EnvConfig(
            seed=12345.0,  # float -> int
            map_scale=2,  # int -> float
            num_agents=2.0,  # float -> int
            ego_index=1.0,  # float -> int
        )
        self.assertIsInstance(cfg.seed, int)
        self.assertIsInstance(cfg.map_scale, float)
        self.assertIsInstance(cfg.num_agents, int)
        self.assertIsInstance(cfg.ego_index, int)


class TestObservationConfigValidation(unittest.TestCase):
    """Tests for ObservationConfig."""

    def test_valid_config(self):
        """Test creating valid ObservationConfig."""
        cfg = ObservationConfig()
        self.assertIsNotNone(cfg.type)

    def test_with_updates(self):
        """Test with_updates method."""
        cfg = ObservationConfig(type=ObservationType.FEATURES, features=("scan",))
        updated = cfg.with_updates(features=("scan", "pose"))
        self.assertEqual(cfg.features, ("scan",))
        self.assertEqual(updated.features, ("scan", "pose"))

    def test_features_requires_features_type(self):
        """`features` with a non-FEATURES type is a config error, not a silent no-op."""
        with self.assertRaises(ValueError):
            ObservationConfig(type=ObservationType.KINEMATIC_STATE, features=("pose_x",))
        # allowed with the FEATURES type
        ObservationConfig(type=ObservationType.FEATURES, features=("pose_x",))


class TestResetConfigValidation(unittest.TestCase):
    """Tests for ResetConfig."""

    def test_valid_config(self):
        """Test creating valid ResetConfig."""
        cfg = ResetConfig()
        self.assertIsNotNone(cfg.strategy)

    def test_with_updates(self):
        """Test with_updates method."""
        from f1tenth_gym.envs.reset import ResetStrategy

        cfg = ResetConfig(strategy=ResetStrategy.RL_GRID_STATIC)
        updated = cfg.with_updates(strategy=ResetStrategy.RL_RANDOM_STATIC)
        self.assertNotEqual(cfg.strategy, updated.strategy)


class TestLiDARConfigValidation(unittest.TestCase):
    """Tests for LiDARConfig validation."""

    def test_valid_config(self):
        """Test creating valid LiDARConfig."""
        import math
        cfg = LiDARConfig(
            num_beams=270,
            angle_min=-math.pi / 2,
            angle_max=math.pi / 2,
        )
        self.assertEqual(cfg.num_beams, 270)

    def test_angle_min_too_small(self):
        """Test that angle_min < -π raises ValueError (catches degrees instead of radians)."""
        with self.assertRaises(ValueError) as ctx:
            LiDARConfig(angle_min=-135.0, angle_max=135.0)
        self.assertIn("radians", str(ctx.exception))

    def test_angle_max_too_large(self):
        """Test that angle_max > π raises ValueError (catches degrees instead of radians)."""
        import math
        with self.assertRaises(ValueError) as ctx:
            LiDARConfig(angle_min=-math.pi / 2, angle_max=135.0)
        self.assertIn("radians", str(ctx.exception))

    def test_angle_min_greater_than_max(self):
        """Test that angle_min >= angle_max raises ValueError."""
        import math
        with self.assertRaises(ValueError):
            LiDARConfig(angle_min=math.pi / 2, angle_max=-math.pi / 2)

    def test_valid_sick_tim_config(self):
        """Test valid SICK TIM 571 config (270° FOV in radians)."""
        import math
        cfg = LiDARConfig(
            num_beams=819,
            angle_min=math.radians(-135.0),
            angle_max=math.radians(135.0),
            range_max=25.0,
            range_min=0.05,
        )
        self.assertAlmostEqual(cfg.angle_min, -2.356, places=2)
        self.assertAlmostEqual(cfg.angle_max, 2.356, places=2)

    def test_zero_num_beams(self):
        """Test that zero num_beams raises ValueError."""
        with self.assertRaises(ValueError):
            LiDARConfig(num_beams=0)

    def test_negative_range_min(self):
        """Test that negative range_min raises ValueError."""
        with self.assertRaises(ValueError):
            LiDARConfig(range_min=-1.0)

    def test_range_min_greater_than_max(self):
        """Test that range_min >= range_max raises ValueError."""
        with self.assertRaises(ValueError):
            LiDARConfig(range_min=30.0, range_max=10.0)

    def test_negative_noise_std(self):
        """Test that negative noise_std raises ValueError."""
        with self.assertRaises(ValueError):
            LiDARConfig(noise_std=-0.01)


class TestSubstepValidation(unittest.TestCase):
    """`timestep` must divide evenly into `integrator_timestep`.

    The check lives in ``F110Simulator.__init__``, not in ``SimulationConfig``,
    so it fires at ``gym.make`` rather than at config construction. It is
    validated against the substep ratio the simulator actually uses; an earlier
    version tested ``timestep % integrator_timestep`` against zero, which
    rejected exact multiples that IEEE-754 cannot represent (``0.03 % 0.01`` is
    ``0.00999999999999999847``).
    """

    def _substeps(self, timestep, integrator_timestep):
        from f1tenth_gym.envs.action import LongitudinalActionType, SteerActionType
        from f1tenth_gym.envs.dynamic_models import DynamicModel
        from f1tenth_gym.envs.integrators import integrator_from_type
        from f1tenth_gym.envs.simulator import F110Simulator

        cfg = EnvConfig(
            simulation_config=SimulationConfig(
                timestep=timestep, integrator_timestep=integrator_timestep
            ),
            render_enabled=False,
        )
        sim = F110Simulator(
            env_config=cfg,
            vehicle_params=cfg.params,
            model=DynamicModel.ST,
            dynamics_fn=DynamicModel.ST.f_dynamics,
            integrator_fn=integrator_from_type(cfg.simulation_config.integrator),
            longitudinal_type=LongitudinalActionType.SPEED,
            steering_type=SteerActionType.STEERING_ANGLE,
            track=None,
            seed=0,
        )
        return sim.substeps

    def test_float_inexact_multiples_are_accepted(self):
        """Regression: these are exact multiples in real arithmetic, not in IEEE-754."""
        for timestep, integrator_timestep, expected in [
            (0.03, 0.01, 3),
            (0.06, 0.02, 3),
            (0.3, 0.1, 3),
            (0.09, 0.03, 3),
        ]:
            with self.subTest(timestep=timestep, integrator_timestep=integrator_timestep):
                self.assertEqual(self._substeps(timestep, integrator_timestep), expected)

    def test_exactly_representable_multiples_still_accepted(self):
        for timestep, integrator_timestep, expected in [
            (0.01, 0.01, 1),
            (0.02, 0.01, 2),
            (0.05, 0.01, 5),
            (0.1, 0.01, 10),
            (0.03, 0.015, 2),
        ]:
            with self.subTest(timestep=timestep, integrator_timestep=integrator_timestep):
                self.assertEqual(self._substeps(timestep, integrator_timestep), expected)

    def test_non_multiples_are_rejected(self):
        for timestep, integrator_timestep in [(0.025, 0.01), (0.01, 0.003), (0.007, 0.002)]:
            with self.subTest(timestep=timestep, integrator_timestep=integrator_timestep):
                with self.assertRaises(ValueError):
                    self._substeps(timestep, integrator_timestep)

    def test_the_rule_is_enforced_at_simulator_build_not_config_build(self):
        """A bad pair constructs fine as config and only raises when the sim is built."""
        SimulationConfig(timestep=0.025, integrator_timestep=0.01)
        with self.assertRaises(ValueError):
            self._substeps(0.025, 0.01)


class TestMultiBodyParameterGate(unittest.TestCase):
    """`DynamicModel.MB` is only usable with a preset that populates its ABI.

    The two small-scale presets leave every multi-body field at ``nan``, which
    used to yield an all-NaN state with no exception. The gate fires at config
    construction — before the map download and the numba JIT — so the failure is
    cheap and names the cause.
    """

    def _cfg(self, params):
        from f1tenth_gym.envs.dynamic_models import DynamicModel

        return EnvConfig(
            params=params,
            simulation_config=SimulationConfig(dynamics_model=DynamicModel.MB),
            render_enabled=False,
        )

    def test_f1tenth_params_are_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self._cfg(F1TENTH_VEHICLE_PARAMETERS)
        self.assertIn("multi-body", str(ctx.exception))

    def test_f1fifth_params_are_rejected(self):
        from f1tenth_gym.envs.dynamic_models import F1FIFTH_VEHICLE_PARAMETERS

        with self.assertRaises(ValueError):
            self._cfg(F1FIFTH_VEHICLE_PARAMETERS)

    def test_fullscale_params_are_accepted(self):
        from f1tenth_gym.envs.dynamic_models import FULLSCALE_VEHICLE_PARAMETERS

        cfg = self._cfg(FULLSCALE_VEHICLE_PARAMETERS)
        self.assertIs(cfg.simulation_config.dynamics_model.name, "MB")

    def test_the_gate_does_not_fire_for_ks_or_st(self):
        """Only MB reads the multi-body block; KS/ST must stay unaffected."""
        from f1tenth_gym.envs.dynamic_models import DynamicModel

        for model in (DynamicModel.KS, DynamicModel.ST):
            with self.subTest(model=model.name):
                EnvConfig(
                    params=F1TENTH_VEHICLE_PARAMETERS,
                    simulation_config=SimulationConfig(dynamics_model=model),
                    render_enabled=False,
                )

    def test_missing_mb_parameters_reports_the_whole_block(self):
        self.assertEqual(len(F1TENTH_VEHICLE_PARAMETERS.missing_mb_parameters()), 69)
        from f1tenth_gym.envs.dynamic_models import FULLSCALE_VEHICLE_PARAMETERS

        self.assertEqual(FULLSCALE_VEHICLE_PARAMETERS.missing_mb_parameters(), ())
