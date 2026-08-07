"""DomainRandomizationConfig: per-episode vehicle-param randomization."""
import unittest

import gymnasium as gym
import numpy as np

from f1tenth_gym.envs.env_config import (
    EnvConfig, SimulationConfig, DomainRandomizationConfig,
)


def _mk(dr):
    return gym.make(
        "f1tenth_gym:f1tenth-v0",
        config=EnvConfig(
            simulation_config=SimulationConfig(max_laps=None),
            domain_randomization_config=dr,
            render_enabled=False,
        ),
    )


class TestDomainRandomization(unittest.TestCase):
    def test_disabled_keeps_params_constant(self):
        env = _mk(DomainRandomizationConfig())  # disabled
        vals = []
        for s in range(4):
            env.reset(seed=s)
            vals.append(float(env.unwrapped.sim.vehicle_params.m))
        self.assertEqual(len(set(vals)), 1)
        env.close()

    def test_samples_within_range_and_varies(self):
        env = _mk(DomainRandomizationConfig(
            enabled=True, param_ranges={"m": (3.0, 4.0), "mu": (0.9, 1.1)}
        ))
        ms, mus, arrays = [], [], []
        for s in range(8):
            env.reset(seed=s)
            p = env.unwrapped.sim.vehicle_params
            ms.append(float(p.m))
            mus.append(float(p.mu))
            arrays.append(env.unwrapped.sim.params_array.copy())
        self.assertTrue(all(3.0 <= m <= 4.0 for m in ms))
        self.assertTrue(all(0.9 <= mu <= 1.1 for mu in mus))
        self.assertGreater(len(set(ms)), 1, "mass did not vary")
        # the params_array (what the njit dynamics index) must actually change
        self.assertFalse(np.array_equal(arrays[0], arrays[1]))
        env.close()

    def test_reproducible_with_reset_seed(self):
        env = _mk(DomainRandomizationConfig(enabled=True, param_ranges={"m": (3.0, 4.0)}))
        env.reset(seed=7)
        a = env.unwrapped.sim.params_array.copy()
        env.reset(seed=7)
        b = env.unwrapped.sim.params_array.copy()
        np.testing.assert_array_equal(a, b)
        env.close()

    def test_validation(self):
        with self.assertRaises(ValueError):
            DomainRandomizationConfig(param_ranges={"not_a_param": (1.0, 2.0)})
        with self.assertRaises(ValueError):
            DomainRandomizationConfig(param_ranges={"m": (4.0, 3.0)})  # low > high


if __name__ == "__main__":
    unittest.main()


class TestWidestParamSpaces(unittest.TestCase):
    """ISSUES_PLAN.md #6: spaces are a fixed superset of every DR episode."""

    def test_randomized_limits_stay_inside_the_spaces(self):
        from f1tenth_gym.envs.dynamic_models import VehicleParamRanges

        cfg = EnvConfig(
            domain_randomization_config=DomainRandomizationConfig(
                enabled=True,
                param_ranges=VehicleParamRanges(s_max=(1.2, 1.4), s_min=(-1.4, -1.2)),
            ),
            render_enabled=False,
        )
        env = gym.make("f1tenth_gym:f1tenth-v0", config=cfg)
        obs, _ = env.reset(seed=1)
        for t in range(60):
            action = np.array([[1.4 if t % 2 else -1.4, 3.0]], dtype=np.float32)
            obs, *_ = env.step(action)
            self.assertTrue(env.observation_space.contains(obs), f"step {t} left the space")
        env.close()

    def test_disabled_dr_leaves_spaces_untouched(self):
        plain = gym.make("f1tenth_gym:f1tenth-v0", config=EnvConfig(render_enabled=False))
        off = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config=EnvConfig(
                domain_randomization_config=DomainRandomizationConfig(enabled=False),
                render_enabled=False,
            ),
        )
        self.assertEqual(plain.observation_space, off.observation_space)
        self.assertEqual(plain.action_space, off.action_space)
        plain.close()
        off.close()

    def test_update_params_rebuilds_the_observation_space(self):
        env = gym.make("f1tenth_gym:f1tenth-v0", config=EnvConfig(render_enabled=False))
        before = float(env.observation_space["agent_0"]["std_state"].high[3])
        env.unwrapped.update_params(
            env.unwrapped.vehicle_params.with_updates(v_max=40.0)
        )
        after = float(env.observation_space["agent_0"]["std_state"].high[3])
        self.assertGreater(after, before + 15.0)
        env.close()
