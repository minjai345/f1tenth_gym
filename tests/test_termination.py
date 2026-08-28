"""Termination and truncation configuration (TerminationConfig, upstream #90)."""
import unittest

import gymnasium as gym
import numpy as np

from f1tenth_gym.envs.env_config import (
    AgentTerminationMode,
    EnvConfig,
    SimulationConfig,
    TerminationConfig,
)


def _mk(num_agents=1, max_laps=None, **termination):
    return gym.make(
        "f1tenth_gym:f1tenth-v0",
        config=EnvConfig(
            num_agents=num_agents,
            simulation_config=SimulationConfig(max_laps=max_laps),
            termination_config=TerminationConfig(**termination),
            render_enabled=False,
        ),
    )


class TestTerminationConfig(unittest.TestCase):
    def test_max_episode_steps_truncates(self):
        env = _mk(max_episode_steps=10)
        env.reset(seed=1)
        for i in range(1, 15):
            _, _, terminated, truncated, _ = env.step(np.array([[0.0, 1.0]], dtype=np.float32))
            if i < 10:
                self.assertFalse(truncated)
            if truncated:
                self.assertEqual(i, 10)
                self.assertFalse(terminated, "time-limit is truncation, not termination")
                break
        else:
            self.fail("never truncated")
        env.close()

    def test_default_never_truncates(self):
        env = _mk()  # max_episode_steps None
        env.reset(seed=1)
        for _ in range(30):
            _, _, _, truncated, info = env.step(np.array([[0.0, 1.0]], dtype=np.float32))
            self.assertFalse(truncated)
        self.assertIn("collisions", info)
        self.assertEqual(info["collisions"].shape, (1,))
        env.close()

    def test_terminate_on_collision_false(self):
        env = _mk(terminate_on_collision=False)
        env.reset(seed=1)
        saw_collision = False
        for _ in range(200):
            _, _, terminated, _, info = env.step(np.array([[0.4, 7.0]], dtype=np.float32))
            saw_collision = saw_collision or bool(info["collisions"][0])
            self.assertFalse(terminated, "terminated despite terminate_on_collision=False")
        self.assertTrue(saw_collision, "test setup: expected a collision")
        env.close()

    def test_default_terminates_on_ego_collision(self):
        env = _mk()
        env.reset(seed=1)
        for _ in range(200):
            _, _, terminated, _, _ = env.step(np.array([[0.4, 7.0]], dtype=np.float32))
            if terminated:
                break
        else:
            self.fail("default config did not terminate on collision")
        env.close()

    def test_agent_termination_modes_reduce_the_whole_environment(self):
        for mode, first_done in (
            (AgentTerminationMode.EGO, False),
            (AgentTerminationMode.ANY, True),
            (AgentTerminationMode.ALL, False),
        ):
            env = _mk(num_agents=2, agent_mode=mode)
            env.reset(seed=1)
            unwrapped = env.unwrapped
            unwrapped.sim.state.collisions[:] = (0.0, 1.0)
            self.assertEqual(unwrapped._check_done(), first_done, f"mode={mode.name}")

            # Contact events are per-step. The opponent's terminal status stays
            # latched while the next transition reports an ego-only contact,
            # making ALL useful without freezing either car.
            unwrapped.sim.state.collisions[:] = (1.0, 0.0)
            self.assertTrue(unwrapped._check_done(), f"mode={mode.name}")
            np.testing.assert_array_equal(unwrapped.terminated_agents, (True, True))
            env.close()

    def test_agent_mode_also_reduces_lap_completion(self):
        env = _mk(
            num_agents=2,
            max_laps=1,
            terminate_on_collision=False,
            agent_mode=AgentTerminationMode.ALL,
        )
        env.reset(seed=1)
        unwrapped = env.unwrapped
        unwrapped.lap_counts[1] = 1
        self.assertFalse(unwrapped._check_done())
        unwrapped.lap_counts[0] = 1
        self.assertTrue(unwrapped._check_done())
        env.close()

    def test_termination_bookkeeping_does_not_mutate_vehicle_state(self):
        env = _mk(num_agents=2, terminate_on_collision=False)
        env.reset(seed=1)
        unwrapped = env.unwrapped
        before = unwrapped.sim.state.state.copy()
        unwrapped.sim.state.collisions[:] = 1.0
        self.assertFalse(unwrapped._check_done())
        np.testing.assert_array_equal(unwrapped.sim.state.state, before)
        np.testing.assert_array_equal(unwrapped.terminated_agents, (False, False))
        env.close()

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            TerminationConfig(max_episode_steps=0)
        with self.assertRaises(TypeError):
            TerminationConfig(agent_mode="any")


if __name__ == "__main__":
    unittest.main()
