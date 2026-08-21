"""FRENET_BASED lap-counter regressions.

A lap is one crossing of the s=0 finish line, not one loop back to the spawn
pose. ``cumulative_s`` is seeded with the spawn arclength so both hold at once:
the first step accrues no spurious progress, and the datum stays the finish line.
"""
import unittest

import gymnasium as gym
import numpy as np

from f1tenth_gym.envs.env_config import EnvConfig, SimulationConfig, LoopCounterMode


def _env(count_partial_first_lap=True, max_laps=None):
    return gym.make(
        "f1tenth_gym:f1tenth-v0",
        config=EnvConfig(
            simulation_config=SimulationConfig(
                max_laps=max_laps,
                loop_counter=LoopCounterMode.FRENET_BASED,
                count_partial_first_lap=count_partial_first_lap,
            ),
            render_enabled=False,
        ),
    )


def length_of(env):
    return env.unwrapped.track.centerline.spline.s_frame_max


def _spawn_at(env, fraction):
    """Reset onto the centerline ``fraction`` of the way round; returns spawn s."""
    cl = env.unwrapped.track.centerline
    cx = np.asarray(cl.xs, dtype=float)
    cy = np.asarray(cl.ys, dtype=float)
    yaw = np.arctan2(np.gradient(cy), np.gradient(cx))
    k = int(len(cx) * fraction)
    obs, _ = env.reset(options={"poses": np.array([[cx[k], cy[k], yaw[k]]], dtype=np.float32)})
    return float(obs["agent_0"]["frenet_pose"][0])


def _roll(env, spawn_s, distance, step=0.5):
    """Walk the Frenet arclength forward, driving the counter one step at a time.

    Returns ``(events, length)`` where each event is
    ``(metres_travelled, lap_counts, lap_time)`` at a lap increment.
    """
    u = env.unwrapped
    length = u.track.centerline.spline.s_frame_max
    s, travelled, events = spawn_s, 0.0, []
    seen = int(u.lap_counts[0])
    while travelled < distance:
        s = (s + step) % length
        travelled += step
        u.sim_time += u.timestep
        u.sim.state.frenet[0, 0] = s
        u._check_done()
        if int(u.lap_counts[0]) > seen:
            seen = int(u.lap_counts[0])
            events.append((travelled, seen, float(u.lap_times[0])))
    return events, length


class TestFrenetLapCounter(unittest.TestCase):
    def test_first_step_does_not_accrue_the_spawn_arclength(self):
        env = _env()
        spawn_s = _spawn_at(env, 0.5)
        self.assertGreater(spawn_s, 20.0, "test setup: spawn should be well past s=0")
        u = env.unwrapped

        # Both references start at the spawn arclength.
        self.assertAlmostEqual(float(u.agents_prev_s[0]), spawn_s, delta=1.0)
        self.assertAlmostEqual(float(u.cumulative_s[0]), spawn_s, delta=1.0)

        # A tiny forward move must not bank another spawn_s of progress.
        env.step(np.array([[0.0, 1.0]], dtype=np.float32))
        self.assertAlmostEqual(float(u.cumulative_s[0]), spawn_s, delta=1.0)
        self.assertEqual(int(u.lap_counts[0]), 0)
        env.close()

    def test_lap_completes_at_the_finish_line_not_the_spawn_pose(self):
        env = _env()
        spawn_s = _spawn_at(env, 0.5)
        events, length = _roll(env, spawn_s, 1.2 * length_of(env))
        self.assertTrue(events, "no lap was ever counted")
        first, laps, _ = events[0]
        to_line = length - spawn_s
        self.assertAlmostEqual(first, to_line, delta=2.0)
        self.assertEqual(laps, 1)
        # ...and it is genuinely earlier than a full loop from the spawn pose.
        self.assertLess(first, length - 20.0)
        env.close()

    def test_second_lap_is_a_full_circuit(self):
        env = _env()
        spawn_s = _spawn_at(env, 0.5)
        events, length = _roll(env, spawn_s, 2.2 * length_of(env))
        self.assertGreaterEqual(len(events), 2)
        self.assertAlmostEqual(events[1][0] - events[0][0], length, delta=2.0)
        env.close()

    def test_out_lap_rule_skips_the_partial_first_lap(self):
        env = _env(count_partial_first_lap=False)
        spawn_s = _spawn_at(env, 0.5)
        events, length = _roll(env, spawn_s, 2.2 * length_of(env))
        self.assertTrue(events, "no lap was ever counted")
        first, laps, lap_time = events[0]
        self.assertEqual(laps, 1)
        # Lap 1 lands on the SECOND crossing, and its time is a full circuit.
        self.assertAlmostEqual(first, (length - spawn_s) + length, delta=2.0)
        self.assertAlmostEqual(lap_time, length / 0.5 * env.unwrapped.timestep, delta=0.2)
        env.close()

    def test_partial_first_lap_is_shorter_than_a_circuit(self):
        env = _env(count_partial_first_lap=True)
        spawn_s = _spawn_at(env, 0.5)
        events, length = _roll(env, spawn_s, 1.2 * length_of(env))
        _, _, lap_time = events[0]
        full = length / 0.5 * env.unwrapped.timestep
        self.assertLess(lap_time, 0.75 * full)
        env.close()

    def test_grid_spawn_still_takes_one_circuit(self):
        """The shipped default spawns on the line, so lap 1 must stay ~one lap."""
        env = _env(max_laps=None)
        obs, _ = env.reset(seed=42)
        spawn_s = float(obs["agent_0"]["frenet_pose"][0])
        self.assertLess(spawn_s, 5.0, "test setup: default spawn sits on the line")
        events, length = _roll(env, spawn_s, 1.5 * length_of(env))
        self.assertTrue(events)
        self.assertAlmostEqual(events[0][0], length - spawn_s, delta=2.0)
        env.close()


if __name__ == "__main__":
    unittest.main()
