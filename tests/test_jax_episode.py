"""Functional Frenet progress, rewards, and episode-end semantics."""

from dataclasses import replace
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from f1tenth_gym.envs.collision_models import CollisionCheckMode
from f1tenth_gym.envs.env_config import (
    EnvConfig,
    RewardConfig,
    RewardMode,
    SimulationConfig,
    TerminationConfig,
)
from f1tenth_gym.envs.lidar import LiDARConfig
from f1tenth_gym.envs.episode import (
    BuiltinRewardMode,
    EpisodeConfig,
    EpisodeParams,
    TerminationMode,
    advance_episode,
    reset_episode_state,
    wrap_progress_delta,
)


def frenet_at(*values):
    values = jnp.asarray(values, dtype=jnp.float32)
    return jnp.stack(
        (values, jnp.zeros_like(values), jnp.zeros_like(values)), axis=1
    )


def params(**updates):
    base = EpisodeParams(lap_limit_enabled=False)
    return replace(base, **updates)


def advance(state, s_values, config, *, collisions=None, speeds=None,
            before=1.0, after=None, length=10.0, timestep=0.1,
            episode_params=None):
    if collisions is None:
        collisions = jnp.zeros((config.num_agents,), dtype=jnp.bool_)
    if speeds is None:
        speeds = jnp.zeros((config.num_agents,), dtype=jnp.float32)
    if after is None:
        after = before + timestep
    if episode_params is None:
        episode_params = params()
    return advance_episode(
        state,
        frenet_at(*s_values),
        jnp.asarray(collisions),
        jnp.asarray(speeds, dtype=jnp.float32),
        jnp.asarray(length, dtype=jnp.float32),
        jnp.asarray(before, dtype=jnp.float32),
        jnp.asarray(after, dtype=jnp.float32),
        jnp.asarray(timestep, dtype=jnp.float32),
        config,
        episode_params,
    )


class TestResetAndProgress(unittest.TestCase):
    def test_reset_seeds_both_arclength_references_from_spawn(self):
        config = EpisodeConfig(num_agents=2)
        initial = frenet_at(2.0, 7.0)
        state = reset_episode_state(initial, config)
        np.testing.assert_array_equal(state.frenet, initial)
        np.testing.assert_array_equal(state.progress_previous_s, [2.0, 7.0])
        np.testing.assert_array_equal(state.lap_previous_s, [2.0, 7.0])
        np.testing.assert_array_equal(state.cumulative_s, [2.0, 7.0])
        np.testing.assert_array_equal(state.lap_counts, 0)
        np.testing.assert_array_equal(state.terminated_agents, False)
        self.assertEqual(int(state.elapsed_steps), 0)

    def test_signed_progress_is_wrap_corrected_in_both_directions(self):
        current = jnp.asarray([0.2, 9.8, 5.0, -5.0], dtype=jnp.float32)
        previous = jnp.asarray([9.8, 0.2, 0.0, 0.0], dtype=jnp.float32)
        actual = wrap_progress_delta(current, previous, jnp.float32(10.0))
        np.testing.assert_allclose(
            actual, [0.4, -0.4, 5.0, -5.0], atol=1.0e-6
        )

        config = EpisodeConfig(num_agents=2)
        state = reset_episode_state(frenet_at(9.8, 0.2), config)
        state, _rewards, _events, metrics, _status = advance(
            state, (0.2, 9.8), config
        )
        np.testing.assert_allclose(metrics.progress, [0.4, -0.4], atol=1.0e-6)
        np.testing.assert_allclose(state.cumulative_s, [10.2, -0.2], atol=1.0e-6)

    def test_shape_errors_name_the_input(self):
        config = EpisodeConfig(num_agents=2)
        with self.assertRaisesRegex(ValueError, "frenet"):
            reset_episode_state(jnp.zeros((1, 3)), config)
        state = reset_episode_state(frenet_at(0.0, 0.0), config)
        with self.assertRaisesRegex(ValueError, "collisions"):
            advance_episode(
                state, frenet_at(0.0, 0.0), jnp.zeros((1,)), jnp.zeros((2,)),
                10.0, 1.0, 1.1, 0.1, config, params(),
            )
        with self.assertRaisesRegex(ValueError, "speeds"):
            advance_episode(
                state, frenet_at(0.0, 0.0), jnp.zeros((2,)), jnp.zeros((1,)),
                10.0, 1.0, 1.1, 0.1, config, params(),
            )


class TestFinishLineAndTiming(unittest.TestCase):
    def test_partial_first_lap_counts_at_the_finish_line(self):
        config = EpisodeConfig(num_agents=1, count_partial_first_lap=True)
        state = reset_episode_state(frenet_at(8.0), config)
        state, _rewards, events, metrics, status = advance(
            state, (0.2,), config, before=1.5, after=1.6
        )
        self.assertTrue(bool(events.finish_crossed[0]))
        self.assertTrue(bool(events.lap_completed[0]))
        self.assertEqual(int(state.finish_crossings[0]), 1)
        self.assertEqual(int(metrics.lap_counts[0]), 1)
        self.assertAlmostEqual(float(metrics.lap_times[0]), 1.5)
        self.assertAlmostEqual(float(metrics.sim_time), 1.6)
        self.assertFalse(bool(status.terminated))

    def test_out_lap_banks_the_first_split_then_times_a_full_lap(self):
        config = EpisodeConfig(num_agents=1, count_partial_first_lap=False)
        state = reset_episode_state(frenet_at(8.0), config)
        state, _rewards, first, _metrics, _status = advance(
            state, (0.2,), config, before=1.0
        )
        self.assertTrue(bool(first.finish_crossed[0]))
        self.assertFalse(bool(first.lap_completed[0]))
        self.assertEqual(int(state.lap_counts[0]), 0)
        self.assertAlmostEqual(float(state.lap_times_finish[0]), 1.0)

        for s_value, clock in ((4.0, 2.0), (8.0, 3.0), (0.2, 4.0)):
            state, _rewards, events, _metrics, _status = advance(
                state, (s_value,), config, before=clock
            )
        self.assertTrue(bool(events.finish_crossed[0]))
        self.assertTrue(bool(events.lap_completed[0]))
        self.assertEqual(int(state.lap_counts[0]), 1)
        self.assertAlmostEqual(float(state.lap_times[0]), 3.0)

    def test_the_lagged_clock_guard_defers_an_early_crossing(self):
        config = EpisodeConfig(num_agents=1)
        state = reset_episode_state(frenet_at(9.8), config)
        state, _rewards, events, _metrics, _status = advance(
            state, (0.2,), config, before=0.1, timestep=0.1
        )
        self.assertFalse(bool(events.finish_crossed[0]))
        self.assertEqual(int(state.finish_crossings[0]), 0)
        state, _rewards, events, _metrics, _status = advance(
            state, (0.3,), config, before=0.2, timestep=0.1
        )
        self.assertTrue(bool(events.finish_crossed[0]))
        self.assertEqual(int(state.lap_counts[0]), 1)


class TestTerminationAndTruncation(unittest.TestCase):
    def test_modes_reduce_latched_status_without_touching_frenet_state(self):
        for mode, first_done in (
            (TerminationMode.EGO, False),
            (TerminationMode.ANY, True),
            (TerminationMode.ALL, False),
        ):
            config = EpisodeConfig(num_agents=2, termination_mode=mode)
            state = reset_episode_state(frenet_at(2.0, 7.0), config)
            state, _rewards, events, _metrics, status = advance(
                state, (2.1, 7.1), config, collisions=(False, True)
            )
            self.assertEqual(bool(status.terminated), first_done)
            np.testing.assert_array_equal(events.newly_terminated, [False, True])
            np.testing.assert_array_equal(state.frenet, frenet_at(2.1, 7.1))

            state, _rewards, events, _metrics, status = advance(
                state, (2.2, 7.2), config, collisions=(True, False)
            )
            self.assertTrue(bool(status.terminated))
            np.testing.assert_array_equal(events.newly_terminated, [True, False])
            np.testing.assert_array_equal(state.terminated_agents, [True, True])
            np.testing.assert_array_equal(state.frenet, frenet_at(2.2, 7.2))

    def test_collision_can_continue_and_laps_can_still_terminate(self):
        config = EpisodeConfig(num_agents=1)
        episode_params = params(
            terminate_on_collision=False,
            lap_limit_enabled=True,
            max_laps=1,
        )
        state = reset_episode_state(frenet_at(8.0), config)
        state, _rewards, events, _metrics, status = advance(
            state, (8.1,), config, collisions=(True,), before=1.0,
            episode_params=episode_params,
        )
        self.assertTrue(bool(events.collisions[0]))
        self.assertFalse(bool(events.newly_terminated[0]))
        self.assertFalse(bool(status.terminated))

        state, _rewards, events, _metrics, status = advance(
            state, (0.2,), config, before=2.0,
            episode_params=episode_params,
        )
        self.assertTrue(bool(events.lap_completed[0]))
        self.assertTrue(bool(events.newly_terminated[0]))
        self.assertTrue(bool(status.terminated))

    def test_timeout_is_truncation_and_can_coincide_with_termination(self):
        config = EpisodeConfig(num_agents=1)
        episode_params = params(
            step_limit_enabled=True,
            max_episode_steps=3,
        )
        state = reset_episode_state(frenet_at(0.0), config)
        for step in range(1, 4):
            collisions = (step == 3,)
            state, _rewards, _events, _metrics, status = advance(
                state, (0.1 * step,), config, collisions=collisions,
                episode_params=episode_params,
            )
            self.assertEqual(bool(status.truncated), step >= 3)
            self.assertEqual(bool(status.terminated), step >= 3)
        self.assertEqual(int(state.elapsed_steps), 3)

    def test_all_can_mix_one_lap_terminal_and_one_collision_terminal(self):
        config = EpisodeConfig(
            num_agents=2,
            termination_mode=TerminationMode.ALL,
        )
        episode_params = params(lap_limit_enabled=True, max_laps=1)
        state = reset_episode_state(frenet_at(8.0, 4.0), config)
        state, _rewards, events, _metrics, status = advance(
            state,
            (0.2, 4.1),
            config,
            collisions=(False, True),
            before=1.0,
            episode_params=episode_params,
        )
        np.testing.assert_array_equal(events.lap_completed, [True, False])
        np.testing.assert_array_equal(events.newly_terminated, [True, True])
        np.testing.assert_array_equal(state.terminated_agents, [True, True])
        self.assertTrue(bool(status.terminated))

    def test_configuration_validation(self):
        for changes in (
            {"num_agents": 0},
            {"num_agents": 1, "ego_index": 1},
        ):
            with self.assertRaises(ValueError):
                EpisodeConfig(**changes)
        with self.assertRaises(TypeError):
            EpisodeConfig(num_agents=1, termination_mode="any")
        with self.assertRaises(TypeError):
            EpisodeConfig(num_agents=1, reward_mode="progress")


class TestBuiltinRewards(unittest.TestCase):
    def test_survival_reward_is_timestep_for_every_agent(self):
        config = EpisodeConfig(num_agents=2)
        state = reset_episode_state(frenet_at(1.0, 2.0), config)
        _state, rewards, _events, _metrics, _status = advance(
            state, (1.2, 2.3), config, timestep=0.02
        )
        np.testing.assert_allclose(rewards, [0.02, 0.02], atol=1.0e-7)

    def test_progress_reward_uses_fresh_per_agent_signals(self):
        config = EpisodeConfig(
            num_agents=2,
            ego_index=1,
            reward_mode=BuiltinRewardMode.PROGRESS,
        )
        episode_params = params(
            progress_weight=2.0,
            velocity_weight=0.5,
            timestep_weight=3.0,
            collision_penalty=4.0,
        )
        state = reset_episode_state(frenet_at(1.0, 2.0), config)
        state, rewards, _events, _metrics, _status = advance(
            state, (1.2, 2.3), config, collisions=(False, True),
            speeds=(2.0, -1.0), timestep=0.1,
            episode_params=episode_params,
        )
        np.testing.assert_allclose(rewards, [1.7, -3.6], atol=1.0e-6)

        _state, rewards, events, _metrics, _status = advance(
            state, (1.4, 2.6), config, collisions=(False, False),
            speeds=(2.0, -1.0), timestep=0.1,
            episode_params=episode_params,
        )
        self.assertFalse(bool(events.newly_terminated[1]))
        np.testing.assert_allclose(rewards, [1.7, 0.4], atol=1.0e-6)


class TestTransformability(unittest.TestCase):
    def test_jit_environment_vmap_and_lax_scan_with_traced_limits(self):
        config = EpisodeConfig(num_agents=1)
        states = jax.vmap(
            lambda initial: reset_episode_state(frenet_at(initial), config)
        )(jnp.asarray([1.0, 6.0], dtype=jnp.float32))
        current = jnp.asarray([[1.2], [6.3]], dtype=jnp.float32)
        collisions = jnp.asarray([[False], [True]])
        batch_params = jax.tree.map(
            lambda value: jnp.asarray([value, value]), params()
        )
        batch_params = replace(
            batch_params,
            terminate_on_collision=jnp.asarray([False, True]),
            step_limit_enabled=jnp.asarray([True, True]),
            max_episode_steps=jnp.asarray([1, 3], dtype=jnp.int32),
        )
        run = jax.jit(
            jax.vmap(
                lambda state, values, hits, one_params: advance(
                    state, values, config, collisions=hits,
                    episode_params=one_params,
                )
            )
        )
        next_states, _rewards, events, metrics, status = run(
            states, current, collisions, batch_params
        )
        self.assertEqual(next_states.frenet.shape, (2, 1, 3))
        self.assertEqual(events.collisions.shape, (2, 1))
        np.testing.assert_allclose(metrics.progress[:, 0], [0.2, 0.3],
                                   atol=1.0e-6)
        np.testing.assert_array_equal(status.terminated, [False, True])
        np.testing.assert_array_equal(status.truncated, [True, False])

        scan_values = jnp.asarray([1.1, 1.2, 1.3, 1.4], dtype=jnp.float32)
        state = reset_episode_state(frenet_at(1.0), config)
        episode_params = params(
            step_limit_enabled=True,
            max_episode_steps=4,
        )

        def body(carry, value):
            next_state, _rewards, _events, _metrics, step_status = advance(
                carry, (value,), config, episode_params=episode_params
            )
            return next_state, step_status

        state, rollout = jax.jit(
            lambda initial: jax.lax.scan(body, initial, scan_values)
        )(state)
        self.assertTrue(bool(rollout.truncated[-1]))
        self.assertEqual(int(state.elapsed_steps), 4)


class TestMutableEnvironmentParity(unittest.TestCase):
    def test_scripted_out_lap_reward_and_status_sequence(self):
        host = EnvConfig(
            simulation_config=SimulationConfig(
                max_laps=1,
                count_partial_first_lap=False,
            ),
            termination_config=TerminationConfig(max_episode_steps=4),
            reward_config=RewardConfig(
                mode=RewardMode.PROGRESS,
                progress_weight=2.0,
                velocity_weight=0.5,
                timestep_weight=3.0,
                collision_penalty=4.0,
            ),
            lidar_config=LiDARConfig(enabled=False),
            collision_check=CollisionCheckMode.NONE,
            render_enabled=False,
        )
        from f1tenth_gym.envs.f110_env import F110Env

        env = F110Env(config=host)
        try:
            env.reset(seed=3)
            length = float(env.track.centerline.spline.s_frame_max)
            initial_s = 0.8 * length
            env.sim.state.frenet[0, 0] = initial_s
            env._reward_prev_s[0] = initial_s
            env.agents_prev_s[0] = initial_s
            env.cumulative_s[0] = initial_s
            env._line_crossings[0] = 0
            env.lap_counts[0] = 0
            env.lap_times[0] = 0
            env.lap_times_finish[0] = 0
            env.terminated_agents[0] = False
            env._elapsed_steps = 0

            config = EpisodeConfig(
                num_agents=1,
                count_partial_first_lap=False,
                reward_mode=BuiltinRewardMode.PROGRESS,
            )
            episode_params = EpisodeParams(
                terminate_on_collision=True,
                lap_limit_enabled=True,
                max_laps=1,
                step_limit_enabled=True,
                max_episode_steps=4,
                progress_weight=2.0,
                velocity_weight=0.5,
                timestep_weight=3.0,
                collision_penalty=4.0,
            )
            state = reset_episode_state(frenet_at(initial_s), config)
            sequence = (
                (0.05 * length, False, 2.0, 0.02),
                (0.40 * length, False, 2.0, 0.03),
                (0.80 * length, False, 2.0, 0.04),
                (0.05 * length, True, 2.0, 0.05),
            )
            action = np.zeros((1, 2), dtype=np.float32)
            for current_s, collision, speed, before in sequence:
                env.sim.state.frenet[0, 0] = current_s
                env.sim.state.collisions[0] = float(collision)
                env.sim.state.standard_state[0, 3] = speed
                env.sim_time = before
                env._elapsed_steps += 1
                host_terminated = env._check_done()
                host_progress = env._compute_progress()
                host_truncated = env._elapsed_steps >= 4
                host_reward = env._compute_reward(
                    {}, action, host_progress, {}, host_terminated,
                    host_truncated,
                )

                state, rewards, _events, metrics, status = advance_episode(
                    state,
                    frenet_at(current_s),
                    jnp.asarray([collision]),
                    jnp.asarray([speed], dtype=jnp.float32),
                    jnp.asarray(length, dtype=jnp.float32),
                    jnp.asarray(before, dtype=jnp.float32),
                    jnp.asarray(before + env.timestep, dtype=jnp.float32),
                    jnp.asarray(env.timestep, dtype=jnp.float32),
                    config,
                    episode_params,
                )
                np.testing.assert_allclose(metrics.progress, host_progress,
                                           atol=2.0e-5)
                np.testing.assert_array_equal(metrics.lap_counts,
                                              env.lap_counts.astype(np.int32))
                np.testing.assert_allclose(metrics.lap_times, env.lap_times,
                                           atol=2.0e-5)
                np.testing.assert_array_equal(metrics.terminated_agents,
                                              env.terminated_agents)
                self.assertAlmostEqual(float(rewards[0]), host_reward, places=4)
                self.assertEqual(bool(status.terminated), host_terminated)
                self.assertEqual(bool(status.truncated), host_truncated)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
