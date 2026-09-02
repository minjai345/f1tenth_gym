"""Focused mathematical and execution gates for the native JAX PPO example."""

import argparse
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from examples.jax_ppo_training import (
    ActorCritic,
    PPOConfig,
    circular_track,
    generalized_advantage_estimate,
    latent_gaussian_log_prob,
    make_train,
    training_config,
    validate_args,
)
from f1tenth_gym.envs.jax_simulator import JaxSimulator


class TestNativeJaxPPO(unittest.TestCase):
    def test_cli_rejects_a_single_sample_update(self):
        args = argparse.Namespace(
            num_envs=1,
            rollout_steps=1,
            total_timesteps=1,
            update_epochs=1,
            minibatches=1,
            hidden_size=8,
            episode_steps=1,
            learning_rate=3.0e-4,
            map_scale=1.0,
        )

        with self.assertRaisesRegex(ValueError, "at least two samples"):
            validate_args(args)

    def test_policy_uses_joint_latent_probability_and_bounded_actions(self):
        network = ActorCritic(hidden_size=8)
        observations = jnp.asarray(
            [[0.1, -0.2, 0.3, -0.4, 0.5, -0.6]],
            dtype=jnp.float32,
        )
        params = network.init(jax.random.key(1), observations)
        means, log_std, values = network.apply(params, observations)
        latent = means + 0.25
        normalized = jnp.tanh(latent)
        log_prob = latent_gaussian_log_prob(latent, means, log_std)

        coordinate_log_prob = -0.5 * (
            jnp.square((latent - means) * jnp.exp(-log_std))
            + 2.0 * log_std
            + np.log(2.0 * np.pi)
        )
        np.testing.assert_allclose(
            log_prob,
            jnp.sum(coordinate_log_prob, axis=-1),
            rtol=1.0e-6,
        )
        self.assertEqual(log_prob.shape, (1,))
        self.assertEqual(values.shape, (1,))
        self.assertTrue(np.all(np.abs(normalized) < 1.0))

    def test_gae_distinguishes_termination_from_timeout(self):
        rewards = jnp.asarray(
            [[0.5, 0.5], [1.0, 1.0], [2.0, 2.0]],
            dtype=jnp.float32,
        )
        values = jnp.asarray(
            [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]],
            dtype=jnp.float32,
        )
        transition_values = jnp.asarray(
            [[0.2, 0.2], [0.3, 0.3], [9.0, 4.0]],
            dtype=jnp.float32,
        )
        terminated = jnp.asarray(
            [[False, False], [False, False], [True, False]]
        )
        truncated = jnp.asarray(
            [[False, False], [False, False], [False, True]]
        )
        gamma = 0.9
        gae_lambda = 0.8

        advantages, returns = jax.jit(
            generalized_advantage_estimate,
            static_argnums=(5, 6),
        )(
            rewards,
            values,
            transition_values,
            terminated,
            truncated,
            gamma,
            gae_lambda,
        )

        deltas = np.asarray(
            [
                [0.5 + gamma * 0.2 - 0.1, 0.5 + gamma * 0.2 - 0.1],
                [1.0 + gamma * 0.3 - 0.2, 1.0 + gamma * 0.3 - 0.2],
                [2.0 - 0.3, 2.0 + gamma * 4.0 - 0.3],
            ],
            dtype=np.float32,
        )
        expected = deltas.copy()
        expected[1] += gamma * gae_lambda * expected[2]
        expected[0] += gamma * gae_lambda * expected[1]
        np.testing.assert_allclose(advantages, expected, rtol=1.0e-6)
        np.testing.assert_allclose(returns, expected + values, rtol=1.0e-6)

    def test_one_compiled_update_is_finite_and_randomized_per_row(self):
        track = circular_track()
        env_config = training_config(
            track,
            "cpu",
            episode_steps=1,
            smoke_test=True,
            randomize=True,
        )
        simulator = JaxSimulator(env_config, track, device="cpu")
        ppo_config = PPOConfig(
            num_envs=4,
            rollout_steps=2,
            updates=1,
            update_epochs=1,
            minibatches=2,
            hidden_size=8,
            learning_rate=3.0e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            value_coefficient=0.5,
            entropy_coefficient=1.0e-3,
            max_gradient_norm=0.5,
        )

        result = make_train(simulator, ppo_config)(jax.random.key(3))
        result = jax.tree.map(
            lambda value: value.block_until_ready(),
            result,
        )

        self.assertEqual(int(result.train_state.step), 2)
        for leaf in jax.tree.leaves(
            (result.train_state.params, result.metrics)
        ):
            self.assertTrue(np.all(np.isfinite(np.asarray(leaf))))
        self.assertGreater(float(result.metrics.gradient_norm[-1]), 0.0)
        masses = np.asarray(
            result.environment_state.params.dynamics.vehicle.m
        )
        randomization = env_config.domain_randomization_config
        self.assertTrue(np.all(masses >= randomization.low.m))
        self.assertTrue(np.all(masses <= randomization.high.m))
        self.assertGreater(float(np.ptp(masses)), 0.0)


if __name__ == "__main__":
    unittest.main()
