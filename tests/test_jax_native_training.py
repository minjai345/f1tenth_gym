"""Fast mathematical gates for the repository-only native PPO validation."""

from dataclasses import replace
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from validation.jax_native_ppo import (
    RolloutBatch,
    TrainingConfig,
    actor_critic,
    generalized_advantage_estimate,
    initialize_actor_critic,
    initialize_adam,
    latent_gaussian_log_prob,
    ppo_loss,
    ppo_update,
    prepare_training_batch,
    sample_policy,
)


class TestNativePPOContracts(unittest.TestCase):
    def test_policy_stores_latent_action_and_joint_log_probability(self):
        params = initialize_actor_critic(jax.random.key(1), 2, 8)
        observations = jnp.asarray([[0.1, -0.2], [0.3, 0.4]])
        latent, normalized, log_probs, values = jax.jit(sample_policy)(
            jax.random.key(2),
            params,
            observations,
        )

        means, log_std, expected_values = actor_critic(params, observations)
        coordinate_log_probs = -0.5 * (
            jnp.square((latent - means) * jnp.exp(-log_std))
            + 2.0 * log_std
            + np.log(2.0 * np.pi)
        )
        np.testing.assert_allclose(
            log_probs,
            jnp.sum(coordinate_log_probs, axis=-1),
            rtol=1.0e-6,
        )
        self.assertEqual(log_probs.shape, (2,))
        self.assertEqual(latent.shape, (2, 2))
        self.assertTrue(np.all(np.abs(normalized) < 1.0))
        np.testing.assert_allclose(normalized, jnp.tanh(latent))
        np.testing.assert_allclose(values, expected_values)

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
        self.assertLess(advantages[-1, 0], advantages[-1, 1])

    def test_rollout_samples_are_stopped_and_one_ppo_update_is_finite(self):
        time_steps = 2
        batch_size = 4
        observations = jnp.linspace(
            -0.5,
            0.5,
            time_steps * batch_size * 2,
        ).reshape(time_steps, batch_size, 2)
        params = initialize_actor_critic(jax.random.key(3), 2, 8)
        flat_observations = observations.reshape(-1, 2)
        means, log_std, values = actor_critic(params, flat_observations)
        latent = means + 0.1
        log_probs = latent_gaussian_log_prob(latent, means, log_std)
        rollout = RolloutBatch(
            observations=observations,
            latent_actions=latent.reshape(time_steps, batch_size, 2),
            old_log_probs=log_probs.reshape(time_steps, batch_size),
            old_values=values.reshape(time_steps, batch_size),
            rewards=jnp.full((time_steps, batch_size), 0.25),
            terminated=jnp.zeros((time_steps, batch_size), dtype=jnp.bool_),
            truncated=jnp.zeros((time_steps, batch_size), dtype=jnp.bool_),
            transition_values=jnp.zeros((time_steps, batch_size)),
            transition_speeds=jnp.ones((time_steps, batch_size)),
        )
        training = prepare_training_batch(rollout, 0.99, 0.95)
        total, diagnostics = ppo_loss(params, training, 0.2, 0.5, 0.001)
        self.assertTrue(np.isfinite(total))
        self.assertEqual(diagnostics.shape, (6,))
        self.assertAlmostEqual(float(diagnostics[4]), 0.0, places=6)
        self.assertAlmostEqual(float(diagnostics[5]), 0.0, places=6)

        config = replace(
            TrainingConfig(),
            batch_size=batch_size,
            rollout_steps=time_steps,
            update_epochs=1,
            minibatches=2,
            hidden_size=8,
            updates=1,
        )
        updated, optimizer, update_metrics = jax.jit(
            ppo_update,
            static_argnums=4,
        )(
            params,
            initialize_adam(params),
            training,
            jax.random.key(4),
            config,
        )
        for leaf in jax.tree.leaves((updated, optimizer, update_metrics)):
            self.assertTrue(np.all(np.isfinite(np.asarray(leaf))))
        self.assertEqual(update_metrics.shape, (7,))
        self.assertGreaterEqual(float(update_metrics[-1]), 0.0)

        def stopped_reward_gradient(reward_values):
            changed = replace(rollout, rewards=reward_values)
            return jnp.sum(prepare_training_batch(changed, 0.99, 0.95).returns)

        gradient = jax.grad(stopped_reward_gradient)(rollout.rewards)
        np.testing.assert_array_equal(gradient, jnp.zeros_like(gradient))


if __name__ == "__main__":
    unittest.main()
