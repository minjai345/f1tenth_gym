"""Run a reproducible device-native PPO validation against the JAX core.

This is a repository validation job, not a supported trainer.  It deliberately
uses only JAX and the public functional batching surface, so running it does not
add a training-library dependency to :mod:`f1tenth_gym`.

Run the default CPU gate from the repository root with::

    uv run --no-sync python validation/jax_native_ppo.py --device cpu

The program exits nonzero when any compiled value is non-finite or when the
fixed-key deterministic evaluation fails to improve by ``--min-improvement``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from f1tenth_gym.envs.action import (
    LongitudinalActionType,
    SteerActionType,
)
from f1tenth_gym.envs.collision_models import CollisionCheckMode
from f1tenth_gym.envs.dynamic_models import DynamicModel
from f1tenth_gym.envs.env_config import (
    ControlConfig,
    EnvConfig,
    ResetConfig,
    SimulationConfig,
    TerminationConfig,
)
from f1tenth_gym.envs.integrators import IntegratorType
from f1tenth_gym.envs.lidar import LiDARConfig
from f1tenth_gym.envs.reset import ReferenceLine, ResetStrategy
from f1tenth_gym.envs.track import Track
from f1tenth_gym.jax.batched import (
    PolicyField,
    PolicyLayout,
    policy_observation,
    reset_batch,
    scale_normalized_actions,
    select_ego_rewards,
    step_batch_autoreset,
)
from f1tenth_gym.jax.builder import CoreBundle, build_core


LOG_TWO_PI = math.log(2.0 * math.pi)
KINEMATIC_LAYOUT = PolicyLayout((PolicyField.KINEMATIC_STATE,))


@dataclass(frozen=True)
class TrainingConfig:
    """Static PPO and validation-job dimensions."""

    seed: int = 7
    batch_size: int = 64
    rollout_steps: int = 64
    updates: int = 24
    update_epochs: int = 4
    minibatches: int = 8
    hidden_size: int = 64
    episode_steps: int = 128
    evaluation_steps: int = 128
    target_speed: float = 3.0
    learning_rate: float = 3.0e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 1.0e-3
    max_gradient_norm: float = 0.5
    min_improvement: float = 0.10
    device: str = "cpu"

    def __post_init__(self) -> None:
        integer_fields = (
            "batch_size",
            "rollout_steps",
            "updates",
            "update_epochs",
            "minibatches",
            "hidden_size",
            "episode_steps",
            "evaluation_steps",
        )
        for name in integer_fields:
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.batch_size * self.rollout_steps % self.minibatches:
            raise ValueError(
                "batch_size * rollout_steps must be divisible by minibatches"
            )
        for name in (
            "learning_rate",
            "gamma",
            "gae_lambda",
            "clip_epsilon",
            "value_coefficient",
            "max_gradient_norm",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be > 0")
        if self.target_speed <= 0.0:
            raise ValueError("target_speed must be > 0")
        if self.entropy_coefficient < 0.0:
            raise ValueError("entropy_coefficient must be >= 0")
        if self.device not in ("cpu", "gpu"):
            raise ValueError("device must be 'cpu' or 'gpu'")


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DenseParams:
    """One affine layer."""

    weight: jax.Array
    bias: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ActorCriticParams:
    """Small shared-trunk actor critic used only by this validation."""

    hidden_1: DenseParams
    hidden_2: DenseParams
    actor: DenseParams
    critic: DenseParams
    log_std: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class AdamState:
    """Minimal local Adam state, avoiding an optimizer dependency."""

    count: jax.Array
    first_moment: ActorCriticParams
    second_moment: ActorCriticParams


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RolloutBatch:
    """Time-major immutable PPO samples from one native rollout."""

    observations: jax.Array
    latent_actions: jax.Array
    old_log_probs: jax.Array
    old_values: jax.Array
    rewards: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    transition_values: jax.Array
    transition_speeds: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class TrainingBatch:
    """Flattened, stopped-gradient samples consumed by PPO."""

    observations: jax.Array
    latent_actions: jax.Array
    old_log_probs: jax.Array
    old_values: jax.Array
    advantages: jax.Array
    returns: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class IterationMetrics:
    """Synchronized scalar outputs that keep the update live."""

    total_loss: jax.Array
    policy_loss: jax.Array
    value_loss: jax.Array
    entropy: jax.Array
    approximate_kl: jax.Array
    clip_fraction: jax.Array
    gradient_norm: jax.Array
    rollout_reward: jax.Array
    rollout_speed: jax.Array
    checksum: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Evaluation:
    """Fixed-key deterministic policy measurements."""

    score: jax.Array
    mean_speed: jax.Array
    mean_absolute_speed_error: jax.Array
    checksum: jax.Array


@dataclass(frozen=True)
class ValidationResult:
    """Host report emitted by :func:`run_validation`."""

    backend: str
    device: str
    environment_steps: int
    compile_seconds: float
    steady_seconds: float
    steady_environment_steps_per_second: float
    baseline_score: float
    final_score: float
    score_improvement: float
    baseline_mean_speed: float
    final_mean_speed: float
    final_speed_error: float
    final_loss: float
    final_policy_loss: float
    final_value_loss: float
    final_entropy: float
    final_approximate_kl: float
    final_clip_fraction: float
    final_gradient_norm: float
    checksum: float
    passed: bool


def _dense(
    key: jax.Array,
    input_size: int,
    output_size: int,
    scale: float,
) -> DenseParams:
    weight = (
        scale
        * jax.random.normal(key, (input_size, output_size), dtype=jnp.float32)
        / math.sqrt(input_size)
    )
    return DenseParams(
        weight=weight,
        bias=jnp.zeros((output_size,), dtype=jnp.float32),
    )


def initialize_actor_critic(
    key: jax.Array,
    input_size: int,
    hidden_size: int,
) -> ActorCriticParams:
    """Initialize a two-layer actor critic with a near-zero actor mean."""
    hidden_1_key, hidden_2_key, actor_key, critic_key = jax.random.split(key, 4)
    return ActorCriticParams(
        hidden_1=_dense(hidden_1_key, input_size, hidden_size, math.sqrt(2.0)),
        hidden_2=_dense(hidden_2_key, hidden_size, hidden_size, math.sqrt(2.0)),
        actor=_dense(actor_key, hidden_size, 2, 0.01),
        critic=_dense(critic_key, hidden_size, 1, 1.0),
        log_std=jnp.full((2,), -0.5, dtype=jnp.float32),
    )


def _affine(inputs: jax.Array, params: DenseParams) -> jax.Array:
    return inputs @ params.weight + params.bias


def actor_critic(
    params: ActorCriticParams,
    observations: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return latent Gaussian means, bounded log standard deviations and value."""
    hidden = jnp.tanh(_affine(observations, params.hidden_1))
    hidden = jnp.tanh(_affine(hidden, params.hidden_2))
    means = _affine(hidden, params.actor)
    values = jnp.squeeze(_affine(hidden, params.critic), axis=-1)
    log_std = jnp.clip(params.log_std, -5.0, 2.0)
    return means, log_std, values


def latent_gaussian_log_prob(
    latent_actions: jax.Array,
    means: jax.Array,
    log_std: jax.Array,
) -> jax.Array:
    """Return one joint latent-Gaussian log probability per action event.

    PPO stores the pre-tanh latent action.  The fixed tanh and affine physical
    transforms therefore cancel in the old/new likelihood ratio, avoiding an
    unstable inverse tanh while still using a bounded environment action.
    """
    standardized = (latent_actions - means) * jnp.exp(-log_std)
    coordinate_log_probs = -0.5 * (
        jnp.square(standardized) + 2.0 * log_std + LOG_TWO_PI
    )
    return jnp.sum(coordinate_log_probs, axis=-1)


def latent_gaussian_entropy(log_std: jax.Array) -> jax.Array:
    """Return the joint entropy of the latent two-dimensional Gaussian."""
    return jnp.sum(log_std + 0.5 * (1.0 + LOG_TWO_PI), axis=-1)


def sample_policy(
    key: jax.Array,
    params: ActorCriticParams,
    observations: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Sample pre-tanh actions and their normalized bounded realization."""
    means, log_std, values = actor_critic(params, observations)
    noise = jax.random.normal(key, means.shape, dtype=means.dtype)
    latent_actions = means + jnp.exp(log_std) * noise
    normalized_actions = jnp.tanh(latent_actions)
    log_probs = latent_gaussian_log_prob(latent_actions, means, log_std)
    return latent_actions, normalized_actions, log_probs, values


def generalized_advantage_estimate(
    rewards: jax.Array,
    values: jax.Array,
    transition_values: jax.Array,
    terminated: jax.Array,
    truncated: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Compute Gymnasium-correct time-major GAE.

    ``transition_values`` must be evaluated from the post-step transition
    observation, never the auto-reset next observation.  Natural termination
    zero-bootstraps.  Timeout truncation bootstraps that transition value, but
    both status types stop the recurrence across the episode boundary.
    """
    shapes = {
        rewards.shape,
        values.shape,
        transition_values.shape,
        terminated.shape,
        truncated.shape,
    }
    if len(shapes) != 1 or rewards.ndim != 2:
        raise ValueError("GAE inputs must share shape (time, batch)")
    terminated = jnp.asarray(terminated, dtype=jnp.bool_)
    truncated = jnp.asarray(truncated, dtype=jnp.bool_)
    bootstrap = jnp.where(terminated, 0.0, transition_values)
    deltas = rewards + gamma * bootstrap - values
    continues = ~(terminated | truncated)

    def reverse_step(next_advantage, inputs):
        delta, continues_here = inputs
        advantage = (
            delta
            + gamma
            * gae_lambda
            * continues_here.astype(delta.dtype)
            * next_advantage
        )
        return advantage, advantage

    initial = jnp.zeros_like(deltas[-1])
    _, reversed_advantages = jax.lax.scan(
        reverse_step,
        initial,
        (deltas[::-1], continues[::-1]),
    )
    advantages = reversed_advantages[::-1]
    return advantages, advantages + values


def prepare_training_batch(
    rollout: RolloutBatch,
    gamma: float,
    gae_lambda: float,
) -> TrainingBatch:
    """Stop rollout gradients, form GAE targets and flatten time/environment."""
    rollout = jax.tree.map(jax.lax.stop_gradient, rollout)
    advantages, returns = generalized_advantage_estimate(
        rollout.rewards,
        rollout.old_values,
        rollout.transition_values,
        rollout.terminated,
        rollout.truncated,
        gamma,
        gae_lambda,
    )
    advantages = (advantages - jnp.mean(advantages)) / (
        jnp.std(advantages) + 1.0e-8
    )

    def flatten(value):
        return jnp.reshape(value, (-1,) + value.shape[2:])

    batch = TrainingBatch(
        observations=flatten(rollout.observations),
        latent_actions=flatten(rollout.latent_actions),
        old_log_probs=flatten(rollout.old_log_probs),
        old_values=flatten(rollout.old_values),
        advantages=flatten(advantages),
        returns=flatten(returns),
    )
    return jax.tree.map(jax.lax.stop_gradient, batch)


def initialize_adam(params: ActorCriticParams) -> AdamState:
    """Create zero first and second moments matching ``params``."""
    zeros = jax.tree.map(jnp.zeros_like, params)
    return AdamState(
        count=jnp.asarray(0, dtype=jnp.int32),
        first_moment=zeros,
        second_moment=zeros,
    )


def tree_global_norm(tree: Any) -> jax.Array:
    """Compute one L2 norm over every array leaf."""
    squared = [jnp.sum(jnp.square(leaf)) for leaf in jax.tree.leaves(tree)]
    return jnp.sqrt(jnp.sum(jnp.stack(squared)))


def clip_gradients(tree: Any, maximum_norm: float) -> tuple[Any, jax.Array]:
    """Clip a gradient pytree by global norm and return its original norm."""
    norm = tree_global_norm(tree)
    coefficient = jnp.minimum(1.0, maximum_norm / (norm + 1.0e-8))
    return jax.tree.map(lambda leaf: coefficient * leaf, tree), norm


def adam_step(
    params: ActorCriticParams,
    gradients: ActorCriticParams,
    state: AdamState,
    learning_rate: float,
) -> tuple[ActorCriticParams, AdamState]:
    """Apply one bias-corrected Adam update."""
    beta_1 = 0.9
    beta_2 = 0.999
    count = state.count + jnp.asarray(1, dtype=state.count.dtype)
    first = jax.tree.map(
        lambda moment, gradient: beta_1 * moment + (1.0 - beta_1) * gradient,
        state.first_moment,
        gradients,
    )
    second = jax.tree.map(
        lambda moment, gradient: (
            beta_2 * moment + (1.0 - beta_2) * jnp.square(gradient)
        ),
        state.second_moment,
        gradients,
    )
    first_scale = 1.0 - beta_1**count.astype(jnp.float32)
    second_scale = 1.0 - beta_2**count.astype(jnp.float32)
    updated = jax.tree.map(
        lambda parameter, first_moment, second_moment: parameter
        - learning_rate
        * (first_moment / first_scale)
        / (jnp.sqrt(second_moment / second_scale) + 1.0e-8),
        params,
        first,
        second,
    )
    return updated, AdamState(count, first, second)


def ppo_loss(
    params: ActorCriticParams,
    batch: TrainingBatch,
    clip_epsilon: float,
    value_coefficient: float,
    entropy_coefficient: float,
) -> tuple[jax.Array, jax.Array]:
    """Return clipped PPO loss and six diagnostics."""
    means, log_std, values = actor_critic(params, batch.observations)
    log_probs = latent_gaussian_log_prob(
        batch.latent_actions,
        means,
        log_std,
    )
    log_ratios = log_probs - batch.old_log_probs
    ratios = jnp.exp(log_ratios)
    clipped_ratios = jnp.clip(
        ratios,
        1.0 - clip_epsilon,
        1.0 + clip_epsilon,
    )
    policy_loss = jnp.mean(
        jnp.maximum(
            -batch.advantages * ratios,
            -batch.advantages * clipped_ratios,
        )
    )

    clipped_values = batch.old_values + jnp.clip(
        values - batch.old_values,
        -clip_epsilon,
        clip_epsilon,
    )
    value_errors = jnp.square(values - batch.returns)
    clipped_value_errors = jnp.square(clipped_values - batch.returns)
    value_loss = 0.5 * jnp.mean(jnp.maximum(value_errors, clipped_value_errors))
    entropy = jnp.mean(latent_gaussian_entropy(log_std))
    total = (
        policy_loss
        + value_coefficient * value_loss
        - entropy_coefficient * entropy
    )
    approximate_kl = jnp.mean((ratios - 1.0) - log_ratios)
    clip_fraction = jnp.mean(
        (jnp.abs(ratios - 1.0) > clip_epsilon).astype(values.dtype)
    )
    metrics = jnp.stack(
        (
            total,
            policy_loss,
            value_loss,
            entropy,
            approximate_kl,
            clip_fraction,
        )
    )
    return total, metrics


def ppo_update(
    params: ActorCriticParams,
    optimizer: AdamState,
    batch: TrainingBatch,
    key: jax.Array,
    config: TrainingConfig,
) -> tuple[ActorCriticParams, AdamState, jax.Array]:
    """Run shuffled PPO epochs and return seven averaged diagnostics."""
    sample_count = batch.observations.shape[0]
    if sample_count % config.minibatches:
        raise ValueError("sample count must be divisible by minibatches")
    minibatch_size = sample_count // config.minibatches

    def update_minibatch(carry, indices):
        current_params, current_optimizer = carry
        minibatch = jax.tree.map(lambda value: value[indices], batch)
        (loss, metrics), gradients = jax.value_and_grad(
            ppo_loss,
            has_aux=True,
        )(
            current_params,
            minibatch,
            config.clip_epsilon,
            config.value_coefficient,
            config.entropy_coefficient,
        )
        del loss
        gradients, gradient_norm = clip_gradients(
            gradients,
            config.max_gradient_norm,
        )
        next_params, next_optimizer = adam_step(
            current_params,
            gradients,
            current_optimizer,
            config.learning_rate,
        )
        return (next_params, next_optimizer), jnp.concatenate(
            (metrics, gradient_norm[None])
        )

    def update_epoch(carry, _unused):
        current_params, current_optimizer, current_key = carry
        current_key, permutation_key = jax.random.split(current_key)
        indices = jax.random.permutation(permutation_key, sample_count)
        indices = jnp.reshape(
            indices,
            (config.minibatches, minibatch_size),
        )
        (next_params, next_optimizer), metrics = jax.lax.scan(
            update_minibatch,
            (current_params, current_optimizer),
            indices,
        )
        return (next_params, next_optimizer, current_key), metrics

    (params, optimizer, _), metrics = jax.lax.scan(
        update_epoch,
        (params, optimizer, key),
        xs=None,
        length=config.update_epochs,
    )
    return params, optimizer, jnp.mean(metrics, axis=(0, 1))


def _circle_track(point_count: int = 64, radius: float = 8.0) -> Track:
    theta = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
    return Track.from_refline(
        x=radius * np.cos(theta),
        y=radius * np.sin(theta),
        velx=np.full(point_count, 3.0),
    )


def build_training_bundle(config: TrainingConfig) -> CoreBundle:
    """Build a one-agent, state-only core for the reproducible PPO gate."""
    track = _circle_track()
    env_config = EnvConfig(
        map_name=track,
        num_agents=1,
        control_config=ControlConfig(
            longitudinal_mode=LongitudinalActionType.ACCL,
            steering_mode=SteerActionType.STEERING_SPEED,
        ),
        simulation_config=SimulationConfig(
            timestep=0.05,
            integrator_timestep=0.05,
            integrator=IntegratorType.RK4,
            dynamics_model=DynamicModel.KS,
            max_laps=None,
        ),
        reset_config=ResetConfig(
            strategy=ResetStrategy.RL_GRID_STATIC,
            reference_line=ReferenceLine.CENTERLINE,
        ),
        lidar_config=LiDARConfig(enabled=False),
        termination_config=TerminationConfig(
            max_episode_steps=config.episode_steps,
            terminate_on_collision=False,
        ),
        collision_check=CollisionCheckMode.NONE,
        render_enabled=False,
    )
    return build_core(env_config, track, target_device=config.device)


def _policy_features(observation, bundle: CoreBundle) -> jax.Array:
    """Select ego steering angle and speed from the canonical policy layout."""
    kinematic = policy_observation(
        observation,
        bundle.config,
        KINEMATIC_LAYOUT,
    )
    ego = bundle.config.episode.ego_index
    return kinematic[:, ego, 2:4]


def _speed_reward(target_speed: float):
    """Build a bounded speed-tracking reward used only by the validation."""

    def reward(observation, actions, events, metrics, params):
        del events, metrics, params
        speed = observation.standard_state[:, 3]
        speed_error = (speed - target_speed) / target_speed
        steering_effort = actions[:, 0] / 3.5
        return 1.0 - jnp.square(speed_error) - 0.02 * jnp.square(
            steering_effort
        )

    return reward


def _tree_checksum(tree: Any) -> jax.Array:
    leaves = jax.tree.leaves(tree)
    return jnp.sum(jnp.stack([jnp.sum(leaf) for leaf in leaves]))


def _block_tree(tree: Any) -> Any:
    return jax.tree.map(lambda value: value.block_until_ready(), tree)


def _tree_is_finite(tree: Any) -> bool:
    return all(
        bool(np.all(np.isfinite(np.asarray(leaf))))
        for leaf in jax.tree.leaves(tree)
    )


def make_iteration(bundle: CoreBundle, config: TrainingConfig):
    """Create one compiled rollout-plus-PPO-update program."""
    reward_fn = _speed_reward(config.target_speed)

    def collect_rollout(params, batch_state, observation, key):
        def rollout_step(carry, _unused):
            current_state, current_observation, current_key = carry
            current_key, action_key, step_root, reset_root = jax.random.split(
                current_key,
                4,
            )
            features = _policy_features(current_observation, bundle)
            latent, normalized, log_prob, value = sample_policy(
                action_key,
                params,
                features,
            )
            normalized = normalized[:, None, :]
            actions = scale_normalized_actions(
                normalized,
                current_state.params,
                bundle.config,
            )
            step_keys = jax.random.split(step_root, config.batch_size)
            reset_keys = jax.random.split(reset_root, config.batch_size)
            transition = step_batch_autoreset(
                step_keys,
                reset_keys,
                current_state,
                actions,
                bundle.tables,
                bundle.config,
                bundle.params,
                bundle.randomization,
                reward_fn,
            )
            terminal_features = _policy_features(
                transition.transition_observation,
                bundle,
            )
            _, _, transition_value = actor_critic(params, terminal_features)
            rewards = select_ego_rewards(
                transition.rewards,
                bundle.config,
            )
            ego = bundle.config.episode.ego_index
            transition_speed = transition.transition_observation.standard_state[
                :, ego, 3
            ]
            sample = RolloutBatch(
                observations=features,
                latent_actions=latent,
                old_log_probs=log_prob,
                old_values=value,
                rewards=rewards,
                terminated=transition.metrics.status.terminated,
                truncated=transition.metrics.status.truncated,
                transition_values=transition_value,
                transition_speeds=transition_speed,
            )
            carry = (
                transition.state,
                transition.next_observation,
                current_key,
            )
            return carry, sample

        carry, rollout = jax.lax.scan(
            rollout_step,
            (batch_state, observation, key),
            xs=None,
            length=config.rollout_steps,
        )
        return carry, rollout

    def iteration(params, optimizer, batch_state, observation, key):
        key, rollout_key, update_key = jax.random.split(key, 3)
        (batch_state, observation, _), rollout = collect_rollout(
            params,
            batch_state,
            observation,
            rollout_key,
        )
        training_batch = prepare_training_batch(
            rollout,
            config.gamma,
            config.gae_lambda,
        )
        params, optimizer, metrics = ppo_update(
            params,
            optimizer,
            training_batch,
            update_key,
            config,
        )
        checksum = (
            _tree_checksum(params)
            + _tree_checksum(batch_state)
            + jnp.sum(metrics)
        )
        output_metrics = IterationMetrics(
            total_loss=metrics[0],
            policy_loss=metrics[1],
            value_loss=metrics[2],
            entropy=metrics[3],
            approximate_kl=metrics[4],
            clip_fraction=metrics[5],
            gradient_norm=metrics[6],
            rollout_reward=jnp.mean(rollout.rewards),
            rollout_speed=jnp.mean(rollout.transition_speeds),
            checksum=checksum,
        )
        return params, optimizer, batch_state, observation, key, output_metrics

    return jax.jit(iteration)


def make_evaluator(bundle: CoreBundle, config: TrainingConfig):
    """Create a deterministic evaluator whose explicit root key is replayable."""
    reward_fn = _speed_reward(config.target_speed)

    def evaluate(params, root_key):
        reset_root = jax.random.fold_in(root_key, 0)
        reset_keys = jax.random.split(reset_root, config.batch_size)
        observation, batch_state = reset_batch(
            reset_keys,
            bundle.tables,
            bundle.config,
            bundle.params,
            bundle.randomization,
        )

        def evaluation_step(carry, step_index):
            current_state, current_observation = carry
            features = _policy_features(current_observation, bundle)
            means, _, _ = actor_critic(params, features)
            normalized = jnp.tanh(means)[:, None, :]
            actions = scale_normalized_actions(
                normalized,
                current_state.params,
                bundle.config,
            )
            step_root = jax.random.fold_in(root_key, 1 + 2 * step_index)
            next_reset_root = jax.random.fold_in(root_key, 2 + 2 * step_index)
            step_keys = jax.random.split(step_root, config.batch_size)
            next_reset_keys = jax.random.split(
                next_reset_root,
                config.batch_size,
            )
            transition = step_batch_autoreset(
                step_keys,
                next_reset_keys,
                current_state,
                actions,
                bundle.tables,
                bundle.config,
                bundle.params,
                bundle.randomization,
                reward_fn,
            )
            rewards = select_ego_rewards(
                transition.rewards,
                bundle.config,
            )
            ego = bundle.config.episode.ego_index
            speeds = transition.transition_observation.standard_state[:, ego, 3]
            return (
                transition.state,
                transition.next_observation,
            ), (rewards, speeds)

        (batch_state, _), (rewards, speeds) = jax.lax.scan(
            evaluation_step,
            (batch_state, observation),
            jnp.arange(config.evaluation_steps, dtype=jnp.int32),
        )
        return Evaluation(
            score=jnp.mean(rewards),
            mean_speed=jnp.mean(speeds),
            mean_absolute_speed_error=jnp.mean(
                jnp.abs(speeds - config.target_speed)
            ),
            checksum=_tree_checksum(batch_state) + jnp.sum(rewards),
        )

    return jax.jit(evaluate)


def run_validation(config: TrainingConfig) -> ValidationResult:
    """Train, synchronize and evaluate one complete native PPO job."""
    bundle = build_training_bundle(config)
    root_key = jax.random.key(config.seed)
    init_key, train_reset_root, train_key, evaluation_key = jax.random.split(
        root_key,
        4,
    )
    params = initialize_actor_critic(init_key, 2, config.hidden_size)
    params = jax.device_put(params, bundle.device)
    optimizer = initialize_adam(params)
    train_reset_keys = jax.random.split(
        train_reset_root,
        config.batch_size,
    )
    reset = jax.jit(
        lambda keys: reset_batch(
            keys,
            bundle.tables,
            bundle.config,
            bundle.params,
            bundle.randomization,
        )
    )
    observation, batch_state = _block_tree(reset(train_reset_keys))
    iteration = make_iteration(bundle, config)
    evaluator = make_evaluator(bundle, config)

    baseline = _block_tree(evaluator(params, evaluation_key))

    compile_start = time.perf_counter()
    outputs = iteration(
        params,
        optimizer,
        batch_state,
        observation,
        train_key,
    )
    outputs = _block_tree(outputs)
    compile_seconds = time.perf_counter() - compile_start
    params, optimizer, batch_state, observation, train_key, metrics = outputs

    steady_start = time.perf_counter()
    for _ in range(1, config.updates):
        outputs = iteration(
            params,
            optimizer,
            batch_state,
            observation,
            train_key,
        )
        params, optimizer, batch_state, observation, train_key, metrics = outputs
    outputs = _block_tree(
        (params, optimizer, batch_state, observation, train_key, metrics)
    )
    steady_seconds = time.perf_counter() - steady_start
    params, optimizer, batch_state, observation, train_key, metrics = outputs
    del optimizer, batch_state, observation, train_key

    final = _block_tree(evaluator(params, evaluation_key))
    finite = _tree_is_finite((params, metrics, baseline, final))
    improvement = float(final.score - baseline.score)
    passed = finite and improvement >= config.min_improvement
    steady_steps = (
        max(0, config.updates - 1)
        * config.batch_size
        * config.rollout_steps
    )
    throughput = steady_steps / steady_seconds if steady_seconds else math.inf
    total_steps = (
        config.updates * config.batch_size * config.rollout_steps
    )
    return ValidationResult(
        backend=jax.default_backend(),
        device=str(bundle.device),
        environment_steps=total_steps,
        compile_seconds=compile_seconds,
        steady_seconds=steady_seconds,
        steady_environment_steps_per_second=throughput,
        baseline_score=float(baseline.score),
        final_score=float(final.score),
        score_improvement=improvement,
        baseline_mean_speed=float(baseline.mean_speed),
        final_mean_speed=float(final.mean_speed),
        final_speed_error=float(final.mean_absolute_speed_error),
        final_loss=float(metrics.total_loss),
        final_policy_loss=float(metrics.policy_loss),
        final_value_loss=float(metrics.value_loss),
        final_entropy=float(metrics.entropy),
        final_approximate_kl=float(metrics.approximate_kl),
        final_clip_fraction=float(metrics.clip_fraction),
        final_gradient_norm=float(metrics.gradient_norm),
        checksum=float(metrics.checksum + final.checksum),
        passed=passed,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=TrainingConfig.batch_size,
    )
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=TrainingConfig.rollout_steps,
    )
    parser.add_argument("--updates", type=int, default=TrainingConfig.updates)
    parser.add_argument(
        "--update-epochs",
        type=int,
        default=TrainingConfig.update_epochs,
    )
    parser.add_argument(
        "--minibatches",
        type=int,
        default=TrainingConfig.minibatches,
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=TrainingConfig.hidden_size,
    )
    parser.add_argument(
        "--episode-steps",
        type=int,
        default=TrainingConfig.episode_steps,
    )
    parser.add_argument(
        "--evaluation-steps",
        type=int,
        default=TrainingConfig.evaluation_steps,
    )
    parser.add_argument(
        "--target-speed",
        type=float,
        default=TrainingConfig.target_speed,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=TrainingConfig.learning_rate,
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=TrainingConfig.min_improvement,
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "gpu"),
        default=TrainingConfig.device,
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    config = TrainingConfig(**vars(args))
    result = run_validation(config)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
