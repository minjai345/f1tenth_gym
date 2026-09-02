"""Train a single F1TENTH policy with an end-to-end JAX PPO loop.

The environment batch, policy, rollout, advantage calculation, minibatches,
and optimizer all remain on one JAX device.  The structure follows the
PureJaxRL single-file style while using the native F1TENTH batch API directly.

Install the training dependencies and launch the default GPU job with::

    uv sync --extra train
    uv run --extra train python examples/jax_ppo_training.py

Use ``--smoke-test --device cpu`` for a small offline circular-track run.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import NamedTuple

from flax import linen as nn
from flax import serialization
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax

from f1tenth_gym.envs.action import (
    LongitudinalActionType,
    SteerActionType,
)
from f1tenth_gym.envs.batching import (
    BatchState,
    scale_normalized_actions,
    select_ego_rewards,
)
from f1tenth_gym.envs.collision_models import CollisionCheckMode
from f1tenth_gym.envs.contact import ContactConfig
from f1tenth_gym.envs.dynamic_models import (
    DynamicModel,
    F1TENTH_VEHICLE_PARAMETERS,
)
from f1tenth_gym.envs.env_config import (
    ControlConfig,
    DomainRandomizationConfig,
    EnvConfig,
    ResetConfig,
    RewardConfig,
    RewardMode,
    SimulationConfig,
    TerminationConfig,
)
from f1tenth_gym.envs.integrators import IntegratorType
from f1tenth_gym.envs.jax_core import CoreObservation
from f1tenth_gym.envs.jax_simulator import JaxSimulator
from f1tenth_gym.envs.lidar import LiDARConfig
from f1tenth_gym.envs.reset import ResetStrategy
from f1tenth_gym.envs.track import Track


LOG_TWO_PI = math.log(2.0 * math.pi)


class Transition(NamedTuple):
    """One time-major PPO sample for every environment row."""

    observations: jax.Array
    latent_actions: jax.Array
    old_log_probs: jax.Array
    old_values: jax.Array
    rewards: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    transition_values: jax.Array
    speeds: jax.Array


class TrainingBatch(NamedTuple):
    """Flattened samples consumed by one PPO minibatch update."""

    observations: jax.Array
    latent_actions: jax.Array
    old_log_probs: jax.Array
    old_values: jax.Array
    advantages: jax.Array
    returns: jax.Array


class UpdateMetrics(NamedTuple):
    """One scalar summary per PPO rollout and update."""

    total_loss: jax.Array
    policy_loss: jax.Array
    value_loss: jax.Array
    entropy: jax.Array
    approximate_kl: jax.Array
    clip_fraction: jax.Array
    gradient_norm: jax.Array
    rollout_reward: jax.Array
    rollout_speed: jax.Array


class TrainingResult(NamedTuple):
    """Final device-resident training carry and per-update metrics."""

    train_state: TrainState
    environment_state: BatchState
    observations: jax.Array
    key: jax.Array
    metrics: UpdateMetrics


class ActorCritic(nn.Module):
    """Two-layer continuous actor and critic in the PureJaxRL style."""

    hidden_size: int

    @nn.compact
    def __call__(
        self,
        observations: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        hidden = observations
        for _ in range(2):
            hidden = nn.Dense(
                self.hidden_size,
                kernel_init=nn.initializers.orthogonal(math.sqrt(2.0)),
                bias_init=nn.initializers.zeros_init(),
            )(hidden)
            hidden = nn.tanh(hidden)
        means = nn.Dense(
            2,
            kernel_init=nn.initializers.orthogonal(0.01),
            bias_init=nn.initializers.zeros_init(),
        )(hidden)
        log_std = self.param(
            "log_std",
            nn.initializers.constant(-0.5),
            (2,),
        )
        log_std = jnp.broadcast_to(jnp.clip(log_std, -5.0, 2.0), means.shape)

        hidden = observations
        for _ in range(2):
            hidden = nn.Dense(
                self.hidden_size,
                kernel_init=nn.initializers.orthogonal(math.sqrt(2.0)),
                bias_init=nn.initializers.zeros_init(),
            )(hidden)
            hidden = nn.tanh(hidden)
        values = nn.Dense(
            1,
            kernel_init=nn.initializers.orthogonal(1.0),
            bias_init=nn.initializers.zeros_init(),
        )(hidden)
        return means, log_std, jnp.squeeze(values, axis=-1)


class PPOConfig(NamedTuple):
    """Static dimensions and hyperparameters captured by the compiled job."""

    num_envs: int
    rollout_steps: int
    updates: int
    update_epochs: int
    minibatches: int
    hidden_size: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_epsilon: float
    value_coefficient: float
    entropy_coefficient: float
    max_gradient_norm: float


def circular_track() -> Track:
    """Return a mapless track for the offline smoke test."""
    theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    return Track.from_refline(
        x=8.0 * np.cos(theta),
        y=8.0 * np.sin(theta),
        velx=np.full(theta.shape, 3.0),
    )


def resolve_track(source: str, scale: float) -> Track:
    """Resolve a named track or filesystem track path."""
    path = Path(source)
    if "/" in source or "\\" in source or path.suffix:
        return Track.from_track_path(path, track_scale=scale)
    return Track.from_track_name(source, track_scale=scale)


def domain_randomization(enabled: bool) -> DomainRandomizationConfig:
    """Return modest independent episode ranges for core vehicle physics."""
    if not enabled:
        return DomainRandomizationConfig()
    nominal = F1TENTH_VEHICLE_PARAMETERS
    low = nominal.with_updates(
        mu=0.85 * nominal.mu,
        m=0.90 * nominal.m,
        I=0.90 * nominal.I,
    )
    high = nominal.with_updates(
        mu=1.15 * nominal.mu,
        m=1.10 * nominal.m,
        I=1.10 * nominal.I,
    )
    return DomainRandomizationConfig(enabled=True, low=low, high=high)


def training_config(
    track: Track,
    device: str,
    *,
    episode_steps: int,
    smoke_test: bool,
    randomize: bool,
) -> EnvConfig:
    """Build the native single-policy racing task."""
    collision_mode = (
        CollisionCheckMode.NONE
        if smoke_test
        else CollisionCheckMode.SEGMENT_CONTACT
    )
    return EnvConfig(
        map_name=track,
        num_agents=1,
        control_config=ControlConfig(
            longitudinal_mode=LongitudinalActionType.ACCL,
            steering_mode=SteerActionType.STEERING_SPEED,
        ),
        simulation_config=SimulationConfig(
            dynamics_model=(
                DynamicModel.KS if smoke_test else DynamicModel.ST
            ),
            integrator=IntegratorType.EULER,
            max_laps=None,
        ),
        reset_config=ResetConfig(strategy=ResetStrategy.RL_RANDOM_STATIC),
        lidar_config=LiDARConfig(enabled=False),
        contact_config=ContactConfig(device=device),
        termination_config=TerminationConfig(
            max_episode_steps=episode_steps,
            terminate_on_collision=not smoke_test,
        ),
        reward_config=RewardConfig(
            mode=RewardMode.PROGRESS,
            progress_weight=1.0,
            velocity_weight=0.02,
            timestep_weight=0.1,
            collision_penalty=5.0,
        ),
        domain_randomization_config=domain_randomization(randomize),
        collision_check=collision_mode,
        render_enabled=False,
    )


def policy_features(
    observation: CoreObservation,
    simulator: JaxSimulator,
) -> jax.Array:
    """Return normalized ego-local dynamics and Frenet errors as ``(B, 6)``."""
    ego = simulator.config.episode.ego_index
    standard = observation.standard_state[:, ego]
    frenet = observation.frenet[:, ego]
    features = jnp.stack(
        (
            standard[:, 2],
            standard[:, 3],
            standard[:, 5],
            standard[:, 6],
            frenet[:, 1],
            frenet[:, 2],
        ),
        axis=-1,
    )
    scale = jnp.asarray((0.5, 10.0, 5.0, 1.0, 5.0, math.pi))
    return features / scale


def latent_gaussian_log_prob(
    latent_actions: jax.Array,
    means: jax.Array,
    log_std: jax.Array,
) -> jax.Array:
    """Return the joint log probability of each pre-tanh action."""
    standardized = (latent_actions - means) * jnp.exp(-log_std)
    coordinate_log_probs = -0.5 * (
        jnp.square(standardized) + 2.0 * log_std + LOG_TWO_PI
    )
    return jnp.sum(coordinate_log_probs, axis=-1)


def latent_gaussian_entropy(log_std: jax.Array) -> jax.Array:
    """Return joint latent-Gaussian entropy for every observation row."""
    return jnp.sum(log_std + 0.5 * (1.0 + LOG_TWO_PI), axis=-1)


def generalized_advantage_estimate(
    rewards: jax.Array,
    values: jax.Array,
    transition_values: jax.Array,
    terminated: jax.Array,
    truncated: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    """Calculate timeout-correct time-major GAE.

    Natural termination has no value bootstrap.  A timeout bootstraps from the
    terminal transition observation, while both statuses stop the recurrence
    before the selectively reset next episode.
    """
    shapes = {
        rewards.shape,
        values.shape,
        transition_values.shape,
        terminated.shape,
        truncated.shape,
    }
    if len(shapes) != 1 or rewards.ndim != 2:
        raise ValueError("GAE inputs must share shape (time, environments)")
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
    trajectory: Transition,
    config: PPOConfig,
) -> TrainingBatch:
    """Stop rollout gradients, calculate GAE, and flatten time/environment."""
    trajectory = jax.tree.map(jax.lax.stop_gradient, trajectory)
    advantages, returns = generalized_advantage_estimate(
        trajectory.rewards,
        trajectory.old_values,
        trajectory.transition_values,
        trajectory.terminated,
        trajectory.truncated,
        config.gamma,
        config.gae_lambda,
    )
    advantages = (advantages - jnp.mean(advantages)) / (
        jnp.std(advantages) + 1.0e-8
    )

    def flatten(value):
        return jnp.reshape(value, (-1,) + value.shape[2:])

    return TrainingBatch(
        observations=flatten(trajectory.observations),
        latent_actions=flatten(trajectory.latent_actions),
        old_log_probs=flatten(trajectory.old_log_probs),
        old_values=flatten(trajectory.old_values),
        advantages=flatten(advantages),
        returns=flatten(returns),
    )


def make_train(
    simulator: JaxSimulator,
    config: PPOConfig,
):
    """Return one compiled end-to-end PPO training program."""
    samples_per_update = config.num_envs * config.rollout_steps
    minibatch_size = samples_per_update // config.minibatches
    network = ActorCritic(config.hidden_size)
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_gradient_norm),
        optax.adam(config.learning_rate, eps=1.0e-5),
    )

    def loss_function(params, minibatch: TrainingBatch):
        means, log_std, values = network.apply(
            params,
            minibatch.observations,
        )
        log_probs = latent_gaussian_log_prob(
            minibatch.latent_actions,
            means,
            log_std,
        )
        log_ratios = log_probs - minibatch.old_log_probs
        ratios = jnp.exp(log_ratios)
        clipped_ratios = jnp.clip(
            ratios,
            1.0 - config.clip_epsilon,
            1.0 + config.clip_epsilon,
        )
        policy_loss = jnp.mean(
            jnp.maximum(
                -minibatch.advantages * ratios,
                -minibatch.advantages * clipped_ratios,
            )
        )

        clipped_values = minibatch.old_values + jnp.clip(
            values - minibatch.old_values,
            -config.clip_epsilon,
            config.clip_epsilon,
        )
        value_errors = jnp.square(values - minibatch.returns)
        clipped_value_errors = jnp.square(
            clipped_values - minibatch.returns
        )
        value_loss = 0.5 * jnp.mean(
            jnp.maximum(value_errors, clipped_value_errors)
        )
        entropy = jnp.mean(latent_gaussian_entropy(log_std))
        total_loss = (
            policy_loss
            + config.value_coefficient * value_loss
            - config.entropy_coefficient * entropy
        )
        approximate_kl = jnp.mean((ratios - 1.0) - log_ratios)
        clip_fraction = jnp.mean(
            (jnp.abs(ratios - 1.0) > config.clip_epsilon).astype(
                values.dtype
            )
        )
        diagnostics = jnp.stack(
            (
                total_loss,
                policy_loss,
                value_loss,
                entropy,
                approximate_kl,
                clip_fraction,
            )
        )
        return total_loss, diagnostics

    def train(root_key: jax.Array) -> TrainingResult:
        root_key, reset_root, initialization_key = jax.random.split(
            root_key,
            3,
        )
        reset_keys = jax.random.split(reset_root, config.num_envs)
        observation, environment_state = simulator.reset_batch(reset_keys)
        features = policy_features(observation, simulator)
        network_params = network.init(initialization_key, features)
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=optimizer,
        )

        def update_step(runner, _unused):
            train_state, environment_state, features, key = runner

            def environment_step(carry, _unused):
                environment_state, features, key = carry
                key, action_key, step_root, reset_root = jax.random.split(
                    key,
                    4,
                )
                means, log_std, values = train_state.apply_fn(
                    train_state.params,
                    features,
                )
                noise = jax.random.normal(
                    action_key,
                    means.shape,
                    dtype=means.dtype,
                )
                latent_actions = means + jnp.exp(log_std) * noise
                normalized_actions = jnp.tanh(latent_actions)
                log_probs = latent_gaussian_log_prob(
                    latent_actions,
                    means,
                    log_std,
                )
                physical_actions = scale_normalized_actions(
                    normalized_actions[:, None, :],
                    environment_state,
                    simulator.config,
                )
                step_keys = jax.random.split(step_root, config.num_envs)
                reset_keys = jax.random.split(reset_root, config.num_envs)
                transition = simulator.step_batch_autoreset(
                    step_keys,
                    reset_keys,
                    environment_state,
                    physical_actions,
                )
                terminal_features = policy_features(
                    transition.transition_observation,
                    simulator,
                )
                _, _, transition_values = train_state.apply_fn(
                    train_state.params,
                    terminal_features,
                )
                rewards = select_ego_rewards(
                    transition.rewards,
                    simulator.config,
                )
                ego = simulator.config.episode.ego_index
                speeds = transition.transition_observation.standard_state[
                    :, ego, 3
                ]
                sample = Transition(
                    observations=features,
                    latent_actions=latent_actions,
                    old_log_probs=log_probs,
                    old_values=values,
                    rewards=rewards,
                    terminated=transition.metrics.status.terminated,
                    truncated=transition.metrics.status.truncated,
                    transition_values=transition_values,
                    speeds=speeds,
                )
                next_features = policy_features(
                    transition.next_observation,
                    simulator,
                )
                return (
                    transition.state,
                    next_features,
                    key,
                ), sample

            (
                environment_state,
                features,
                key,
            ), trajectory = jax.lax.scan(
                environment_step,
                (environment_state, features, key),
                xs=None,
                length=config.rollout_steps,
            )
            training_batch = prepare_training_batch(trajectory, config)

            def update_minibatch(train_state, minibatch):
                (loss, diagnostics), gradients = jax.value_and_grad(
                    loss_function,
                    has_aux=True,
                )(train_state.params, minibatch)
                del loss
                gradient_norm = optax.tree.norm(gradients)
                train_state = train_state.apply_gradients(grads=gradients)
                return train_state, jnp.concatenate(
                    (diagnostics, gradient_norm[None])
                )

            def update_epoch(carry, _unused):
                train_state, key = carry
                key, permutation_key = jax.random.split(key)
                permutation = jax.random.permutation(
                    permutation_key,
                    samples_per_update,
                )
                shuffled = jax.tree.map(
                    lambda value: jnp.take(value, permutation, axis=0),
                    training_batch,
                )
                minibatches = jax.tree.map(
                    lambda value: jnp.reshape(
                        value,
                        (config.minibatches, minibatch_size)
                        + value.shape[1:],
                    ),
                    shuffled,
                )
                train_state, metrics = jax.lax.scan(
                    update_minibatch,
                    train_state,
                    minibatches,
                )
                return (train_state, key), metrics

            (train_state, key), loss_metrics = jax.lax.scan(
                update_epoch,
                (train_state, key),
                xs=None,
                length=config.update_epochs,
            )
            loss_metrics = jnp.mean(loss_metrics, axis=(0, 1))
            metrics = UpdateMetrics(
                total_loss=loss_metrics[0],
                policy_loss=loss_metrics[1],
                value_loss=loss_metrics[2],
                entropy=loss_metrics[3],
                approximate_kl=loss_metrics[4],
                clip_fraction=loss_metrics[5],
                gradient_norm=loss_metrics[6],
                rollout_reward=jnp.mean(trajectory.rewards),
                rollout_speed=jnp.mean(trajectory.speeds),
            )
            runner = (train_state, environment_state, features, key)
            return runner, metrics

        runner, metrics = jax.lax.scan(
            update_step,
            (train_state, environment_state, features, root_key),
            xs=None,
            length=config.updates,
        )
        train_state, environment_state, features, root_key = runner
        return TrainingResult(
            train_state=train_state,
            environment_state=environment_state,
            observations=features,
            key=root_key,
            metrics=metrics,
        )

    return jax.jit(train)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", default="Spielberg", help="track name or path")
    parser.add_argument("--map-scale", type=float, default=1.0)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--total-timesteps", type=int, default=1_048_576)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatches", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--episode-steps", type=int, default=1_000)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("jax_f1tenth_ppo.msgpack"),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="use a synthetic circular track and disable contact",
    )
    parser.add_argument(
        "--no-domain-randomization",
        action="store_true",
        help="keep nominal vehicle parameters for every episode",
    )
    parser.add_argument("--no-save", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> PPOConfig:
    """Validate CLI dimensions and return the static compiled configuration."""
    integer_names = (
        "num_envs",
        "rollout_steps",
        "total_timesteps",
        "update_epochs",
        "minibatches",
        "hidden_size",
        "episode_steps",
    )
    for name in integer_names:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    samples_per_update = args.num_envs * args.rollout_steps
    if samples_per_update < 2:
        raise ValueError(
            "--num-envs * --rollout-steps must provide at least two samples"
        )
    if args.total_timesteps % samples_per_update:
        raise ValueError(
            "--total-timesteps must be divisible by --num-envs * "
            f"--rollout-steps ({samples_per_update})"
        )
    if samples_per_update % args.minibatches:
        raise ValueError(
            "--minibatches must divide --num-envs * --rollout-steps"
        )
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")
    if args.map_scale <= 0.0:
        raise ValueError("--map-scale must be positive")
    return PPOConfig(
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        updates=args.total_timesteps // samples_per_update,
        update_epochs=args.update_epochs,
        minibatches=args.minibatches,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coefficient=0.5,
        entropy_coefficient=1.0e-3,
        max_gradient_norm=0.5,
    )


def main() -> None:
    args = parse_args()
    ppo_config = validate_args(args)
    try:
        device = jax.devices(args.device)[0]
    except (IndexError, RuntimeError) as error:
        raise SystemExit(
            f"No JAX {args.device} device is available. Install the CUDA "
            "extra for GPU use or pass `--device cpu`."
        ) from error

    track = circular_track() if args.smoke_test else resolve_track(
        args.map,
        args.map_scale,
    )
    env_config = training_config(
        track,
        args.device,
        episode_steps=args.episode_steps,
        smoke_test=args.smoke_test,
        randomize=not args.no_domain_randomization,
    )
    simulator = JaxSimulator(env_config, track, device=device)
    train = make_train(simulator, ppo_config)
    root_key = jax.device_put(jax.random.key(args.seed), device)

    print(f"Training device: {simulator.device}")
    print(
        "Randomized fields: "
        f"{env_config.domain_randomization_config.randomized_fields() or 'none'}"
    )
    compile_start = time.perf_counter()
    compiled_train = train.lower(root_key).compile()
    compile_seconds = time.perf_counter() - compile_start

    run_start = time.perf_counter()
    result = compiled_train(root_key)
    result = jax.tree.map(
        lambda value: value.block_until_ready(),
        result,
    )
    train_seconds = time.perf_counter() - run_start
    environment_steps = (
        ppo_config.num_envs
        * ppo_config.rollout_steps
        * ppo_config.updates
    )
    final = jax.device_get(jax.tree.map(lambda value: value[-1], result.metrics))
    report = {
        "compile_seconds": compile_seconds,
        "environment_steps": environment_steps,
        "environment_steps_per_second": environment_steps / train_seconds,
        "final_approximate_kl": float(final.approximate_kl),
        "final_clip_fraction": float(final.clip_fraction),
        "final_entropy": float(final.entropy),
        "final_gradient_norm": float(final.gradient_norm),
        "final_policy_loss": float(final.policy_loss),
        "final_rollout_reward": float(final.rollout_reward),
        "final_rollout_speed": float(final.rollout_speed),
        "final_total_loss": float(final.total_loss),
        "final_value_loss": float(final.value_loss),
        "train_seconds": train_seconds,
        "updates": ppo_config.updates,
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    if not args.no_save:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(serialization.to_bytes(result.train_state.params))
        print(f"Saved policy parameters to {args.output}")


if __name__ == "__main__":
    main()
