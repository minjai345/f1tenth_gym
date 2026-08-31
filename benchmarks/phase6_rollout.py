#!/usr/bin/env python3
"""Measure Phase 6 device batches against equivalent mutable Gym rollouts.

Run from the repository root, for example::

    uv run --no-sync python -m benchmarks.phase6_rollout \
        --batch-sizes 1,16,64 --rollout-length 256 --agents 1

Use ``--scenario lidar``, ``contact`` or ``full`` to include exact sensing or
contact, and ``--unique-maps`` to exercise equal-shape indexed map selection.

The only stdout payload is JSON.  Preprocessing, construction, reset, key
generation, compilation, and the first synchronized execution are excluded
from steady timing.  The JAX program returns a checksum depending on every
transition leaf, and every timed call is explicitly synchronized.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import sys
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
    ObservationConfig,
    ResetConfig,
    SimulationConfig,
    TerminationConfig,
)
from f1tenth_gym.envs.f110_env import F110Env
from f1tenth_gym.envs.integrators import IntegratorType
from f1tenth_gym.envs.lidar import LiDARConfig
from f1tenth_gym.envs.observation import ObservationType
from f1tenth_gym.envs.reset import ReferenceLine, ResetStrategy
from f1tenth_gym.envs.track import Track
from f1tenth_gym.jax import (
    reset_batch_from_poses,
    reset_indexed_batch_from_poses,
    step_batch,
    step_indexed_batch,
)
from f1tenth_gym.jax.builder import (
    CoreBundle,
    IndexedCoreBundle,
    build_core,
    build_indexed_core,
)

try:
    from ._phase6_helpers import (
        BENCHMARK_NAME,
        SCHEMA_VERSION,
        device_peak_memory,
        parse_positive_csv,
        timing_summary,
        unavailable_memory,
        validate_backend_result,
        validate_report,
    )
except ImportError:  # direct ``python benchmarks/phase6_rollout.py`` execution
    from _phase6_helpers import (  # type: ignore[no-redef]
        BENCHMARK_NAME,
        SCHEMA_VERSION,
        device_peak_memory,
        parse_positive_csv,
        timing_summary,
        unavailable_memory,
        validate_backend_result,
        validate_report,
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be >= 1")
    return parsed


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-sizes",
        default="1,16,64",
        help="comma-separated environment batch sizes (default: 1,16,64)",
    )
    parser.add_argument("--rollout-length", type=_positive_int, default=256)
    parser.add_argument("--agents", type=_positive_int, default=1)
    parser.add_argument(
        "--model",
        choices=("KS", "ST"),
        default="KS",
    )
    parser.add_argument(
        "--scenario",
        choices=("state", "lidar", "contact", "full"),
        default="state",
    )
    parser.add_argument(
        "--beams",
        type=_positive_int,
        default=1080,
        help="beam count for lidar/full scenarios (default: 1080)",
    )
    parser.add_argument(
        "--unique-maps",
        type=_positive_int,
        default=1,
        help="translated exact-shape maps assigned round-robin to rows",
    )
    parser.add_argument(
        "--track-points",
        type=_positive_int,
        default=96,
        help="reference-line samples used to build each synthetic map",
    )
    parser.add_argument("--repetitions", type=_positive_int, default=5)
    parser.add_argument("--warmup-runs", type=_positive_int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--backend",
        choices=("both", "jax", "mutable"),
        default="both",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="authoritative JAX device request; auto uses JAX's default device",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write JSON to this path instead of stdout",
    )
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.batch_sizes = parse_positive_csv(
            args.batch_sizes,
            name="batch sizes",
        )
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    if args.unique_maps > min(args.batch_sizes):
        parser.error("unique maps must not exceed the smallest batch size")
    if args.track_points < 8:
        parser.error("track points must be >= 8")
    return args


def _make_track(
    agents: int,
    point_count: int,
    *,
    center_x: float = 0.0,
) -> Track:
    minimum_circumference = max(2.0 * np.pi * 8.0, 3.0 * agents)
    radius = minimum_circumference / (2.0 * np.pi)
    theta = np.linspace(0.0, 2.0 * np.pi, point_count, endpoint=False)
    return Track.from_refline(
        x=center_x + radius * np.cos(theta),
        y=radius * np.sin(theta),
        velx=np.full(point_count, 3.0),
    )


def _make_config(
    track: Track,
    agents: int,
    seed: int,
    scenario: str,
    beams: int,
    model: str,
) -> EnvConfig:
    lidar_enabled = scenario in ("lidar", "full")
    contact_enabled = scenario in ("contact", "full")
    return EnvConfig(
        seed=seed,
        map_name=track,
        num_agents=agents,
        control_config=ControlConfig(
            longitudinal_mode=LongitudinalActionType.ACCL,
            steering_mode=SteerActionType.STEERING_SPEED,
        ),
        simulation_config=SimulationConfig(
            timestep=0.01,
            integrator_timestep=0.01,
            integrator=IntegratorType.EULER,
            dynamics_model=(
                DynamicModel.KS if model == "KS" else DynamicModel.ST
            ),
            max_laps=None,
        ),
        observation_config=ObservationConfig(
            type=ObservationType.KINEMATIC_STATE,
        ),
        reset_config=ResetConfig(
            strategy=ResetStrategy.RL_GRID_STATIC,
            shuffle=False,
            move_laterally=False,
            reference_line=ReferenceLine.CENTERLINE,
        ),
        lidar_config=LiDARConfig(
            enabled=lidar_enabled,
            num_beams=beams if lidar_enabled else 1,
            range_max=12.0,
            noise_std=0.0,
        ),
        termination_config=TerminationConfig(
            max_episode_steps=None,
            terminate_on_collision=False,
        ),
        collision_check=(
            CollisionCheckMode.SEGMENT_CONTACT
            if contact_enabled
            else CollisionCheckMode.NONE
        ),
        render_enabled=False,
    )


def _explicit_poses(track: Track, agents: int) -> np.ndarray:
    indices = np.linspace(
        0,
        track.centerline.n,
        num=agents,
        endpoint=False,
        dtype=np.int32,
    )
    return np.stack(
        (
            track.centerline.xs[indices],
            track.centerline.ys[indices],
            track.centerline.yaws[indices],
        ),
        axis=1,
    ).astype(np.float32)


def _resolve_device(requested: str) -> jax.Device:
    if requested == "auto":
        devices = jax.devices()
    else:
        try:
            devices = jax.devices(requested)
        except RuntimeError as error:
            raise RuntimeError(
                f"requested JAX platform {requested!r} is unavailable"
            ) from error
    if not devices:
        raise RuntimeError(f"requested JAX platform {requested!r} is unavailable")
    return devices[0]


def _tree_probe(tree: Any) -> jax.Array:
    """Make one scalar depend on every element of every array leaf."""
    total = jnp.asarray(0.0, dtype=jnp.float32)
    for index, leaf in enumerate(jax.tree.leaves(tree)):
        value = jnp.asarray(leaf)
        if value.size:
            weight = jnp.float32(1.0 + (index % 13) / 16.0)
            total = total + jnp.sum(value.astype(jnp.float32)) * weight
    return total


def _rollout_program(
    bundle: CoreBundle | IndexedCoreBundle,
    tables: Any,
    map_indices: jax.Array | None,
):
    config = bundle.config

    def rollout(state, step_keys, actions):
        def one_step(carry, keys):
            current_state, checksum = carry
            if map_indices is None:
                transition = step_batch(
                    keys,
                    current_state,
                    actions,
                    tables,
                    config,
                )
            else:
                transition = step_indexed_batch(
                    keys,
                    map_indices,
                    current_state,
                    actions,
                    tables,
                    config,
                )
            return (
                transition.state,
                checksum + _tree_probe(transition),
            ), None

        (_, checksum), _ = jax.lax.scan(
            one_step,
            (state, jnp.asarray(0.0, dtype=jnp.float32)),
            step_keys,
        )
        return checksum

    return rollout


def _tree_nbytes(tree: Any) -> int:
    total = 0
    for leaf in jax.tree.leaves(tree):
        value = jnp.asarray(leaf)
        total += int(value.nbytes)
    return total


def _device_record(device: jax.Device) -> dict[str, Any]:
    return {
        "platform": str(device.platform),
        "kind": str(getattr(device, "device_kind", "unknown")),
        "id": int(device.id),
    }


def _benchmark_jax(
    config: EnvConfig,
    tracks: tuple[Track, ...],
    poses: np.ndarray,
    device: jax.Device,
    *,
    batch_size: int,
    rollout_length: int,
    agents: int,
    repetitions: int,
    warmup_runs: int,
    seed: int,
    scenario: str,
    active_lidar_beams: int,
    unique_maps: int,
    track_points: int,
) -> dict[str, Any]:
    if unique_maps == 1:
        bundle: CoreBundle | IndexedCoreBundle = build_core(
            config,
            tracks[0],
            target_device=device,
        )
        tables = bundle.tables
        map_indices = None
    else:
        bundle = build_indexed_core(
            config,
            tracks,
            target_device=device,
        )
        if len(bundle.buckets) != 1:
            raise RuntimeError(
                "translated benchmark maps did not form one exact-shape bucket"
            )
        bucket = bundle.buckets[0]
        if bucket.source_indices != tuple(range(batch_size)):
            raise RuntimeError("indexed benchmark routing changed row order")
        tables = bucket.tables
        map_indices = bucket.map_indices

    root_key = jax.random.key(seed)
    reset_key = jax.random.fold_in(root_key, batch_size)
    reset_keys = jax.device_put(
        jax.random.split(reset_key, batch_size),
        bundle.device,
    )
    batch_poses = jnp.broadcast_to(
        jax.device_put(jnp.asarray(poses), bundle.device),
        (batch_size, agents, 3),
    )

    if map_indices is None:
        reset_fn = jax.jit(
            lambda keys, value: reset_batch_from_poses(
                keys,
                value,
                tables,
                bundle.config,
                bundle.params,
                bundle.randomization,
            )
        )
    else:
        reset_fn = jax.jit(
            lambda keys, value: reset_indexed_batch_from_poses(
                keys,
                map_indices,
                value,
                tables,
                bundle.config,
                bundle.params,
                bundle.randomization,
            )
        )
    _observation, state = reset_fn(reset_keys, batch_poses)
    state.core.dynamics.model.block_until_ready()

    step_root = jax.random.fold_in(root_key, batch_size + 1_000_003)
    step_keys = jax.random.split(
        step_root,
        rollout_length * batch_size,
    ).reshape((rollout_length, batch_size))
    step_keys = jax.device_put(step_keys, bundle.device)
    actions = jax.device_put(
        jnp.zeros((batch_size, agents, 2), dtype=jnp.float32),
        bundle.device,
    )

    rollout = jax.jit(_rollout_program(bundle, tables, map_indices))
    compile_started = time.perf_counter()
    executable = rollout.lower(state, step_keys, actions).compile()
    compile_seconds = time.perf_counter() - compile_started

    warmup_seconds = []
    checksum = None
    for _ in range(warmup_runs):
        started = time.perf_counter()
        checksum = executable(state, step_keys, actions)
        checksum.block_until_ready()
        warmup_seconds.append(time.perf_counter() - started)

    steady_seconds = []
    for _ in range(repetitions):
        started = time.perf_counter()
        checksum = executable(state, step_keys, actions)
        checksum.block_until_ready()
        steady_seconds.append(time.perf_counter() - started)

    if checksum is None:
        raise RuntimeError("the synchronized JAX rollout did not execute")
    checksum_value = float(np.asarray(checksum))
    if not math.isfinite(checksum_value):
        raise RuntimeError(f"the JAX checksum is not finite: {checksum_value}")
    timing = timing_summary(
        steady_seconds,
        batch_size=batch_size,
        rollout_length=rollout_length,
        agents=agents,
    )
    try:
        memory_stats = bundle.device.memory_stats()
    except (RuntimeError, NotImplementedError):
        memory_stats = None
    result = {
        "backend": "jax_device_batch",
        "batch_size": batch_size,
        "agents": agents,
        "rollout_length": rollout_length,
        "scenario": scenario,
        "unique_maps": unique_maps,
        "track_points": track_points,
        "active_lidar_beams": active_lidar_beams,
        "lidar_enabled": scenario in ("lidar", "full"),
        "contact_enabled": scenario in ("contact", "full"),
        "dtype": str(state.core.dynamics.model.dtype),
        "device": _device_record(bundle.device),
        "compile_seconds": compile_seconds,
        "warmup_seconds": warmup_seconds,
        **timing,
        "checksum": checksum_value,
        "resident_input_bytes": _tree_nbytes((state, step_keys, actions)),
        "resident_table_bytes": _tree_nbytes(tables),
        "peak_memory": device_peak_memory(memory_stats),
    }
    validate_backend_result(result)
    return result


def _numpy_probe(value: Any) -> float:
    if isinstance(value, dict):
        return sum(_numpy_probe(value[key]) for key in sorted(value))
    if isinstance(value, (tuple, list)):
        return sum(_numpy_probe(item) for item in value)
    array = np.asarray(value)
    if not array.size or not np.issubdtype(array.dtype, np.number):
        return 0.0
    return float(array.reshape(-1)[0])


def _mutable_rollout(
    environments: list[F110Env],
    actions: np.ndarray,
    rollout_length: int,
) -> float:
    checksum = 0.0
    for _ in range(rollout_length):
        for environment in environments:
            transition = environment.step(actions)
            if transition[2] or transition[3]:
                raise RuntimeError(
                    "the non-terminating benchmark configuration ended an episode"
                )
            checksum += _numpy_probe(transition)
    return checksum


def _benchmark_mutable(
    config: EnvConfig,
    tracks: tuple[Track, ...],
    poses: np.ndarray,
    *,
    batch_size: int,
    rollout_length: int,
    agents: int,
    repetitions: int,
    warmup_runs: int,
    seed: int,
    scenario: str,
    active_lidar_beams: int,
    unique_maps: int,
    track_points: int,
) -> dict[str, Any]:
    environments = [
        F110Env(replace(config, map_name=track)) for track in tracks
    ]
    actions = np.zeros((agents, 2), dtype=np.float32)

    def reset_environments():
        for index, environment in enumerate(environments):
            environment.reset(
                seed=seed + index,
                options={"poses": poses[index].copy()},
            )

    try:
        warmup_seconds = []
        checksum = None
        for _ in range(warmup_runs):
            reset_environments()
            started = time.perf_counter()
            checksum = _mutable_rollout(
                environments,
                actions,
                rollout_length,
            )
            warmup_seconds.append(time.perf_counter() - started)

        steady_seconds = []
        for _ in range(repetitions):
            reset_environments()
            started = time.perf_counter()
            checksum = _mutable_rollout(
                environments,
                actions,
                rollout_length,
            )
            steady_seconds.append(time.perf_counter() - started)
    finally:
        for environment in environments:
            environment.close()

    if checksum is None or not math.isfinite(checksum):
        raise RuntimeError(f"the mutable checksum is not finite: {checksum}")
    timing = timing_summary(
        steady_seconds,
        batch_size=batch_size,
        rollout_length=rollout_length,
        agents=agents,
    )
    result = {
        "backend": "mutable_numpy_gym",
        "batch_size": batch_size,
        "agents": agents,
        "rollout_length": rollout_length,
        "scenario": scenario,
        "unique_maps": unique_maps,
        "track_points": track_points,
        "active_lidar_beams": active_lidar_beams,
        "lidar_enabled": scenario in ("lidar", "full"),
        "contact_enabled": scenario in ("contact", "full"),
        "dtype": "float32",
        "device": {
            "platform": "cpu",
            "kind": platform.processor() or platform.machine() or "unknown",
            "id": None,
        },
        "compile_seconds": None,
        "warmup_seconds": warmup_seconds,
        **timing,
        "checksum": float(checksum),
        "resident_input_bytes": None,
        "resident_table_bytes": None,
        "peak_memory": unavailable_memory(
            "mutable NumPy/Gym backend",
            "no portable benchmark-local peak allocator metric is exposed",
        ),
    }
    validate_backend_result(result)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    unique_tracks = tuple(
        _make_track(
            args.agents,
            args.track_points,
            center_x=50.0 * index,
        )
        for index in range(args.unique_maps)
    )
    config = _make_config(
        unique_tracks[0],
        args.agents,
        args.seed,
        args.scenario,
        args.beams,
        args.model,
    )
    active_lidar_beams = (
        args.beams if args.scenario in ("lidar", "full") else 0
    )
    include_jax = args.backend in ("both", "jax")
    include_mutable = args.backend in ("both", "mutable")
    device = _resolve_device(args.device) if include_jax else None

    results = []
    for batch_size in args.batch_sizes:
        tracks = tuple(
            unique_tracks[index % args.unique_maps]
            for index in range(batch_size)
        )
        poses = np.stack(
            [_explicit_poses(track, args.agents) for track in tracks]
        )
        if device is not None:
            results.append(
                _benchmark_jax(
                    config,
                    tracks,
                    poses,
                    device,
                    batch_size=batch_size,
                    rollout_length=args.rollout_length,
                    agents=args.agents,
                    repetitions=args.repetitions,
                    warmup_runs=args.warmup_runs,
                    seed=args.seed,
                    scenario=args.scenario,
                    active_lidar_beams=active_lidar_beams,
                    unique_maps=args.unique_maps,
                    track_points=args.track_points,
                )
            )
        if include_mutable:
            results.append(
                _benchmark_mutable(
                    config,
                    tracks,
                    poses,
                    batch_size=batch_size,
                    rollout_length=args.rollout_length,
                    agents=args.agents,
                    repetitions=args.repetitions,
                    warmup_runs=args.warmup_runs,
                    seed=args.seed,
                    scenario=args.scenario,
                    active_lidar_beams=active_lidar_beams,
                    unique_maps=args.unique_maps,
                    track_points=args.track_points,
                )
            )

    report = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "python": platform.python_version(),
            "logical_cpu_count": os.cpu_count(),
        },
        "jax": {
            "version": jax.__version__,
            "default_backend": jax.default_backend(),
            "x64_enabled": bool(jax.config.x64_enabled),
        },
        "configuration": {
            "batch_sizes": list(args.batch_sizes),
            "rollout_length": args.rollout_length,
            "agents": args.agents,
            "scenario": args.scenario,
            "unique_maps": args.unique_maps,
            "track_points": args.track_points,
            "repetitions": args.repetitions,
            "warmup_runs": args.warmup_runs,
            "seed": args.seed,
            "model": args.model,
            "integrator": "EULER",
            "timestep_seconds": 0.01,
            "controls": ["STEERING_SPEED", "ACCL"],
            "observation": "KINEMATIC_STATE",
            "state_only": args.scenario == "state",
            "lidar_enabled": args.scenario in ("lidar", "full"),
            "active_lidar_beams": active_lidar_beams,
            "internal_disabled_lidar_placeholder_beams": (
                0 if active_lidar_beams else 1
            ),
            "collision_mode": (
                "SEGMENT_CONTACT"
                if args.scenario in ("contact", "full")
                else "NONE"
            ),
            "frenet_enabled": True,
            "episode_limits_enabled": False,
            "actions": "zero physical commands",
            "initialization": "identical explicit centerline poses",
            "comparison": (
                "one shared/exact-shape-indexed JAX device batch versus the "
                "same number of independent mutable F110Env instances"
            ),
        },
        "results": results,
    }
    validate_report(report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run(args)
    payload = json.dumps(
        report,
        indent=None if args.compact else 2,
        sort_keys=True,
        allow_nan=False,
    )
    if args.output is None:
        print(payload)
    else:
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
