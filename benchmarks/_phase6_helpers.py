"""Standard-library result helpers for the Phase 6 rollout benchmark.

Keeping these helpers separate lets the normal test suite validate the JSON
contract without importing JAX or constructing simulation environments.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import numbers
import statistics
from typing import Any


SCHEMA_VERSION = 2
BENCHMARK_NAME = "f1tenth_gym_phase6_rollout"


def parse_positive_csv(value: str, *, name: str) -> tuple[int, ...]:
    """Parse a non-empty, duplicate-free CSV of positive integers."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a comma-separated string")
    fields = tuple(field.strip() for field in value.split(","))
    if not fields or any(not field for field in fields):
        raise ValueError(f"{name} must be a comma-separated list of integers")
    try:
        parsed = tuple(int(field) for field in fields)
    except ValueError as error:
        raise ValueError(f"{name} must contain only integers") from error
    if any(item < 1 for item in parsed):
        raise ValueError(f"{name} values must all be >= 1")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"{name} must not contain duplicate values")
    return parsed


def unavailable_memory(source: str, note: str) -> dict[str, Any]:
    """Return an explicit unavailable peak-memory record."""
    return {
        "available": False,
        "bytes": None,
        "source": str(source),
        "scope": None,
        "note": str(note),
    }


def device_peak_memory(stats: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize a JAX device allocator's lifetime peak, when exposed.

    CPU backends commonly return ``None`` from ``Device.memory_stats()``.  The
    benchmark reports that absence rather than substituting an incomparable
    process-RSS or an estimate from array shapes.
    """
    source = "jax.Device.memory_stats"
    if stats is None:
        return unavailable_memory(
            source,
            "the selected JAX backend does not expose allocator memory statistics",
        )
    value = stats.get("peak_bytes_in_use")
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return unavailable_memory(
            source,
            "peak_bytes_in_use is absent from the device allocator statistics",
        )
    if not math.isfinite(float(value)) or value < 0:
        return unavailable_memory(
            source,
            "peak_bytes_in_use is not a finite non-negative value",
        )
    return {
        "available": True,
        "bytes": int(value),
        "source": source,
        "scope": "device_allocator_lifetime",
        "note": "includes setup and compilation performed in this process",
    }


def timing_summary(
    seconds: Sequence[float],
    *,
    batch_size: int,
    rollout_length: int,
    agents: int,
) -> dict[str, Any]:
    """Summarize synchronized steady runs and their two throughput units."""
    values = tuple(float(value) for value in seconds)
    if not values:
        raise ValueError("at least one steady timing is required")
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("steady timings must be finite and > 0")
    for name, value in (
        ("batch_size", batch_size),
        ("rollout_length", rollout_length),
        ("agents", agents),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1")

    median_seconds = float(statistics.median(values))
    environment_steps = int(batch_size * rollout_length)
    agent_steps = int(environment_steps * agents)
    return {
        "steady_seconds": list(values),
        "steady_median_seconds": median_seconds,
        "steady_min_seconds": min(values),
        "environment_steps_per_run": environment_steps,
        "agent_steps_per_run": agent_steps,
        "environment_steps_per_second": environment_steps / median_seconds,
        "agent_steps_per_second": agent_steps / median_seconds,
    }


_BACKEND_KEYS = {
    "backend",
    "batch_size",
    "agents",
    "rollout_length",
    "scenario",
    "unique_maps",
    "track_points",
    "active_lidar_beams",
    "lidar_enabled",
    "contact_enabled",
    "dtype",
    "device",
    "compile_seconds",
    "warmup_seconds",
    "steady_seconds",
    "steady_median_seconds",
    "steady_min_seconds",
    "environment_steps_per_run",
    "agent_steps_per_run",
    "environment_steps_per_second",
    "agent_steps_per_second",
    "checksum",
    "resident_input_bytes",
    "resident_table_bytes",
    "peak_memory",
}


def validate_backend_result(result: Mapping[str, Any]) -> None:
    """Reject incomplete or non-finite backend records before JSON output."""
    missing = sorted(_BACKEND_KEYS.difference(result))
    if missing:
        raise ValueError("backend result is missing keys: " + ", ".join(missing))
    if result["backend"] not in ("jax_device_batch", "mutable_numpy_gym"):
        raise ValueError(f"unknown backend result {result['backend']!r}")
    for name in ("batch_size", "agents", "rollout_length"):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1")
    if (
        isinstance(result["active_lidar_beams"], bool)
        or not isinstance(result["active_lidar_beams"], int)
        or result["active_lidar_beams"] < 0
    ):
        raise ValueError("active_lidar_beams must be an integer >= 0")
    if result["scenario"] not in ("state", "lidar", "contact", "full"):
        raise ValueError(f"unknown benchmark scenario {result['scenario']!r}")
    for name in ("unique_maps", "track_points"):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1")
    for name in ("lidar_enabled", "contact_enabled"):
        if not isinstance(result[name], bool):
            raise ValueError(f"{name} must be boolean")
    expected_lidar = result["scenario"] in ("lidar", "full")
    expected_contact = result["scenario"] in ("contact", "full")
    if result["lidar_enabled"] is not expected_lidar:
        raise ValueError("lidar_enabled does not match the scenario")
    if result["contact_enabled"] is not expected_contact:
        raise ValueError("contact_enabled does not match the scenario")
    if expected_lidar != (result["active_lidar_beams"] > 0):
        raise ValueError("active_lidar_beams does not match the scenario")
    for name in (
        "steady_median_seconds",
        "steady_min_seconds",
        "environment_steps_per_second",
        "agent_steps_per_second",
        "checksum",
    ):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(f"{name} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
        if name != "checksum" and value <= 0.0:
            raise ValueError(f"{name} must be > 0")
    compile_seconds = result["compile_seconds"]
    if compile_seconds is not None and (
        not isinstance(compile_seconds, numbers.Real)
        or isinstance(compile_seconds, bool)
        or not math.isfinite(float(compile_seconds))
        or compile_seconds < 0.0
    ):
        raise ValueError("compile_seconds must be finite and >= 0, or null")
    if result["backend"] == "jax_device_batch" and compile_seconds is None:
        raise ValueError("a JAX result must report compile_seconds")
    if result["backend"] == "mutable_numpy_gym" and compile_seconds is not None:
        raise ValueError("a mutable result must use null compile_seconds")
    for name in ("warmup_seconds", "steady_seconds"):
        timings = result[name]
        if not isinstance(timings, list) or not timings:
            raise ValueError(f"{name} must be a non-empty list")
        if any(
            isinstance(value, bool)
            or not isinstance(value, numbers.Real)
            or not math.isfinite(float(value))
            or value <= 0.0
            for value in timings
        ):
            raise ValueError(f"{name} values must be finite and > 0")
    expected_environment_steps = (
        result["batch_size"] * result["rollout_length"]
    )
    if result["environment_steps_per_run"] != expected_environment_steps:
        raise ValueError("environment_steps_per_run does not match the dimensions")
    if result["agent_steps_per_run"] != (
        expected_environment_steps * result["agents"]
    ):
        raise ValueError("agent_steps_per_run does not match the dimensions")
    if not isinstance(result["device"], Mapping):
        raise ValueError("device must be an object")
    memory = result["peak_memory"]
    if not isinstance(memory, Mapping):
        raise ValueError("peak_memory must be an object")
    if memory.get("available") is False and memory.get("bytes") is not None:
        raise ValueError("unavailable peak memory must use a null byte count")
    if memory.get("available") is True:
        byte_count = memory.get("bytes")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, numbers.Integral)
            or byte_count < 0
        ):
            raise ValueError("available peak memory must have non-negative bytes")


def validate_report(report: Mapping[str, Any]) -> None:
    """Validate the stable top-level result contract."""
    required = {
        "schema_version",
        "benchmark",
        "generated_at_utc",
        "host",
        "jax",
        "configuration",
        "results",
    }
    missing = sorted(required.difference(report))
    if missing:
        raise ValueError("report is missing keys: " + ", ".join(missing))
    if report["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if report["benchmark"] != BENCHMARK_NAME:
        raise ValueError(f"benchmark must be {BENCHMARK_NAME!r}")
    results = report["results"]
    if not isinstance(results, list) or not results:
        raise ValueError("results must be a non-empty list")
    for result in results:
        if not isinstance(result, Mapping):
            raise ValueError("each result must be an object")
        validate_backend_result(result)


__all__ = [
    "BENCHMARK_NAME",
    "SCHEMA_VERSION",
    "device_peak_memory",
    "parse_positive_csv",
    "timing_summary",
    "unavailable_memory",
    "validate_backend_result",
    "validate_report",
]
