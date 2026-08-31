"""Fast contracts for the opt-in Phase 6 performance result helpers."""

import unittest

from benchmarks._phase6_helpers import (
    BENCHMARK_NAME,
    SCHEMA_VERSION,
    device_peak_memory,
    parse_positive_csv,
    timing_summary,
    unavailable_memory,
    validate_backend_result,
    validate_report,
)


def backend_result():
    timing = timing_summary(
        [2.0, 1.0, 3.0],
        batch_size=4,
        rollout_length=8,
        agents=2,
    )
    return {
        "backend": "jax_device_batch",
        "batch_size": 4,
        "agents": 2,
        "rollout_length": 8,
        "scenario": "state",
        "unique_maps": 1,
        "track_points": 96,
        "active_lidar_beams": 0,
        "lidar_enabled": False,
        "contact_enabled": False,
        "dtype": "float32",
        "device": {"platform": "cpu", "kind": "test", "id": 0},
        "compile_seconds": 0.5,
        "warmup_seconds": [2.5],
        **timing,
        "checksum": 1.25,
        "collision_events_per_run": 0,
        "resident_input_bytes": 1024,
        "resident_table_bytes": 2048,
        "peak_memory": unavailable_memory("test", "not exposed"),
    }


class TestParsing(unittest.TestCase):
    def test_positive_csv_preserves_order(self):
        self.assertEqual(
            parse_positive_csv("1, 16,64", name="batches"),
            (1, 16, 64),
        )

    def test_positive_csv_rejects_empty_nonpositive_and_duplicate_items(self):
        for value in ("", "1,", "0,2", "2,-1", "2,two", "2,2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_positive_csv(value, name="batches")


class TestTimingSummary(unittest.TestCase):
    def test_median_drives_both_step_rates(self):
        result = timing_summary(
            [2.0, 1.0, 3.0],
            batch_size=4,
            rollout_length=8,
            agents=2,
        )
        self.assertEqual(result["steady_median_seconds"], 2.0)
        self.assertEqual(result["steady_min_seconds"], 1.0)
        self.assertEqual(result["environment_steps_per_run"], 32)
        self.assertEqual(result["agent_steps_per_run"], 64)
        self.assertEqual(result["environment_steps_per_second"], 16.0)
        self.assertEqual(result["agent_steps_per_second"], 32.0)

    def test_invalid_timings_and_dimensions_are_rejected(self):
        for seconds in ([], [0.0], [float("nan")], [float("inf")]):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                timing_summary(
                    seconds,
                    batch_size=1,
                    rollout_length=1,
                    agents=1,
                )
        with self.assertRaises(ValueError):
            timing_summary([1.0], batch_size=0, rollout_length=1, agents=1)


class TestMemoryRecords(unittest.TestCase):
    def test_device_peak_is_normalized_when_available(self):
        result = device_peak_memory({"peak_bytes_in_use": 4096})
        self.assertTrue(result["available"])
        self.assertEqual(result["bytes"], 4096)
        self.assertEqual(result["scope"], "device_allocator_lifetime")

    def test_missing_device_stats_are_explicitly_unavailable(self):
        result = device_peak_memory(None)
        self.assertFalse(result["available"])
        self.assertIsNone(result["bytes"])
        self.assertIn("does not expose", result["note"])


class TestResultSchema(unittest.TestCase):
    def test_complete_backend_and_report_are_accepted(self):
        result = backend_result()
        validate_backend_result(result)
        validate_report(
            {
                "schema_version": SCHEMA_VERSION,
                "benchmark": BENCHMARK_NAME,
                "generated_at_utc": "2026-08-31T00:00:00+00:00",
                "host": {},
                "jax": {},
                "configuration": {},
                "results": [result],
            }
        )

    def test_missing_fields_and_nonfinite_checksum_are_rejected(self):
        result = backend_result()
        del result["device"]
        with self.assertRaisesRegex(ValueError, "device"):
            validate_backend_result(result)

        result = backend_result()
        result["checksum"] = float("nan")
        with self.assertRaisesRegex(ValueError, "checksum"):
            validate_backend_result(result)

    def test_unavailable_memory_cannot_report_a_byte_count(self):
        result = backend_result()
        result["peak_memory"]["bytes"] = 123
        with self.assertRaisesRegex(ValueError, "null"):
            validate_backend_result(result)

    def test_collision_event_count_must_match_the_scenario(self):
        result = backend_result()
        result["collision_events_per_run"] = 1
        with self.assertRaisesRegex(ValueError, "collision event"):
            validate_backend_result(result)


if __name__ == "__main__":
    unittest.main()
