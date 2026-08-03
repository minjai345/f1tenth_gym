"""Pin the VehicleParameters -> flat float32 ABI that the njit kernels index.

Every ``@njit`` dynamics kernel reads the array from ``VehicleParameters.to_array``
positionally, so the ABI is a wire format. It used to be defined implicitly by the
dataclass field order, and inserting ``collision_body_center_x/y`` at positions
18/19 silently shifted every multi-body parameter by +2 -- which is what made the
MB model return NaN. The order now lives in ``_BASE_PARAM_ABI`` / ``_MB_PARAM_ABI``
and these tests fail loudly if it drifts again.
"""
import dataclasses
import math
import unittest

import numpy as np

from f1tenth_gym.envs.dynamic_models import (
    DynamicModel,
    F1FIFTH_VEHICLE_PARAMETERS,
    F1TENTH_VEHICLE_PARAMETERS,
    FULLSCALE_VEHICLE_PARAMETERS,
    VehicleParameters,
    _BASE_PARAM_ABI,
    _MB_PARAM_ABI,
)

# The KS/ST layout, spelled out. Any change here is a breaking ABI change and must
# be matched by every kernel that indexes the array.
EXPECTED_BASE_ABI = (
    "mu", "C_Sf", "C_Sr", "lf", "lr", "h", "m", "I",
    "s_min", "s_max", "sv_min", "sv_max",
    "v_switch", "a_max", "v_min", "v_max",
    "width", "length",
)

ALL_PRESETS = (
    F1TENTH_VEHICLE_PARAMETERS,
    F1FIFTH_VEHICLE_PARAMETERS,
    FULLSCALE_VEHICLE_PARAMETERS,
)


class TestParameterABI(unittest.TestCase):
    def test_base_abi_is_exactly_as_documented(self):
        self.assertEqual(tuple(_BASE_PARAM_ABI), EXPECTED_BASE_ABI)

    def test_base_abi_has_18_entries(self):
        self.assertEqual(len(_BASE_PARAM_ABI), 18)
        self.assertEqual(DynamicModel.KS.parameter_count(), 18)
        self.assertEqual(DynamicModel.ST.parameter_count(), 18)

    def test_mb_abi_has_87_entries(self):
        """MB kernels read indices 0..86 contiguously, so the ABI is 87 long."""
        self.assertEqual(len(_MB_PARAM_ABI), 87)
        self.assertEqual(DynamicModel.MB.parameter_count(), 87)

    def test_mb_abi_excludes_the_collision_body_offsets(self):
        """These are Python-level geometry, never read by a kernel.

        Including them is precisely the +2 shift that broke MB.
        """
        self.assertNotIn("collision_body_center_x", _MB_PARAM_ABI)
        self.assertNotIn("collision_body_center_y", _MB_PARAM_ABI)

    def test_mb_abi_extends_the_base_abi(self):
        self.assertEqual(tuple(_MB_PARAM_ABI[:18]), EXPECTED_BASE_ABI)

    def test_every_abi_name_is_a_real_field(self):
        fields = {f.name for f in dataclasses.fields(VehicleParameters)}
        for name in set(_BASE_PARAM_ABI) | set(_MB_PARAM_ABI):
            self.assertIn(name, fields, f"{name!r} is not a VehicleParameters field")

    def test_no_duplicate_slots(self):
        self.assertEqual(len(set(_MB_PARAM_ABI)), len(_MB_PARAM_ABI))

    def test_indices_the_kernels_hardcode(self):
        """Spot-check the slots that appear literally in the kernels."""
        for index, name in ((0, "mu"), (3, "lf"), (4, "lr"), (6, "m"), (7, "I"),
                            (13, "a_max"), (15, "v_max"), (16, "width"), (17, "length")):
            self.assertEqual(_BASE_PARAM_ABI[index], name, f"base slot {index}")
        # multi_body.py reads kappa_dot_max at 18 and K_zt at 39.
        self.assertEqual(_MB_PARAM_ABI[18], "kappa_dot_max")
        self.assertEqual(_MB_PARAM_ABI[39], "K_zt")


class TestToArray(unittest.TestCase):
    def test_ks_and_st_match_the_documented_order(self):
        for params in ALL_PRESETS:
            expected = np.asarray(
                [getattr(params, n) for n in EXPECTED_BASE_ABI], dtype=np.float32
            )
            for model in (DynamicModel.KS, DynamicModel.ST):
                got = params.to_array(model)
                np.testing.assert_array_equal(got, expected, err_msg=f"{model.name}")

    def test_dtype_and_shape(self):
        for model, size in ((DynamicModel.KS, 18), (DynamicModel.ST, 18), (DynamicModel.MB, 87)):
            arr = FULLSCALE_VEHICLE_PARAMETERS.to_array(model)
            self.assertEqual(arr.dtype, np.float32)
            self.assertEqual(arr.shape, (size,))

    def test_array_owns_its_data(self):
        """Kernels keep the array; it must not be a view of a temporary."""
        arr = F1TENTH_VEHICLE_PARAMETERS.to_array(DynamicModel.ST)
        self.assertTrue(arr.flags.owndata)

    def test_marshalling_is_by_name_not_position(self):
        """A value set on a field must surface at that field's ABI slot."""
        params = F1TENTH_VEHICLE_PARAMETERS.with_updates(mu=0.5, v_max=33.0, length=1.25)
        arr = params.to_array(DynamicModel.ST)
        self.assertAlmostEqual(float(arr[_BASE_PARAM_ABI.index("mu")]), 0.5, places=5)
        self.assertAlmostEqual(float(arr[_BASE_PARAM_ABI.index("v_max")]), 33.0, places=5)
        self.assertAlmostEqual(float(arr[_BASE_PARAM_ABI.index("length")]), 1.25, places=5)

    def test_fullscale_mb_array_is_finite(self):
        arr = FULLSCALE_VEHICLE_PARAMETERS.to_array(DynamicModel.MB)
        self.assertTrue(np.all(np.isfinite(arr)), "FULLSCALE should fully populate the MB ABI")

    def test_k_zt_is_not_zero_for_fullscale(self):
        """K_zt is a divisor in init_mb; reading the wrong slot gave 0.0 -> inf."""
        arr = FULLSCALE_VEHICLE_PARAMETERS.to_array(DynamicModel.MB)
        self.assertGreater(float(arr[39]), 0.0)


class TestMissingMultiBodyParameters(unittest.TestCase):
    def test_small_scale_presets_do_not_support_mb(self):
        for params in (F1TENTH_VEHICLE_PARAMETERS, F1FIFTH_VEHICLE_PARAMETERS):
            missing = params.missing_mb_parameters()
            self.assertEqual(len(missing), 69)
            self.assertTrue(all(math.isnan(getattr(params, n)) for n in missing))

    def test_fullscale_supports_mb(self):
        self.assertEqual(FULLSCALE_VEHICLE_PARAMETERS.missing_mb_parameters(), ())

    def test_base_parameters_are_never_reported_missing(self):
        """The base block is populated by every preset, so it must never be flagged."""
        for params in ALL_PRESETS:
            self.assertFalse(set(params.missing_mb_parameters()) & set(_BASE_PARAM_ABI))


class TestMultiBodyRuns(unittest.TestCase):
    def test_fullscale_initial_state_is_finite(self):
        arr = FULLSCALE_VEHICLE_PARAMETERS.to_array(DynamicModel.MB)
        state = DynamicModel.MB.get_initial_state(pose=np.zeros(3), params=arr)
        self.assertEqual(state.shape, (29,))
        self.assertTrue(np.all(np.isfinite(state)), f"non-finite at {np.where(~np.isfinite(state))[0]}")

    def test_fullscale_integrates_without_going_non_finite(self):
        from f1tenth_gym.envs.integrators import rk4_integration

        arr = FULLSCALE_VEHICLE_PARAMETERS.to_array(DynamicModel.MB)
        state = DynamicModel.MB.get_initial_state(pose=np.zeros(3), params=arr)
        control = np.array([0.05, 2.0], dtype=np.float32)
        for step in range(50):
            state = rk4_integration(DynamicModel.MB.f_dynamics, state, control, 0.01, arr)
            self.assertTrue(np.all(np.isfinite(state)), f"non-finite at step {step + 1}")
        self.assertGreater(float(state[3]), 0.0, "the car should have accelerated")


if __name__ == "__main__":
    unittest.main()
