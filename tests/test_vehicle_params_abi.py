"""Pin the VehicleParameters -> flat float32 wire format that the kernels index.

Every dynamics kernel reads the array from ``VehicleParameters.to_array``
POSITIONALLY, so the field order is a wire format. Inserting
``collision_body_center_x/y`` at positions 18/19 once silently shifted every
multi-body parameter by +2, which is what made the MB model return NaN.

There is deliberately no hand-maintained ABI tuple any more: the parameters
describe the vehicle, not the model, so ``to_array`` emits every field in
declaration order and each model reads the slots it cares about. That keeps the
source simple but puts the whole guard here -- ``EXPECTED_PARAMETER_ORDER``
below is the contract, and it is checked name-by-name so a reorder fails with
the moved field named rather than as a mysterious physics change.
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
    PARAMETER_ORDER,
    VehicleParameters,
)

# THE WIRE FORMAT, spelled out. Changing this is a breaking change and must be
# matched by every kernel that indexes the array. Slots 0-17 are read by
# KS/ST/MB; 18-19 are Python-side collision geometry that no kernel reads;
# 20-87 are the multi-body block. There is no multi-body copy of the
# total-mass CoG height -- MB shares the base `h` at slot 5.
EXPECTED_PARAMETER_ORDER = (
    "mu", "C_Sf", "C_Sr", "lf", "lr", "h", "m", "I",
    "s_min", "s_max", "sv_min", "sv_max",
    "v_switch", "a_max", "v_min", "v_max",
    "width", "length",
    "collision_body_center_x", "collision_body_center_y",
    "kappa_dot_max", "kappa_dot_dot_max", "j_max", "j_dot_max",
    "m_s", "m_uf", "m_ur",
    "I_Phi_s", "I_y_s", "I_z", "I_xz_s",
    "K_sf", "K_sdf", "K_sr", "K_sdr",
    "T_f", "T_r",
    "K_ras", "K_tsf", "K_tsr", "K_rad", "K_zt",
    "h_raf", "h_rar", "h_s",
    "I_uf", "I_ur", "I_y_w",
    "K_lt", "R_w",
    "T_sb", "T_se",
    "D_f", "D_r", "E_f", "E_r",
    "tire_p_cx1", "tire_p_dx1", "tire_p_dx3", "tire_p_ex1", "tire_p_kx1",
    "tire_p_hx1", "tire_p_vx1",
    "tire_r_bx1", "tire_r_bx2", "tire_r_cx1", "tire_r_ex1", "tire_r_hx1",
    "tire_p_cy1", "tire_p_dy1", "tire_p_dy3", "tire_p_ey1", "tire_p_ky1",
    "tire_p_hy1", "tire_p_hy3", "tire_p_vy1", "tire_p_vy3",
    "tire_r_by1", "tire_r_by2", "tire_r_by3", "tire_r_cy1", "tire_r_ey1",
    "tire_r_hy1", "tire_r_vy1", "tire_r_vy3", "tire_r_vy4", "tire_r_vy5",
    "tire_r_vy6",
)

# Slots the kernels hardcode, checked by name so a shift is caught at the exact
# index rather than as a silent change in the physics.
KERNEL_SLOTS = {
    0: "mu", 1: "C_Sf", 2: "C_Sr", 3: "lf", 4: "lr", 5: "h", 6: "m", 7: "I",
    8: "s_min", 9: "s_max", 10: "sv_min", 11: "sv_max",
    12: "v_switch", 13: "a_max", 14: "v_min", 15: "v_max",
    16: "width", 17: "length",
    20: "kappa_dot_max", 41: "K_zt", 49: "R_w", 87: "tire_r_vy6",
}

ALL_PRESETS = (
    F1TENTH_VEHICLE_PARAMETERS,
    F1FIFTH_VEHICLE_PARAMETERS,
    FULLSCALE_VEHICLE_PARAMETERS,
)


class TestParameterOrder(unittest.TestCase):
    def test_order_is_exactly_as_documented(self):
        # Compared as lists so a mismatch names the field that moved.
        self.assertEqual(list(PARAMETER_ORDER), list(EXPECTED_PARAMETER_ORDER))

    def test_order_has_89_entries(self):
        self.assertEqual(len(PARAMETER_ORDER), 88)
        self.assertEqual(len(EXPECTED_PARAMETER_ORDER), 88)

    def test_order_is_the_dataclass_declaration_order(self):
        self.assertEqual(
            list(PARAMETER_ORDER),
            [f.name for f in dataclasses.fields(VehicleParameters)],
        )

    def test_every_name_is_a_real_field(self):
        valid = {f.name for f in dataclasses.fields(VehicleParameters)}
        for name in PARAMETER_ORDER:
            self.assertIn(name, valid)

    def test_no_duplicate_slots(self):
        self.assertEqual(len(set(PARAMETER_ORDER)), len(PARAMETER_ORDER))

    def test_slots_the_kernels_hardcode(self):
        for index, name in KERNEL_SLOTS.items():
            self.assertEqual(PARAMETER_ORDER[index], name, f"slot {index}")


class TestToArray(unittest.TestCase):
    def test_one_array_for_every_model(self):
        # The parameters describe the vehicle, not the model, so there is no
        # per-model length any more.
        for params in ALL_PRESETS:
            arr = params.to_array()
            self.assertEqual(arr.shape, (88,))
            self.assertEqual(arr.dtype, np.float32)

    def test_marshals_by_name_in_order(self):
        for params in ALL_PRESETS:
            arr = params.to_array()
            for index, name in enumerate(PARAMETER_ORDER):
                expected = getattr(params, name)
                if math.isnan(expected):
                    self.assertTrue(math.isnan(float(arr[index])), f"{name} at {index}")
                else:
                    # exact: the only transformation to_array applies is the
                    # float64 -> float32 narrowing
                    self.assertEqual(
                        arr[index], np.float32(expected), f"{name} at {index}"
                    )

    def test_values_land_at_their_documented_index(self):
        params = F1TENTH_VEHICLE_PARAMETERS.with_updates(mu=0.5, v_max=33.0, length=1.25)
        arr = params.to_array()
        self.assertAlmostEqual(float(arr[0]), 0.5, places=5)
        self.assertAlmostEqual(float(arr[15]), 33.0, places=5)
        self.assertAlmostEqual(float(arr[17]), 1.25, places=5)

    def test_collision_offsets_are_present_but_unread_by_kernels(self):
        # They ride along at 18/19 rather than being filtered out; the kernels
        # simply start the multi-body block at 20.
        arr = FULLSCALE_VEHICLE_PARAMETERS.with_updates(
            collision_body_center_x=0.4, collision_body_center_y=0.1
        ).to_array()
        self.assertAlmostEqual(float(arr[18]), 0.4, places=5)
        self.assertAlmostEqual(float(arr[19]), 0.1, places=5)


class TestMultiBodyParameters(unittest.TestCase):
    def test_small_scale_presets_are_missing_the_mb_block(self):
        for params in (F1TENTH_VEHICLE_PARAMETERS, F1FIFTH_VEHICLE_PARAMETERS):
            missing = params.missing_mb_parameters()
            self.assertEqual(len(missing), 68)

    def test_fullscale_has_every_parameter(self):
        self.assertEqual(FULLSCALE_VEHICLE_PARAMETERS.missing_mb_parameters(), ())

    def test_missing_never_reports_a_slot_the_base_models_read(self):
        base_slots = set(PARAMETER_ORDER[:18])
        for params in ALL_PRESETS:
            self.assertFalse(set(params.missing_mb_parameters()) & base_slots)

    def test_fullscale_mb_array_is_finite(self):
        arr = FULLSCALE_VEHICLE_PARAMETERS.to_array()
        self.assertTrue(np.all(np.isfinite(arr)))

    def test_k_zt_is_not_zero_for_fullscale(self):
        # K_zt is a divisor in the multi-body suspension math; reading the wrong
        # slot used to land on a 0.0 and raise ZeroDivisionError.
        arr = FULLSCALE_VEHICLE_PARAMETERS.to_array()
        self.assertEqual(PARAMETER_ORDER[41], "K_zt")
        self.assertGreater(float(arr[41]), 0.0)


class TestMultiBodyRuns(unittest.TestCase):
    def test_fullscale_mb_integrates_without_nan(self):
        from f1tenth_gym.envs.integrators import rk4_integration

        params = FULLSCALE_VEHICLE_PARAMETERS
        arr = params.to_array()
        model = DynamicModel.MB
        state = model.get_initial_state(pose=np.zeros(3), params=arr)
        control = np.array([0.1, 2.0], dtype=np.float32)
        for _ in range(50):
            state = rk4_integration(model.f_dynamics, state, control, 0.01, arr)
        self.assertTrue(np.all(np.isfinite(state)))
        self.assertGreater(float(state[3]), 0.0)


class TestNoPerModelParameterAPI(unittest.TestCase):
    """The per-model split is gone; these pin that it stays gone."""

    def test_to_array_takes_no_model_argument(self):
        with self.assertRaises(TypeError):
            F1TENTH_VEHICLE_PARAMETERS.to_array(DynamicModel.ST)

    def test_parameter_count_is_removed(self):
        self.assertFalse(hasattr(DynamicModel.ST, "parameter_count"))

    def test_from_string_is_removed(self):
        self.assertFalse(hasattr(DynamicModel, "from_string"))

    def test_vehicle_param_ranges_is_removed(self):
        import f1tenth_gym.envs.dynamic_models as dm

        self.assertFalse(hasattr(dm, "VehicleParamRanges"))


if __name__ == "__main__":
    unittest.main()
