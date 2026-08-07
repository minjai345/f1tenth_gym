"""
Dynamic model definitions and strongly typed vehicle parameters for the F1TENTH gym.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, fields, replace
from enum import IntEnum
from typing import Optional, Tuple

import numpy as np

from .kinematic import vehicle_dynamics_ks, get_standardized_state_ks
from .single_track import vehicle_dynamics_st, get_standardized_state_st
from .multi_body import init_mb, vehicle_dynamics_mb, get_standardized_state_mb
from .utils import pid_steer, pid_accl

__all__ = [
    "DynamicModel",
    "VehicleParameters",
    "F1TENTH_VEHICLE_PARAMETERS",
    "F1FIFTH_VEHICLE_PARAMETERS",
    "FULLSCALE_VEHICLE_PARAMETERS",
    "get_f1tenth_vehicle_parameters",
    "get_f1fifth_vehicle_parameters",
    "get_fullscale_vehicle_parameters",
    "pid_steer",
    "pid_accl",
]


@dataclass(frozen=True)
class VehicleParameters:
    """Typed view over the vehicle parameters used by the dynamics models."""

    # Core parameters required by the simplified single-track models
    mu: float = math.nan
    C_Sf: float = math.nan
    C_Sr: float = math.nan
    lf: float = math.nan
    lr: float = math.nan
    h: float = math.nan
    m: float = math.nan
    I: float = math.nan
    s_min: float = math.nan
    s_max: float = math.nan
    sv_min: float = math.nan
    sv_max: float = math.nan
    v_switch: float = math.nan
    a_max: float = math.nan
    v_min: float = math.nan
    v_max: float = math.nan
    width: float = math.nan
    length: float = math.nan

    # Collision body center offset from base_link (rear axle) in meters.
    # For symmetric overhang: x = wheelbase/2 = (lf + lr)/2
    collision_body_center_x: float = 0.0
    collision_body_center_y: float = 0.0

    # Additional parameters for the multi-body model (defaulted to NaN)
    kappa_dot_max: float = math.nan
    kappa_dot_dot_max: float = math.nan
    j_max: float = math.nan
    j_dot_max: float = math.nan
    m_s: float = math.nan
    m_uf: float = math.nan
    m_ur: float = math.nan
    I_Phi_s: float = math.nan
    I_y_s: float = math.nan
    I_z_body: float = math.nan
    I_xz_s: float = math.nan
    K_sf: float = math.nan
    K_sdf: float = math.nan
    K_sr: float = math.nan
    K_sdr: float = math.nan
    T_f: float = math.nan
    T_r: float = math.nan
    K_ras: float = math.nan
    K_tsf: float = math.nan
    K_tsr: float = math.nan
    K_rad: float = math.nan
    K_zt: float = math.nan
    h_cg_mb: float = math.nan
    h_raf: float = math.nan
    h_rar: float = math.nan
    h_s: float = math.nan
    I_uf: float = math.nan
    I_ur: float = math.nan
    I_y_w: float = math.nan
    K_lt: float = math.nan
    R_w: float = math.nan
    T_sb: float = math.nan
    T_se: float = math.nan
    D_f: float = math.nan
    D_r: float = math.nan
    E_f: float = math.nan
    E_r: float = math.nan
    tire_p_cx1: float = math.nan
    tire_p_dx1: float = math.nan
    tire_p_dx3: float = math.nan
    tire_p_ex1: float = math.nan
    tire_p_kx1: float = math.nan
    tire_p_hx1: float = math.nan
    tire_p_vx1: float = math.nan
    tire_r_bx1: float = math.nan
    tire_r_bx2: float = math.nan
    tire_r_cx1: float = math.nan
    tire_r_ex1: float = math.nan
    tire_r_hx1: float = math.nan
    tire_p_cy1: float = math.nan
    tire_p_dy1: float = math.nan
    tire_p_dy3: float = math.nan
    tire_p_ey1: float = math.nan
    tire_p_ky1: float = math.nan
    tire_p_hy1: float = math.nan
    tire_p_hy3: float = math.nan
    tire_p_vy1: float = math.nan
    tire_p_vy3: float = math.nan
    tire_r_by1: float = math.nan
    tire_r_by2: float = math.nan
    tire_r_by3: float = math.nan
    tire_r_cy1: float = math.nan
    tire_r_ey1: float = math.nan
    tire_r_hy1: float = math.nan
    tire_r_vy1: float = math.nan
    tire_r_vy3: float = math.nan
    tire_r_vy4: float = math.nan
    tire_r_vy5: float = math.nan
    tire_r_vy6: float = math.nan

    def with_updates(self, **updates: float) -> "VehicleParameters":
        return replace(self, **updates)

    def missing_mb_parameters(self) -> tuple[str, ...]:
        """Names of multi-body ABI fields that are not finite.

        ``F1TENTH_VEHICLE_PARAMETERS`` and ``F1FIFTH_VEHICLE_PARAMETERS`` leave the
        entire multi-body block at ``nan``, so selecting ``DynamicModel.MB`` with
        either of them yields a NaN state rather than a trajectory. Only
        ``FULLSCALE_VEHICLE_PARAMETERS`` populates it.

        Returns:
            The unusable field names, empty if the parameters support MB.
        """
        return tuple(
            name for name in _MB_PARAM_ABI if not math.isfinite(getattr(self, name))
        )

    def to_array(self, model: "DynamicModel") -> np.ndarray:
        """Marshal the parameters into the flat float32 array the njit kernels index.

        The layout is defined by ``_BASE_PARAM_ABI`` / ``_MB_PARAM_ABI``, not by the
        dataclass field order, so fields may be added or reordered above without
        disturbing the kernels.

        Args:
            model: The dynamics model whose ABI to marshal for.

        Returns:
            A contiguous ``float32`` array of ``model.parameter_count()`` values.
        """
        abi = _MB_PARAM_ABI if model is DynamicModel.MB else _BASE_PARAM_ABI
        return np.asarray([getattr(self, name) for name in abi], dtype=np.float32)


# THE ABI. Every @njit kernel indexes the flat array from `to_array` POSITIONALLY
# (`mu = params[0]` ... `v_max = params[15]`), so these tuples -- not the dataclass
# field order -- are the wire format. Reordering or inserting a dataclass field is
# now safe; reordering THESE is not.
#
# Verified against the kernels: kinematic.py and single_track.py read indices 0-17
# contiguously; multi_body/ and tire_model.py read 0-86 contiguously.
_BASE_PARAM_ABI: tuple[str, ...] = (
    "mu", "C_Sf", "C_Sr", "lf", "lr", "h", "m", "I",
    "s_min", "s_max", "sv_min", "sv_max",
    "v_switch", "a_max", "v_min", "v_max",
    "width", "length",
)

_ALL_PARAMETER_FIELDS = tuple(field.name for field in fields(VehicleParameters))

# collision_body_center_x/y are deliberately absent: they are consumed by the
# simulator's Python-level geometry helpers, never by a kernel. Including them --
# which is what happened when they were inserted at dataclass positions 18/19 --
# shifts every multi-body parameter by +2 and is what broke the MB model.
_MB_PARAM_ABI: tuple[str, ...] = _BASE_PARAM_ABI + tuple(
    name for name in _ALL_PARAMETER_FIELDS
    if name not in _BASE_PARAM_ABI
    and name not in ("collision_body_center_x", "collision_body_center_y")
)

_BASE_PARAMETER_COUNT = len(_BASE_PARAM_ABI)
_MB_PARAMETER_COUNT = len(_MB_PARAM_ABI)

F1TENTH_VEHICLE_PARAMETERS = VehicleParameters(
    mu=1.0489,
    C_Sf=4.718,
    C_Sr=5.4562,
    lf=0.15875,
    lr=0.17145,
    h=0.074,
    m=3.74,
    I=0.04712,
    s_min=-0.4189,
    s_max=0.4189,
    sv_min=-3.2,
    sv_max=3.2,
    v_switch=7.319,
    a_max=9.51,
    v_min=-5.0,
    v_max=20.0,
    width=0.31,
    length=0.58,
    collision_body_center_x=(0.15875 + 0.17145) / 2,  # wheelbase/2 = 0.1651m
)

F1FIFTH_VEHICLE_PARAMETERS = VehicleParameters(
    mu=1.1,
    C_Sf=5.3507,
    C_Sr=5.3507,
    lf=0.2725,
    lr=0.2585,
    h=0.1825,
    m=15.32,
    I=0.64332,
    s_min=-0.4189,
    s_max=0.4189,
    sv_min=-3.2,
    sv_max=3.2,
    v_switch=7.319,
    a_max=9.51,
    v_min=-5.0,
    v_max=20.0,
    width=0.55,
    length=0.8,
    collision_body_center_x=(0.2725 + 0.2585) / 2,  # wheelbase/2 = 0.2655m
)

FULLSCALE_VEHICLE_PARAMETERS = VehicleParameters(
    mu=1.0489,
    C_Sf=20.89,
    C_Sr=20.89,
    lf=0.88392,
    lr=1.50876,
    h=0.557,
    m=1225.8878467253344,
    I=1538.8533713561394,
    s_min=-0.91,
    s_max=0.91,
    sv_min=-0.4,
    sv_max=0.4,
    v_switch=4.755,
    a_max=11.5,
    v_min=-13.9,
    v_max=45.8,
    width=1.674,
    length=4.298,
    collision_body_center_x=(0.88392 + 1.50876) / 2,  # wheelbase/2 = 1.1963m
    kappa_dot_max=0.4,
    kappa_dot_dot_max=20.0,
    j_max=10_000.0,
    j_dot_max=10_000.0,
    m_s=1094.542720290477,
    m_uf=65.67256321742863,
    m_ur=65.67256321742863,
    I_Phi_s=244.04723069965206,
    I_y_s=1342.2597688480864,
    I_z_body=1538.8533713561394,
    I_xz_s=0.0,
    K_sf=21898.332429625985,
    K_sdf=1459.3902937206362,
    K_sr=21898.332429625985,
    K_sdr=1459.3902937206362,
    T_f=1.389888,
    T_r=1.423416,
    K_ras=175186.65943700788,
    K_tsf=-12880.270509148304,
    K_tsr=0.0,
    K_rad=10215.732056044453,
    K_zt=189785.5477234252,
    h_cg_mb=0.5577840000000001,
    h_raf=0.0,
    h_rar=0.0,
    h_s=0.59436,
    I_uf=32.53963075995361,
    I_ur=32.53963075995361,
    I_y_w=1.7,
    K_lt=1.0278264878518764e-05,
    R_w=0.344,
    T_sb=0.76,
    T_se=1.0,
    D_f=-0.6233595800524934,
    D_r=-0.20997375328083986,
    E_f=0.0,
    E_r=0.0,
    tire_p_cx1=1.6411,
    tire_p_dx1=1.1739,
    tire_p_dx3=0.0,
    tire_p_ex1=0.46403,
    tire_p_kx1=22.303,
    tire_p_hx1=0.0012297,
    tire_p_vx1=-8.8098e-06,
    tire_r_bx1=13.276,
    tire_r_bx2=-13.778,
    tire_r_cx1=1.2568,
    tire_r_ex1=0.65225,
    tire_r_hx1=0.0050722,
    tire_p_cy1=1.3507,
    tire_p_dy1=1.0489,
    tire_p_dy3=-2.8821,
    tire_p_ey1=-0.0074722,
    tire_p_ky1=-21.92,
    tire_p_hy1=0.0026747,
    tire_p_hy3=0.031415,
    tire_p_vy1=0.037318,
    tire_p_vy3=-0.32931,
    tire_r_by1=7.1433,
    tire_r_by2=9.1916,
    tire_r_by3=-0.027856,
    tire_r_cy1=1.0719,
    tire_r_ey1=-0.27572,
    tire_r_hy1=5.7448e-06,
    tire_r_vy1=-0.027825,
    tire_r_vy3=-0.27568,
    tire_r_vy4=12.12,
    tire_r_vy5=1.9,
    tire_r_vy6=-10.704,
)


@dataclass(frozen=True)
class VehicleParamRanges:
    """Typed (low, high) randomization ranges, one optional slot per
    ``VehicleParameters`` field — the autocomplete-friendly face of
    ``DomainRandomizationConfig.param_ranges``. A field left ``None`` is not
    randomized. ``tests/test_vehicle_params_abi.py`` pins field parity with
    ``VehicleParameters``, so the two cannot drift apart silently."""

    mu: Optional[Tuple[float, float]] = None
    C_Sf: Optional[Tuple[float, float]] = None
    C_Sr: Optional[Tuple[float, float]] = None
    lf: Optional[Tuple[float, float]] = None
    lr: Optional[Tuple[float, float]] = None
    h: Optional[Tuple[float, float]] = None
    m: Optional[Tuple[float, float]] = None
    I: Optional[Tuple[float, float]] = None
    s_min: Optional[Tuple[float, float]] = None
    s_max: Optional[Tuple[float, float]] = None
    sv_min: Optional[Tuple[float, float]] = None
    sv_max: Optional[Tuple[float, float]] = None
    v_switch: Optional[Tuple[float, float]] = None
    a_max: Optional[Tuple[float, float]] = None
    v_min: Optional[Tuple[float, float]] = None
    v_max: Optional[Tuple[float, float]] = None
    width: Optional[Tuple[float, float]] = None
    length: Optional[Tuple[float, float]] = None
    collision_body_center_x: Optional[Tuple[float, float]] = None
    collision_body_center_y: Optional[Tuple[float, float]] = None
    kappa_dot_max: Optional[Tuple[float, float]] = None
    kappa_dot_dot_max: Optional[Tuple[float, float]] = None
    j_max: Optional[Tuple[float, float]] = None
    j_dot_max: Optional[Tuple[float, float]] = None
    m_s: Optional[Tuple[float, float]] = None
    m_uf: Optional[Tuple[float, float]] = None
    m_ur: Optional[Tuple[float, float]] = None
    I_Phi_s: Optional[Tuple[float, float]] = None
    I_y_s: Optional[Tuple[float, float]] = None
    I_z_body: Optional[Tuple[float, float]] = None
    I_xz_s: Optional[Tuple[float, float]] = None
    K_sf: Optional[Tuple[float, float]] = None
    K_sdf: Optional[Tuple[float, float]] = None
    K_sr: Optional[Tuple[float, float]] = None
    K_sdr: Optional[Tuple[float, float]] = None
    T_f: Optional[Tuple[float, float]] = None
    T_r: Optional[Tuple[float, float]] = None
    K_ras: Optional[Tuple[float, float]] = None
    K_tsf: Optional[Tuple[float, float]] = None
    K_tsr: Optional[Tuple[float, float]] = None
    K_rad: Optional[Tuple[float, float]] = None
    K_zt: Optional[Tuple[float, float]] = None
    h_cg_mb: Optional[Tuple[float, float]] = None
    h_raf: Optional[Tuple[float, float]] = None
    h_rar: Optional[Tuple[float, float]] = None
    h_s: Optional[Tuple[float, float]] = None
    I_uf: Optional[Tuple[float, float]] = None
    I_ur: Optional[Tuple[float, float]] = None
    I_y_w: Optional[Tuple[float, float]] = None
    K_lt: Optional[Tuple[float, float]] = None
    R_w: Optional[Tuple[float, float]] = None
    T_sb: Optional[Tuple[float, float]] = None
    T_se: Optional[Tuple[float, float]] = None
    D_f: Optional[Tuple[float, float]] = None
    D_r: Optional[Tuple[float, float]] = None
    E_f: Optional[Tuple[float, float]] = None
    E_r: Optional[Tuple[float, float]] = None
    tire_p_cx1: Optional[Tuple[float, float]] = None
    tire_p_dx1: Optional[Tuple[float, float]] = None
    tire_p_dx3: Optional[Tuple[float, float]] = None
    tire_p_ex1: Optional[Tuple[float, float]] = None
    tire_p_kx1: Optional[Tuple[float, float]] = None
    tire_p_hx1: Optional[Tuple[float, float]] = None
    tire_p_vx1: Optional[Tuple[float, float]] = None
    tire_r_bx1: Optional[Tuple[float, float]] = None
    tire_r_bx2: Optional[Tuple[float, float]] = None
    tire_r_cx1: Optional[Tuple[float, float]] = None
    tire_r_ex1: Optional[Tuple[float, float]] = None
    tire_r_hx1: Optional[Tuple[float, float]] = None
    tire_p_cy1: Optional[Tuple[float, float]] = None
    tire_p_dy1: Optional[Tuple[float, float]] = None
    tire_p_dy3: Optional[Tuple[float, float]] = None
    tire_p_ey1: Optional[Tuple[float, float]] = None
    tire_p_ky1: Optional[Tuple[float, float]] = None
    tire_p_hy1: Optional[Tuple[float, float]] = None
    tire_p_hy3: Optional[Tuple[float, float]] = None
    tire_p_vy1: Optional[Tuple[float, float]] = None
    tire_p_vy3: Optional[Tuple[float, float]] = None
    tire_r_by1: Optional[Tuple[float, float]] = None
    tire_r_by2: Optional[Tuple[float, float]] = None
    tire_r_by3: Optional[Tuple[float, float]] = None
    tire_r_cy1: Optional[Tuple[float, float]] = None
    tire_r_ey1: Optional[Tuple[float, float]] = None
    tire_r_hy1: Optional[Tuple[float, float]] = None
    tire_r_vy1: Optional[Tuple[float, float]] = None
    tire_r_vy3: Optional[Tuple[float, float]] = None
    tire_r_vy4: Optional[Tuple[float, float]] = None
    tire_r_vy5: Optional[Tuple[float, float]] = None
    tire_r_vy6: Optional[Tuple[float, float]] = None

    def as_dict(self) -> dict:
        """The non-None ranges as {field_name: (low, high)}."""
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if getattr(self, f.name) is not None
        }

    def with_updates(self, **updates) -> "VehicleParamRanges":
        return replace(self, **updates)

class DynamicModel(IntEnum):
    KS = 1  # Kinematic Single Track
    ST = 2  # Single Track
    MB = 3  # Multi-body Model

    def parameter_count(self) -> int:
        return _MB_PARAMETER_COUNT if self is DynamicModel.MB else _BASE_PARAMETER_COUNT

    def velocity_indices(self) -> tuple[int, ...]:
        """State indices holding velocities/rates — what a collision halt zeroes.

        Pose-like states (positions, angles, ride heights, tire deflections)
        are deliberately absent: zeroing MB's z-heights breaks the suspension
        math with a divide-by-zero on the next step.
        """
        if self is DynamicModel.KS:
            return (3,)
        elif self is DynamicModel.ST:
            return (3, 5, 6)
        elif self is DynamicModel.MB:
            # vx, yaw/roll/pitch rates, vy, vz, front/rear rates and vy/vz,
            # four wheel speeds
            return (3, 5, 7, 9, 10, 12, 14, 15, 17, 19, 20, 22, 23, 24, 25, 26)
        else:
            raise ValueError(f"no velocity indices for model {self!r}")

    @staticmethod
    def from_string(model: str):
        if model == "ks":
            warnings.warn(
                "Chosen model is KS. This is different from previous versions of the gym."
            )
            return DynamicModel.KS
        elif model == "st":
            return DynamicModel.ST
        elif model == "mb":
            return DynamicModel.MB
        else:
            raise ValueError(f"Unknown model type {model}")

    def get_initial_state(self, pose=None, params=None):
        if self == DynamicModel.MB and params is None:
            raise ValueError("MultiBody model requires parameters to be provided.")
        if self == DynamicModel.KS:
            self.state_dim = 5
            self.control_dim = 2
        elif self == DynamicModel.ST:
            self.state_dim = 7
            self.control_dim = 2
        elif self == DynamicModel.MB:
            self.state_dim = 29
            self.control_dim = 2
        else:
            raise ValueError(f"Unknown model type {self}")
        state = np.zeros(self.state_dim)

        if pose is not None:
            state[0:2] = pose[0:2]
            state[4] = pose[2]

        if self == DynamicModel.MB:
            state = init_mb(state, params)
        return state

    @property
    def f_dynamics(self):
        if self == DynamicModel.KS:
            return vehicle_dynamics_ks
        elif self == DynamicModel.ST:
            return vehicle_dynamics_st
        elif self == DynamicModel.MB:
            return vehicle_dynamics_mb
        else:
            raise ValueError(f"Unknown model type {self}")

    def get_standardized_state_fn(self):
        if self == DynamicModel.KS:
            return get_standardized_state_ks
        elif self == DynamicModel.ST:
            return get_standardized_state_st
        elif self == DynamicModel.MB:
            return get_standardized_state_mb
        else:
            raise ValueError(f"Unknown model type {self}")


def get_f1tenth_vehicle_parameters() -> VehicleParameters:
    return F1TENTH_VEHICLE_PARAMETERS


def get_f1fifth_vehicle_parameters() -> VehicleParameters:
    return F1FIFTH_VEHICLE_PARAMETERS


def get_fullscale_vehicle_parameters() -> VehicleParameters:
    return FULLSCALE_VEHICLE_PARAMETERS
