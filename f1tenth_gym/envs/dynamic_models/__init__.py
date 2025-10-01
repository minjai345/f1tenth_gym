"""
This module contains the dynamic models available in the F1TENTH Gym.
Each submodule contains a single model, and the equations or their source is documented alongside it. Many of the models are from the CommonRoad repository, available here: https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from enum import IntEnum
from types import MappingProxyType
from typing import Mapping, Optional

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


_VEHICLE_BASE_KEYS: tuple[str, ...] = (
    "mu",
    "C_Sf",
    "C_Sr",
    "lf",
    "lr",
    "h",
    "m",
    "I",
    "s_min",
    "s_max",
    "sv_min",
    "sv_max",
    "v_switch",
    "a_max",
    "v_min",
    "v_max",
    "width",
    "length",
)


def _empty_mapping() -> Mapping[str, float]:
    return MappingProxyType({})


def _freeze_mapping(data: Mapping[str, float]) -> Mapping[str, float]:
    return MappingProxyType(dict(data))


@dataclass(frozen=True, slots=True)
class VehicleParameters:
    """Typed view over the vehicle parameters used by the dynamics models."""

    mu: float
    cornering_stiffness_front: float
    cornering_stiffness_rear: float
    lf: float
    lr: float
    cg_height: float
    mass: float
    inertia_z: float
    steering_angle_min: float
    steering_angle_max: float
    steering_velocity_min: float
    steering_velocity_max: float
    switch_velocity: float
    acceleration_max: float
    velocity_min: float
    velocity_max: float
    width: float
    length: float
    extra: Mapping[str, float] = field(default_factory=_empty_mapping)

    def as_mapping(self) -> dict[str, float]:
        """Return a mutable dictionary representation of the parameters."""

        data = {
            "mu": self.mu,
            "C_Sf": self.cornering_stiffness_front,
            "C_Sr": self.cornering_stiffness_rear,
            "lf": self.lf,
            "lr": self.lr,
            "h": self.cg_height,
            "m": self.mass,
            "I": self.inertia_z,
            "s_min": self.steering_angle_min,
            "s_max": self.steering_angle_max,
            "sv_min": self.steering_velocity_min,
            "sv_max": self.steering_velocity_max,
            "v_switch": self.switch_velocity,
            "a_max": self.acceleration_max,
            "v_min": self.velocity_min,
            "v_max": self.velocity_max,
            "width": self.width,
            "length": self.length,
        }
        data.update(self.extra)
        return data

    def with_updates(self, **updates: float) -> "VehicleParameters":
        """Return a copy of these parameters with the provided overrides applied."""

        mapping = self.as_mapping()
        mapping.update({key: float(value) for key, value in updates.items()})
        return VehicleParameters.from_mapping(mapping)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, float]) -> "VehicleParameters":
        """Construct parameters from a mapping (e.g. legacy dictionary configuration)."""

        if isinstance(raw, VehicleParameters):
            return raw

        missing = [key for key in _VEHICLE_BASE_KEYS if key not in raw]
        if missing:
            raise KeyError(
                f"Vehicle parameter dictionary missing required keys: {missing}"
            )

        base_values = {key: float(raw[key]) for key in _VEHICLE_BASE_KEYS}
        extras = {key: float(value) for key, value in raw.items() if key not in _VEHICLE_BASE_KEYS}

        return cls(
            mu=base_values["mu"],
            cornering_stiffness_front=base_values["C_Sf"],
            cornering_stiffness_rear=base_values["C_Sr"],
            lf=base_values["lf"],
            lr=base_values["lr"],
            cg_height=base_values["h"],
            mass=base_values["m"],
            inertia_z=base_values["I"],
            steering_angle_min=base_values["s_min"],
            steering_angle_max=base_values["s_max"],
            steering_velocity_min=base_values["sv_min"],
            steering_velocity_max=base_values["sv_max"],
            switch_velocity=base_values["v_switch"],
            acceleration_max=base_values["a_max"],
            velocity_min=base_values["v_min"],
            velocity_max=base_values["v_max"],
            width=base_values["width"],
            length=base_values["length"],
            extra=_freeze_mapping(extras),
        )


F1TENTH_VEHICLE_PARAMETERS = VehicleParameters(
    mu=1.0489,
    cornering_stiffness_front=4.718,
    cornering_stiffness_rear=5.4562,
    lf=0.15875,
    lr=0.17145,
    cg_height=0.074,
    mass=3.74,
    inertia_z=0.04712,
    steering_angle_min=-0.4189,
    steering_angle_max=0.4189,
    steering_velocity_min=-3.2,
    steering_velocity_max=3.2,
    switch_velocity=7.319,
    acceleration_max=9.51,
    velocity_min=-5.0,
    velocity_max=20.0,
    width=0.31,
    length=0.58,
)

F1FIFTH_VEHICLE_PARAMETERS = VehicleParameters(
    mu=1.1,
    cornering_stiffness_front=5.3507,
    cornering_stiffness_rear=5.3507,
    lf=0.2725,
    lr=0.2585,
    cg_height=0.1825,
    mass=15.32,
    inertia_z=0.64332,
    steering_angle_min=-0.4189,
    steering_angle_max=0.4189,
    steering_velocity_min=-3.2,
    steering_velocity_max=3.2,
    switch_velocity=7.319,
    acceleration_max=9.51,
    velocity_min=-5.0,
    velocity_max=20.0,
    width=0.55,
    length=0.8,
)

FULLSCALE_VEHICLE_PARAMETERS = VehicleParameters(
    mu=1.0489,
    cornering_stiffness_front=20.89,
    cornering_stiffness_rear=20.89,
    lf=0.88392,
    lr=1.50876,
    cg_height=0.557,
    mass=1225.8878467253344,
    inertia_z=1538.8533713561394,
    steering_angle_min=-0.91,
    steering_angle_max=0.91,
    steering_velocity_min=-0.4,
    steering_velocity_max=0.4,
    switch_velocity=4.755,
    acceleration_max=11.5,
    velocity_min=-13.9,
    velocity_max=45.8,
    width=1.674,
    length=4.298,
    extra=_freeze_mapping(
        {
            "kappa_dot_max": 0.4,
            "kappa_dot_dot_max": 20.0,
            "j_max": 10.0e3,
            "j_dot_max": 10.0e3,
            "m_s": 1094.542720290477,
            "m_uf": 65.67256321742863,
            "m_ur": 65.67256321742863,
            "I_Phi_s": 244.04723069965206,
            "I_y_s": 1342.2597688480864,
            "I_z": 1538.8533713561394,
            "I_xz_s": 0.0,
            "K_sf": 21898.332429625985,
            "K_sdf": 1459.3902937206362,
            "K_sr": 21898.332429625985,
            "K_sdr": 1459.3902937206362,
            "T_f": 1.389888,
            "T_r": 1.423416,
            "K_ras": 175186.65943700788,
            "K_tsf": -12880.270509148304,
            "K_tsr": 0.0,
            "K_rad": 10215.732056044453,
            "K_zt": 189785.5477234252,
            "h_cg": 0.5577840000000001,
            "h_raf": 0.0,
            "h_rar": 0.0,
            "h_s": 0.59436,
            "I_uf": 32.53963075995361,
            "I_ur": 32.53963075995361,
            "I_y_w": 1.7,
            "K_lt": 1.0278264878518764e-05,
            "R_w": 0.344,
            "T_sb": 0.76,
            "T_se": 1.0,
            "D_f": -0.6233595800524934,
            "D_r": -0.20997375328083986,
            "E_f": 0.0,
            "E_r": 0.0,
            "tire_p_cx1": 1.6411,
            "tire_p_dx1": 1.1739,
            "tire_p_dx3": 0.0,
            "tire_p_ex1": 0.46403,
            "tire_p_kx1": 22.303,
            "tire_p_hx1": 0.0012297,
            "tire_p_vx1": -8.8098e-06,
            "tire_r_bx1": 13.276,
            "tire_r_bx2": -13.778,
            "tire_r_cx1": 1.2568,
            "tire_r_ex1": 0.65225,
            "tire_r_hx1": 0.0050722,
            "tire_p_cy1": 1.3507,
            "tire_p_dy1": 1.0489,
            "tire_p_dy3": -2.8821,
            "tire_p_ey1": -0.0074722,
            "tire_p_ky1": -21.92,
            "tire_p_hy1": 0.0026747,
            "tire_p_hy3": 0.031415,
            "tire_p_vy1": 0.037318,
            "tire_p_vy3": -0.32931,
            "tire_r_by1": 7.1433,
            "tire_r_by2": 9.1916,
            "tire_r_by3": -0.027856,
            "tire_r_cy1": 1.0719,
            "tire_r_ey1": -0.27572,
            "tire_r_hy1": 5.7448e-06,
            "tire_r_vy1": -0.027825,
            "tire_r_vy3": -0.27568,
            "tire_r_vy4": 12.12,
            "tire_r_vy5": 1.9,
            "tire_r_vy6": -10.704,
        }
    ),
)


def get_f1tenth_vehicle_parameters() -> VehicleParameters:
    return F1TENTH_VEHICLE_PARAMETERS


def get_f1fifth_vehicle_parameters() -> VehicleParameters:
    return F1FIFTH_VEHICLE_PARAMETERS


def get_fullscale_vehicle_parameters() -> VehicleParameters:
    return FULLSCALE_VEHICLE_PARAMETERS


class DynamicModel(IntEnum):
    KS = 1  # Kinematic Single Track
    ST = 2  # Single Track
    MB = 3  # Multi-body Model

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
        # Assert that if self is MB, params is not None
        if self == DynamicModel.MB and params is None:
            raise ValueError("MultiBody model requires parameters to be provided.")
        # initialize zero state
        if self == DynamicModel.KS:
            # state is [x, y, steer_angle, vel, yaw_angle]
            self.state_dim = 5
            self.control_dim = 2
        elif self == DynamicModel.ST:
            # state is [x, y, steer_angle, vel, yaw_angle, yaw_rate, slip_angle]
            self.state_dim = 7
            self.control_dim = 2
        elif self == DynamicModel.MB:
            # state is a 29D vector
            self.state_dim = 29
            self.control_dim = 2
        else:
            raise ValueError(f"Unknown model type {self}")
        state = np.zeros(self.state_dim)

        # set initial pose if provided
        if pose is not None:
            state[0:2] = pose[0:2]
            state[4] = pose[2]

        # If state is MultiBody, we must inflate the state to 29D
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
        """
        This function returns the standardized state information for the model.
        This needs to be a function, because the state information is different for each model.
        Slip is not directly available from the MB model.
        """
        if self == DynamicModel.KS:
            return get_standardized_state_ks
        elif self == DynamicModel.ST:
            return get_standardized_state_st
        elif self == DynamicModel.MB:
            return get_standardized_state_mb
        else:
            raise ValueError(f"Unknown model type {self}")
