"""Pure, correlated vehicle domain randomization for functional rollouts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import jax
import jax.numpy as jnp

from .dynamics import DynamicsParams
from .environment import CoreParams
from .geometry import BodyParams


# This literal mirrors the active prefix of the host VehicleParameters wire ABI.
# Keeping the tuple local lets this module remain independent of NumPy and the
# mutable environment package while still giving field-key derivation a stable
# positional contract.
ACTIVE_VEHICLE_FIELDS = (
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
    "collision_body_center_x",
    "collision_body_center_y",
)

# ASCII ``F1DR`` interpreted as one unsigned 32-bit fold-in tag. This is an RNG
# ABI: changing it changes every device-domain-randomization episode draw.
_DOMAIN_RANDOMIZATION_TAG = 0x46314452


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ActiveVehicleParams:
    """The twenty finite vehicle leaves consumed by supported JAX features."""

    mu: Any
    C_Sf: Any
    C_Sr: Any
    lf: Any
    lr: Any
    h: Any
    m: Any
    I: Any
    s_min: Any
    s_max: Any
    sv_min: Any
    sv_max: Any
    v_switch: Any
    a_max: Any
    v_min: Any
    v_max: Any
    width: Any
    length: Any
    collision_body_center_x: Any
    collision_body_center_y: Any

    def as_array(self) -> jax.Array:
        """Stack the ABI-ordered leaves along a final field axis."""
        leaves = tuple(
            jnp.asarray(getattr(self, name)) for name in ACTIVE_VEHICLE_FIELDS
        )
        return jnp.stack(leaves, axis=-1)

    @classmethod
    def from_array(cls, values: jax.Array) -> "ActiveVehicleParams":
        """Build active parameters from an array with a 20-value final axis."""
        values = jnp.asarray(values)
        if values.ndim < 1 or values.shape[-1] != len(ACTIVE_VEHICLE_FIELDS):
            raise ValueError(
                "values must have a final axis of length "
                f"{len(ACTIVE_VEHICLE_FIELDS)}, got {values.shape}"
            )
        return cls(*(values[..., index] for index in range(len(ACTIVE_VEHICLE_FIELDS))))

    def to_dynamics(self) -> DynamicsParams:
        """Return the correlated dynamics view of this vehicle draw."""
        return DynamicsParams(
            mu=self.mu,
            C_Sf=self.C_Sf,
            C_Sr=self.C_Sr,
            lf=self.lf,
            lr=self.lr,
            h=self.h,
            m=self.m,
            I=self.I,
            s_min=self.s_min,
            s_max=self.s_max,
            sv_min=self.sv_min,
            sv_max=self.sv_max,
            v_switch=self.v_switch,
            a_max=self.a_max,
            v_min=self.v_min,
            v_max=self.v_max,
        )

    def to_body(self) -> BodyParams:
        """Return body geometry in the supported models' CoG pose frame."""
        return BodyParams(
            length=self.length,
            width=self.width,
            centre_x=-self.lr + self.collision_body_center_x,
            centre_y=self.collision_body_center_y,
        )


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class VehicleRandomizationParams:
    """Nominal vehicle and traced bounds for one shared episode draw."""

    nominal: ActiveVehicleParams
    low: ActiveVehicleParams
    high: ActiveVehicleParams
    enabled: Any


def domain_randomization_key(reset_key: jax.Array) -> jax.Array:
    """Derive the named randomization stream without splitting reset streams."""
    return jax.random.fold_in(reset_key, _DOMAIN_RANDOMIZATION_TAG)


def sample_vehicle_params(
    key: jax.Array,
    spec: VehicleRandomizationParams,
) -> ActiveVehicleParams:
    """Draw one correlated vehicle using stable field-index substreams."""
    nominal = spec.nominal.as_array()
    low = jnp.asarray(spec.low.as_array(), dtype=nominal.dtype)
    high = jnp.asarray(spec.high.as_array(), dtype=nominal.dtype)

    def sample_enabled(_unused: None) -> ActiveVehicleParams:
        draws = []
        for index in range(len(ACTIVE_VEHICLE_FIELDS)):
            field_key = jax.random.fold_in(key, index)
            draw = jax.random.uniform(
                field_key,
                shape=(),
                dtype=nominal.dtype,
                minval=low[index],
                maxval=high[index],
            )
            draws.append(jnp.where(low[index] == high[index], low[index], draw))
        return ActiveVehicleParams.from_array(jnp.stack(draws))

    return jax.lax.cond(
        jnp.asarray(spec.enabled, dtype=jnp.bool_),
        sample_enabled,
        lambda _unused: spec.nominal,
        operand=None,
    )


def replace_core_vehicle_params(
    base: CoreParams,
    vehicle: ActiveVehicleParams,
) -> CoreParams:
    """Replace only the correlated dynamics and body leaves of core params."""
    transition = replace(base.transition, dynamics=vehicle.to_dynamics())
    return replace(base, transition=transition, body=vehicle.to_body())


def sample_core_params(
    key: jax.Array,
    base: CoreParams,
    spec: VehicleRandomizationParams,
) -> tuple[CoreParams, ActiveVehicleParams]:
    """Sample one vehicle and install both of its correlated core views."""
    vehicle = sample_vehicle_params(key, spec)
    return replace_core_vehicle_params(base, vehicle), vehicle


__all__ = [
    "ACTIVE_VEHICLE_FIELDS",
    "ActiveVehicleParams",
    "VehicleRandomizationParams",
    "domain_randomization_key",
    "replace_core_vehicle_params",
    "sample_core_params",
    "sample_vehicle_params",
]
