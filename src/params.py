# src/params.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from .geometry import Geometry


@dataclass
class WallLayer:
    """Single wall/lining layer description (SI units)."""
    k: float
    rho: float
    cp: float
    thickness: float
    epsilon: float = 0.85


@dataclass
class WallEnergy:
    """Two-node wall energy model configuration."""
    outer: WallLayer
    inner: WallLayer
    h_amb: float
    T_amb_C: float
    zones: int = 3


@dataclass
class MixtureEnergy:
    """Effective thermal properties of the reacting mixture."""
    lambda_eff: float
    cp_eff: float
    h0_mix: float
    alpha_mdot: float
    alpha_p: float
    mdot_ref: float
    p_ref: float


@dataclass
class ReactionEnergy:
    """Effective heats of reaction (per kilogram of VR)."""
    dH_dist: float = 0.0
    dH_coke: float = 0.0


def default_wall_energy() -> WallEnergy:
    """Factory helper for typical delayed coking drum wall properties."""
    outer = WallLayer(k=30.0, rho=7800.0, cp=600.0, thickness=0.03, epsilon=0.85)
    inner = WallLayer(k=1.5, rho=3500.0, cp=800.0, thickness=0.12, epsilon=0.80)
    return WallEnergy(outer=outer, inner=inner, h_amb=15.0, T_amb_C=25.0)


def default_mixture_energy() -> MixtureEnergy:
    """Factory helper for an effective mixture energy configuration."""
    return MixtureEnergy(
        lambda_eff=0.5,
        cp_eff=2500.0,
        h0_mix=150.0,
        alpha_mdot=0.5,
        alpha_p=0.1,
        mdot_ref=1.0,
        p_ref=1.0,
    )


def default_reaction_energy() -> ReactionEnergy:
    return ReactionEnergy(dH_dist=0.0, dH_coke=0.0)

@dataclass
class Inlet:
    T_in_C: float = 370.0
    m_dot_kg_s: float = 5e-3 / 60.0   # 5 g/min
    rho_vr: float = 1050.0            # кг/м³ (VR3)
    v_gas_base_factor: float = 8.0    # псевдоскорость газа относительно жидкости

    def velocity(self, geom: Geometry) -> float:
        return self.m_dot_kg_s / (self.rho_vr * geom.A)

    def velocity_gas(self, geom: Geometry, porosity: float = 1.0) -> float:
        eff = self.v_gas_base_factor / max(porosity, 0.3)
        return self.velocity(geom) * min(eff, 25.0)

@dataclass
class Walls:
    T_wall_C: float = 510.0
    tau_heat_bottom_s: float = 2.0 * 3600.0
    tau_heat_top_s:    float = 14.0 * 3600.0   # ← было 6 ч, вернули 14 ч
    tau_profile_beta:  float = 2.5             # можешь оставить 2.8 — не критично

@dataclass
class Materials:
    rho_coke_bulk: float = 950.0   # кг/м³
    rho_dist_vap: float = 2.5      # кг/м³ (условный пар)
    porosity_min: float = 0.30     # минимальная пористость в слое
    porosity_initial: float = 1.0
    d_particle_mm: float = 1.0

@dataclass
class TimeSetup:
    total_hours: float = 12.0
    dt: float = 0.05
    snapshots_h: tuple[float, ...] = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0)
    contour_every_s: float = 10.0 * 60.0

def defaults():
    return Geometry(), Inlet(), Walls(), Materials(), TimeSetup()
