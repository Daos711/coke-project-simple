# src/params.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
from .geometry import Geometry

@dataclass
class Inlet:
    T_in_C: float = 370.0             # °C
    m_dot_kg_s: float = 5e-3 / 60.0   # кг/с (5 g/min)
    rho_vr: float = 1050.0            # кг/м^3
    v_gas_factor: float = 25.0        # ← газ бежит быстрее жидкости (25×)

    def velocity(self, geom: Geometry) -> float:
        return self.m_dot_kg_s / (self.rho_vr * geom.A)

    def velocity_gas(self, geom: Geometry) -> float:
        return self.velocity(geom) * self.v_gas_factor

@dataclass
class Walls:
    T_wall_C: float = 510.0
    # профиль времени прогрева: снизу быстрее, сверху медленнее
    tau_heat_s: float = 10.0 * 60.0
    tau_heat_bottom_s: float = 10.0 * 60.0
    tau_heat_top_s: float = 120.0 * 60.0
    tau_profile_beta: float = 2.0

@dataclass
class Materials:
    rho_coke_bulk: float = 950.0        # кг/м^3 (bulk)
    rho_dist_vap: float = 3.0           # кг/м^3 (условная для паров)
    alpha_c_threshold: float = 0.03     # для совместимости (в высоте больше не нужен)

@dataclass
class TimeSetup:
    total_hours: float = 12.0
    dt: float = 0.05
    snapshots_h: tuple[float, ...] = (0.0, 3.0, 6.0, 9.0, 12.0)
    contour_every_s: float = 10.0 * 60.0

def defaults():
    from .geometry import Geometry
    return Geometry(), Inlet(), Walls(), Materials(), TimeSetup()