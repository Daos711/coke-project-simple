# src/kinetics.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field
import math

R_GAS = 8.314462618  # Дж/(моль·К)

@dataclass
class Regime:
    order: float
    A_dist: float; Ea_dist: float
    A_coke: float; Ea_coke: float

@dataclass
class VR3Kinetics:
    """
    Кинетика VR-3 с переключением порядка и масштабированием ветвей.
    Масштабы scale_* передаются прямо в numba-ядро решателя.
    """
    T1_C: float = 488.0
    T2_C: float = 570.0

    # ---- НАСТРОЙКИ ДЛЯ ДОТЯЖКИ ДО ЭКСПЕРИМЕНТА ----
    scale_dist: float = 0.25   # было 0.30
    scale_coke: float = 5.00   # было 4.20

    # Базовые параметры (как раньше)
    reg1:  Regime = field(default_factory=lambda: Regime(order=1.0,  A_dist=1.2e+01, Ea_dist=92e3, A_coke=1.5e+01, Ea_coke=85e3))
    reg15: Regime = field(default_factory=lambda: Regime(order=1.5,  A_dist=1.2e+01, Ea_dist=92e3, A_coke=1.5e+01, Ea_coke=85e3))
    reg2:  Regime = field(default_factory=lambda: Regime(order=2.0,  A_dist=1.2e+01, Ea_dist=92e3, A_coke=1.5e+01, Ea_coke=85e3))

    def _regime(self, T_C: float) -> Regime:
        if T_C < self.T1_C:   return self.reg1
        elif T_C < self.T2_C: return self.reg15
        else:                 return self.reg2

    def rates(self, T_C: float) -> tuple[float, float, float]:
        """(k_dist [1/с], k_coke [1/с], order) — используется питон-веткой; в numba-ядро масштабы передаются отдельно."""
        r = self._regime(T_C); Tk = T_C + 273.15
        k_dist = self.scale_dist * r.A_dist * math.exp(-r.Ea_dist / (R_GAS * Tk))
        k_coke = self.scale_coke * r.A_coke * math.exp(-r.Ea_coke / (R_GAS * Tk))
        return k_dist, k_coke, r.order