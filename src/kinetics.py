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
    """Упрощённая кинетика VR→(дистилляты + кокс) с трёхрежимной зависимостью порядка."""
    # Температуры переключения (°C) для VR3 из статьи
    T1_C: float = 487.8
    T2_C: float = 570.1

    # Масштабы ветвей (без калибровки)
    scale_dist: float = 0.002
    scale_coke: float = 0.50

    # Arrhenius-пары и порядки для трёх диапазонов
    reg1:  Regime = field(default_factory=lambda: Regime(1.0, 7.8408e4, 1.1045e5, 7.8408e4, 1.1045e5))
    reg15: Regime = field(default_factory=lambda: Regime(1.5, 3.3909e18, 3.0287e5, 3.3909e17, 3.0287e5))
    reg2:  Regime = field(default_factory=lambda: Regime(2.0, 2.1660e11, 1.8316e5, 2.1660e10, 1.8316e5))

    def _regime(self, T_C: float) -> Regime:
        if T_C < self.T1_C:   return self.reg1
        elif T_C < self.T2_C: return self.reg15
        else:                 return self.reg2

    def rates(self, T_C: float) -> tuple[float, float, float]:
        """Возвращает (k_dist, k_coke, order) при температуре в °C."""
        r = self._regime(T_C)
        Tk = T_C + 273.15
        k_dist = self.scale_dist * r.A_dist * math.exp(-r.Ea_dist / (R_GAS * Tk))
        k_coke = self.scale_coke * r.A_coke * math.exp(-r.Ea_coke / (R_GAS * Tk))
        return k_dist, k_coke, r.order
