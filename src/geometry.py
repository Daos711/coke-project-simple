# src/geometry.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

@dataclass
class Geometry:
    H: float = 0.5692  # м
    D: float = 0.0602  # м
    NZ: int = 50

    @property
    def A(self) -> float:
        return math.pi * (self.D ** 2) / 4.0

    @property
    def dz(self) -> float:
        return self.H / self.NZ

    @property
    def z(self) -> np.ndarray:
        return np.linspace(0.0, self.H, self.NZ)
