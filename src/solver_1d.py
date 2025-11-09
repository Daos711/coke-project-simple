# src/solver_1d.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math

import numpy as np

from .geometry import Geometry
from .params import (
    Inlet,
    Walls,
    Materials,
    TimeSetup,
    WallEnergy,
    MixtureEnergy,
    ReactionEnergy,
    default_wall_energy,
    default_mixture_energy,
    default_reaction_energy,
)
from .kinetics import VR3Kinetics

# Порог начала коксования (ниже реакции выключены)
T_COKE_ONSET_C = 415.0
SIGMA = 5.670374419e-8  # Постоянная Стефана—Больцмана, W/(m²·K⁴)


def h_mix(
    mdot_kg_s: float,
    pressure: float,
    h0_mix: float,
    alpha_mdot: float,
    alpha_p: float,
    mdot_ref: float,
    p_ref: float,
) -> float:
    """Корреляция коэффициента теплоотдачи стенка→смесь."""
    mdot = max(float(mdot_kg_s), 1e-6)
    pref = max(float(mdot_ref), 1e-6)
    pres = max(float(pressure), 1e-6)
    pref_p = max(float(p_ref), 1e-6)
    h0 = max(float(h0_mix), 1e-6)
    return h0 * (mdot / pref) ** float(alpha_mdot) * (pres / pref_p) ** float(alpha_p)


class Coking1DSolver:
    def __init__(
        self,
        geom: Geometry,
        inlet: Inlet,
        walls: Walls,
        mats: Materials,
        tcfg: TimeSetup,
        kin: VR3Kinetics,
        wall_energy: WallEnergy | None = None,
        mixture_energy: MixtureEnergy | None = None,
        reaction_energy: ReactionEnergy | None = None,
    ):
        self.g, self.inlet, self.walls, self.mats, self.tcfg, self.kin = (
            geom,
            inlet,
            walls,
            mats,
            tcfg,
            kin,
        )
        self.wall_energy = wall_energy or default_wall_energy()
        self.mixture_energy = mixture_energy or default_mixture_energy()
        self.reaction_energy = reaction_energy or default_reaction_energy()

        self.NZ = geom.NZ
        NZ = self.NZ

        # Температура смеси (K) и вспомогательное поле в °C для обратной совместимости
        self.T_mix_K = np.full(NZ, self._to_K(inlet.T_in_C), dtype=np.float64)
        self.T = self.T_mix_K - 273.15

        # Объёмные доли фаз
        self.aR = np.ones(NZ, dtype=np.float64)
        self.aC = np.zeros(NZ, dtype=np.float64)
        self.aD = np.zeros(NZ, dtype=np.float64)
        self.gamma = 1.0 - self.aC

        self.porosity_min = float(getattr(mats, "porosity_min", 0.3))
        self.time_s = 0.0

        self.vol_cell = self.g.A * max(self.g.dz, 1e-12)
        self.perimeter = math.pi * self.g.D
        self.base_area = math.pi * (self.g.D / 2.0) ** 2

        # Динамические условия процесса
        self.mdot_kg_s = max(float(self.inlet.m_dot_kg_s), 0.0)
        self.pressure = max(1.0, getattr(self, "pressure", 1.0))
        self.T_inlet_K = self._to_K(self.inlet.T_in_C)
        self.ambient_temperature_K = self._to_K(self.wall_energy.T_amb_C)

        # Параметры двухузловой стенки
        self.zone_count = max(int(self.wall_energy.zones), 1)
        self.zone_index = self._build_zone_index(NZ, self.zone_count)
        self.zone_area_mix = self._build_zone_areas()
        self.cell_area_exchange = self._build_cell_areas()

        outer, inner = self.wall_energy.outer, self.wall_energy.inner
        self.wall_eps = np.full(self.zone_count, float(outer.epsilon), dtype=float)

        self.T_shell_out_K = np.full(self.zone_count, self.ambient_temperature_K, dtype=np.float64)
        self.T_shell_in_K = np.full(self.zone_count, self.T_inlet_K, dtype=np.float64)

        self.wall_conductance = np.zeros(self.zone_count, dtype=np.float64)
        self.wall_C_out = np.zeros(self.zone_count, dtype=np.float64)
        self.wall_C_in = np.zeros(self.zone_count, dtype=np.float64)
        for i in range(self.zone_count):
            area = max(self.zone_area_mix[i], 1e-6)
            R_total = outer.thickness / max(outer.k, 1e-6) + inner.thickness / max(inner.k, 1e-6)
            self.wall_conductance[i] = area / max(R_total, 1e-9)
            self.wall_C_out[i] = outer.rho * outer.cp * area * outer.thickness
            self.wall_C_in[i] = inner.rho * inner.cp * area * inner.thickness

        # Наблюдения внешней поверхности
        self.shell_obs_K = np.full(self.zone_count, np.nan, dtype=np.float64)
        self.shell_obs_valid = np.zeros(self.zone_count, dtype=bool)
        self.bottom_zone = 0
        self.top_zone = self.zone_count - 1

        # «жёсткий» инвентарь VR в стартовый момент (для балансной нормировки)
        self.m_vr0 = float(np.sum(self.inlet.rho_vr * self.aR) * self.vol_cell)

        # Истории / снимки
        self.snap_times_h = list(tcfg.snapshots_h)
        self._snap_idx = 0
        self.snapshots = {"t_h": [], "T": [], "aR": [], "aD": [], "aC": []}

        self.time_h_hist: list[float] = []
        self.bed_eq_cm_hist: list[float] = []
        self.bed_front_cm_hist: list[float] = []
        self.contour_every_s = float(tcfg.contour_every_s)
        self.contour_t: list[float] = []
        self.contour_T: list[np.ndarray] = []
        self.contour_aR: list[np.ndarray] = []
        self.contour_aD: list[np.ndarray] = []
        self.contour_aC: list[np.ndarray] = []
        self.porosity_avg = 1.0

        self.k_mix_gas = 0.05

        self._take_snapshot()
        self._maybe_take_contour(force=True)
        if self.snap_times_h and abs(self.snapshots["t_h"][-1] - self.snap_times_h[0]) < 1e-6:
            self._snap_idx = 1

    # --- Вспомогательные преобразования ---
    @staticmethod
    def _to_K(T_C: float) -> float:
        return float(T_C) + 273.15

    @staticmethod
    def _to_C(T_K: float) -> float:
        return float(T_K) - 273.15

    def _build_zone_index(self, NZ: int, zones: int) -> np.ndarray:
        idx = np.zeros(NZ, dtype=int)
        if zones <= 1 or NZ == 0:
            return idx
        if zones == 3 and NZ >= 3:
            idx[:] = 1
            idx[0] = 0
            idx[-1] = 2
            return idx
        cuts = np.linspace(0, NZ, zones + 1, dtype=int)
        for i in range(zones):
            idx[cuts[i]: cuts[i + 1]] = i
        return idx

    def _build_zone_areas(self) -> np.ndarray:
        zone_area = np.zeros(self.zone_count, dtype=np.float64)
        dz = max(self.g.dz, 1e-9)
        counts = [int(np.sum(self.zone_index == i)) for i in range(self.zone_count)]
        for i, n in enumerate(counts):
            if i == 0 or i == self.zone_count - 1:
                zone_area[i] = max(self.base_area, 1e-6)
            else:
                zone_area[i] = max(self.perimeter * dz * max(n, 1), 1e-6)
        return zone_area

    def _build_cell_areas(self) -> np.ndarray:
        cell_area = np.zeros(self.NZ, dtype=np.float64)
        for i in range(self.zone_count):
            idxs = np.where(self.zone_index == i)[0]
            n = max(len(idxs), 1)
            area = self.zone_area_mix[i] / n
            for k in idxs:
                cell_area[k] = area
        return cell_area

    # --- Публичные сервисные методы ---
    def zone_index_for_cell(self, k: int) -> int:
        return int(self.zone_index[int(np.clip(k, 0, self.NZ - 1))])

    def set_shell_observations(
        self,
        t: float,
        T_upper_C: float | None,
        T_lower_C: float | None,
    ) -> None:
        if T_upper_C is not None:
            self.shell_obs_K[self.top_zone] = self._to_K(T_upper_C)
            self.shell_obs_valid[self.top_zone] = True
        if T_lower_C is not None:
            self.shell_obs_K[self.bottom_zone] = self._to_K(T_lower_C)
            self.shell_obs_valid[self.bottom_zone] = True

    def set_operating_conditions(
        self,
        T_in_C: float | None = None,
        m_dot_kg_s: float | None = None,
        pressure: float | None = None,
        T_amb_C: float | None = None,
    ) -> None:
        if T_in_C is not None:
            self.inlet.T_in_C = float(T_in_C)
            self.T_inlet_K = self._to_K(T_in_C)
        if m_dot_kg_s is not None:
            self.mdot_kg_s = max(float(m_dot_kg_s), 0.0)
            self.inlet.m_dot_kg_s = self.mdot_kg_s
        if pressure is not None:
            self.pressure = max(float(pressure), 1e-6)
        if T_amb_C is not None:
            self.ambient_temperature_K = self._to_K(T_amb_C)

    # --- Геометрия и балансы ---
    def bed_height_front(self, thr: float = 0.05) -> float:
        idx = np.where(self.aC > thr)[0]
        if idx.size == 0:
            return 0.0
        top_idx = idx[-1] + 1
        return float(min(top_idx * self.g.dz, self.g.H))

    def bed_height_equiv(self) -> float:
        eps_max = 1.0 - self.porosity_min
        if eps_max <= 1e-12:
            return 0.0
        aC_clip = np.clip(self.aC, 0.0, eps_max)
        return float(min(np.sum(aC_clip / eps_max) * self.g.dz, self.g.H))

    def coke_mass(self) -> float:
        return float(np.sum(self.mats.rho_coke_bulk * self.aC) * self.vol_cell)

    def inlet_mass_total(self) -> float:
        return float(self.mdot_kg_s * self.time_s)

    def vr_inventory_mass(self) -> float:
        return float(np.sum(self.inlet.rho_vr * self.aR) * self.vol_cell)

    def get_porosity_profile(self) -> np.ndarray:
        return np.maximum(1.0 - self.aC, self.porosity_min)

    def update_porosity_effects(self) -> None:
        zone = self.aC > 0.01
        por = 1.0 - self.aC
        if np.any(zone):
            self.porosity_avg = max(float(np.mean(por[zone])), self.porosity_min)
        else:
            self.porosity_avg = 1.0

    def coke_yield_pct_feed(self) -> float:
        return 100.0 * self.coke_mass() / max(self.inlet_mass_total(), 1e-9)

    def coke_yield_pct_balance(self) -> float:
        return 100.0 * self.coke_mass() / max(self.m_vr0 + self.inlet_mass_total(), 1e-9)

    def outlet_temperature_C(self) -> float:
        return float(self.T[-1])

    def shell_temperatures_C(self) -> tuple[float, ...]:
        return tuple(self._to_C(TK) for TK in self.T_shell_out_K)

    # --- Истории ---
    def _take_snapshot(self) -> None:
        t_h = self.time_s / 3600.0
        self.snapshots["t_h"].append(t_h)
        self.snapshots["T"].append(self.T.copy())
        self.snapshots["aR"].append(self.aR.copy())
        self.snapshots["aD"].append(self.aD.copy())
        self.snapshots["aC"].append(self.aC.copy())

    def _maybe_take_snapshot(self) -> None:
        if self._snap_idx >= len(self.snap_times_h):
            return
        t_h = self.time_s / 3600.0
        if t_h + 1e-4 >= self.snap_times_h[self._snap_idx]:
            self._take_snapshot()
            self._snap_idx += 1

    def _maybe_take_contour(self, force: bool = False) -> None:
        if force or (len(self.contour_t) == 0) or (
            self.time_s - self.contour_t[-1] >= self.contour_every_s - 1e-12
        ):
            self.contour_t.append(self.time_s)
            self.contour_T.append(self.T.copy())
            self.contour_aR.append(self.aR.copy())
            self.contour_aD.append(self.aD.copy())
            self.contour_aC.append(self.aC.copy())

    # --- Шаг моделирования ---
    def _liquid_velocity(self) -> float:
        return self.mdot_kg_s / (max(self.inlet.rho_vr, 1e-9) * max(self.g.A, 1e-12))

    def _advect(self, vR: float, vD: float, dt: float) -> None:
        dz = max(self.g.dz, 1e-12)
        NZ = self.NZ
        if NZ == 0:
            return

        max_cfl = max(vR, vD) * dt / dz
        nsub = int(max(1.0, math.ceil(max_cfl))) if max_cfl > 0 else 1
        dt_sub = dt / nsub

        for _ in range(nsub):
            sR = max(vR, 0.0) * dt_sub / dz
            aR_old = self.aR.copy()
            prevR = self.gamma[0]
            for k in range(NZ):
                cur = aR_old[k]
                self.aR[k] = cur - sR * (cur - prevR)
                if self.aR[k] < 0.0:
                    self.aR[k] = 0.0
                maxR = self.gamma[k]
                if self.aR[k] > maxR:
                    self.aR[k] = maxR
                prevR = cur

            avg_por = max(self.porosity_avg, self.porosity_min)
            sD = max(vD, 0.0) * dt_sub / dz
            if avg_por > 1e-12:
                sD /= avg_por
            aD_old = self.aD.copy()
            prevD = 0.0
            for k in range(NZ):
                cur = aD_old[k]
                self.aD[k] = cur - sD * (cur - prevD)
                if self.aD[k] < 0.0:
                    self.aD[k] = 0.0
                prevD = cur

            for k in range(NZ):
                maxD = max(self.gamma[k] - self.aR[k], 0.0)
                if self.aD[k] > maxD:
                    self.aD[k] = maxD

    def _reaction_step(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        NZ = self.NZ
        rho_vr = max(self.inlet.rho_vr, 1e-9)
        rho_coke = max(self.mats.rho_coke_bulk, 1e-9)
        rho_dist = max(self.mats.rho_dist_vap, 1e-9)
        rate_dist = np.zeros(NZ, dtype=np.float64)
        rate_coke = np.zeros(NZ, dtype=np.float64)

        for k in range(NZ):
            T_C = self.T[k]
            if T_C < T_COKE_ONSET_C:
                continue

            k_dist, k_coke, order = self.kin.rates(T_C)
            g = self.gamma[k]
            r0 = self.aR[k]
            if g < 0.4:
                phi = float(np.clip(self.kin.phi_por, 0.0, 1.0))
                factor = phi + (1.0 - phi) * (g / 0.4)
                k_dist *= factor
                k_coke *= factor

            k_tot = k_dist + k_coke
            if k_tot <= 0.0:
                continue
            if k_tot * dt > 0.1:
                ratio = 0.1 / (k_tot * dt)
                k_dist *= ratio
                k_coke *= ratio
                k_tot = k_dist + k_coke

            if r0 <= 1e-6 or g <= 0.01:
                continue

            base = r0 ** (order - 1.0)
            r_mass = rho_vr * g * (r0 ** order)
            rate_dist_k = r_mass * k_dist
            rate_coke_k = r_mass * k_coke

            dR = g * r0 * (k_tot * base) * dt
            if dR > r0:
                dR = r0
            self.aR[k] = max(r0 - dR, 0.0)

            dC = rate_coke_k * dt / rho_coke
            self.aC[k] = min(self.aC[k] + dC, 1.0 - self.porosity_min)
            self.gamma[k] = 1.0 - self.aC[k]

            dD = rate_dist_k * dt / rho_dist
            self.aD[k] = min(self.aD[k] + dD, max(self.gamma[k] - self.aR[k], 0.0))

            rate_dist[k] = rate_dist_k
            rate_coke[k] = rate_coke_k

        return rate_dist, rate_coke

    def _update_mixture_energy(
        self,
        dt: float,
        v_liq: float,
        rate_dist: np.ndarray,
        rate_coke: np.ndarray,
        T_shell_in_prev: np.ndarray,
        h_coeff: float,
    ) -> None:
        NZ = self.NZ
        if NZ == 0:
            return
        dz = max(self.g.dz, 1e-9)
        rho = max(self.inlet.rho_vr, 1e-9)
        cp = max(self.mixture_energy.cp_eff, 1e-9)
        lam = max(self.mixture_energy.lambda_eff, 1e-9)
        rho_cp = rho * cp

        T_conv = self.T_mix_K.copy()

        cfl = max(v_liq, 0.0) * dt / dz
        nsub = int(max(1.0, math.ceil(cfl))) if cfl > 0 else 1
        dt_sub = dt / nsub
        for _ in range(nsub):
            s = max(v_liq, 0.0) * dt_sub / dz
            T_prev = T_conv.copy()
            T_conv[0] = self.T_inlet_K
            for k in range(1, NZ):
                cur = T_prev[k]
                T_conv[k] = cur - s * (cur - T_prev[k - 1])
            T_conv[0] = self.T_inlet_K

        q_rxn = -self.reaction_energy.dH_dist * rate_dist - self.reaction_energy.dH_coke * rate_coke

        alpha = lam / (dz ** 2)
        coeff_wall = h_coeff * self.cell_area_exchange / self.vol_cell

        A = np.zeros(NZ, dtype=np.float64)
        B = np.zeros(NZ, dtype=np.float64)
        C = np.zeros(NZ, dtype=np.float64)
        RHS = rho_cp / dt * T_conv + coeff_wall * T_shell_in_prev + q_rxn

        if NZ == 1:
            B[0] = rho_cp / dt + coeff_wall[0]
        else:
            for k in range(NZ):
                B[k] = rho_cp / dt + coeff_wall[k] + 2.0 * alpha
            for k in range(1, NZ):
                A[k] = -alpha
            for k in range(NZ - 1):
                C[k] = -alpha
            B[0] = rho_cp / dt + coeff_wall[0] + alpha
            B[-1] = rho_cp / dt + coeff_wall[-1] + alpha
        T_new = self._solve_tridiagonal(A, B, C, RHS)
        if not np.all(np.isfinite(T_new)):
            raise FloatingPointError("Non-finite temperatures in energy update")
        self.T_mix_K[:] = np.clip(T_new, 200.0, 2500.0)

    def _update_wall_energy(self, dt: float, h_coeff: float) -> None:
        T_mix_zone = np.zeros(self.zone_count, dtype=np.float64)
        for i in range(self.zone_count):
            idxs = np.where(self.zone_index == i)[0]
            if idxs.size:
                T_mix_zone[i] = float(np.mean(self.T_mix_K[idxs]))
            else:
                T_mix_zone[i] = float(np.mean(self.T_mix_K))

        for i in range(self.zone_count):
            area = self.zone_area_mix[i]
            K_wall = self.wall_conductance[i]
            C_out = max(self.wall_C_out[i], 1e-9)
            C_in = max(self.wall_C_in[i], 1e-9)
            T_obs = self.shell_obs_K[i] if self.shell_obs_valid[i] else None
            T_out = self.T_shell_out_K[i]
            T_in = self.T_shell_in_K[i]

            q_cond = K_wall * (T_out - T_in)
            q_conv_amb = self.wall_energy.h_amb * area * (T_out - self.ambient_temperature_K)
            T_sky = max(self.ambient_temperature_K - 10.0, 1.0)
            q_rad = SIGMA * self.wall_eps[i] * area * ((T_out ** 4) - (T_sky ** 4))
            dT_out = (-q_cond - q_conv_amb - q_rad) * dt / C_out
            if T_obs is not None:
                tau_obs = max(300.0, 6.0 * dt)
                dT_out += (T_obs - T_out) * dt / tau_obs
            self.T_shell_out_K[i] = T_out + dT_out

            q_cond_inner = K_wall * (self.T_shell_out_K[i] - T_in)
            q_conv_mix = h_coeff * area * (T_in - T_mix_zone[i])
            dT_in = (q_cond_inner - q_conv_mix) * dt / C_in
            self.T_shell_in_K[i] = T_in + dT_in

        self.shell_obs_valid[:] = False

    @staticmethod
    def _solve_tridiagonal(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
        n = len(d)
        cp = c.copy()
        dp = d.copy()
        bp = b.copy()
        for i in range(1, n):
            m = a[i] / bp[i - 1] if bp[i - 1] != 0 else 0.0
            bp[i] = bp[i] - m * cp[i - 1]
            dp[i] = dp[i] - m * dp[i - 1]
        x = np.zeros(n, dtype=np.float64)
        if n:
            x[-1] = dp[-1] / bp[-1]
            for i in range(n - 2, -1, -1):
                x[i] = (dp[i] - cp[i] * x[i + 1]) / bp[i]
        return x

    def step(self, dt: float | None = None) -> None:
        dt_val = float(self.tcfg.dt if dt is None else dt)
        self.update_porosity_effects()

        vR_base = self._liquid_velocity()
        vD_gas = self.inlet.velocity_gas(self.g, self.porosity_avg)
        vR_eff = vR_base + self.k_mix_gas * vD_gas

        self._advect(vR_eff, vD_gas, dt_val)
        rate_dist, rate_coke = self._reaction_step(dt_val)

        h_coeff = h_mix(
            self.mdot_kg_s,
            self.pressure,
            self.mixture_energy.h0_mix,
            self.mixture_energy.alpha_mdot,
            self.mixture_energy.alpha_p,
            self.mixture_energy.mdot_ref,
            self.mixture_energy.p_ref,
        )

        T_shell_in_prev = self.T_shell_in_K.copy()
        self._update_mixture_energy(dt_val, vR_eff, rate_dist, rate_coke, T_shell_in_prev, h_coeff)
        self._update_wall_energy(dt_val, h_coeff)

        self.T = self.T_mix_K - 273.15
        self.time_s += dt_val

    def run(self, verbose_hourly: bool = True) -> dict:
        steps = int(self.tcfg.total_hours * 3600.0 / self.tcfg.dt)
        next_hour_s = 3600.0
        for _ in range(steps):
            self.step()
            if self.time_s + 1e-12 >= next_hour_s:
                self.time_h_hist.append(self.time_s / 3600.0)
                self.bed_eq_cm_hist.append(self.bed_height_equiv() * 100.0)
                self.bed_front_cm_hist.append(self.bed_height_front() * 100.0)
                if verbose_hourly:
                    y_feed = self.coke_yield_pct_feed()
                    y_bal = self.coke_yield_pct_balance()
                    info = (
                        f"t = {self.time_s/3600.0:5.1f} ч | H_eq={self.bed_eq_cm_hist[-1]:.1f} см | "
                        f"H_front={self.bed_front_cm_hist[-1]:.1f} см | Yбал={y_bal:5.2f}% | "
                        f"T_avg={np.mean(self.T):.1f}°C | ε_avg={self.porosity_avg:.3f}"
                    )
                    print(info)
                next_hour_s += 3600.0
            self._maybe_take_snapshot()
            self._maybe_take_contour()

        return {
            "H_bed_m": self.bed_height_equiv(),
            "H_front_m": self.bed_height_front(),
            "yield_pct": self.coke_yield_pct_feed(),
            "T_avg_C": float(np.mean(self.T)),
            "porosity_avg": float(self.porosity_avg),
            "final": {
                "T": self.T.copy(),
                "aR": self.aR.copy(),
                "aD": self.aD.copy(),
                "aC": self.aC.copy(),
            },
            "z": self.g.z.copy(),
            "snapshots": {
                "t_h": np.array(self.snapshots["t_h"], dtype=float),
                "T": np.stack(self.snapshots["T"], axis=0),
                "aR": np.stack(self.snapshots["aR"], axis=0),
                "aD": np.stack(self.snapshots["aD"], axis=0),
                "aC": np.stack(self.snapshots["aC"], axis=0),
            },
            "growth": {
                "t_h": np.array(self.time_h_hist, dtype=float),
                "H_cm": np.array(self.bed_eq_cm_hist, dtype=float),
                "H_front_cm": np.array(self.bed_front_cm_hist, dtype=float),
            },
            "contours": {
                "t_s": np.array(self.contour_t, dtype=float),
                "T": np.stack(self.contour_T, axis=0) if self.contour_T else np.zeros((0, self.NZ)),
                "aR": np.stack(self.contour_aR, axis=0) if self.contour_aR else np.zeros((0, self.NZ)),
                "aD": np.stack(self.contour_aD, axis=0) if self.contour_aD else np.zeros((0, self.NZ)),
                "aC": np.stack(self.contour_aC, axis=0) if self.contour_aC else np.zeros((0, self.NZ)),
            },
        }

