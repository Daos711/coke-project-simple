# src/solver_1d.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import math, numpy as np

try:
    from numba import njit
    NUMBA = True
except Exception:
    NUMBA = False

from .geometry import Geometry
from .params import Inlet, Walls, Materials, TimeSetup
from .kinetics import VR3Kinetics

# Порог начала коксования (ниже реакции выключены)
T_COKE_ONSET_C = 415.0

def _make_tau_profile(geom: Geometry, walls: Walls) -> np.ndarray:
    tau_bottom = getattr(walls, "tau_heat_bottom_s", 2.0*3600.0)
    tau_top    = getattr(walls, "tau_heat_top_s",    6.0*3600.0)
    beta = float(getattr(walls, "tau_profile_beta", 2.0))
    z = geom.z
    s = (z/geom.H)**beta if geom.H > 0 else np.zeros_like(z)
    return (tau_bottom + (tau_top - tau_bottom) * s).astype(np.float64)

if NUMBA:
    @njit(cache=True, fastmath=True)
    def _nb_step_advect_react(T, aR, aC, aD, gamma,
                              dt, rho_vr, rho_coke, rho_dist,
                              T_wall, tau_z, vR, vD, dz,
                              T1_C, T2_C, T_onset_C,
                              A1d, E1d, A1c, E1c, o1,
                              A15d, E15d, A15c, E15c, o15,
                              A2d, E2d, A2c, E2c, o2,
                              scale_dist, scale_coke,
                              porosity_min):
        R = 8.314462618
        NZ = T.shape[0]

        # 1) прогрев
        for k in range(NZ):
            T[k] += (T_wall - T[k]) * (dt / tau_z[k])

        # 2) адвекция (upwind, с подсубшагами)
        sigR = max(0.0, vR) * dt / dz
        sigD = max(0.0, vD) * dt / dz
        nsub = 1
        if sigR > 1.0 or sigD > 1.0:
            nsub = int(math.ceil(max(sigR, sigD)))

        for _ in range(nsub):
            dt_sub = dt / nsub
            sR = max(0.0, vR) * dt_sub / dz

            # усиление газовой скорости при низкой пористости
            sum_por = 0.0;
            n = 0
            for k in range(NZ):
                if gamma[k] < 0.95:
                    sum_por += gamma[k];
                    n += 1
            avg_por = max((sum_por / n) if n > 0 else 1.0, porosity_min)
            sD = max(0.0, vD / avg_por) * dt_sub / dz

            aR_in, aD_in = gamma[0], 0.0
            aR0 = aR.copy(); aD0 = aD.copy()
            prevR, prevD = aR_in, aD_in
            for k in range(NZ):
                curR, curD = aR0[k], aD0[k]
                aR[k] = curR - sR * (curR - prevR)
                aD[k] = curD - sD * (curD - prevD)
                if aR[k] < 0.0: aR[k] = 0.0
                if aR[k] > gamma[k]: aR[k] = gamma[k]
                if aD[k] < 0.0: aD[k] = 0.0
                prevR, prevD = curR, curD

        # 3) реакции (с замедлением в плотном коксе)
        for k in range(NZ):
            Tc = T[k]
            if Tc < T_onset_C:
                continue

            if Tc < T1_C:
                order, Ad, Ed, Ac, Ec = o1, A1d, E1d, A1c, E1c
            elif Tc < T2_C:
                order, Ad, Ed, Ac, Ec = o15, A15d, E15d, A15c, E15c
            else:
                order, Ad, Ed, Ac, Ec = o2, A2d, E2d, A2c, E2c

            Tk = Tc + 273.15
            k_dist = scale_dist * Ad * math.exp(-Ed / (R * Tk))
            k_coke = scale_coke * Ac * math.exp(-Ec / (R * Tk))

            g = gamma[k]; r0 = aR[k]

            # одинаковая формула для numba/python: factor ∈ [0.5, 1.0]
            if g < 0.4:
                factor = 0.5 + 0.5*(g/0.4)
                k_dist *= factor; k_coke *= factor

            k_tot = k_dist + k_coke
            if k_tot * dt > 0.1:
                ratio = 0.1/(k_tot*dt)
                k_dist *= ratio; k_coke *= ratio

            if r0 > 1e-6 and g > 0.01:
                r_total = k_tot * (r0 ** (order - 1.0))
                dR = g * r0 * r_total * dt
                if dR > r0: dR = r0
                aR[k] = r0 - dR

                dC = (rho_vr * g * (r0 ** order) * k_coke * dt) / rho_coke
                aC[k] += dC
                max_coke = 1.0 - porosity_min
                if aC[k] > max_coke: aC[k] = max_coke
                gamma[k] = 1.0 - aC[k]

                aD[k] += (rho_vr * g * (r0 ** order) * k_dist * dt) / rho_dist
                maxD = gamma[k] - aR[k]
                if maxD < 0.0: maxD = 0.0
                if aD[k] > maxD: aD[k] = maxD

        return T, aR, aC, aD, gamma
else:
    # python-only версия с той же логикой
    def _nb_step_advect_react(*args, **kwargs):
        # просто переиспользуем реализацию из блока выше через копию кода без @njit
        T, aR, aC, aD, gamma, \
        dt, rho_vr, rho_coke, rho_dist, \
        T_wall, tau_z, vR, vD, dz, \
        T1_C, T2_C, T_onset_C, \
        A1d, E1d, A1c, E1c, o1, \
        A15d, E15d, A15c, E15c, o15, \
        A2d, E2d, A2c, E2c, o2, \
        scale_dist, scale_coke, porosity_min = args

        R = 8.314462618
        NZ = T.shape[0]
        for k in range(NZ):
            T[k] += (T_wall - T[k]) * (dt / tau_z[k])

        sigR = max(0.0, vR) * dt / dz
        sigD = max(0.0, vD) * dt / dz
        nsub = int(max(1.0, math.ceil(max(sigR, sigD))))
        for _ in range(nsub):
            dt_sub = dt / nsub
            sR = max(0.0, vR) * dt_sub / dz
            porous = gamma < 0.95
            avg_por = max(float(np.mean(gamma[porous])) if np.any(porous) else 1.0, porosity_min)
            sD = max(0.0, vD / avg_por) * dt_sub / dz

            aR_in, aD_in = gamma[0], 0.0
            aR0 = aR.copy(); aD0 = aD.copy()
            prevR, prevD = aR_in, aD_in
            for k in range(NZ):
                curR, curD = aR0[k], aD0[k]
                aR[k] = curR - sR * (curR - prevR)
                aD[k] = curD - sD * (curD - prevD)
                if aR[k] < 0.0: aR[k] = 0.0
                if aR[k] > gamma[k]: aR[k] = gamma[k]
                if aD[k] < 0.0: aD[k] = 0.0
                prevR, prevD = curR, curD

        for k in range(NZ):
            Tc = T[k]
            if Tc < T_onset_C: continue
            if Tc < T1_C:   order, Ad, Ed, Ac, Ec = o1,  A1d,  E1d,  A1c,  E1c
            elif Tc < T2_C: order, Ad, Ed, Ac, Ec = o15, A15d, E15d, A15c, E15c
            else:           order, Ad, Ed, Ac, Ec = o2,  A2d,  E2d,  A2c,  E2c

            Tk = Tc + 273.15
            k_dist = scale_dist * Ad * math.exp(-Ed / (R * Tk))
            k_coke = scale_coke * Ac * math.exp(-Ec / (R * Tk))
            g = gamma[k]; r0 = aR[k]

            if g < 0.4:
                factor = 0.5 + 0.5*(g/0.4)
                k_dist *= factor; k_coke *= factor

            k_tot = k_dist + k_coke
            if k_tot*dt > 0.1:
                ratio = 0.1/(k_tot*dt)
                k_dist *= ratio; k_coke *= ratio

            if r0 > 1e-6 and g > 0.01:
                r_total = (k_dist + k_coke) * (r0 ** (order - 1.0))
                dR = g * r0 * r_total * dt
                if dR > r0: dR = r0
                aR[k] = r0 - dR
                dC = (rho_vr * g * (r0 ** order) * k_coke * dt) / rho_coke
                aC[k] += dC
                max_coke = 1.0 - porosity_min
                if aC[k] > max_coke: aC[k] = max_coke
                gamma[k] = 1.0 - aC[k]
                aD[k] += (rho_vr * g * (r0 ** order) * k_dist * dt) / rho_dist
                maxD = gamma[k] - aR[k];  maxD = 0.0 if maxD < 0 else maxD
                if aD[k] > maxD: aD[k] = maxD

        return T, aR, aC, aD, gamma

class Coking1DSolver:
    def __init__(self, geom: Geometry, inlet: Inlet, walls: Walls,
                 mats: Materials, tcfg: TimeSetup, kin: VR3Kinetics):
        self.g, self.inlet, self.walls, self.mats, self.tcfg, self.kin = geom, inlet, walls, mats, tcfg, kin
        NZ = geom.NZ

        # поля
        self.T  = np.full(NZ, inlet.T_in_C, dtype=np.float64)
        self.aR = np.ones(NZ, dtype=np.float64)
        self.aC = np.zeros(NZ, dtype=np.float64)
        self.aD = np.zeros(NZ, dtype=np.float64)
        self.gamma = 1.0 - self.aC

        self.porosity_min = float(getattr(mats, 'porosity_min', 0.3))
        self.time_s = 0.0

        self.vol_cell = self.g.A * self.g.dz

        # базовый профиль прогрева от стенки
        self.tau_z = _make_tau_profile(self.g, self.walls)

        # === ВЛИЯНИЕ ГАЗА НА ТЕПЛОПЕРЕДАЧУ (ускоряем прогрев) ===
        # эмпирический множитель; 0.06 даёт умеренный эффект (подбирается по установке)
        k_ht = 0.02
        ht_mult = 1.0 + k_ht * float(self.inlet.v_gas_base_factor)
        # уменьшаем характерное время прогрева
        self.tau_z = self.tau_z / ht_mult

        # «жёсткий» инвентарь VR в стартовый момент (для балансной нормировки)
        self.m_vr0 = float(np.sum(self.inlet.rho_vr * self.aR) * self.vol_cell)

        # снимки/истории
        self.snap_times_h = list(tcfg.snapshots_h); self._snap_idx = 0
        self.snapshots = {"t_h": [], "T": [], "aR": [], "aD": [], "aC": []}

        self.time_h_hist = []; self.bed_eq_cm_hist = []; self.bed_front_cm_hist = []
        self.contour_every_s = float(tcfg.contour_every_s)
        self.contour_t, self.contour_aR, self.contour_aD, self.contour_aC, self.contour_T = [], [], [], [], []
        self.porosity_avg = 1.0

        # коэффициент «подхвата» жидкой фазы газом в адвекции VR (0.1–0.2 даёт ощутимый, но адекватный эффект)
        self.k_mix_gas = 0.05

        # первый снимок
        self._take_snapshot(); self._maybe_take_contour(force=True)
        if self.snap_times_h and abs(self.snapshots["t_h"][-1]-self.snap_times_h[0]) < 1e-6:
            self._snap_idx = 1

    # сервис
    def bed_height_front(self, thr: float = 0.05) -> float:
        idx = np.where(self.aC > thr)[0]
        if idx.size == 0: return 0.0
        top_idx = idx[-1] + 1
        return float(min(top_idx * self.g.dz, self.g.H))

    def bed_height_equiv(self) -> float:
        eps_max = 1.0 - self.porosity_min
        if eps_max <= 1e-12: return 0.0
        aC_clip = np.clip(self.aC, 0.0, eps_max)
        return float(min(np.sum(aC_clip/eps_max) * self.g.dz, self.g.H))

    def coke_mass(self) -> float:
        return float(np.sum(self.mats.rho_coke_bulk * self.aC) * self.vol_cell)

    def inlet_mass_total(self) -> float:
        return float(self.inlet.m_dot_kg_s * self.time_s)

    def vr_inventory_mass(self) -> float:
        return float(np.sum(self.inlet.rho_vr * self.aR) * self.vol_cell)

    def get_porosity_profile(self) -> np.ndarray:
        return np.maximum(self.gamma, self.porosity_min)

    def update_porosity_effects(self):
        zone = self.aC > 0.01
        self.porosity_avg = max(float(np.mean(self.gamma[zone])) if np.any(zone) else 1.0,
                                self.porosity_min)

    def coke_yield_pct_feed(self) -> float:
        return 100.0 * self.coke_mass() / max(self.inlet_mass_total(), 1e-9)

    def coke_yield_pct_balance(self) -> float:
        return 100.0 * self.coke_mass() / max(self.m_vr0 + self.inlet_mass_total(), 1e-9)

    def _take_snapshot(self):
        t_h = self.time_s/3600.0
        self.snapshots["t_h"].append(t_h)
        self.snapshots["T"].append(self.T.copy())
        self.snapshots["aR"].append(self.aR.copy())
        self.snapshots["aD"].append(self.aD.copy())
        self.snapshots["aC"].append(self.aC.copy())

    def _maybe_take_snapshot(self):
        if self._snap_idx >= len(self.snap_times_h): return
        t_h = self.time_s/3600.0
        if t_h + 1e-4 >= self.snap_times_h[self._snap_idx]:
            self._take_snapshot(); self._snap_idx += 1

    def _maybe_take_contour(self, force: bool = False):
        if force or (len(self.contour_t) == 0) or (self.time_s - self.contour_t[-1] >= self.contour_every_s - 1e-12):
            self.contour_t.append(self.time_s)
            self.contour_T.append(self.T.copy())
            self.contour_aR.append(self.aR.copy())
            self.contour_aD.append(self.aD.copy())
            self.contour_aC.append(self.aC.copy())

    def step(self):
        # пересчитали среднюю пористость «рабочей» зоны
        self.update_porosity_effects()

        # доступ к регимам кинетики
        r1, r15, r2 = self.kin.reg1, self.kin.reg15, self.kin.reg2

        # базовая скорость жидкости (VR) и газа (по параметру v_gas_base_factor)
        vR_base = self.inlet.velocity(self.g)  # м/с (жидк. подача)
        vD_gas = self.inlet.velocity_gas(self.g, self.porosity_avg)  # м/с (газовая «псевдоскорость»)

        # === ВЛИЯНИЕ ГАЗА НА АДВЕКЦИЮ VR ===
        # газ «подхватывает» жидкость: эффективная скорость VR растёт с газом
        vR_eff = vR_base + self.k_mix_gas * vD_gas

        # единичный шаг «прогрев + адвекция + реакции»
        self.T, self.aR, self.aC, self.aD, self.gamma = _nb_step_advect_react(
            self.T, self.aR, self.aC, self.aD, self.gamma,
            self.tcfg.dt,
            self.inlet.rho_vr, self.mats.rho_coke_bulk, self.mats.rho_dist_vap,
            self.walls.T_wall_C, self.tau_z,
            vR_eff,  # ← использовали эффективную скорость VR
            vD_gas,  # ← газовая скорость для дистиллятов
            self.g.dz,
            self.kin.T1_C, self.kin.T2_C, T_COKE_ONSET_C,
            r1.A_dist, r1.Ea_dist, r1.A_coke, r1.Ea_coke, r1.order,
            r15.A_dist, r15.Ea_dist, r15.A_coke, r15.Ea_coke, r15.order,
            r2.A_dist, r2.Ea_dist, r2.A_coke, r2.Ea_coke, r2.order,
            self.kin.scale_dist, self.kin.scale_coke,
            self.porosity_min
        )

        self.time_s += self.tcfg.dt

    def run(self, verbose_hourly: bool = True):
        steps = int(self.tcfg.total_hours * 3600.0 / self.tcfg.dt)
        next_hour_s = 3600.0
        for _ in range(steps):
            self.step()
            if self.time_s + 1e-12 >= next_hour_s:
                self.time_h_hist.append(self.time_s/3600.0)
                self.bed_eq_cm_hist.append(self.bed_height_equiv()*100.0)
                self.bed_front_cm_hist.append(self.bed_height_front()*100.0)
                if verbose_hourly:
                    y_feed = self.coke_yield_pct_feed(); y_bal = self.coke_yield_pct_balance()
                    if self.time_s < 2*3600.0:
                        print(f"t = {self.time_s/3600.0:5.1f} ч | H_eq={self.bed_eq_cm_hist[-1]:.1f} см | "
                              f"H_front={self.bed_front_cm_hist[-1]:.1f} см | Yбал={y_bal:5.2f}% | "
                              f"T_avg={np.mean(self.T):.1f}°C | ε_avg={self.porosity_avg:.3f}")
                    else:
                        print(f"t = {self.time_s/3600.0:5.1f} ч | H_eq={self.bed_eq_cm_hist[-1]:.1f} см | "
                              f"H_front={self.bed_front_cm_hist[-1]:.1f} см | Yfeed={y_feed:5.2f}% | "
                              f"Yбал={y_bal:5.2f}% | T_avg={np.mean(self.T):.1f}°C | ε_avg={self.porosity_avg:.3f}")
                next_hour_s += 3600.0
            self._maybe_take_snapshot(); self._maybe_take_contour()

        return {
            "H_bed_m": self.bed_height_equiv(),
            "H_front_m": self.bed_height_front(),
            "yield_pct": self.coke_yield_pct_feed(),
            "T_avg_C": float(np.mean(self.T)),
            "porosity_avg": float(self.porosity_avg),
            "final": {"T": self.T.copy(), "aR": self.aR.copy(), "aD": self.aD.copy(), "aC": self.aC.copy()},
            "z": self.g.z.copy(),
            "snapshots": {
                "t_h": np.array(self.snapshots["t_h"], dtype=float),
                "T":   np.stack(self.snapshots["T"], axis=0),
                "aR":  np.stack(self.snapshots["aR"], axis=0),
                "aD":  np.stack(self.snapshots["aD"], axis=0),
                "aC":  np.stack(self.snapshots["aC"], axis=0),
            },
            "growth": {
                "t_h": np.array(self.time_h_hist, dtype=float),
                "H_cm": np.array(self.bed_eq_cm_hist, dtype=float),
                "H_front_cm": np.array(self.bed_front_cm_hist, dtype=float),
            },
            "contours": {
                "t_s": np.array(self.contour_t, dtype=float),
                "T":   np.stack(self.contour_T, axis=0),
                "aR":  np.stack(self.contour_aR, axis=0),
                "aD":  np.stack(self.contour_aD, axis=0),
                "aC":  np.stack(self.contour_aC, axis=0),
            },
        }
