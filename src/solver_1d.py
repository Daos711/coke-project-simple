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

def _make_tau_profile(geom: Geometry, walls: Walls) -> np.ndarray:
    tau_bottom = getattr(walls, "tau_heat_bottom_s", getattr(walls, "tau_heat_s", 10.0*60.0))
    tau_top    = getattr(walls, "tau_heat_top_s",    max(tau_bottom, 12.0*tau_bottom))
    beta       = float(getattr(walls, "tau_profile_beta", 2.0))
    z = geom.z
    s = (z/geom.H)**beta if geom.H > 0 else np.zeros_like(z)
    return (tau_bottom + (tau_top - tau_bottom)*s).astype(np.float64)

if NUMBA:
    @njit(cache=True, fastmath=True)
    def _nb_step_advect_react(
        T, aR, aC, aD, gamma,
        dt, rho_vr, rho_coke, rho_dist,
        T_wall, tau_z, vR, vD, dz,
        # --- кинетика + МАСШТАБЫ ---
        T1_C, T2_C,
        A1d, E1d, A1c, E1c, o1,
        A15d, E15d, A15c, E15c, o15,
        A2d, E2d, A2c, E2c, o2,
        scale_dist, scale_coke
    ):
        R = 8.314462618
        NZ = T.shape[0]

        # (1) прогрев
        for k in range(NZ):
            T[k] += (T_wall - T[k]) * (dt / tau_z[k])

        # (2) адвекция aR (vR) и aD (vD), upwind
        sigR = max(0.0, vR) * dt / dz
        sigD = max(0.0, vD) * dt / dz
        nsub = 1
        smax = sigR if sigR > sigD else sigD
        if smax > 1.0:
            nsub = int(math.ceil(smax))
        for _ in range(nsub):
            dt_sub = dt / nsub
            sR = max(0.0, vR) * dt_sub / dz
            sD = max(0.0, vD) * dt_sub / dz
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

        # (3) реакции + пересчёт фаз
        for k in range(NZ):
            Tc = T[k]
            if Tc < T1_C:
                order = o1; Ad=A1d; Ed=E1d; Ac=A1c; Ec=E1c
            elif Tc < T2_C:
                order = o15; Ad=A15d; Ed=E15d; Ac=A15c; Ec=E15c
            else:
                order = o2; Ad=A2d; Ed=E2d; Ac=A2c; Ec=E2c

            Tk = Tc + 273.15
            # --- ВАЖНО: применяем масштабы к константам ---
            k_dist = scale_dist * Ad * math.exp(-Ed/(R*Tk))
            k_coke = scale_coke * Ac * math.exp(-Ec/(R*Tk))

            g  = gamma[k]
            r0 = aR[k]

            k_tot = k_dist + k_coke
            r_total = k_tot * (r0 ** (order - 1.0))
            dR = g * r0 * r_total * dt
            if dR > r0: dR = r0
            aR[k] = r0 - dR

            aC[k] += (rho_vr * g * (r0 ** order) * k_coke * dt) / rho_coke
            if aC[k] > 1.0: aC[k] = 1.0
            gamma[k] = 1.0 - aC[k]

            aD[k] += (rho_vr * g * (r0 ** order) * k_dist * dt) / rho_dist
            maxD = gamma[k] - aR[k]
            if maxD < 0.0: maxD = 0.0
            if aD[k] > maxD: aD[k] = maxD

        return T, aR, aC, aD, gamma
else:
    # Python-ядро (то же, что выше)
    def _nb_step_advect_react(T,aR,aC,aD,gamma,dt,rho_vr,rho_coke,rho_dist,T_wall,tau_z,vR,vD,dz,
                              T1_C,T2_C,A1d,E1d,A1c,E1c,o1,A15d,E15d,A15c,E15c,o15,A2d,E2d,A2c,E2c,o2,
                              scale_dist, scale_coke):
        R = 8.314462618; NZ = T.shape[0]
        for k in range(NZ):
            T[k] += (T_wall - T[k]) * (dt / tau_z[k])
        sigR = max(0.0, vR) * dt / dz
        sigD = max(0.0, vD) * dt / dz
        nsub = int(max(1.0, math.ceil(max(sigR, sigD))))
        for _ in range(nsub):
            dt_sub = dt / nsub
            sR = max(0.0, vR) * dt_sub / dz
            sD = max(0.0, vD) * dt_sub / dz
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
            if Tc < T1_C:   order,Ad,Ed,Ac,Ec = o1,A1d,E1d,A1c,E1c
            elif Tc < T2_C: order,Ad,Ed,Ac,Ec = o15,A15d,E15d,A15c,E15c
            else:           order,Ad,Ed,Ac,Ec = o2,A2d,E2d,A2c,E2c
            Tk = Tc + 273.15
            k_dist = scale_dist * Ad * math.exp(-Ed/(R*Tk))
            k_coke = scale_coke * Ac * math.exp(-Ec/(R*Tk))
            g, r0 = gamma[k], aR[k]
            r_total = (k_dist + k_coke) * (r0 ** (order - 1.0))
            dR = g * r0 * r_total * dt
            if dR > r0: dR = r0
            aR[k] = r0 - dR
            aC[k] = min(1.0, aC[k] + (rho_vr * g * (r0 ** order) * k_coke * dt) / rho_coke)
            gamma[k] = 1.0 - aC[k]
            aD[k] += (rho_vr * g * (r0 ** order) * k_dist * dt) / rho_dist
            maxD = gamma[k] - aR[k];  maxD = 0.0 if maxD < 0 else maxD
            if aD[k] > maxD: aD[k] = maxD
        return T, aR, aC, aD, gamma

class Coking1DSolver:
    def __init__(self, geom: Geometry, inlet: Inlet, walls: Walls, mats: Materials, tcfg: TimeSetup, kin: VR3Kinetics):
        self.g, self.inlet, self.walls, self.mats, self.tcfg, self.kin = geom, inlet, walls, mats, tcfg, kin
        NZ = geom.NZ
        self.T  = np.full(NZ, inlet.T_in_C, dtype=np.float64)
        self.aR = np.ones(NZ,  dtype=np.float64)
        self.aC = np.zeros(NZ, dtype=np.float64)
        self.aD = np.zeros(NZ, dtype=np.float64)
        self.gamma = 1.0 - self.aC
        self.time_s = 0.0
        self.vol_cell = self.g.A * self.g.dz
        self.tau_z = _make_tau_profile(self.g, self.walls)
        self.snap_times_h = list(tcfg.snapshots_h)
        self.snapshots = {"t_h": [], "T": [], "aR": [], "aD": [], "aC": []}
        self.bed_h_cm_hist, self.time_h_hist = [], []
        self.contour_every_s = float(tcfg.contour_every_s)
        self.contour_t, self.contour_aR, self.contour_aD, self.contour_aC, self.contour_T = [], [], [], [], []
        self._maybe_take_snapshot(force=True); self._maybe_take_contour(force=True)

    # эквивалентная высота: H_eq = ∫ αC dz (согласована с массой)
    def bed_height(self) -> float:
        return float(np.sum(self.aC) * self.g.dz)

    def coke_mass(self) -> float:
        return float(np.sum(self.mats.rho_coke_bulk * self.aC) * self.vol_cell)

    def inlet_mass_total(self) -> float:
        return self.inlet.m_dot_kg_s * self.time_s

    def coke_yield_pct(self) -> float:
        m_in = self.inlet_mass_total()
        return 0.0 if m_in <= 0.0 else 100.0 * self.coke_mass() / m_in

    def vr_inventory_mass(self) -> float:
        return float(np.sum(self.inlet.rho_vr * self.aR) * self.vol_cell)

    def _maybe_take_snapshot(self, force: bool=False):
        t_h = self.time_s/3600.0
        if force or any(abs(t_h - th) < 1e-9 for th in self.snap_times_h):
            self.snapshots["t_h"].append(t_h)
            self.snapshots["T"].append(self.T.copy())
            self.snapshots["aR"].append(self.aR.copy())
            self.snapshots["aD"].append(self.aD.copy())
            self.snapshots["aC"].append(self.aC.copy())

    def _maybe_take_contour(self, force: bool=False):
        if force or (len(self.contour_t)==0) or (self.time_s - self.contour_t[-1] >= self.contour_every_s - 1e-12):
            self.contour_t.append(self.time_s)
            self.contour_T.append(self.T.copy())
            self.contour_aR.append(self.aR.copy())
            self.contour_aD.append(self.aD.copy())
            self.contour_aC.append(self.aC.copy())

    def step(self):
        r1, r15, r2 = self.kin.reg1, self.kin.reg15, self.kin.reg2
        vR = self.inlet.velocity(self.g)
        vD = self.inlet.velocity_gas(self.g)
        self.T, self.aR, self.aC, self.aD, self.gamma = _nb_step_advect_react(
            self.T, self.aR, self.aC, self.aD, self.gamma,
            self.tcfg.dt, self.inlet.rho_vr, self.mats.rho_coke_bulk, self.mats.rho_dist_vap,
            self.walls.T_wall_C, self.tau_z, vR, vD, self.g.dz,
            self.kin.T1_C, self.kin.T2_C,
            r1.A_dist, r1.Ea_dist, r1.A_coke, r1.Ea_coke, r1.order,
            r15.A_dist, r15.Ea_dist, r15.A_coke, r15.Ea_coke, r15.order,
            r2.A_dist, r2.Ea_dist, r2.A_coke, r2.Ea_coke, r2.order,
            self.kin.scale_dist, self.kin.scale_coke   # ← масштабы КИНЕТИКИ в ядро
        )
        self.time_s += self.tcfg.dt

    def run(self, verbose_hourly=True):
        steps = int(self.tcfg.total_hours * 3600.0 / self.tcfg.dt)
        next_hour_s = 3600.0
        for _ in range(steps):
            self.step()
            if self.time_s + 1e-12 >= next_hour_s:
                self.time_h_hist.append(self.time_s/3600.0)
                self.bed_h_cm_hist.append(self.bed_height()*100.0)
                if verbose_hourly:
                    print(f"t = {self.time_s/3600.0:5.1f} ч | Высота: {self.bed_h_cm_hist[-1]:.1f} см | "
                          f"Выход: {self.coke_yield_pct():5.2f}% | T_avg: {np.mean(self.T):.1f}°C | "
                          f"M_VR в колонне: {self.vr_inventory_mass():.3f} кг")
                next_hour_s += 3600.0
            self._maybe_take_snapshot(); self._maybe_take_contour()

        return {
            "H_bed_m": self.bed_height(),
            "yield_pct": self.coke_yield_pct(),
            "T_avg_C": float(np.mean(self.T)),
            "final": {"T": self.T.copy(), "aR": self.aR.copy(), "aD": self.aD.copy(), "aC": self.aC.copy()},
            "z": self.g.z.copy(),
            "snapshots": {
                "t_h": np.array(self.snapshots["t_h"], dtype=float),
                "T":  np.stack(self.snapshots["T"],  axis=0),
                "aR": np.stack(self.snapshots["aR"], axis=0),
                "aD": np.stack(self.snapshots["aD"], axis=0),
                "aC": np.stack(self.snapshots["aC"], axis=0),
            },
            "growth": {"t_h": np.array(self.time_h_hist, dtype=float), "H_cm": np.array(self.bed_h_cm_hist, dtype=float)},
            "contours": {
                "t_s": np.array(self.contour_t, dtype=float),
                "T":   np.stack(self.contour_T,  axis=0),
                "aR":  np.stack(self.contour_aR, axis=0),
                "aD":  np.stack(self.contour_aD, axis=0),
                "aC":  np.stack(self.contour_aC, axis=0),
            }
        }
