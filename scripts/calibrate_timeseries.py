#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Calibration pipeline for delayed coking 1D model against industrial time series."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - explicit message for missing dependency
    raise SystemExit(
        "pandas is required to read Excel files. Install pandas in the current environment."
    ) from exc

from scipy.optimize import differential_evolution, minimize

from src.geometry import Geometry
from src.kinetics import VR3Kinetics
from src.params import (
    Inlet,
    Materials,
    ReactionEnergy,
    TimeSetup,
    WallEnergy,
    WallLayer,
    Walls,
)
from src.solver_1d import Coking1DSolver
from src.visualization import render_all_ru

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (in %)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return 0.0
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    denom = np.maximum(np.abs(y_true), 1e-6)
    return float(np.mean(np.abs((y_pred - y_true) / denom)) * 100.0)


def interpolate_series(t_src: np.ndarray, values: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    mask = np.isfinite(values)
    if np.count_nonzero(mask) < 2:
        return np.full_like(t_new, np.nan, dtype=float)
    return np.interp(t_new, t_src[mask], values[mask])


@dataclass
class MeasuredData:
    t_s: np.ndarray
    flow_m3_h: np.ndarray
    pressure: np.ndarray
    T_in_C: np.ndarray
    T_out_C: np.ndarray
    T_head_upper_C: np.ndarray
    T_head_lower_C: np.ndarray


@dataclass
class SimulationResult:
    loss: float
    metrics: Dict[str, float]
    series: Dict[str, np.ndarray]
    solver: Coking1DSolver


# ---------------------------------------------------------------------------
# Parameter vector utilities
# ---------------------------------------------------------------------------


@dataclass
class ParameterVector:
    names: List[str]
    bounds: List[Tuple[float, float]]

    def clip(self, vec: Sequence[float]) -> np.ndarray:
        arr = np.asarray(vec, dtype=float)
        low = np.array([b[0] for b in self.bounds], dtype=float)
        high = np.array([b[1] for b in self.bounds], dtype=float)
        return np.clip(arr, low, high)


def build_parameter_vector() -> ParameterVector:
    names: List[str] = []
    bounds: List[Tuple[float, float]] = []

    # Wall (outer, inner)
    wall_params = [
        ("wall.outer.k", (1.0, 40.0)),
        ("wall.outer.rho", (3000.0, 8000.0)),
        ("wall.outer.cp", (400.0, 900.0)),
        ("wall.outer.thickness", (0.01, 0.25)),
        ("wall.outer.epsilon", (0.6, 0.95)),
        ("wall.inner.k", (1.0, 40.0)),
        ("wall.inner.rho", (3000.0, 8000.0)),
        ("wall.inner.cp", (400.0, 900.0)),
        ("wall.inner.thickness", (0.01, 0.25)),
        ("wall.inner.epsilon", (0.6, 0.95)),
        ("wall.h_amb", (5.0, 50.0)),
        ("wall.T_amb_C", (0.0, 50.0)),
    ]
    names.extend([n for n, _ in wall_params])
    bounds.extend([b for _, b in wall_params])

    mixture_params = [
        ("mix.lambda_eff", (0.1, 3.0)),
        ("mix.cp_eff", (1500.0, 3500.0)),
        ("mix.h0_mix", (20.0, 800.0)),
        ("mix.alpha_mdot", (0.2, 0.9)),
        ("mix.alpha_p", (0.0, 0.4)),
        ("mix.mdot_ref", (0.1, 150.0)),
        ("mix.p_ref", (0.1, 25.0)),
    ]
    names.extend([n for n, _ in mixture_params])
    bounds.extend([b for _, b in mixture_params])

    reaction_params = [
        ("rxn.dH_dist", (-3e5, 3e5)),
        ("rxn.dH_coke", (-3e5, 3e5)),
    ]
    names.extend([n for n, _ in reaction_params])
    bounds.extend([b for _, b in reaction_params])

    kinetics_params = [
        ("kin.A_dist_scale", (0.3, 3.0)),
        ("kin.A_coke_scale", (0.3, 3.0)),
        ("kin.dT1_K", (-20.0, 20.0)),
        ("kin.dT2_K", (-20.0, 20.0)),
        ("kin.phi_por", (0.3, 1.0)),
    ]
    names.extend([n for n, _ in kinetics_params])
    bounds.extend([b for _, b in kinetics_params])

    return ParameterVector(names=names, bounds=bounds)


def vector_to_configs(vec: Sequence[float]) -> Tuple[WallEnergy, MixtureEnergy, ReactionEnergy, VR3Kinetics]:
    v = np.asarray(vec, dtype=float)
    idx = 0

    def take(n: int) -> np.ndarray:
        nonlocal idx
        arr = v[idx : idx + n]
        idx += n
        return arr

    outer = take(5)
    inner = take(5)
    misc = take(2)
    mix_vals = take(7)
    rxn_vals = take(2)
    kin_vals = take(5)

    wall_energy = WallEnergy(
        outer=WallLayer(k=outer[0], rho=outer[1], cp=outer[2], thickness=outer[3], epsilon=outer[4]),
        inner=WallLayer(k=inner[0], rho=inner[1], cp=inner[2], thickness=inner[3], epsilon=inner[4]),
        h_amb=misc[0],
        T_amb_C=misc[1],
    )

    mixture_energy = MixtureEnergy(
        lambda_eff=mix_vals[0],
        cp_eff=mix_vals[1],
        h0_mix=mix_vals[2],
        alpha_mdot=mix_vals[3],
        alpha_p=mix_vals[4],
        mdot_ref=mix_vals[5],
        p_ref=mix_vals[6],
    )

    reaction_energy = ReactionEnergy(dH_dist=rxn_vals[0], dH_coke=rxn_vals[1])

    kinetics = VR3Kinetics(
        A_dist_scale=kin_vals[0],
        A_coke_scale=kin_vals[1],
        dT1_K=kin_vals[2],
        dT2_K=kin_vals[3],
        phi_por=kin_vals[4],
    )

    return wall_energy, mixture_energy, reaction_energy, kinetics


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulate(
    vec: Sequence[float],
    geom: Geometry,
    inlet: Inlet,
    walls: Walls,
    materials: Materials,
    tcfg: TimeSetup,
    data: MeasuredData,
    measured: MeasuredData,
    dt: float,
) -> SimulationResult:
    wall_energy, mixture_energy, reaction_energy, kinetics = vector_to_configs(vec)

    solver = Coking1DSolver(
        geom,
        inlet,
        walls=walls,
        mats=materials,
        tcfg=tcfg,
        kin=kinetics,
        wall_energy=wall_energy,
        mixture_energy=mixture_energy,
        reaction_energy=reaction_energy,
    )

    rho_vr = max(inlet.rho_vr, 1e-9)

    # Reset histories for custom stepping
    solver.time_h_hist = []
    solver.bed_eq_cm_hist = []
    solver.bed_front_cm_hist = []
    next_hour_s = 3600.0

    times = [0.0]
    T_out_model = [solver.outlet_temperature_C()]
    shells0 = solver.shell_temperatures_C()
    T_lower_model = [shells0[solver.bottom_zone]]
    T_upper_model = [shells0[solver.top_zone]]
    H_model = [solver.bed_height_equiv()]

    n_steps = len(data.t_s)
    for i in range(1, n_steps):
        t_prev = data.t_s[i - 1]
        t_cur = data.t_s[i]
        dt_step = max(dt, t_cur - t_prev)

        flow_m3_h = max(float(data.flow_m3_h[i]), 0.0)
        pressure = float(max(data.pressure[i], 1e-6))
        solver.set_operating_conditions(
            T_in_C=float(data.T_in_C[i]),
            m_dot_kg_s=flow_m3_h * rho_vr / 3600.0,
            pressure=pressure,
        )

        upper = None if not np.isfinite(data.T_head_upper_C[i]) else float(data.T_head_upper_C[i])
        lower = None if not np.isfinite(data.T_head_lower_C[i]) else float(data.T_head_lower_C[i])
        solver.set_shell_observations(t_cur, upper, lower)

        solver.step(dt=dt_step)
        solver._maybe_take_snapshot()
        solver._maybe_take_contour()

        while solver.time_s + 1e-12 >= next_hour_s:
            solver.time_h_hist.append(next_hour_s / 3600.0)
            solver.bed_eq_cm_hist.append(solver.bed_height_equiv() * 100.0)
            solver.bed_front_cm_hist.append(solver.bed_height_front() * 100.0)
            next_hour_s += 3600.0

        times.append(solver.time_s)
        T_out_model.append(solver.outlet_temperature_C())
        shells = solver.shell_temperatures_C()
        T_lower_model.append(shells[solver.bottom_zone])
        T_upper_model.append(shells[solver.top_zone])
        H_model.append(solver.bed_height_equiv())

    times_arr = np.asarray(times, dtype=float)
    series = {
        't_s': times_arr,
        'T_out_model': np.asarray(T_out_model, dtype=float),
        'T_lower_model': np.asarray(T_lower_model, dtype=float),
        'T_upper_model': np.asarray(T_upper_model, dtype=float),
        'T_out_meas': np.asarray(data.T_out_C, dtype=float),
        'T_lower_meas': np.asarray(data.T_head_lower_C, dtype=float),
        'T_upper_meas': np.asarray(data.T_head_upper_C, dtype=float),
        'H_m': np.asarray(H_model, dtype=float),
    }

    metrics = {}
    loss = 0.0

    t_meas = measured.t_s
    interp = lambda arr: np.interp(t_meas, times_arr, arr)
    model_out = interp(series['T_out_model'])
    model_top = interp(series['T_upper_model'])
    model_bot = interp(series['T_lower_model'])

    w_out, w_top, w_bot, w_H = 0.35, 0.30, 0.30, 0.05
    m_out = mape(measured.T_out_C, model_out)
    m_top = mape(measured.T_head_upper_C, model_top)
    m_bot = mape(measured.T_head_lower_C, model_bot)
    H_final = float(series['H_m'][-1])
    h_penalty = abs(H_final - 17.5) / 17.5

    loss = w_out * m_out + w_top * m_top + w_bot * m_bot + w_H * h_penalty * 100.0

    metrics.update(
        {
            'MAPE_out': m_out,
            'MAPE_top': m_top,
            'MAPE_bottom': m_bot,
            'MAPE_avg': (m_out + m_top + m_bot) / 3.0,
            'height_final_m': H_final,
            'height_penalty': h_penalty,
        }
    )

    return SimulationResult(loss=loss, metrics=metrics, series=series, solver=solver)


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Calibrate the delayed coking 1D model on time series data.')
    parser.add_argument('--excel', required=True, type=Path, help='Path to Excel file with measurements.')
    parser.add_argument('--sheet', default=None, help='Optional sheet name to read.')
    parser.add_argument('--t-start', dest='t_start', default=None, help='Start timestamp (inclusive).')
    parser.add_argument('--t-end', dest='t_end', default=None, help='End timestamp (inclusive).')
    parser.add_argument('--dt', type=float, default=30.0, help='Integration time step, seconds (default: 30).')
    parser.add_argument('--out', type=Path, default=Path('reports'), help='Output directory for reports.')
    parser.add_argument('--de-iters', type=int, default=60, help='Differential Evolution max iterations.')
    parser.add_argument('--de-pop', type=int, default=30, help='Population size for Differential Evolution.')
    parser.add_argument('--lbfgsb', action='store_true', help='Apply L-BFGS-B refinement after DE.')
    parser.add_argument('--col-time', default='time', help='Time column name (Excel datetime or seconds).')
    parser.add_argument('--col-flow', default='flow_m3_h', help='Feed flow column name (m^3/h).')
    parser.add_argument('--col-pressure', default='pressure', help='Drum pressure column name.')
    parser.add_argument('--col-T-in', default='T_in', help='Inlet temperature column (°C).')
    parser.add_argument('--col-T-out', default='T_out', help='Outlet temperature column (°C).')
    parser.add_argument('--col-T-upper', default='T_head_upper', help='Upper head shell temperature (°C).')
    parser.add_argument('--col-T-lower', default='T_head_lower', help='Lower head shell temperature (°C).')
    return parser.parse_args()


def load_data(args: argparse.Namespace) -> MeasuredData:
    df = pd.read_excel(args.excel, sheet_name=args.sheet)
    if args.col_time not in df.columns:
        raise SystemExit(f"Time column '{args.col_time}' not found in Excel file.")

    time_series = pd.to_datetime(df[args.col_time], errors='ignore')
    if np.issubdtype(time_series.dtype, np.datetime64):
        t_seconds = (time_series - time_series.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
    else:
        t_seconds = pd.to_numeric(time_series, errors='coerce').to_numpy(dtype=float)

    mask = np.isfinite(t_seconds)
    if not np.any(mask):
        raise SystemExit('Time column does not contain valid numeric values.')

    t_seconds = t_seconds[mask]
    df = df.loc[mask].reset_index(drop=True)
    if t_seconds.size:
        t_seconds = t_seconds - t_seconds[0]

    if args.t_start is not None:
        t0 = pd.to_datetime(args.t_start, errors='ignore')
        if np.issubdtype(time_series.dtype, np.datetime64):
            t_ref = time_series.iloc[0]
            start_seconds = (t0 - t_ref).total_seconds()
        else:
            start_seconds = float(pd.to_numeric(args.t_start, errors='coerce'))
        sel = t_seconds >= start_seconds
        t_seconds = t_seconds[sel]
        df = df.loc[sel].reset_index(drop=True)
    if args.t_end is not None:
        t1 = pd.to_datetime(args.t_end, errors='ignore')
        if np.issubdtype(time_series.dtype, np.datetime64):
            t_ref = time_series.iloc[0]
            end_seconds = (t1 - t_ref).total_seconds()
        else:
            end_seconds = float(pd.to_numeric(args.t_end, errors='coerce'))
        sel = t_seconds <= end_seconds
        t_seconds = t_seconds[sel]
        df = df.loc[sel].reset_index(drop=True)

    def col(name: str) -> np.ndarray:
        if name not in df.columns:
            raise SystemExit(f"Column '{name}' not found in Excel file.")
        return pd.to_numeric(df[name], errors='coerce').to_numpy(dtype=float)

    return MeasuredData(
        t_s=t_seconds,
        flow_m3_h=col(args.col_flow),
        pressure=col(args.col_pressure),
        T_in_C=col(args.col_T_in),
        T_out_C=col(args.col_T_out),
        T_head_upper_C=col(args.col_T_upper),
        T_head_lower_C=col(args.col_T_lower),
    )


def resample_data(raw: MeasuredData, dt: float) -> MeasuredData:
    t_grid = np.arange(raw.t_s[0], raw.t_s[-1] + dt, dt)
    interp = lambda arr: interpolate_series(raw.t_s, arr, t_grid)
    def fill(arr: np.ndarray, default: float) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        if not np.any(np.isfinite(arr)):
            return np.full_like(arr, default)
        mask = np.isfinite(arr)
        if np.all(mask):
            return arr
        idx = np.flatnonzero(mask)
        filled = arr.copy()
        filled[~mask] = np.interp(np.flatnonzero(~mask), idx, arr[mask])
        return filled

    def mean_or_default(arr: np.ndarray, default: float = 0.0) -> float:
        arr = np.asarray(arr, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return default
        return float(np.mean(finite))

    return MeasuredData(
        t_s=t_grid,
        flow_m3_h=fill(interp(raw.flow_m3_h), default=mean_or_default(raw.flow_m3_h)),
        pressure=fill(interp(raw.pressure), default=mean_or_default(raw.pressure)),
        T_in_C=fill(interp(raw.T_in_C), default=mean_or_default(raw.T_in_C)),
        T_out_C=fill(interp(raw.T_out_C), default=mean_or_default(raw.T_out_C)),
        T_head_upper_C=fill(interp(raw.T_head_upper_C), default=mean_or_default(raw.T_head_upper_C)),
        T_head_lower_C=fill(interp(raw.T_head_lower_C), default=mean_or_default(raw.T_head_lower_C)),
    )


def main() -> None:
    args = parse_args()
    outdir = args.out
    outdir.mkdir(parents=True, exist_ok=True)

    raw_data = load_data(args)
    data = resample_data(raw_data, args.dt)

    geom = Geometry(H=21.25, D=5.5, NZ=200)
    inlet = Inlet(T_in_C=float(np.nanmean(data.T_in_C)), m_dot_kg_s=5.0, rho_vr=1050.0)
    walls = Walls()
    materials = Materials()
    total_hours = max((data.t_s[-1] - data.t_s[0]) / 3600.0, args.dt / 3600.0)
    snapshot_grid = np.linspace(0.0, total_hours, num=6)
    snapshots_h = tuple(float(x) for x in np.unique(np.round(snapshot_grid, decimals=6)))
    tcfg = TimeSetup(total_hours=total_hours, dt=args.dt, snapshots_h=snapshots_h)

    param_vector = build_parameter_vector()

    def objective(vec: Sequence[float]) -> float:
        try:
            sim = simulate(vec, geom, inlet, walls, materials, tcfg, data, raw_data, args.dt)
        except Exception:
            return 1e9
        if not np.isfinite(sim.loss):
            return 1e9
        if not all(np.all(np.isfinite(v)) for v in sim.series.values()):
            return 1e9
        return sim.loss

    bounds = param_vector.bounds

    de_result = differential_evolution(
        objective,
        bounds=bounds,
        maxiter=args.de_iters,
        popsize=args.de_pop,
        polish=False,
        disp=True,
        updating='deferred',
        workers=1,
    )

    best_vec = param_vector.clip(de_result.x)
    best_loss = float(de_result.fun)

    if args.lbfgsb:
        res = minimize(
            objective,
            best_vec,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100},
        )
        if res.success and res.fun < best_loss:
            best_vec = param_vector.clip(res.x)
            best_loss = float(res.fun)

    best_sim = simulate(best_vec, geom, inlet, walls, materials, tcfg, data, raw_data, args.dt)
    print('Best loss:', best_sim.loss)
    for k, v in best_sim.metrics.items():
        print(f"  {k}: {v:.4f}")

    best_wall, best_mix, best_rxn, best_kin = vector_to_configs(best_vec)

    best_params = {
        'wall_energy': {
            'outer': asdict(best_wall.outer),
            'inner': asdict(best_wall.inner),
            'h_amb': best_wall.h_amb,
            'T_amb_C': best_wall.T_amb_C,
        },
        'mixture_energy': asdict(best_mix),
        'reaction_energy': asdict(best_rxn),
        'kinetics': {
            'A_dist_scale': best_kin.A_dist_scale,
            'A_coke_scale': best_kin.A_coke_scale,
            'dT1_K': best_kin.dT1_K,
            'dT2_K': best_kin.dT2_K,
            'phi_por': best_kin.phi_por,
        },
        'loss': best_sim.loss,
    }

    (outdir / 'best_params.json').write_text(json.dumps(best_params, indent=2, ensure_ascii=False), encoding='utf-8')
    (outdir / 'metrics.json').write_text(json.dumps(best_sim.metrics, indent=2, ensure_ascii=False), encoding='utf-8')

    ts = best_sim.series
    ts_data = np.column_stack(
        [
            ts['t_s'],
            ts['T_out_model'],
            ts['T_out_meas'],
            ts['T_upper_model'],
            ts['T_upper_meas'],
            ts['T_lower_model'],
            ts['T_lower_meas'],
            ts['H_m'],
        ]
    )
    header = 't_s,T_out_model,T_out_meas,T_upper_model,T_upper_meas,T_lower_model,T_lower_meas,H_m'
    np.savetxt(outdir / 'timeseries.csv', ts_data, delimiter=',', header=header, comments='')

    results = {
        'z': best_sim.solver.g.z.copy(),
        'snapshots': {
            't_h': np.array(best_sim.solver.snapshots['t_h'], dtype=float),
            'T': np.stack(best_sim.solver.snapshots['T'], axis=0) if best_sim.solver.snapshots['T'] else np.zeros((0, best_sim.solver.NZ)),
            'aR': np.stack(best_sim.solver.snapshots['aR'], axis=0) if best_sim.solver.snapshots['aR'] else np.zeros((0, best_sim.solver.NZ)),
            'aD': np.stack(best_sim.solver.snapshots['aD'], axis=0) if best_sim.solver.snapshots['aD'] else np.zeros((0, best_sim.solver.NZ)),
            'aC': np.stack(best_sim.solver.snapshots['aC'], axis=0) if best_sim.solver.snapshots['aC'] else np.zeros((0, best_sim.solver.NZ)),
        },
        'contours': {
            't_s': np.array(best_sim.solver.contour_t, dtype=float),
            'T': np.stack(best_sim.solver.contour_T, axis=0) if best_sim.solver.contour_T else np.zeros((0, best_sim.solver.NZ)),
            'aR': np.stack(best_sim.solver.contour_aR, axis=0) if best_sim.solver.contour_aR else np.zeros((0, best_sim.solver.NZ)),
            'aD': np.stack(best_sim.solver.contour_aD, axis=0) if best_sim.solver.contour_aD else np.zeros((0, best_sim.solver.NZ)),
            'aC': np.stack(best_sim.solver.contour_aC, axis=0) if best_sim.solver.contour_aC else np.zeros((0, best_sim.solver.NZ)),
        },
        'growth': {
            't_h': np.array(best_sim.solver.time_h_hist, dtype=float),
            'H_cm': np.array(best_sim.solver.bed_eq_cm_hist, dtype=float),
            'H_front_cm': np.array(best_sim.solver.bed_front_cm_hist, dtype=float),
        },
        'H_bed_m': best_sim.solver.bed_height_equiv(),
        'H_front_m': best_sim.solver.bed_height_front(),
        'yield_pct': best_sim.solver.coke_yield_pct_feed(),
        'T_avg_C': float(np.mean(best_sim.solver.T)),
        'porosity_avg': float(best_sim.solver.porosity_avg),
        'timeseries': ts,
        'meta': {
            'A_m2': float(best_sim.solver.g.A),
            'm_dot_kg_s': float(inlet.m_dot_kg_s),
            'rho_vr': float(inlet.rho_vr),
            'rho_coke': float(materials.rho_coke_bulk),
            'm_vr0': float(best_sim.solver.m_vr0),
        },
    }

    render_all_ru(results, outdir)


if __name__ == '__main__':
    main()
