# src/visualization.py
# -*- coding: utf-8 -*-
"""
Визуализация результатов симуляции (подписи на русском) + расчёт выходов фаз.
Вызывай одной функцией: render_all_ru(results, geom, inlet, mats, tcfg, ...)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Sequence
import os
import numpy as np
import matplotlib.pyplot as plt

# Кириллица и минус
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


# ====================== ВСПОМОГАТЕЛЬНОЕ ======================
def _ensure(outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    return outdir

def _tz(a: np.ndarray, t: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Привести массив либо [Nt,Nz], либо [Nz,Nt] к [Nt,Nz]."""
    Nt, Nz = len(t), len(z)
    a = np.asarray(a)
    if a.shape == (Nt, Nz): return a
    if a.shape == (Nz, Nt): return a.T
    raise ValueError(f"Ожидали (Nt,Nz) или (Nz,Nt), получили {a.shape}")


@dataclass
class PhaseYields:
    time_s: np.ndarray
    time_h: np.ndarray
    y_vr:   np.ndarray  # %
    y_dist: np.ndarray  # %
    y_coke: np.ndarray  # %
    m_feed: np.ndarray  # кг
    m_vr:   np.ndarray  # кг
    m_dist: np.ndarray  # кг (внутри)
    m_coke: np.ndarray  # кг
    m_dist_out: np.ndarray  # кг (ушедшие)


def compute_phase_yields_from_results(
    results: Dict,
    geom,
    inlet,
    mats,
) -> PhaseYields:
    """
    Считает выходы фаз во времени из results решателя.
    Использует карты из results['contours'].
    """
    t_s = np.asarray(results["contours"]["t_s"], dtype=float)
    z_m = np.asarray(results["z"], dtype=float)

    aR = _tz(np.asarray(results["contours"]["aR"], dtype=float), t_s, z_m)
    aD = _tz(np.asarray(results["contours"]["aD"], dtype=float), t_s, z_m)
    aC = _tz(np.asarray(results["contours"]["aC"], dtype=float), t_s, z_m)

    # Массы фаз внутри колонны (кг): M = A ∫ (α * ρ) dz
    # (NumPy 2.0: используем np.trapezoid вместо устаревшего trapz)
    m_vr   = geom.A * np.trapezoid(aR * inlet.rho_vr,         z_m, axis=1)
    m_dist = geom.A * np.trapezoid(aD * mats.rho_dist_vap,    z_m, axis=1)
    m_coke = geom.A * np.trapezoid(aC * mats.rho_coke_bulk,   z_m, axis=1)

    # Подача и ушедшие дистилляты по балансу
    m_feed = inlet.m_dot_kg_s * t_s
    m_dist_out = np.maximum(m_feed - (m_vr + m_dist + m_coke), 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        y_coke = np.where(m_feed > 0.0, 100.0 * m_coke / m_feed, 0.0)
        y_vr   = np.where(m_feed > 0.0, 100.0 * m_vr   / m_feed, 0.0)
        y_dist = np.where(m_feed > 0.0, 100.0 * (m_dist + m_dist_out) / m_feed, 0.0)

    return PhaseYields(
        time_s=t_s,
        time_h=t_s/3600.0,
        y_vr=y_vr, y_dist=y_dist, y_coke=y_coke,
        m_feed=m_feed, m_vr=m_vr, m_dist=m_dist, m_coke=m_coke, m_dist_out=m_dist_out
    )


# ====================== ГРАФИКИ ===============================
def plot_volume_profiles_ru(results: Dict, outdir="results") -> str:
    outdir = _ensure(outdir)
    z_cm = results["z"] * 100.0
    snaps = results["snapshots"]; times_h = snaps["t_h"]

    fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    for i, (name, key) in enumerate([("Вакуумный остаток", "aR"),
                                     ("Дистилляты", "aD"),
                                     ("Кокс", "aC")]):
        for th, prof in zip(times_h, snaps[key]):
            axs[i].plot(prof, z_cm, lw=2, label=f"{th:.1f} ч")
        axs[i].set_title(name); axs[i].set_xlabel("Доля объёма"); axs[i].grid(alpha=0.3)
    axs[0].set_ylabel("Высота реактора (см)")
    axs[0].legend(title="Время")
    fig.suptitle("Эволюция объёмных долей", fontsize=18, y=1.02)
    fig.tight_layout()
    p = os.path.join(outdir, "volume_fractions_evolution_ru.png")
    fig.savefig(p, dpi=150); plt.close(fig); return p


def plot_temperature_profiles_ru(results: Dict, outdir="results") -> str:
    outdir = _ensure(outdir)
    z_cm = results["z"] * 100.0
    snaps = results["snapshots"]; times_h = snaps["t_h"]

    fig, ax = plt.subplots(figsize=(12, 7))
    for th, prof in zip(times_h, snaps["T"]):
        ax.plot(prof, z_cm, lw=2, label=f"{th:.1f} ч")
    ax.set_title("Эволюция температурного профиля")
    ax.set_xlabel("Температура (°C)"); ax.set_ylabel("Высота реактора (см)")
    ax.grid(alpha=0.3); ax.legend(title="Время")
    fig.tight_layout()
    p = os.path.join(outdir, "temperature_evolution_ru.png")
    fig.savefig(p, dpi=150); plt.close(fig); return p


def plot_coke_bed_growth_ru(results: Dict, exp_h_cm: float, outdir="results") -> str:
    outdir = _ensure(outdir)
    growth = results["growth"]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(growth["t_h"], growth["H_cm"], lw=3)
    ax.axhline(exp_h_cm, ls="--")
    ax.set_title("Рост коксового слоя"); ax.set_xlabel("Время (ч)"); ax.set_ylabel("Высота коксового слоя (см)")
    ax.grid(alpha=0.3); fig.tight_layout()
    p = os.path.join(outdir, "coke_bed_growth_ru.png")
    fig.savefig(p, dpi=150); plt.close(fig); return p


def plot_porosity_profiles_ru(results: Dict, times_h: Sequence[float], outdir="results") -> str:
    outdir = _ensure(outdir)
    z_cm = results["z"] * 100.0
    snaps = results["snapshots"]; times = snaps["t_h"]

    idx = [int(np.argmin(np.abs(np.asarray(times) - th))) for th in times_h]

    fig, ax = plt.subplots(figsize=(12, 7))
    for k in idx:
        gamma = 1.0 - np.asarray(snaps["aC"][k])
        ax.plot(gamma, z_cm, lw=2, label=f"{times[k]:.1f} ч")
    ax.set_title("Эволюция пористости коксового пласта")
    ax.set_xlabel("Пористость"); ax.set_ylabel("Высота реактора (см)")
    ax.set_xlim(0, 1); ax.grid(alpha=0.3); ax.legend(title="Время")
    fig.tight_layout()
    p = os.path.join(outdir, "porosity_profile_ru.png")
    fig.savefig(p, dpi=150); plt.close(fig); return p


def plot_contours_ru(results: Dict, outdir="results") -> str:
    outdir = _ensure(outdir)
    z_cm = results["z"] * 100.0
    cont = results["contours"]; t_h = cont["t_s"] / 3600.0

    data = [("Доля VR", cont["aR"]), ("Доля дистиллятов", cont["aD"]),
            ("Доля кокса", cont["aC"]), ("Температура (°C)", cont["T"])]

    fig, axes = plt.subplots(1, 4, figsize=(21, 5), sharey=True)
    for ax, (title, field) in zip(axes, data):
        im = ax.pcolormesh(t_h, z_cm, np.asarray(field).T, shading="auto")
        ax.set_title(title); ax.set_xlabel("Время (ч)"); ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("Высота (см)")
    fig.tight_layout()
    p = os.path.join(outdir, "contour_maps_ru.png")
    fig.savefig(p, dpi=150); plt.close(fig); return p


def plot_phase_yields_ru(ylds: PhaseYields, outdir="results") -> Dict[str, str]:
    outdir = _ensure(outdir)
    # Timeseries
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(ylds.time_h, ylds.y_vr,   lw=2, label="VR (остаток)")
    ax.plot(ylds.time_h, ylds.y_dist, lw=2, label="Дистилляты (всего)")
    ax.plot(ylds.time_h, ylds.y_coke, lw=2, label="Кокс")
    ax.set_title("Выходы фаз во времени"); ax.set_xlabel("Время (ч)"); ax.set_ylabel("Выход, % от подачи")
    ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout()
    p1 = os.path.join(outdir, "phase_yields_timeseries_ru.png")
    fig.savefig(p1, dpi=150); plt.close(fig)

    # Finals
    fig, ax = plt.subplots(figsize=(9, 6))
    vals = [float(ylds.y_vr[-1]), float(ylds.y_dist[-1]), float(ylds.y_coke[-1])]
    labels = ["VR (остаток)", "Дистилляты", "Кокс"]
    bars = ax.bar(labels, vals)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f"{v:.2f} %", ha="center", va="bottom")
    ax.set_title("Финальные выходы фаз"); ax.set_ylabel("Выход, % от подачи")
    ax.grid(axis="y", alpha=0.2); fig.tight_layout()
    p2 = os.path.join(outdir, "phase_yields_final_ru.png")
    fig.savefig(p2, dpi=150); plt.close(fig)

    return {"timeseries": p1, "final": p2}


def plot_comparison_with_experiment_ru(H_cm: float, Y_coke_pct: float,
                                       exp_H_cm: float, exp_Y_pct: float,
                                       outdir="results") -> str:
    outdir = _ensure(outdir)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Высота
    ax = axes[0]
    bars = ax.bar(["Модель", "Эксперимент"], [H_cm, exp_H_cm])
    for b, v in zip(bars, [H_cm, exp_H_cm]):
        ax.text(b.get_x()+b.get_width()/2, v+0.5, f"{v:.2f} см", ha="center")
    dev = 100.0 * abs(H_cm-exp_H_cm)/exp_H_cm
    ax.annotate(f"Отклонение: {dev:.2f}%", xy=(0.5, 0.5), xycoords="axes fraction",
                bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    ax.set_title("Финальная высота коксового слоя"); ax.set_ylabel("Высота, см"); ax.grid(axis="y", alpha=0.2)

    # Выход
    ax = axes[1]
    bars = ax.bar(["Модель", "Эксперимент"], [Y_coke_pct, exp_Y_pct])
    for b, v in zip(bars, [Y_coke_pct, exp_Y_pct]):
        ax.text(b.get_x()+b.get_width()/2, v+0.5, f"{v:.2f} %", ha="center")
    dev = 100.0 * abs(Y_coke_pct-exp_Y_pct)/exp_Y_pct
    ax.annotate(f"Отклонение: {dev:.2f}%", xy=(0.5, 0.5), xycoords="axes fraction",
                bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    ax.set_title("Выход кокса (%)"); ax.set_ylabel("Выход, %"); ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    p = os.path.join(outdir, "comparison_with_experiment_ru.png")
    fig.savefig(p, dpi=150); plt.close(fig); return p


def write_report_ru(results: Dict, ylds: PhaseYields,
                    exp_H_cm: float, exp_Y_pct: float,
                    outdir="results") -> str:
    outdir = _ensure(outdir)
    H_cm = float(results["H_bed_m"]*100.0)
    Y    = float(results["yield_pct"])
    dev_h = 100.0*abs(H_cm-exp_H_cm)/exp_H_cm
    dev_y = 100.0*abs(Y-exp_Y_pct)/exp_Y_pct
    m_feed_total = float(ylds.m_feed[-1])

    path = os.path.join(outdir, "simulation_results.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("======================================================================\n")
        f.write("РЕЗУЛЬТАТЫ СИМУЛЯЦИИ ЗАМЕДЛЕННОГО КОКСОВАНИЯ\n")
        f.write("======================================================================\n\n")
        f.write("Тип сырья: Vacuum Residue 3\n")
        f.write(f"Время симуляции: {ylds.time_h[-1]:.2f} часов\n\n")
        f.write("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ (по коксу):\n")
        f.write(f"  Высота коксового слоя: {H_cm:.2f} см\n")
        f.write(f"  Выход кокса: {Y:.2f} %\n\n")
        f.write("ВЫХОДЫ ФАЗ (по массе подачи):\n")
        f.write(f"  Кокс:       {ylds.y_coke[-1]:.2f} %  (m = {ylds.m_coke[-1]:.3f} кг)\n")
        f.write(f"  Дистилляты: {ylds.y_dist[-1]:.2f} %  (m ≈ {(ylds.m_dist[-1]+ylds.m_dist_out[-1]):.3f} кг)\n")
        f.write(f"  Остаток VR: {ylds.y_vr[-1]:.2f} %  (m = {ylds.m_vr[-1]:.3f} кг)\n")
        f.write(f"  Подано сырья: {m_feed_total:.3f} кг\n\n")
        f.write("СРАВНЕНИЕ С ЭКСПЕРИМЕНТОМ (по коксу):\n")
        f.write(f"  Эксп. высота:  {exp_H_cm:.2f} см\n")
        f.write(f"  Эксп. выход:   {exp_Y_pct:.2f} %\n\n")
        f.write("ОТКЛОНЕНИЯ:\n")
        f.write(f"  По высоте: {dev_h:.2f} %\n")
        f.write(f"  По выходу: {dev_y:.2f} %\n\n")
        f.write("======================================================================\n")
    return path


def render_all_ru(results: Dict, geom, inlet, mats, tcfg,
                  exp_H_cm=48.34, exp_Y_pct=36.57, outdir="results",
                  profile_times_h=(3.0, 6.0, 9.0)) -> Dict[str, str]:
    """
    Высокоуровневый «один вызов»: посчитать выходы фаз, построить ВСЕ графики и записать отчёт.
    Возвращает пути к созданным файлам.
    """
    outdir = _ensure(outdir)
    ylds = compute_phase_yields_from_results(results, geom, inlet, mats)

    paths = {}
    paths["fractions_profiles"] = plot_volume_profiles_ru(results, outdir)
    paths["temperature_profiles"] = plot_temperature_profiles_ru(results, outdir)
    paths["growth"] = plot_coke_bed_growth_ru(results, exp_H_cm, outdir)
    paths["porosity"] = plot_porosity_profiles_ru(results, profile_times_h, outdir)
    paths["contours"] = plot_contours_ru(results, outdir)
    paths.update(plot_phase_yields_ru(ylds, outdir))
    paths["comparison"] = plot_comparison_with_experiment_ru(
        float(results["H_bed_m"]*100.0), float(results["yield_pct"]),
        exp_H_cm, exp_Y_pct, outdir
    )
    paths["report"] = write_report_ru(results, ylds, exp_H_cm, exp_Y_pct, outdir)
    return paths
