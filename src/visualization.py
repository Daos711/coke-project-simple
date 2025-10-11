# src/visualization.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt


# ---------- утилиты ----------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _meta(results: Dict) -> Tuple[float, float, float, float, float]:
    """
    Читает из results метаданные, которые кладёт решатель/раннер:
      A_m2, m_dot_kg_s, rho_vr, rho_dist, rho_coke
    """
    meta = results.get("meta", {})
    A = float(meta["A_m2"])
    m_dot = float(meta["m_dot_kg_s"])
    rho_vr = float(meta["rho_vr"])
    rho_dist = float(meta["rho_dist"])
    rho_coke = float(meta["rho_coke"])
    return A, m_dot, rho_vr, rho_dist, rho_coke


def _phase_masses(results: Dict):
    """
    Инвентарь фаз в колонне как функции времени (кг).
    """
    A, m_dot, rho_vr, rho_dist, rho_coke = _meta(results)
    z_m = np.asarray(results["z"])             # (Nz,)
    cont = results["contours"]
    t_s = np.asarray(cont["t_s"])              # (Nt,)
    aR = np.asarray(cont["aR"])                # (Nt, Nz)
    aD = np.asarray(cont["aD"])
    aC = np.asarray(cont["aC"])

    # масса фазы = A * ∫ alpha * rho dz
    m_vr   = A * np.trapezoid(aR * rho_vr,   z_m, axis=1)
    m_dist = A * np.trapezoid(aD * rho_dist, z_m, axis=1)  # (для отчёта; в графиках используем баланс)
    m_coke = A * np.trapezoid(aC * rho_coke, z_m, axis=1)

    m_feed = m_dot * t_s
    return t_s, m_feed, m_vr, m_dist, m_coke


def _phase_yields_massbalance(results: Dict, start_frac: float = 0.02):
    """
    Временные ряды 'корректных' выходов фаз из баланса масс:
      M_total(t) = m_vr0 + M_feed(t)
      M_dist_total(t) = M_total - M_VR - M_coke
      Y_i = 100 * M_i / M_total
    Плюс маска, отсекающая самую раннюю фазу, пока подача малышка.
    """
    t_s, m_feed, m_vr, _m_dist_inv, m_coke = _phase_masses(results)
    m_vr0 = float(m_vr[0])
    M_total = m_feed + m_vr0
    eps = 1e-12

    m_dist_total = np.maximum(M_total - m_vr - m_coke, 0.0)

    Y_vr   = 100.0 * m_vr         / np.maximum(M_total, eps)
    Y_dist = 100.0 * m_dist_total / np.maximum(M_total, eps)
    Y_coke = 100.0 * m_coke       / np.maximum(M_total, eps)

    # маска: начинаем рисовать, когда подача превысила start_frac * m_vr0
    mask = (m_feed >= start_frac * m_vr0)
    return t_s, Y_vr, Y_dist, Y_coke, mask


# ---------- отрисовка ----------
def _plot_fractions_profiles_ru(results: Dict, outdir: Path):
    z = results["z"] * 100.0  # см
    snaps = results["snapshots"]
    times_h = snaps["t_h"]

    fig, axs = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    panels = [("Вакуумный остаток", "aR"),
              ("Дистилляты",       "aD"),
              ("Кокс",             "aC")]
    for i, (title, key) in enumerate(panels):
        for j, t in enumerate(times_h):
            axs[i].plot(snaps[key][j], z, label=f"{t:.1f} ч")
        axs[i].set_title(title, fontsize=16)
        axs[i].set_xlabel("Доля объёма")
        axs[i].grid(True, linewidth=0.3)
    axs[0].set_ylabel("Высота реактора (см)")
    axs[0].legend(title="Время", loc="lower right", fontsize=9)
    fig.suptitle("Эволюция объёмных долей", fontsize=18, y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "volume_fractions_evolution_ru.png", dpi=150)
    plt.close(fig)


def _plot_temperature_profiles_ru(results: Dict, outdir: Path):
    z = results["z"] * 100.0
    snaps = results["snapshots"]
    times_h = snaps["t_h"]

    plt.figure(figsize=(12, 7))
    for j, t in enumerate(times_h):
        plt.plot(snaps["T"][j], z, label=f"{t:.1f} ч")
    plt.title("Эволюция температурного профиля")
    plt.xlabel("Температура (°C)")
    plt.ylabel("Высота реактора (см)")
    plt.grid(True, linewidth=0.3)
    plt.legend(title="Время")
    plt.tight_layout()
    plt.savefig(outdir / "temperature_evolution_ru.png", dpi=150)
    plt.close()


def _plot_growth_ru(results: Dict, outdir: Path, exp_h_cm: float):
    growth = results["growth"]
    plt.figure(figsize=(12, 7))
    plt.plot(growth["t_h"], growth["H_cm"], linewidth=3)
    plt.axhline(exp_h_cm, linestyle="--")
    plt.title("Рост коксового слоя")
    plt.xlabel("Время (ч)")
    plt.ylabel("Высота коксового слоя (см)")
    plt.grid(True, linewidth=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "coke_bed_growth_ru.png", dpi=150)
    plt.close()


def _plot_porosity_ru(results: Dict, outdir: Path):
    z = results["z"] * 100.0
    snaps = results["snapshots"]
    times_h = snaps["t_h"]

    plt.figure(figsize=(12, 7))
    for j, t in enumerate(times_h[1:]):  # без нулевого
        gamma = 1.0 - snaps["aC"][j + 0]
        plt.plot(gamma, z, label=f"{times_h[j]:.1f} ч")
    plt.title("Эволюция пористости коксового пласта")
    plt.xlabel("Пористость")
    plt.ylabel("Высота реактора (см)")
    plt.xlim(0.0, 1.0)
    plt.grid(True, linewidth=0.3)
    plt.legend(title="Время")
    plt.tight_layout()
    plt.savefig(outdir / "porosity_profile_ru.png", dpi=150)
    plt.close()


def _plot_contours_ru(results: Dict, outdir: Path):
    z = results["z"] * 100.0
    cont = results["contours"]
    t_h = cont["t_s"] / 3600.0

    fig, axes = plt.subplots(1, 4, figsize=(23, 6), sharey=True)
    for ax, data, title in zip(
        axes,
        [cont["aR"], cont["aD"], cont["aC"], cont["T"]],
        ["Доля VR", "Доля дистиллятов", "Доля кокса", "Температура (°C)"],
    ):
        im = ax.pcolormesh(t_h, z, data.T, shading="auto")
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Время (ч)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("Высота (см)")
    fig.tight_layout()
    fig.savefig(outdir / "contour_maps_ru.png", dpi=150)
    plt.close(fig)


def _plot_phase_yields_timeseries_ru(results: Dict, outdir: Path):
    t_s, Y_vr, Y_dist, Y_coke, mask = _phase_yields_massbalance(results)
    t_h = (t_s[mask]) / 3600.0

    plt.figure(figsize=(12, 7))
    plt.plot(t_h, Y_vr[mask],   label="VR (остаток)")
    plt.plot(t_h, Y_dist[mask], label="Дистилляты (накоплено)")
    plt.plot(t_h, Y_coke[mask], label="Кокс")
    plt.title("Выходы фаз во времени")
    plt.xlabel("Время (ч)")
    plt.ylabel("Выход, % (баланс масс)")
    plt.grid(True, linewidth=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "phase_yields_timeseries_ru.png", dpi=150)
    plt.close()


def _plot_phase_yields_final_ru(results: Dict, outdir: Path):
    """
    Финальные значения по нормировке 'на подачу' (как в статье/репорте).
    Дистилляты считаем по балансу: M_feed - M_VR - M_coke (аппроксимация).
    """
    t_s, m_feed, m_vr, _m_dist_inv, m_coke = _phase_masses(results)
    M_feed = float(m_feed[-1])
    eps = 1e-12

    Y_vr_f   = 100.0 * float(m_vr[-1])   / max(M_feed, eps)
    Y_coke_f = 100.0 * float(m_coke[-1]) / max(M_feed, eps)
    Y_dist_f = 100.0 * max(M_feed - float(m_vr[-1]) - float(m_coke[-1]), 0.0) / max(M_feed, eps)

    names = ["VR (остаток)", "Дистилляты", "Кокс"]
    vals  = [Y_vr_f, Y_dist_f, Y_coke_f]

    plt.figure(figsize=(12, 7))
    bars = plt.bar(names, vals)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                 f"{v:.2f} %", ha="center", va="bottom")
    plt.title("Финальные выходы фаз")
    plt.ylabel("Выход, % от подачи")
    plt.tight_layout()
    plt.savefig(outdir / "phase_yields_final_ru.png", dpi=150)
    plt.close()


def _plot_comparison_ru(results: Dict, outdir: Path, exp_h_cm: float, exp_y_pct: float):
    H_cm = float(results["H_bed_m"] * 100.0)
    Y = float(results["yield_pct"])

    fig, axs = plt.subplots(1, 2, figsize=(18, 7))
    # Высота
    axs[0].bar([0], [H_cm]); axs[0].bar([1], [exp_h_cm])
    axs[0].set_xticks([0, 1], ["Модель", "Эксперимент"])
    axs[0].set_title("Финальная высота коксового слоя")
    axs[0].text(0, H_cm + 0.5, f"{H_cm:.2f} см", ha="center")
    axs[0].text(1, exp_h_cm + 0.5, f"{exp_h_cm:.2f} см", ha="center")
    dev_h = abs(H_cm - exp_h_cm) / max(exp_h_cm, 1e-6) * 100.0
    axs[0].annotate(f"Отклонение: {dev_h:.2f}%",
                    xy=(0.5, max(H_cm, exp_h_cm) * 0.6), xycoords="data",
                    ha="center", bbox=dict(boxstyle="round", fc="w"))
    # Выход кокса
    axs[1].bar([0], [Y]); axs[1].bar([1], [exp_y_pct])
    axs[1].set_xticks([0, 1], ["Модель", "Эксперимент"])
    axs[1].set_title("Выход кокса (%)")
    axs[1].text(0, Y + 0.5, f"{Y:.2f} %", ha="center")
    axs[1].text(1, exp_y_pct + 0.5, f"{exp_y_pct:.2f} %", ha="center")
    dev_y = abs(Y - exp_y_pct) / max(exp_y_pct, 1e-6) * 100.0
    axs[1].annotate(f"Отклонение: {dev_y:.2f}%",
                    xy=(0.5, max(Y, exp_y_pct) * 0.6), xycoords="data",
                    ha="center", bbox=dict(boxstyle="round", fc="w"))

    fig.tight_layout()
    fig.savefig(outdir / "comparison_with_experiment_ru.png", dpi=150)
    plt.close(fig)


def _write_report_ru(results: Dict, outdir: Path, exp_h_cm: float, exp_y_pct: float):
    A, m_dot, rho_vr, rho_dist, rho_coke = _meta(results)
    t_s, m_feed, m_vr, _m_dist_inv, m_coke = _phase_masses(results)
    m_vr0 = float(m_vr[0])
    M_feed = float(m_feed[-1])
    M_total = M_feed + m_vr0

    # финал (на подачу)
    Y_vr_f_feed   = 100.0 * float(m_vr[-1])   / max(M_feed, 1e-12)
    Y_coke_f_feed = 100.0 * float(m_coke[-1]) / max(M_feed, 1e-12)
    Y_dist_f_feed = 100.0 * max(M_feed - float(m_vr[-1]) - float(m_coke[-1]), 0.0) / max(M_feed, 1e-12)

    # финал (на M_total) — баланс
    Y_vr_f_total   = 100.0 * float(m_vr[-1])   / max(M_total, 1e-12)
    Y_coke_f_total = 100.0 * float(m_coke[-1]) / max(M_total, 1e-12)
    Y_dist_f_total = 100.0 * max(M_total - float(m_vr[-1]) - float(m_coke[-1]), 0.0) / max(M_total, 1e-12)

    H_cm = float(results["H_bed_m"] * 100.0)
    Y_coke_pct = float(results["yield_pct"])

    with open(outdir / "simulation_results.txt", "w", encoding="utf-8") as f:
        f.write("======================================================================\n")
        f.write("РЕЗУЛЬТАТЫ СИМУЛЯЦИИ ЗАМЕДЛЕННОГО КОКСОВАНИЯ\n")
        f.write("======================================================================\n\n")
        f.write(f"Время симуляции: {t_s[-1]/3600.0:.2f} часов\n\n")
        f.write("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ (по коксу):\n")
        f.write(f"  Высота коксового слоя: {H_cm:.2f} см\n")
        f.write(f"  Выход кокса: {Y_coke_pct:.2f} %\n\n")
        f.write("ВЫХОДЫ ФАЗ (нормировка на подачу):\n")
        f.write(f"  Кокс:       {Y_coke_f_feed:.2f} %\n")
        f.write(f"  Дистилляты: {Y_dist_f_feed:.2f} %\n")
        f.write(f"  Остаток VR: {Y_vr_f_feed:.2f} %\n")
        f.write(f"  Подано сырья: {M_feed:.3f} кг\n\n")
        f.write("ВЫХОДЫ ФАЗ (баланс, на M_total = M_feed + M_VR(0)):\n")
        f.write(f"  Кокс:       {Y_coke_f_total:.2f} %\n")
        f.write(f"  Дистилляты: {Y_dist_f_total:.2f} %\n")
        f.write(f"  Остаток VR: {Y_vr_f_total:.2f} %\n\n")
        f.write("СРАВНЕНИЕ С ЭКСПЕРИМЕНТОМ (по коксу):\n")
        f.write(f"  Эксп. высота: {exp_h_cm:.2f} см\n")
        f.write(f"  Эксп. выход:  {exp_y_pct:.2f} %\n")
        f.write("======================================================================\n")


# ---------- публичное API ----------
def render_all_ru(results: Dict, outdir: Path, exp_h_cm: float, exp_y_pct: float) -> None:
    """
    Единая точка входа: сохраняет все картинки в outdir.
    Требования к results['meta']: A_m2, m_dot_kg_s, rho_vr, rho_dist, rho_coke.
    """
    _ensure_dir(outdir)

    _plot_fractions_profiles_ru(results, outdir)
    _plot_temperature_profiles_ru(results, outdir)
    _plot_growth_ru(results, outdir, exp_h_cm)
    _plot_porosity_ru(results, outdir)
    _plot_contours_ru(results, outdir)
    _plot_phase_yields_timeseries_ru(results, outdir)
    _plot_phase_yields_final_ru(results, outdir)
    _plot_comparison_ru(results, outdir, exp_h_cm, exp_y_pct)
    _write_report_ru(results, outdir, exp_h_cm, exp_y_pct)
