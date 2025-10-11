# examples/run_simple.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path

from src.params import defaults
from src.kinetics import VR3Kinetics
from src.solver_1d import Coking1DSolver
from src.visualization import render_all_ru


# Эталон из статьи (VR3): итоговая высота и выход кокса
EXP_H_CM  = 48.34
EXP_Y_PCT = 36.57


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("\n======================================================================")
    print("    СИМУЛЯЦИЯ РЕАКТОРА ЗАМЕДЛЕННОГО КОКСОВАНИЯ")
    print("======================================================================\n")

    # --- 1) Параметры/объекты ------------------------------------------------
    geom, inlet, walls, mats, tcfg = defaults()
    kin = VR3Kinetics()

    outdir = Path("results")
    _ensure_dir(outdir)

    print("Шаг 1: Инициализация...")
    print("======================================================================")
    print("Геометрия:")
    print(f"  Высота реактора: {geom.H:.3f} м | Диаметр: {geom.D:.4f} м | Площадь: {geom.A:.6f} м²")
    print("Сетка:")
    print(f"  NZ={geom.NZ} | dz={geom.dz*1000:.2f} мм")
    print("Время:")
    print(f"  T_sim={tcfg.total_hours:.1f} ч | dt={tcfg.dt:.4f} с | шагов={int(tcfg.total_hours*3600/tcfg.dt)}")
    v_in = inlet.velocity(geom) * 1000.0
    print("Условия:")
    print(f"  T_in={inlet.T_in_C:.1f} °C | m_dot={inlet.m_dot_kg_s*1000*60:.2f} g/min | v_in={v_in:.5f} мм/с")
    print("======================================================================\n")

    # --- 2) Солвер ------------------------------------------------------------
    print("Шаг 2: Создание решателя...")
    solver = Coking1DSolver(geom, inlet, walls, mats, tcfg, kin)
    print("Решатель инициализирован")

    # --- 3) Запуск -------------------------------------------------------------
    print("Шаг 3: Запуск симуляции...\n")
    print("======================================================================")
    print("ЗАПУСК СИМУЛЯЦИИ (с Numba JIT, если доступна)")
    print("======================================================================")
    results = solver.run(verbose_hourly=True)

    H_cm = float(results["H_bed_m"] * 100.0)
    Y    = float(results["yield_pct"])
    print("\n======================================================================")
    print("СИМУЛЯЦИЯ ЗАВЕРШЕНА")
    print("======================================================================")
    print(f"Финальная высота коксового слоя: {H_cm:.2f} см")
    print(f"Финальный выход кокса:           {Y:.2f} %")
    print("======================================================================\n")

    # --- 3.1) Метаданные для визуализации/отчёта -----------------------------
    # Твоя текущая визуализация использует results['meta'] для расчётов масс.
    results.setdefault("meta", {})
    meta = results["meta"]
    meta["A_m2"]         = float(geom.A)
    meta["m_dot_kg_s"]   = float(inlet.m_dot_kg_s)
    meta["rho_vr"]       = float(getattr(inlet, "rho_vr", 1000.0))
    meta["rho_dist"]     = float(getattr(mats,  "rho_dist_vap", 1.6))
    meta["rho_coke"]     = float(getattr(mats,  "rho_coke_bulk", 700.0))

    # --- 4) Постпроцесс -------------------------------------------------------
    print("Шаг 4: Визуализация и отчёт...")
    render_all_ru(results, outdir=outdir, exp_h_cm=EXP_H_CM, exp_y_pct=EXP_Y_PCT)
    print("Готово. Картинки и отчёт →", outdir)


if __name__ == "__main__":
    main()
