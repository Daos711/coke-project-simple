# examples/run_simple.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from pathlib import Path
import numpy as np

from src.params import defaults
from src.kinetics import VR3Kinetics
from src.solver_1d import Coking1DSolver
from src.visualization import render_all_ru

EXP_H_CM = 48.34
EXP_Y_PCT = 36.57

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main():
    print("\n======================================================================")
    print("    СИМУЛЯЦИЯ РЕАКТОРА ЗАМЕДЛЕННОГО КОКСОВАНИЯ")
    print("======================================================================\n")

    geom, inlet, walls, mats, tcfg = defaults()
    kin = VR3Kinetics()

    outdir = Path("results"); ensure_dir(outdir)

    # ---- Информация
    print("Шаг 1: Инициализация...")
    print("======================================================================")
    print("Геометрия:")
    print(f"  Высота реактора: {geom.H:.3f} м | Диаметр: {geom.D:.4f} м | Площадь: {geom.A:.6f} м²")
    print("Сетка:")
    print(f"  NZ={geom.NZ} | dz={geom.dz*1000:.2f} мм")
    print("Время:")
    print(f"  T_sim={tcfg.total_hours:.1f} ч | dt={tcfg.dt:.4f} с | шагов={int(tcfg.total_hours*3600/tcfg.dt)}")
    print("Условия:")
    v_in = inlet.velocity(geom) * 1000.0
    print(f"  T_in={inlet.T_in_C:.1f} °C | m_dot={inlet.m_dot_kg_s*1000*60:.2f} g/min | v_in={v_in:.5f} мм/с")
    print("======================================================================\n")

    # ---- Решатель
    print("Шаг 2: Создание решателя..."); solver = Coking1DSolver(geom, inlet, walls, mats, tcfg, kin)
    print("Решатель инициализирован")

    print("Шаг 3: Запуск симуляции...\n")
    print("======================================================================")
    print("ЗАПУСК СИМУЛЯЦИИ (с Numba JIT, если доступна)")
    print("======================================================================")
    results = solver.run(verbose_hourly=True)

    H_cm = results["H_bed_m"] * 100.0
    Y    = results["yield_pct"]
    print("\n======================================================================")
    print("СИМУЛЯЦИЯ ЗАВЕРШЕНА")
    print("======================================================================")
    print(f"Финальная высота коксового слоя: {H_cm:.2f} см")
    print(f"Финальный выход кокса:           {Y:.2f} %")
    print("======================================================================\n")

    # ---- Визуализация и отчёт (ОДИН вызов)
    print("Шаг 4: Визуализация и отчёт...")
    paths = render_all_ru(results, geom, inlet, mats, tcfg,
                          exp_H_cm=EXP_H_CM, exp_Y_pct=EXP_Y_PCT, outdir=str(outdir))
    print("Готово. Файлы:")
    for k, v in paths.items():
        print(f"  - {k}: {v}")

if __name__ == "__main__":
    main()
