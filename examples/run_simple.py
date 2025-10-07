"""
Главный скрипт для запуска симуляции замедленного коксования
Main script for running delayed coking simulation
"""

import sys
import os

# Добавляем путь к модулям src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Прямой импорт из модулей (без циклических зависимостей)
import src.params as params
from src.geometry import Geometry1D
from src.kinetics import KineticsModel
from src.solver_1d import Solver1D
from src.visualization import Visualizer


def main():
    """Главная функция"""

    print("\n" + "=" * 70)
    print("    СИМУЛЯЦИЯ РЕАКТОРА ЗАМЕДЛЕННОГО КОКСОВАНИЯ")
    print("    CFD Simulation of Delayed Coking Reactor")
    print("=" * 70 + "\n")

    # ========================================================================
    # 1. ИНИЦИАЛИЗАЦИЯ
    # ========================================================================
    print("Шаг 1: Инициализация...")

    # Вывод параметров
    params.print_parameters()

    # Создание геометрии
    geometry = Geometry1D()
    geometry.print_info()

    # Создание кинетической модели
    kinetics = KineticsModel(vr_params=params.ACTIVE_VR)
    kinetics.print_info()

    # ========================================================================
    # 2. СОЗДАНИЕ РЕШАТЕЛЯ
    # ========================================================================
    print("Шаг 2: Создание решателя...")
    solver = Solver1D(geometry=geometry, kinetics=kinetics)

    # ========================================================================
    # 3. ЗАПУСК СИМУЛЯЦИИ
    # ========================================================================
    print("Шаг 3: Запуск симуляции...")
    solver.run(verbose=True)

    # ========================================================================
    # 4. ПОЛУЧЕНИЕ РЕЗУЛЬТАТОВ
    # ========================================================================
    print("Шаг 4: Обработка результатов...")
    results = solver.get_results()

    print(f"\nФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
    print(f"  Высота коксового слоя: {results['final_coke_height'] * 100:.2f} см")
    print(f"  Выход кокса:           {results['final_coke_yield'] * 100:.2f} %\n")

    # ========================================================================
    # 5. ВИЗУАЛИЗАЦИЯ
    # ========================================================================
    print("Шаг 5: Генерация графиков...")
    visualizer = Visualizer(results)
    visualizer.generate_all_plots()
    visualizer.save_data_to_file()

    # ========================================================================
    # ЗАВЕРШЕНИЕ
    # ========================================================================
    print("\n" + "=" * 70)
    print("    СИМУЛЯЦИЯ УСПЕШНО ЗАВЕРШЕНА!")
    print("=" * 70)
    print(f"\nРезультаты сохранены в папке: {params.OUTPUT_DIR}/")
    print("Графики:")
    print("  - volume_fractions_evolution.png")
    print("  - temperature_evolution.png")
    print("  - coke_bed_growth.png")
    print("  - porosity_profile.png")
    print("  - contour_maps.png")
    print("  - comparison_with_experiment.png")
    print("\nДанные:")
    print("  - simulation_results.txt")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nСимуляция прервана пользователем.")
    except Exception as e:
        print(f"\n\nОШИБКА: {e}")
        import traceback

        traceback.print_exc()