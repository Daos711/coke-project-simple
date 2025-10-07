"""
Решатель для 1D модели реактора замедленного коксования
1D Solver for delayed coking reactor
"""

import numpy as np
import time
from . import params
from .geometry import Geometry1D, create_initial_fields
from .kinetics import KineticsModel


class Solver1D:
    """Класс для решения 1D модели реактора"""

    def __init__(self, geometry=None, kinetics=None):
        """
        Инициализация решателя

        Parameters:
        -----------
        geometry : Geometry1D, optional
            Объект геометрии
        kinetics : KineticsModel, optional
            Объект кинетической модели
        """
        # Геометрия
        if geometry is None:
            geometry = Geometry1D()
        self.geometry = geometry

        # Кинетика
        if kinetics is None:
            kinetics = KineticsModel()
        self.kinetics = kinetics

        # Создание полей
        self.fields = create_initial_fields(geometry)

        # Временные параметры
        self.time = 0.0
        self.dt = params.DT
        self.n_timesteps = params.N_TIMESTEPS
        self.current_step = 0

        # История для сохранения результатов
        self.history = {
            'time': [],
            'temperature': [],
            'alpha_vr': [],
            'alpha_dist': [],
            'alpha_coke': [],
            'porosity': [],
            'coke_height': []
        }

        print("Решатель инициализирован")

    def apply_boundary_conditions(self):
        """Применение граничных условий"""
        # Температура на входе
        self.fields['temperature'].values[0] = params.INLET_TEMPERATURE

        # Температура на стенке (упрощённо - постоянная)
        # В полной модели здесь был бы теплообмен со стенкой

        # Объёмные доли на входе (подаётся чистый вакуумный остаток)
        self.fields['alpha_vr'].values[0] = 1.0
        self.fields['alpha_dist'].values[0] = 0.0
        self.fields['alpha_coke'].values[0] = 0.0

    def solve_temperature(self):
        """
        Решение уравнения теплопроводности

        ρCp * dT/dt = k * d²T/dz² + Q_reaction

        Используется явная схема
        """
        T = self.fields['temperature'].values
        alpha_vr = self.fields['alpha_vr'].values
        alpha_dist = self.fields['alpha_dist'].values
        alpha_coke = self.fields['alpha_coke'].values

        # Эффективные теплофизические свойства смеси
        rho_eff = (alpha_vr * params.VR_DENSITY +
                   alpha_dist * params.DIST_DENSITY +
                   alpha_coke * params.COKE_DENSITY)

        cp_eff = (alpha_vr * params.VR_HEAT_CAPACITY +
                  alpha_dist * params.DIST_HEAT_CAPACITY +
                  alpha_coke * params.COKE_HEAT_CAPACITY)

        k_eff = (alpha_vr * params.VR_THERMAL_CONDUCTIVITY +
                 alpha_dist * params.DIST_THERMAL_CONDUCTIVITY +
                 alpha_coke * params.COKE_THERMAL_CONDUCTIVITY)

        # Тепловыделение от реакций
        Q_reaction = self.kinetics.get_heat_generation(T, alpha_vr)

        # Новое поле температуры
        T_new = T.copy()

        # Решение для внутренних точек
        for i in range(1, self.geometry.n_points - 1):
            # Лапласиан (вторая производная)
            dz2 = self.geometry.dz ** 2
            laplacian = (T[i + 1] - 2 * T[i] + T[i - 1]) / dz2

            # Диффузионный член
            diffusion = k_eff[i] / (rho_eff[i] * cp_eff[i]) * laplacian

            # Источниковый член (тепловыделение от реакций)
            source = Q_reaction[i] / (rho_eff[i] * cp_eff[i])

            # Конвективный перенос (упрощённо)
            v = params.INLET_VELOCITY
            if i > 0:
                convection = -v * (T[i] - T[i - 1]) / self.geometry.dz
            else:
                convection = 0.0

            # Обновление температуры
            T_new[i] = T[i] + self.dt * (diffusion + source + convection)

        # Граничные условия
        T_new[0] = params.INLET_TEMPERATURE  # вход
        T_new[-1] = T_new[-2]  # выход (градиент = 0)

        self.fields['temperature'].values = T_new

    def solve_species(self):
        """Решение уравнений для объёмных долей фаз"""
        T = self.fields['temperature'].values
        alpha_vr = self.fields['alpha_vr'].values
        alpha_dist = self.fields['alpha_dist'].values
        alpha_coke = self.fields['alpha_coke'].values

        # Обновление объёмных долей через кинетику
        new_vr, new_dist, new_coke = self.kinetics.update_volume_fractions(
            alpha_vr, alpha_dist, alpha_coke, T, self.dt
        )

        # Конвективный перенос (упрощённо - явная upwind схема)
        v = params.INLET_VELOCITY
        dz = self.geometry.dz

        for i in range(1, self.geometry.n_points):
            # Перенос снизу вверх
            convection_vr = v * (alpha_vr[i - 1] - alpha_vr[i]) / dz
            convection_dist = v * (alpha_dist[i - 1] - alpha_dist[i]) / dz

            new_vr[i] += self.dt * convection_vr
            new_dist[i] += self.dt * convection_dist
            # Кокс не переносится конвекцией (твёрдая фаза)

        self.fields['alpha_vr'].values = new_vr
        self.fields['alpha_dist'].values = new_dist
        self.fields['alpha_coke'].values = new_coke

        # Обновление пористости
        self.fields['porosity'].values = self.kinetics.get_porosity(new_coke)

    def calculate_coke_height(self):
        """
        Расчёт высоты коксового слоя

        Высота определяется как высота зоны с объёмной долей кокса > 0.01
        """
        alpha_coke = self.fields['alpha_coke'].values
        coke_threshold = 0.01

        # Находим индекс верхней точки с коксом
        coke_indices = np.where(alpha_coke > coke_threshold)[0]

        if len(coke_indices) > 0:
            max_index = np.max(coke_indices)
            coke_height = self.geometry.z[max_index]
        else:
            coke_height = 0.0

        return coke_height

    def calculate_coke_yield(self):
        """
        Расчёт выхода кокса (массовый процент)

        Выход = (масса_кокса / масса_загруженного_сырья) * 100%
        """
        alpha_coke = self.fields['alpha_coke'].values
        cell_volume = self.geometry.get_cell_volume()

        # Масса кокса
        coke_mass = np.sum(alpha_coke * params.COKE_DENSITY * cell_volume)

        # Масса загруженного сырья (за всё время)
        # Упрощённо: объёмный расход * время * плотность
        volume_flow_rate = params.INLET_VELOCITY * self.geometry.area
        total_vr_volume = volume_flow_rate * self.time
        total_vr_mass = total_vr_volume * params.VR_DENSITY

        if total_vr_mass > 0:
            coke_yield = coke_mass / total_vr_mass
        else:
            coke_yield = 0.0

        return coke_yield

    def save_state(self):
        """Сохранение текущего состояния в историю"""
        self.history['time'].append(self.time)
        self.history['temperature'].append(self.fields['temperature'].values.copy())
        self.history['alpha_vr'].append(self.fields['alpha_vr'].values.copy())
        self.history['alpha_dist'].append(self.fields['alpha_dist'].values.copy())
        self.history['alpha_coke'].append(self.fields['alpha_coke'].values.copy())
        self.history['porosity'].append(self.fields['porosity'].values.copy())
        self.history['coke_height'].append(self.calculate_coke_height())

    def timestep(self):
        """Выполнение одного временного шага"""
        # 1. Применение граничных условий
        self.apply_boundary_conditions()

        # 2. Решение уравнений
        self.solve_temperature()
        self.solve_species()

        # 3. Обновление времени
        self.time += self.dt
        self.current_step += 1

    def run(self, verbose=True):
        """
        Запуск симуляции

        Parameters:
        -----------
        verbose : bool
            Выводить прогресс
        """
        print("\n" + "=" * 70)
        print("ЗАПУСК СИМУЛЯЦИИ")
        print("=" * 70)
        print(f"Время симуляции: {params.SIMULATION_TIME / 3600:.1f} часов")
        print(f"Число шагов: {self.n_timesteps}")
        print(f"Шаг по времени: {self.dt:.4f} с")
        print("=" * 70 + "\n")

        # Сохранение начального состояния
        self.save_state()

        start_time = time.time()
        save_interval_steps = int(params.SAVE_INTERVAL / self.dt)

        for step in range(self.n_timesteps):
            self.timestep()

            # Сохранение состояния
            if (step + 1) % save_interval_steps == 0:
                self.save_state()

                if verbose:
                    hours = self.time / 3600
                    coke_height = self.calculate_coke_height()
                    coke_yield = self.calculate_coke_yield()
                    T_avg = np.mean(self.fields['temperature'].values) - 273.15

                    print(f"t = {hours:5.1f} ч | "
                          f"Высота кокса: {coke_height*100:.1f} см | "
                          f"Выход: {coke_yield*100:.2f}% | "
                          f"T_avg: {T_avg:.1f}°C")

        # Финальное сохранение
        if self.current_step % save_interval_steps != 0:
            self.save_state()

        elapsed_time = time.time() - start_time

        print("\n" + "=" * 70)
        print("СИМУЛЯЦИЯ ЗАВЕРШЕНА")
        print("=" * 70)
        print(f"Время выполнения: {elapsed_time:.2f} с")
        print(f"Финальная высота кокса: {self.calculate_coke_height()*100:.2f} см")
        print(f"Финальный выход кокса: {self.calculate_coke_yield()*100:.2f}%")
        print("=" * 70 + "\n")

    def get_results(self):
        """
        Получить результаты симуляции

        Returns:
        --------
        dict : словарь с результатами
        """
        results = {
            'geometry': self.geometry,
            'history': self.history,
            'final_coke_height': self.calculate_coke_height(),
            'final_coke_yield': self.calculate_coke_yield(),
            'simulation_time': self.time,
            'kinetics_info': self.kinetics.get_info()
        }
        return results


if __name__ == "__main__":
    # Тест модуля
    print("Тестирование модуля solver_1d...")

    solver = Solver1D()

    # Короткая тестовая симуляция
    solver.n_timesteps = 1000
    solver.run(verbose=True)

    print("\nТест пройден успешно!")