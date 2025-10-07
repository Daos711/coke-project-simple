"""
Модуль кинетики реакций коксования
Reaction kinetics for delayed coking process
"""

import numpy as np
from . import params


class KineticsModel:
    """Класс для расчёта кинетики реакций коксования"""

    def __init__(self, vr_params=None):
        """
        Инициализация модели кинетики

        Parameters:
        -----------
        vr_params : dict, optional
            Параметры вакуумного остатка (по умолчанию из params.ACTIVE_VR)
        """
        if vr_params is None:
            vr_params = params.ACTIVE_VR

        self.vr_params = vr_params
        self.R = params.R_GAS

        print(f"Инициализация кинетической модели: {vr_params['name']}")

    def get_rate_constant(self, temperature, k0, E_activation):
        """
        Расчёт константы скорости реакции (уравнение Аррениуса)

        k = k0 * exp(-E / RT)

        Parameters:
        -----------
        temperature : float or ndarray
            Температура в K
        k0 : float
            Предэкспоненциальный множитель
        E_activation : float
            Энергия активации в Дж/моль

        Returns:
        --------
        float or ndarray : константа скорости реакции, 1/с
        """
        return k0 * np.exp(-E_activation / (self.R * temperature))

    def get_distillation_rate(self, temperature, alpha_vr):
        """
        Расчёт скорости образования дистиллятов из вакуумного остатка

        Реакция: VR -> Distillables

        Parameters:
        -----------
        temperature : float or ndarray
            Температура в K
        alpha_vr : float or ndarray
            Объёмная доля вакуумного остатка

        Returns:
        --------
        float or ndarray : скорость реакции, 1/с
        """
        k_dist = self.get_rate_constant(
            temperature,
            self.vr_params['k_dist_0'],
            self.vr_params['E_dist']
        )
        return k_dist * alpha_vr

    def get_coking_rate(self, temperature, alpha_vr):
        """
        Расчёт скорости образования кокса из вакуумного остатка

        Реакция: VR -> Coke

        Parameters:
        -----------
        temperature : float or ndarray
            Температура в K
        alpha_vr : float or ndarray
            Объёмная доля вакуумного остатка

        Returns:
        --------
        float or ndarray : скорость реакции, 1/с
        """
        k_coke = self.get_rate_constant(
            temperature,
            self.vr_params['k_coke_0'],
            self.vr_params['E_coke']
        )
        return k_coke * alpha_vr

    def get_total_consumption_rate(self, temperature, alpha_vr):
        """
        Полная скорость расхода вакуумного остатка

        d(alpha_vr)/dt = -(r_dist + r_coke)

        Parameters:
        -----------
        temperature : float or ndarray
            Температура в K
        alpha_vr : float or ndarray
            Объёмная доля вакуумного остатка

        Returns:
        --------
        float or ndarray : скорость расхода VR, 1/с
        """
        r_dist = self.get_distillation_rate(temperature, alpha_vr)
        r_coke = self.get_coking_rate(temperature, alpha_vr)
        return r_dist + r_coke

    def get_heat_generation(self, temperature, alpha_vr):
        """
        Расчёт тепловыделения от химических реакций

        Parameters:
        -----------
        temperature : float or ndarray
            Температура в K
        alpha_vr : float or ndarray
            Объёмная доля вакуумного остатка

        Returns:
        --------
        float or ndarray : тепловыделение, Вт/м³
        """
        r_dist = self.get_distillation_rate(temperature, alpha_vr)
        r_coke = self.get_coking_rate(temperature, alpha_vr)

        # Тепловыделение (отрицательные delta_H означают экзотермические реакции)
        Q_dist = -self.vr_params['delta_H_dist'] * r_dist * params.VR_DENSITY
        Q_coke = -self.vr_params['delta_H_coke'] * r_coke * params.VR_DENSITY

        return Q_dist + Q_coke

    def update_volume_fractions(self, alpha_vr, alpha_dist, alpha_coke,
                                 temperature, dt):
        """
        Обновление объёмных долей фаз на временном шаге

        Parameters:
        -----------
        alpha_vr : ndarray
            Объёмная доля вакуумного остатка
        alpha_dist : ndarray
            Объёмная доля дистиллятов
        alpha_coke : ndarray
            Объёмная доля кокса
        temperature : ndarray
            Температура в K
        dt : float
            Шаг по времени, с

        Returns:
        --------
        tuple : (новое alpha_vr, новое alpha_dist, новое alpha_coke)
        """
        # Скорости реакций
        r_dist = self.get_distillation_rate(temperature, alpha_vr)
        r_coke = self.get_coking_rate(temperature, alpha_vr)

        # Обновление объёмных долей (явная схема Эйлера)
        new_alpha_vr = alpha_vr - (r_dist + r_coke) * dt
        new_alpha_dist = alpha_dist + r_dist * dt
        new_alpha_coke = alpha_coke + r_coke * dt

        # Проверка ограничений (объёмные доли не могут быть отрицательными)
        new_alpha_vr = np.maximum(new_alpha_vr, 0.0)
        new_alpha_dist = np.maximum(new_alpha_dist, 0.0)
        new_alpha_coke = np.maximum(new_alpha_coke, 0.0)

        # Нормализация (сумма должна быть <= 1)
        total = new_alpha_vr + new_alpha_dist + new_alpha_coke
        mask = total > 1.0
        if np.any(mask):
            new_alpha_vr[mask] /= total[mask]
            new_alpha_dist[mask] /= total[mask]
            new_alpha_coke[mask] /= total[mask]

        return new_alpha_vr, new_alpha_dist, new_alpha_coke

    def get_porosity(self, alpha_coke):
        """
        Расчёт пористости коксового слоя

        Пористость линейно уменьшается с увеличением объёмной доли кокса

        Parameters:
        -----------
        alpha_coke : float or ndarray
            Объёмная доля кокса

        Returns:
        --------
        float or ndarray : пористость
        """
        # Линейная интерполяция между максимальной и минимальной пористостью
        porosity = params.COKE_POROSITY_MAX - alpha_coke * (
            params.COKE_POROSITY_MAX - params.COKE_POROSITY_MIN
        )
        return np.clip(porosity, params.COKE_POROSITY_MIN, params.COKE_POROSITY_MAX)

    def get_info(self):
        """Получить информацию о кинетической модели"""
        info = {
            'name': self.vr_params['name'],
            'k_dist_0': self.vr_params['k_dist_0'],
            'E_dist': self.vr_params['E_dist'],
            'k_coke_0': self.vr_params['k_coke_0'],
            'E_coke': self.vr_params['E_coke'],
            'delta_H_dist': self.vr_params['delta_H_dist'],
            'delta_H_coke': self.vr_params['delta_H_coke']
        }
        return info

    def print_info(self):
        """Вывести информацию о кинетической модели"""
        info = self.get_info()
        print("\n" + "=" * 60)
        print("ИНФОРМАЦИЯ О КИНЕТИЧЕСКОЙ МОДЕЛИ")
        print("=" * 60)
        print(f"Тип сырья: {info['name']}")
        print(f"\nОбразование дистиллятов (VR → Distillables):")
        print(f"  k0 = {info['k_dist_0']:.2e} 1/с")
        print(f"  E  = {info['E_dist']/1000:.1f} кДж/моль")
        print(f"  ΔH = {info['delta_H_dist']/1000:.1f} кДж/кг")
        print(f"\nОбразование кокса (VR → Coke):")
        print(f"  k0 = {info['k_coke_0']:.2e} 1/с")
        print(f"  E  = {info['E_coke']/1000:.1f} кДж/моль")
        print(f"  ΔH = {info['delta_H_coke']/1000:.1f} кДж/кг")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    # Тест модуля
    print("Тестирование модуля kinetics...")

    kinetics = KineticsModel()
    kinetics.print_info()

    # Тест расчёта констант скорости
    T = 773.15  # K (500°C)
    alpha_vr = 0.5

    r_dist = kinetics.get_distillation_rate(T, alpha_vr)
    r_coke = kinetics.get_coking_rate(T, alpha_vr)

    print(f"При T = {T-273.15:.1f}°C и alpha_VR = {alpha_vr}:")
    print(f"  Скорость образования дистиллятов: {r_dist:.6e} 1/с")
    print(f"  Скорость образования кокса:       {r_coke:.6e} 1/с")
    print(f"  Тепловыделение:                    {kinetics.get_heat_generation(T, alpha_vr):.2e} Вт/м³")

    print("\nТест пройден успешно!")