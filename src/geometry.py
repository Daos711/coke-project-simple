"""
Модуль геометрии и сетки для 1D модели реактора
"""

import numpy as np
from . import params


class Geometry1D:
    """Класс для работы с 1D геометрией реактора"""

    def __init__(self):
        """Инициализация геометрии"""
        self.height = params.REACTOR_HEIGHT
        self.diameter = params.REACTOR_DIAMETER
        self.area = params.REACTOR_AREA
        self.n_points = params.N_POINTS
        self.dz = params.DZ

        # Создание сетки
        self.z = np.linspace(0, self.height, self.n_points)

        # Координаты центров ячеек (для конечных объемов)
        self.z_centers = (self.z[:-1] + self.z[1:]) / 2

    def get_cell_volume(self, index=None):
        """
        Получить объем ячейки

        Parameters:
        -----------
        index : int, optional
            Индекс ячейки. Если None, возвращает все объемы

        Returns:
        --------
        float or ndarray : объем ячейки(ек) в м³
        """
        if index is None:
            return self.area * self.dz * np.ones(self.n_points)
        return self.area * self.dz

    def get_cell_area(self):
        """
        Получить площадь поперечного сечения ячейки

        Returns:
        --------
        float : площадь в м²
        """
        return self.area

    def get_wall_area(self, index=None):
        """
        Получить площадь боковой поверхности ячейки

        Parameters:
        -----------
        index : int, optional
            Индекс ячейки

        Returns:
        --------
        float : площадь боковой поверхности в м²
        """
        wall_area = np.pi * self.diameter * self.dz
        if index is None:
            return wall_area * np.ones(self.n_points)
        return wall_area

    def is_inlet(self, index):
        """Проверка, является ли ячейка входной"""
        return index == 0

    def is_outlet(self, index):
        """Проверка, является ли ячейка выходной"""
        return index == self.n_points - 1

    def get_info(self):
        """Получить информацию о геометрии"""
        info = {
            'height': self.height,
            'diameter': self.diameter,
            'area': self.area,
            'n_points': self.n_points,
            'dz': self.dz,
            'total_volume': self.area * self.height
        }
        return info

    def print_info(self):
        """Вывести информацию о геометрии"""
        info = self.get_info()
        print("\n" + "=" * 50)
        print("ИНФОРМАЦИЯ О ГЕОМЕТРИИ")
        print("=" * 50)
        print(f"Высота реактора:          {info['height']:.4f} м")
        print(f"Диаметр реактора:         {info['diameter']:.4f} м")
        print(f"Площадь сечения:          {info['area']:.6f} м²")
        print(f"Полный объем:             {info['total_volume']:.6f} м³")
        print(f"Число расчетных точек:    {info['n_points']}")
        print(f"Шаг сетки (dz):           {info['dz'] * 1000:.2f} мм")
        print("=" * 50 + "\n")


class Field1D:
    """Класс для работы со скалярными полями на 1D сетке"""

    def __init__(self, geometry, initial_value=0.0, name="field"):
        """
        Инициализация поля

        Parameters:
        -----------
        geometry : Geometry1D
            Объект геометрии
        initial_value : float or ndarray
            Начальное значение поля
        name : str
            Имя поля
        """
        self.geometry = geometry
        self.name = name
        self.n_points = geometry.n_points

        if isinstance(initial_value, (int, float)):
            self.values = np.ones(self.n_points) * initial_value
        else:
            self.values = np.array(initial_value).copy()

        # История значений для анализа сходимости
        self.history = []

    def set_value(self, index, value):
        """Установить значение в точке"""
        self.values[index] = value

    def get_value(self, index):
        """Получить значение в точке"""
        return self.values[index]

    def set_all(self, value):
        """Установить значение во всех точках"""
        if isinstance(value, (int, float)):
            self.values[:] = value
        else:
            self.values[:] = value

    def get_all(self):
        """Получить все значения"""
        return self.values.copy()

    def get_gradient(self, index):
        """
        Рассчитать градиент в точке (центральная разность)

        Parameters:
        -----------
        index : int
            Индекс точки

        Returns:
        --------
        float : градиент
        """
        if index == 0:
            # Передняя разность
            return (self.values[1] - self.values[0]) / self.geometry.dz
        elif index == self.n_points - 1:
            # Задняя разность
            return (self.values[-1] - self.values[-2]) / self.geometry.dz
        else:
            # Центральная разность
            return (self.values[index + 1] - self.values[index - 1]) / (2 * self.geometry.dz)

    def get_laplacian(self, index):
        """
        Рассчитать лапласиан в точке (вторая производная)

        Parameters:
        -----------
        index : int
            Индекс точки

        Returns:
        --------
        float : лапласиан
        """
        if index == 0 or index == self.n_points - 1:
            return 0.0  # граничные условия

        dz2 = self.geometry.dz ** 2
        return (self.values[index + 1] - 2 * self.values[index] + self.values[index - 1]) / dz2

    def save_to_history(self):
        """Сохранить текущее состояние в историю"""
        self.history.append(self.values.copy())

    def get_max(self):
        """Получить максимальное значение"""
        return np.max(self.values)

    def get_min(self):
        """Получить минимальное значение"""
        return np.min(self.values)

    def get_mean(self):
        """Получить среднее значение"""
        return np.mean(self.values)


def create_initial_fields(geometry):
    """
    Создать начальные поля для симуляции

    Parameters:
    -----------
    geometry : Geometry1D
        Объект геометрии

    Returns:
    --------
    dict : словарь с полями
    """
    fields = {
        # Температура
        'temperature': Field1D(geometry, params.INITIAL_TEMPERATURE, "Temperature"),

        # Объемные доли фаз (начально весь реактор заполнен вакуумным остатком)
        'alpha_vr': Field1D(geometry, 1.0, "VR volume fraction"),
        'alpha_dist': Field1D(geometry, 0.0, "Distillables volume fraction"),
        'alpha_coke': Field1D(geometry, 0.0, "Coke volume fraction"),

        # Пористость коксового слоя
        'porosity': Field1D(geometry, params.COKE_POROSITY_MAX, "Coke porosity"),

        # Скорость (упрощенно - постоянная)
        'velocity': Field1D(geometry, params.INLET_VELOCITY, "Velocity"),
    }

    return fields


if __name__ == "__main__":
    # Тест модуля
    print("Тестирование модуля geometry...")

    geom = Geometry1D()
    geom.print_info()

    # Создание тестового поля
    temp_field = Field1D(geom, 500.0, "Test Temperature")
    print(f"Тестовое поле создано:")
    print(f"  Среднее значение: {temp_field.get_mean():.2f}")
    print(f"  Мин/Макс: {temp_field.get_min():.2f} / {temp_field.get_max():.2f}")

    print("\nТест пройден успешно!")