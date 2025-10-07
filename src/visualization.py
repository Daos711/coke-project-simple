"""
Модуль визуализации результатов симуляции
Visualization module for simulation results
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
from . import params


class Visualizer:
    """Класс для визуализации результатов симуляции"""

    def __init__(self, results):
        """
        Инициализация визуализатора

        Parameters:
        -----------
        results : dict
            Словарь с результатами из Solver1D.get_results()
        """
        self.results = results
        self.geometry = results['geometry']
        self.history = results['history']

        # Создание папки для результатов
        self.output_dir = params.OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"Визуализатор инициализирован. Результаты будут сохранены в '{self.output_dir}/'")

    def plot_volume_fractions_evolution(self, save=True):
        """
        График эволюции объёмных долей фаз во времени (как Figure 5-7 из статьи)
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        times_hours = np.array(self.history['time']) / 3600
        z_cm = self.geometry.z * 100  # перевод в см

        # Выбираем несколько моментов времени для отображения
        time_indices = [0, len(times_hours) // 4, len(times_hours) // 2,
                        3 * len(times_hours) // 4, len(times_hours) - 1]

        for idx in time_indices:
            t_hour = times_hours[idx]
            label = f'{t_hour:.1f}h'

            # VR
            axes[0].plot(self.history['alpha_vr'][idx], z_cm, label=label, linewidth=2)
            # Distillables
            axes[1].plot(self.history['alpha_dist'][idx], z_cm, label=label, linewidth=2)
            # Coke
            axes[2].plot(self.history['alpha_coke'][idx], z_cm, label=label, linewidth=2)

        titles = ['Vacuum Residue', 'Distillables', 'Coke']
        for i, ax in enumerate(axes):
            ax.set_xlabel('Volume Fraction', fontsize=12)
            ax.set_ylabel('Reactor Height (cm)', fontsize=12)
            ax.set_title(titles[i], fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])

        plt.tight_layout()

        if save:
            filepath = os.path.join(self.output_dir, 'volume_fractions_evolution.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Сохранён график: {filepath}")

        plt.show()
        return fig

    def plot_temperature_evolution(self, save=True):
        """
        График эволюции температуры по высоте реактора
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        times_hours = np.array(self.history['time']) / 3600
        z_cm = self.geometry.z * 100

        time_indices = [0, len(times_hours) // 4, len(times_hours) // 2,
                        3 * len(times_hours) // 4, len(times_hours) - 1]

        for idx in time_indices:
            t_hour = times_hours[idx]
            T_celsius = self.history['temperature'][idx] - 273.15
            ax.plot(T_celsius, z_cm, label=f'{t_hour:.1f}h', linewidth=2)

        ax.set_xlabel('Temperature (°C)', fontsize=12)
        ax.set_ylabel('Reactor Height (cm)', fontsize=12)
        ax.set_title('Temperature Profile Evolution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save:
            filepath = os.path.join(self.output_dir, 'temperature_evolution.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Сохранён график: {filepath}")

        plt.show()
        return fig

    def plot_coke_bed_growth(self, save=True):
        """
        График роста высоты коксового слоя во времени (как Figure 11)
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        times_hours = np.array(self.history['time']) / 3600
        coke_heights_cm = np.array(self.history['coke_height']) * 100

        ax.plot(times_hours, coke_heights_cm, linewidth=3, color='darkblue', label='Simulation')

        # Добавление экспериментальных данных если есть
        vr_name = self.results['kinetics_info']['name']
        if 'VR3' in vr_name or 'Vacuum Residue 3' in vr_name:
            exp_height = params.EXPERIMENTAL_DATA['VR3']['final_height'] * 100
            ax.axhline(y=exp_height, color='red', linestyle='--', linewidth=2,
                       label=f'Experimental ({exp_height:.1f} cm)')

        ax.set_xlabel('Time (hours)', fontsize=12)
        ax.set_ylabel('Coke Bed Height (cm)', fontsize=12)
        ax.set_title('Coke Bed Growth', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, times_hours[-1]])

        plt.tight_layout()

        if save:
            filepath = os.path.join(self.output_dir, 'coke_bed_growth.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Сохранён график: {filepath}")

        plt.show()
        return fig

    def plot_porosity_profile(self, save=True):
        """
        График профиля пористости коксового слоя (как Figure 8-10)
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        times_hours = np.array(self.history['time']) / 3600
        z_cm = self.geometry.z * 100

        time_indices = [len(times_hours) // 4, len(times_hours) // 2,
                        3 * len(times_hours) // 4, len(times_hours) - 1]

        for idx in time_indices:
            t_hour = times_hours[idx]
            ax.plot(self.history['porosity'][idx], z_cm,
                    label=f'{t_hour:.1f}h', linewidth=2)

        ax.set_xlabel('Porosity', fontsize=12)
        ax.set_ylabel('Reactor Height (cm)', fontsize=12)
        ax.set_title('Coke Bed Porosity Evolution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])

        plt.tight_layout()

        if save:
            filepath = os.path.join(self.output_dir, 'porosity_profile.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Сохранён график: {filepath}")

        plt.show()
        return fig

    def plot_contour_maps(self, save=True):
        """
        Контурные карты (как Figure 2-4 из статьи)
        Показывает распределение фаз и температуры в пространстве и времени
        """
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        times_hours = np.array(self.history['time']) / 3600
        z_cm = self.geometry.z * 100

        # Создание meshgrid для contourf
        T_grid, Z_grid = np.meshgrid(times_hours, z_cm)

        # Преобразование данных в 2D массивы
        alpha_vr_2d = np.array(self.history['alpha_vr']).T
        alpha_dist_2d = np.array(self.history['alpha_dist']).T
        alpha_coke_2d = np.array(self.history['alpha_coke']).T
        temp_2d = np.array(self.history['temperature']).T - 273.15

        # VR
        im0 = axes[0].contourf(T_grid, Z_grid, alpha_vr_2d, levels=20, cmap='Blues')
        axes[0].set_title('VR Volume Fraction', fontweight='bold')
        plt.colorbar(im0, ax=axes[0])

        # Distillables
        im1 = axes[1].contourf(T_grid, Z_grid, alpha_dist_2d, levels=20, cmap='Reds')
        axes[1].set_title('Distillables Volume Fraction', fontweight='bold')
        plt.colorbar(im1, ax=axes[1])

        # Coke
        im2 = axes[2].contourf(T_grid, Z_grid, alpha_coke_2d, levels=20, cmap='Greys')
        axes[2].set_title('Coke Volume Fraction', fontweight='bold')
        plt.colorbar(im2, ax=axes[2])

        # Temperature
        im3 = axes[3].contourf(T_grid, Z_grid, temp_2d, levels=20, cmap='hot')
        axes[3].set_title('Temperature (°C)', fontweight='bold')
        plt.colorbar(im3, ax=axes[3])

        for ax in axes:
            ax.set_xlabel('Time (hours)', fontsize=11)
            ax.set_ylabel('Height (cm)', fontsize=11)

        plt.tight_layout()

        if save:
            filepath = os.path.join(self.output_dir, 'contour_maps.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Сохранён график: {filepath}")

        plt.show()
        return fig

    def plot_comparison_with_experiment(self, save=True):
        """
        Сравнение результатов симуляции с экспериментальными данными
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        vr_name = self.results['kinetics_info']['name']

        # Определяем тип сырья
        if 'VR1' in vr_name or '1' in vr_name:
            exp_key = 'VR1'
        elif 'VR2' in vr_name or '2' in vr_name:
            exp_key = 'VR2'
        else:
            exp_key = 'VR3'

        exp_data = params.EXPERIMENTAL_DATA[exp_key]

        # График 1: Высота коксового слоя
        sim_height = self.results['final_coke_height'] * 100  # в см
        exp_height = exp_data['final_height'] * 100

        ax1.bar(['Simulation', 'Experiment'], [sim_height, exp_height],
                color=['steelblue', 'coral'], width=0.5)
        ax1.set_ylabel('Coke Bed Height (cm)', fontsize=12)
        ax1.set_title('Final Coke Bed Height', fontsize=14, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)

        # Вывод значений на столбцах
        ax1.text(0, sim_height + 1, f'{sim_height:.2f} cm', ha='center', fontsize=11)
        ax1.text(1, exp_height + 1, f'{exp_height:.2f} cm', ha='center', fontsize=11)

        # Процент отклонения
        deviation_height = abs(sim_height - exp_height) / exp_height * 100
        ax1.text(0.5, max(sim_height, exp_height) * 0.5,
                 f'Deviation: {deviation_height:.2f}%',
                 ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat'))

        # График 2: Выход кокса
        sim_yield = self.results['final_coke_yield'] * 100  # в %
        exp_yield = exp_data['coke_yield'] * 100

        ax2.bar(['Simulation', 'Experiment'], [sim_yield, exp_yield],
                color=['steelblue', 'coral'], width=0.5)
        ax2.set_ylabel('Coke Yield (%)', fontsize=12)
        ax2.set_title('Coke Yield', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        ax2.text(0, sim_yield + 1, f'{sim_yield:.2f}%', ha='center', fontsize=11)
        ax2.text(1, exp_yield + 1, f'{exp_yield:.2f}%', ha='center', fontsize=11)

        deviation_yield = abs(sim_yield - exp_yield) / exp_yield * 100
        ax2.text(0.5, max(sim_yield, exp_yield) * 0.5,
                 f'Deviation: {deviation_yield:.2f}%',
                 ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat'))

        plt.tight_layout()

        if save:
            filepath = os.path.join(self.output_dir, 'comparison_with_experiment.png')
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            print(f"Сохранён график: {filepath}")

        plt.show()
        return fig

    def generate_all_plots(self):
        """Генерация всех графиков"""
        print("\n" + "=" * 70)
        print("ГЕНЕРАЦИЯ ГРАФИКОВ")
        print("=" * 70 + "\n")

        self.plot_volume_fractions_evolution(save=True)
        self.plot_temperature_evolution(save=True)
        self.plot_coke_bed_growth(save=True)
        self.plot_porosity_profile(save=True)
        self.plot_contour_maps(save=True)
        self.plot_comparison_with_experiment(save=True)

        print("\n" + "=" * 70)
        print(f"ВСЕ ГРАФИКИ СОХРАНЕНЫ В ПАПКЕ: {self.output_dir}/")
        print("=" * 70 + "\n")

    def save_data_to_file(self):
        """Сохранение численных данных в текстовый файл"""
        filepath = os.path.join(self.output_dir, 'simulation_results.txt')

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("РЕЗУЛЬТАТЫ СИМУЛЯЦИИ ЗАМЕДЛЕННОГО КОКСОВАНИЯ\n")
            f.write("=" * 70 + "\n\n")

            f.write(f"Тип сырья: {self.results['kinetics_info']['name']}\n")
            f.write(f"Время симуляции: {self.results['simulation_time'] / 3600:.2f} часов\n\n")

            f.write("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ:\n")
            f.write(f"  Высота коксового слоя: {self.results['final_coke_height'] * 100:.2f} см\n")
            f.write(f"  Выход кокса: {self.results['final_coke_yield'] * 100:.2f} %\n\n")

            # Сравнение с экспериментом
            vr_name = self.results['kinetics_info']['name']
            if 'VR1' in vr_name or '1' in vr_name:
                exp_key = 'VR1'
            elif 'VR2' in vr_name or '2' in vr_name:
                exp_key = 'VR2'
            else:
                exp_key = 'VR3'

            exp_data = params.EXPERIMENTAL_DATA[exp_key]

            f.write("СРАВНЕНИЕ С ЭКСПЕРИМЕНТОМ:\n")
            f.write(f"  Эксп. высота:  {exp_data['final_height'] * 100:.2f} см\n")
            f.write(f"  Эксп. выход:   {exp_data['coke_yield'] * 100:.2f} %\n\n")

            dev_height = abs(self.results['final_coke_height'] - exp_data['final_height']) / exp_data[
                'final_height'] * 100
            dev_yield = abs(self.results['final_coke_yield'] - exp_data['coke_yield']) / exp_data['coke_yield'] * 100

            f.write("ОТКЛОНЕНИЯ:\n")
            f.write(f"  По высоте: {dev_height:.2f} %\n")
            f.write(f"  По выходу: {dev_yield:.2f} %\n\n")

            f.write("=" * 70 + "\n")

        print(f"Результаты сохранены в файл: {filepath}")


if __name__ == "__main__":
    print("Модуль визуализации загружен успешно!")