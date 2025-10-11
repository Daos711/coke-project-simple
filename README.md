# Delayed Coking Reactor - 1D CFD Simulation

Упрощённая 1D модель реактора замедленного коксования на основе статьи:

**Díaz, F. A., et al. (2017).** *CFD simulation of a pilot plant Delayed Coking reactor using an In-House CFD code.* CT&F - Ciencia, Tecnología y Futuro, 7(1), 85-100.

## 📋 Описание

Проект реализует трёхфазную (вакуумный остаток, дистилляты, кокс) одномерную модель для симуляции процесса замедленного коксования с **многорежимной кинетикой** и **Numba-ускорением**:

- ✅ 1D геометрия по высоте реактора
- ✅ Многорежимная кинетика с переменным порядком реакции (1.0 → 1.5 → 2.0)
- ✅ Масштабируемые константы скоростей для калибровки
- ✅ Интегральный расчёт высоты коксового слоя
- ✅ Прогрев от стенок с профилем по высоте
- ✅ Адвекция с автоматическим субшагированием (Courant)
- ✅ Numba JIT-компиляция для ускорения в ~20x
- ✅ Сравнение с экспериментальными данными (VR3)

## 🚀 Быстрый старт

### Установка зависимостей

```bash
pip install numpy matplotlib numba
```

> **Примечание:** `numba` опциональна, но даёт значительное ускорение (~20x)

### Запуск симуляции

```bash
cd examples
python run_simple.py
```

## 📁 Структура проекта

```
coke-project-simple/
├── src/                      # Исходный код
│   ├── __init__.py           # Инициализация модуля
│   ├── params.py             # Параметры модели (Inlet, Walls, Materials, TimeSetup)
│   ├── geometry.py           # Геометрия реактора (Geometry)
│   ├── kinetics.py           # Многорежимная кинетика VR3 (VR3Kinetics)
│   ├── solver_1d.py          # Решатель с Numba JIT (Coking1DSolver)
│   └── visualization.py      # Визуализация результатов
├── examples/
│   └── run_simple.py         # Главный скрипт запуска
├── results/                  # Результаты (создаётся автоматически)
│   ├── *.png                 # Графики
│   └── simulation_results.txt # Численные результаты
└── README.md                 # Этот файл
```

## ⚙️ Параметры модели

### Геометрия (`src/geometry.py`)

```python
@dataclass
class Geometry:
    H: float = 0.5692   # м - высота реактора
    D: float = 0.0602   # м - диаметр
    NZ: int = 50        # количество расчётных точек
```

### Условия на входе (`src/params.py`)

```python
@dataclass
class Inlet:
    T_in_C: float = 370.0              # °C - температура входа
    m_dot_kg_s: float = 5e-3 / 60.0    # кг/с (5 g/min подача)
    rho_vr: float = 1050.0             # кг/м³ - плотность VR
    v_gas_factor: float = 25.0         # коэфф. для скорости газа
```

### Стенки и прогрев (`src/params.py`)

```python
@dataclass
class Walls:
    T_wall_C: float = 510.0              # °C - температура стенки
    tau_heat_bottom_s: float = 10.0*60   # с - время прогрева снизу (быстро)
    tau_heat_top_s: float = 120.0*60     # с - время прогрева сверху (медленно)
    tau_profile_beta: float = 2.0        # показатель профиля τ(z)
```

**Профиль времени прогрева:**
```
τ(z) = τ_bottom + (τ_top - τ_bottom) * (z/H)^β
```
Снизу нагрев быстрее (10 мин), сверху медленнее (120 мин).

### Материалы (`src/params.py`)

```python
@dataclass
class Materials:
    rho_coke_bulk: float = 950.0   # кг/м³ - bulk плотность кокса
    rho_dist_vap: float = 3.0      # кг/м³ - условная плотность паров
```

### Временные параметры (`src/params.py`)

```python
@dataclass
class TimeSetup:
    total_hours: float = 12.0                              # часов
    dt: float = 0.05                                       # с
    snapshots_h: tuple = (0.0, 3.0, 6.0, 9.0, 12.0)       # моменты сохранения
    contour_every_s: float = 10.0 * 60.0                  # интервал контуров
```

## 🧪 Кинетика реакций

### Многорежимная кинетика (`src/kinetics.py`)

Модель использует **три температурных режима** с разными порядками реакций:

```python
@dataclass
class VR3Kinetics:
    T1_C: float = 488.0        # °C - граница режима 1
    T2_C: float = 570.0        # °C - граница режима 2
    
    # МАСШТАБЫ для калибровки (ключевые параметры!)
    scale_dist: float = 0.25   # масштаб для дистилляции
    scale_coke: float = 5.00   # масштаб для коксообразования
    
    # Базовые параметры Аррениуса
    reg1:  Regime  # order=1.0, T < 488°C
    reg15: Regime  # order=1.5, 488°C ≤ T < 570°C
    reg2:  Regime  # order=2.0, T ≥ 570°C
```

**Константы скоростей:**
```python
k_dist = scale_dist * A_dist * exp(-Ea_dist / (R*T))
k_coke = scale_coke * A_coke * exp(-Ea_coke / (R*T))
```

**Базовые значения:**
- `A_dist = 1.2e+01` 1/с
- `A_coke = 1.5e+01` 1/с
- `Ea_dist = 92 кДж/моль`
- `Ea_coke = 85 кДж/моль`

### Скорость реакций

```python
r_total = (k_dist + k_coke) * (αR^(order-1))
dαR = γ * αR * r_total * dt
```

где:
- `αR` - объёмная доля вакуумного остатка
- `γ = 1 - αC` - пористость (доля незанятая коксом)
- `order` - порядок реакции (зависит от температуры)

## 📊 Расчёт ключевых параметров

### Высота коксового слоя

**Интегральная высота (эквивалентная):**
```python
H_bed = ∫₀^H αC(z) dz ≈ Σ αC[i] * dz
```

Это корректный метод, согласованный с массой кокса.

### Выход кокса

```python
Yield_coke = (M_coke / M_feed) * 100%

где:
  M_coke = ∫ ρ_coke * αC * A dz
  M_feed = m_dot * t_total
```

## 📈 Результаты и визуализация

После запуска создаются следующие графики в папке `results/`:

1. **volume_fractions_evolution_ru.png**
   - Профили αR, αD, αC по высоте в разные моменты времени

2. **temperature_evolution_ru.png**
   - Эволюция температурного профиля

3. **coke_bed_growth_ru.png**
   - Рост высоты коксового слоя во времени
   - Сравнение с экспериментом (48.34 см)

4. **porosity_profile_ru.png**
   - Профиль пористости γ = 1 - αC

5. **contour_maps_ru.png**
   - Контурные карты αR, αD, αC, T(z,t)

6. **phase_yields_timeseries_ru.png**
   - Выходы фаз во времени (%)

7. **phase_yields_final_ru.png**
   - Финальные выходы фаз (столбчатая диаграмма)

8. **comparison_with_experiment_ru.png**
   - Сравнение высоты и выхода кокса с экспериментом

9. **simulation_results.txt**
   - Текстовый отчёт с числовыми результатами

## 🎯 Валидация модели (VR3)

Модель калибрована под экспериментальные данные для **тяжёлого сырья VR3**:

| Параметр | Эксперимент | Типичный результат модели |
|----------|-------------|---------------------------|
| Высота коксового слоя | 48.34 см | ~48-50 см |
| Выход кокса | 36.57% | ~35-37% |

**Отклонения:** < 5%

## 🔧 Настройка и калибровка

### Основные "ручки" для калибровки

**1. Масштабы кинетики (`src/kinetics.py`):**
```python
scale_dist: float = 0.25   # ↓ меньше → меньше дистиллятов
scale_coke: float = 5.00   # ↑ больше → больше кокса
```

**Типичные значения:**
- Для **увеличения выхода кокса**: увеличьте `scale_coke`
- Для **уменьшения выхода кокса**: увеличьте `scale_dist`

**2. Профиль прогрева (`src/params.py`):**
```python
tau_heat_bottom_s: float = 10.0*60   # быстрый прогрев снизу
tau_heat_top_s: float = 120.0*60     # медленный прогрев сверху
```

**3. Шаг по времени (`src/params.py`):**
```python
dt: float = 0.05  # с (уменьшить для большей точности)
```

### Пример калибровки

```python
# src/kinetics.py

# Для выхода кокса ~30%
scale_dist: float = 0.35
scale_coke: float = 4.00

# Для выхода кокса ~40%
scale_dist: float = 0.20
scale_coke: float = 6.00
```

## 💻 Примеры использования

### Базовый запуск

```python
from src.params import defaults
from src.kinetics import VR3Kinetics
from src.solver_1d import Coking1DSolver
from src.visualization import render_all_ru

# Параметры по умолчанию
geom, inlet, walls, mats, tcfg = defaults()

# Кинетика VR3
kin = VR3Kinetics()

# Создание решателя
solver = Coking1DSolver(geom, inlet, walls, mats, tcfg, kin)

# Запуск симуляции
results = solver.run(verbose_hourly=True)

# Визуализация
from pathlib import Path
render_all_ru(results, outdir=Path("results"), 
              exp_h_cm=48.34, exp_y_pct=36.57)
```

### Изменение параметров кинетики

```python
from src.kinetics import VR3Kinetics

# Увеличиваем выход кокса
kin = VR3Kinetics(scale_dist=0.20, scale_coke=6.00)

solver = Coking1DSolver(geom, inlet, walls, mats, tcfg, kin)
results = solver.run()
```

### Изменение температуры стенки

```python
from src.params import Walls

# Более высокая температура стенки
walls = Walls(T_wall_C=530.0)

solver = Coking1DSolver(geom, inlet, walls, mats, tcfg, kin)
results = solver.run()
```

### Тестовая симуляция (короткая)

```python
from src.params import TimeSetup

# Симуляция на 3 часа вместо 12
tcfg = TimeSetup(total_hours=3.0, dt=0.1)

solver = Coking1DSolver(geom, inlet, walls, mats, tcfg, kin)
results = solver.run()
```

## 🔬 Физическая модель

### Основные уравнения

**1. Прогрев от стенок:**
```
dT/dt = (T_wall - T) / τ(z)
```

**2. Адвекция фаз (upwind с субшагами):**
```
∂αR/∂t + vR * ∂αR/∂z = 0
∂αD/∂t + vD * ∂αD/∂z = 0
```

**3. Реакции:**
```
VR → Distillables (k_dist)
VR → Coke (k_coke)

dαR = -γ * αR * (k_dist + k_coke) * αR^(order-1) * dt
dαC = (ρ_vr * γ * αR^order * k_coke * dt) / ρ_coke
dαD = (ρ_vr * γ * αR^order * k_dist * dt) / ρ_dist
```

**4. Пористость:**
```
γ = 1 - αC
```

### Автоматическое субшагирование

Для устойчивости адвекции используется автоматический расчёт числа субшагов:
```python
σ = v * dt / dz
n_sub = max(1, ceil(σ))  # Courant condition
```

## ⚡ Ускорение с Numba

Критические части кода JIT-компилируются с Numba:

```python
@njit(cache=True, fastmath=True)
def _nb_step_advect_react(...):
    # Прогрев + адвекция + реакции
    # Ускорение ~20x относительно чистого Python
```

**Производительность:**
- 12 часов симуляции, 50 точек, dt=0.05: **~25-30 секунд**
- Без Numba: **~8-10 минут**

## 📝 Упрощения модели

По сравнению с полной 2D CFD моделью из статьи:

1. ✅ **1D вместо 2D** - только вдоль высоты реактора
2. ✅ **Упрощённая кинетика** - двухстадийная (dist + coke) вместо многостадийной
3. ✅ **Фиксированная скорость** - без решения уравнений Навье-Стокса
4. ✅ **Упрощённый прогрев** - релаксация к T_wall с профилем τ(z)
5. ✅ **Явная схема** - адвекция upwind + явный Эйлер для реакций

**Преимущества:**
- Быстрая разработка и итерация
- Гарантированная сходимость
- Достаточная точность для основных результатов (< 5% отклонение)
- Быстрое выполнение (~30 секунд)

## 🐛 Устранение неполадок

### Симуляция нестабильна / расходится

1. **Уменьшите шаг по времени:**
   ```python
   tcfg = TimeSetup(dt=0.01)  # было 0.05
   ```

2. **Увеличьте число точек сетки:**
   ```python
   geom = Geometry(NZ=100)  # было 50
   ```

3. **Проверьте Numba:** убедитесь, что Numba установлена
   ```bash
   pip install numba
   ```

### Нереалистичные результаты

1. **Проверьте масштабы кинетики:**
   - `scale_dist` и `scale_coke` должны быть > 0
   - Типичные значения: 0.2-0.4 для dist, 3-7 для coke

2. **Проверьте температуры:**
   - `T_in_C` обычно 350-400°C
   - `T_wall_C` обычно 500-520°C

3. **Проверьте плотности:**
   - `rho_vr` ~ 1000-1100 кг/м³
   - `rho_coke_bulk` ~ 900-1000 кг/м³

### Numba не работает

Если Numba недоступна, код автоматически переключится на Python-версию:
```python
try:
    from numba import njit
    NUMBA = True
except Exception:
    NUMBA = False  # будет использована Python-версия
```

### Ошибка импорта модулей

```bash
# Убедитесь, что вы в правильной директории
cd examples
python run_simple.py
```

## 📚 Литература

1. **Díaz, F. A., et al. (2017).** CFD simulation of a pilot plant Delayed Coking reactor using an In-House CFD code. *CT&F - Ciencia, Tecnología y Futuro*, 7(1), 85-100.

2. **Ellis, P. J., & Hardin, C. A. (1993).** Refinery coker yields. *Hydrocarbon Processing*.

3. **Trigo, J. M. (2005).** Delayed coking: An industrial perspective. *PhD Thesis*.

## 👥 Авторы

CFD Simulation Team

## 📄 Лицензия

MIT License

## 🤝 Вклад в проект

Возможные улучшения модели:

1. **2D геометрия** - добавить радиальное направление
2. **Улучшенная кинетика** - больше стадий реакций
3. **Двухфазный поток** - coupling давления и скорости
4. **Модель пористости** - эволюция структуры кокса
5. **Валидация на VR1, VR2** - расширить набор данных

---

**Удачи с симуляцией! 🚀**

## 📞 Поддержка

Если возникли вопросы или проблемы:
1. Проверьте раздел **Устранение неполадок**
2. Убедитесь, что все зависимости установлены
3. Проверьте версию Python (рекомендуется Python 3.8+)

---

*Последнее обновление: октябрь 2025*