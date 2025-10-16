# gui/app.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from pathlib import Path
import json
import threading
from queue import Queue

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# Модель
from src.params import Geometry, Inlet, Walls, Materials, TimeSetup
from src.kinetics import VR3Kinetics
from src.solver_1d import Coking1DSolver


# ----------------------------- Руководство (тексты) -----------------------------
USER_GUIDE_SECTIONS = {
"quickstart": """
БЫСТРЫЙ СТАРТ (5 шагов)

1) Минимальные параметры:
   • Стенка: T стенки, τ прогрева низа/верха, профиль β.
   • Вход: расход (г/мин), T входа.
   • Расчёт: длительность и dt (0.05–0.10 с для 12–36 ч).

2) Запуск:
   Нажмите «Запустить расчёт». Прогресс показывает заполнение по эквивалентной высоте H_eq/H.

3) Вкладка «Графики»:
   • «Рост»: синяя — H_eq(t) (эквивалентная высота слоя по объёму твёрдого);
     красный пунктир «Фронт» — диагностическая граница (aC > порог).
   • «Выходы фаз»: VR + Дистилляты + Кокс = 100 % (баланс масс).
   • «Профили», «Температура», «Пористость», «Тепловые карты».

4) Вкладка «Результаты»:
   Сводка по высотам, выходам, средним T и пористости. «Балансный выход кокса» пересчитывается по
   M_total = M_feed + M_VR(0).

5) Экспорт:
   Меню → «Экспорт результатов». (Визуализация/отчёт в папку.)
""",

"inlet": """
ПАРАМЕТРЫ ВХОДА (сырьё/подача)

• Температура входа T_in (°C):
  Чем выше T_in, тем короче индукционный участок до начала интенсивного коксования.

• Массовый расход ṁ (г/мин):
  Чем меньше ṁ, тем выше % кокса по подаче (деноминатор меньше), но рост H_eq идёт медленнее.

• Плотность VR (кг/м³):
  Участвует в переводе долей объёма (aR) в массу и обратно при балансах.

• Фактор газа v_gas_base_factor (безразмерный):
  Усиливает газовый вынос/перемешивание и теплоотдачу.
  В модели:
   – Газовая «скорость» ∝ (v_gas_base_factor / пористость) × скорость подачи, с ограничением.
   – Эффект на теплообмен: τ(z) уменьшается как τ/ (1 + k_ht·v_gas), k_ht ≈ 0.02.
   – Эффект на адвекцию VR: vR_eff = vR_base + k_mix_gas · v_газа, k_mix_gas ≈ 0.05.
  Типичные значения: 0.5–12 (1 — нейтрально).

Подсказка: начни с v_gas=1–3 для «мягкого» режима, 6–10 — для ускоренного прогрева/выноса.
""",

"walls": """
ТЕПЛОПЕРЕДАЧА (стенка и профиль прогрева)

• T стенки (°C): целевая температура прогрева слоя.

• τ прогрева низа/верха (ч):
  Формируют характерные времена релаксации по высоте.

• Профиль β:
  Задаёт форму распределения τ(z). В модели:
     s = (z/H)^β; τ(z) = τ_низ + (τ_верх − τ_низ)·s.
  β≈1 — почти линейно; β=2…3 — сильнее замедляет прогрев у верха.

Важно: в коде дополнительно ускоряется прогрев газом: τ(z) ← τ(z)/(1 + 0.02·v_gas).
Главные «рычаги» длительности: τ_верх и T стенки.
""",

"materials": """
МАТЕРИАЛЫ

• Плотность кокса (кг/м³):
  Переводит массу твёрдой фазы в объём (напрямую влияет на H_eq).

• Плотность паров (кг/м³):
  Используется для объёмной доли «дистиллятов»; на % по балансу влияет слабо.

• Мин. пористость ε_min:
  Нижняя граница пористости слоя. Максимум aC ограничен 1−ε_min.
  Типично 0.25–0.40. Ниже — более «плотный» кокс, выше H_eq при той же массе.

Замечание: d_particle_mm сейчас не используется явно в расчёте.
""",

"kinetics": """
КИНЕТИКА (VR3, 3 температурные области)

• Порог включения реакций: T_onset ≈ 415 °C (ниже реакции «выключены»).

• Режимы:
   – T < T1: порядок 1.0
   – T1 ≤ T < T2: порядок 1.5
   – T ≥ T2: порядок 2.0
  Для каждой области заданы пары Аррениуса (A, Ea) для ветвей «дистилляты» и «кокс».

• Масштабы:
  scale_dist, scale_coke — грубые множители скоростей ветвей.
  Увеличение scale_coke — больше кокса в начале; итоговый % по балансу зависит и от M_total.

• Замедление в плотном слое:
  при низкой пористости реакции замедляются (см. раздел «Как устроена модель»).
""",

"run_stop": """
РАСЧЁТ И УСЛОВИЯ ОСТАНОВА

• «Время расчёта»: общая длительность прогона.

• «Стоп при заполнении (aC > порог)»:
   В текущей версии стоп выполняется по ЭКВИВАЛЕНТНОЙ высоте H_eq (реальная высота слоя,
   вычисленная по объёму твёрдого) при достижении H_eq ≈ H.
   Порог «Фронта» используется только для диагностики (красный пунктир на графике «Рост»).

• Эквивалентная высота:
   H_eq = min( H, Σ (aC / (1 − ε_min)) · Δz ).
""",

"plots": """
ГРАФИКИ, МЕТРИКИ, ИНТЕРПРЕТАЦИЯ

• «Рост»:
   – Синяя кривая — H_eq(t): что фактически «занято» коксом.
   – Красный пунктир — фронт aC>порог: диагностический маркер, НЕ высота слоя.

• «Выходы фаз» (баланс):
   – VR(t) = 100·M_VR / (M_feed + M_VR(0));
   – Кокс(t) = 100·M_coke / (M_feed + M_VR(0));
   – Дистилляты = 100 − VR − Кокс (как остаток, без явной химспецификации).

• «Профили / Температура / Пористость»:
   Проверяй физичность: T(z) должно монотонно греться, γ(z)=1−aC ∈ [ε_min, 1].

• «Тепловые карты»:
   Быстрый контроль «где и когда» идёт коксование/дистилляция и как прогревается слой.
""",

"presets": """
ТИПОВЫЕ РЕЖИМЫ (стартовые пресеты)

• Короткий (12 ч):
  Tст=510 °C; τ низ/верх = 2/14 ч; ṁ=5 г/мин; v_gas=8; dt=0.05 с.

• Длинный (36 ч):
  Tст=500 °C; τ низ/верх = 6/36 ч; ṁ=5 г/мин; v_gas=1; dt=0.10 с.

Чувствительность:
  T_in ±20 °C; ε_min 0.25–0.40; ρ_coke 850–1050; ṁ 3–7 г/мин; β 1.5–3.5; τ 4/24…8/48.
""",

"tips": """
СОВЕТЫ ПО ЭКСПЛУАТАЦИИ

• Длительность цикла больше всего задаёт τ_верх и T стенки.
• % кокса по подаче ↑ при меньшем ṁ, большей T стенки и «раннем» T_in.
• Геометрия:
   – Диаметр влияет на скорость v_R (через площадь A) и перевод массы в H_eq;
   – Высота H определяет максимальный инвентарь.
• Газовый фактор:
   1 — нейтрально; больше — быстрее прогрев и вынос.
• Сходимость:
   NZ≈80–120 даёт гладкие профили; % меняются обычно <1–2 %.
""",

"limits": """
ОГРАНИЧЕНИЯ МОДЕЛИ

• 1D-модель без детальной гидродинамики; газ учтён эффективными поправками.
• «Дистилляты» считаются как остаток (100 − VR − Кокс), без фракционной развертки.
• «Фронт aC>порог» — диагностический маркер; используйте H_eq как высоту слоя.
• Кинетика агрегирована (VR3); параметры scale_* и профиль τ(z) калибруются по данным ПНР.
""",

"theory": """
КАК УСТРОЕНА МОДЕЛЬ (кратко)

• Сетка по высоте (NZ ячеек), шаг Δz = H/NZ.

• Прогрев от стенки:
   dT/dt = (T_wall − T) / τ(z), где τ(z) = τ_низ + (τ_верх − τ_низ)·(z/H)^β,
   затем τ(z) дополнительно делится на (1 + 0.02·v_gas).

• Транспорт (upwind):
   – VR (жидкость): скорость vR_eff = vR_base + 0.05·v_газа.
   – Дистилляты (газ): скорость v_газа усиливается при низкой пористости γ (делится на ⟨γ⟩ активной зоны).

• Реакции (VR → дистилляты + кокс):
   – Порог включения T ≥ 415 °C.
   – Три области T: порядки 1/1.5/2.0; для каждой ветви свои (A, Ea).
   – Скорости ветвей: k = scale · A · exp(−Ea/(R·(T+273.15))).
   – Замедление в плотном слое: при γ < 0.4 скорости уменьшаются (factor∈[0.5,1.0]).
   – Защита шага: если (k_dist+k_coke)·dt слишком велико, коэффициенты масштабируются.

• Эквивалентная высота:
   H_eq = min(H, Σ (aC/(1−ε_min))·Δz).
""",

"stability": """
УСТОЙЧИВОСТЬ И СКОРОСТЬ

• Критерии Куранта (прибл.):
   σ_R = vR_eff·dt/Δz ≤ 1,  σ_D = v_газа·dt/Δz ≤ 1.
  Если σ > 1, шаг делится на подшаги автоматически, но лучше держать dt разумным.

• Практика:
   – NZ=80–120 для гладкости; NZ=50 — быстро, грубо.
   – dt=0.05…0.10 с при H≈0.57 м, ṁ≈5 г/мин.
   – Есть «скачки» фронта — уменьшить dt в 2 раза.

• Производительность:
   – Бóльшие τ и меньший v_gas = медленнее прогрев ⇒ цикл дольше.
   – NZ↑ и dt↓ — точнее, но медленнее.
""",

"calibration": """
КАЛИБРОВКА ПО ДАННЫМ

1) Прогрев без коксования:
   scale_* → 0; подбери τ_низ, τ_верх, β, v_gas по термопарам.

2) Включи кокс:
   scale_dist≈0.002, scale_coke≈0.5. Совмести ранний наклон H_eq(t) — подстрой scale_coke.

3) Переходы T1/T2:
   Если рост «включается» не там — скорректируй T1/T2 на ±10–20 °C.

4) Баланс и финальные %:
   Сверь «Балансный выход кокса». Тонкая правка scale_dist и ρ_coke_bulk.

5) Устойчивость:
   Гладкие профили, σ_R/σ_D ок; при необходимости уменьшай dt.
""",

"troubleshooting": """
ДИАГНОСТИКА

• H_eq = 0:
   T ниже 415 °C; v_gas мал; τ_верх велик; пороги T1/T2 смещены.

• H_eq растёт слишком быстро:
   scale_coke велик, T стенки высока, dt крупный.

• % кокса >100 % (по подаче):
   Возможен в конце, если M_VR(0) ≫ M_feed. Смотри «балансный» показатель.

• Зубчатые профили:
   Уменьши dt, увеличь NZ, проверь v_gas и τ-профиль.
""",

"faq": """
FAQ

• Чем «Фронт» отличается от H_eq?
  Фронт — диагностический маркер aC>порог. Высота слоя — H_eq.

• Почему «Дистилляты» — остаток?
  Модель агрегированная: VR и кокс считаем явно, дистилляты — как остаток до 100 %.

• Почему «по подаче» и «балансный» расходятся?
  По подаче — нормировка на M_feed; балансный — на M_feed + начальный запас VR.

• Диапазоны:
  ε_min 0.25–0.40; ρ_coke 850–1050; v_gas 1–10; β 1.5–3.5; τ 2–8 ч / 14–48 ч.

• Как выбрать dt?
  0.05–0.10 с; есть «скачки» — 0.02–0.05 с.
"""
}


# ----------------------------- UI helpers -----------------------------
class ParameterFrame(ttk.LabelFrame):
    def __init__(self, parent, title, **kwargs):
        super().__init__(parent, text=title, **kwargs)
        self.vars, self.entries = {}, {}

    def add_field(self, label, var_name, default_value, row,
                  tooltip=None, unit="", field_type="float"):
        ttk.Label(self, text=f"{label}:").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        if field_type == "float":
            var = tk.DoubleVar(value=default_value)
        elif field_type == "int":
            var = tk.IntVar(value=default_value)
        else:
            var = tk.StringVar(value=default_value)
        self.vars[var_name] = var
        entry = ttk.Entry(self, textvariable=var, width=10); entry.grid(row=row, column=1, padx=5, pady=2)
        self.entries[var_name] = entry
        if unit:
            ttk.Label(self, text=unit).grid(row=row, column=2, sticky="w", padx=2, pady=2)
        if tooltip:
            self._create_tooltip(entry, tooltip)
        return var

    def _create_tooltip(self, widget, text):
        def on_enter(e):
            tip = tk.Toplevel(); tip.wm_overrideredirect(True)
            tip.wm_geometry(f"+{e.x_root + 10}+{e.y_root + 10}")
            ttk.Label(tip, text=text, background="#ffffe0", relief="solid", borderwidth=1).pack()
            widget.tooltip = tip
        def on_leave(e):
            if hasattr(widget, 'tooltip'): widget.tooltip.destroy(); del widget.tooltip
        widget.bind("<Enter>", on_enter); widget.bind("<Leave>", on_leave)

    def get_values(self):
        return {name: var.get() for name, var in self.vars.items()}

    def set_values(self, values_dict):
        for name, value in values_dict.items():
            if name in self.vars: self.vars[name].set(value)


# ------------------------------ Main GUI ------------------------------
class CokingSimulatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Симулятор замедленного коксования v1.0")
        self.root.geometry("1400x900")

        self.update_queue = Queue()
        self.calculation_thread = None
        self.solver = None
        self.results = None
        self._stop_flag = False

        self.setup_ui()
        self.root.after(100, self.process_queue)

    # ---------- UI ----------
    def setup_ui(self):
        self.create_menu()

        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left_frame = ttk.Frame(main_paned, width=420)
        main_paned.add(left_frame, weight=0)
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=1)

        canvas = tk.Canvas(left_frame, width=400)
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        self.create_parameter_panels(scrollable_frame)

        button_frame = ttk.Frame(left_frame); button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        row1 = ttk.Frame(button_frame); row1.pack(fill=tk.X)
        self.run_button = ttk.Button(row1, text="▶ Запустить расчёт", command=self.run_simulation); self.run_button.pack(side=tk.LEFT, padx=5, pady=2)
        self.stop_button = ttk.Button(row1, text="⏹ Остановить", state=tk.DISABLED, command=self.stop_simulation); self.stop_button.pack(side=tk.LEFT, padx=5, pady=2)
        row2 = ttk.Frame(button_frame); row2.pack(fill=tk.X)
        ttk.Button(row2, text="💾 Сохранить конфиг", command=self.save_config).pack(side=tk.LEFT, padx=5, pady=2)
        ttk.Button(row2, text="📂 Загрузить конфиг", command=self.load_config).pack(side=tk.LEFT, padx=5, pady=2)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        left_frame.pack_propagate(False)

        self.notebook = ttk.Notebook(right_frame); self.notebook.pack(fill=tk.BOTH, expand=True)
        self.progress_frame = ttk.Frame(self.notebook); self.notebook.add(self.progress_frame, text="Прогресс расчёта")
        self.create_progress_tab()
        self.plots_frame = ttk.Frame(self.notebook); self.notebook.add(self.plots_frame, text="Графики")
        ttk.Label(self.plots_frame, text="Запустите расчет для отображения графиков", font=("Arial", 14)).pack(expand=True)
        self.results_frame = ttk.Frame(self.notebook); self.notebook.add(self.results_frame, text="Результаты")

        self.status_var = tk.StringVar(value="Готов к работе")
        ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X)

    def create_menu(self):
        menubar = tk.Menu(self.root); self.root.config(menu=menubar)
        file_menu = tk.Menu(menubar, tearoff=0); menubar.add_cascade(label="Файл", menu=file_menu)
        file_menu.add_command(label="Загрузить конфигурацию", command=self.load_config)
        file_menu.add_command(label="Сохранить конфигурацию", command=self.save_config)
        file_menu.add_separator(); file_menu.add_command(label="Экспорт результатов", command=self.export_results)
        file_menu.add_separator(); file_menu.add_command(label="Выход", command=self.root.quit)

        calc_menu = tk.Menu(menubar, tearoff=0); menubar.add_cascade(label="Расчёт", menu=calc_menu)
        calc_menu.add_command(label="Запустить", command=self.run_simulation)
        calc_menu.add_command(label="Остановить", command=self.stop_simulation)

        help_menu = tk.Menu(menubar, tearoff=0); menubar.add_cascade(label="Справка", menu=help_menu)
        help_menu.add_command(label="Руководство", command=self.open_user_guide)
        help_menu.add_command(label="О программе", command=self.show_about)

    def create_parameter_panels(self, parent):
        self.param_frames = {}

        geom = ParameterFrame(parent, "Геометрия реактора"); geom.pack(fill=tk.X, padx=5, pady=5)
        geom.add_field("Высота", "H", 0.5692, 0, unit="м")
        geom.add_field("Диаметр", "D", 0.0602, 1, unit="м")
        geom.add_field("Число ячеек", "NZ", 50, 2, "Разбиение по высоте", "", "int")
        self.param_frames['geometry'] = geom

        inlet = ParameterFrame(parent, "Параметры входа"); inlet.pack(fill=tk.X, padx=5, pady=5)
        inlet.add_field("Температура входа", "T_in_C", 370, 0, unit="°C")
        inlet.add_field("Массовый расход", "m_dot_g_min", 5.0, 1, "Расход сырья", "г/мин")
        inlet.add_field("Плотность VR", "rho_vr", 1050, 2, unit="кг/м³")
        inlet.add_field("Фактор газа", "v_gas_base_factor", 8.0, 3)
        self.param_frames['inlet'] = inlet

        walls = ParameterFrame(parent, "Теплообмен"); walls.pack(fill=tk.X, padx=5, pady=5)
        walls.add_field("Температура стенки", "T_wall_C", 510, 0, unit="°C")
        walls.add_field("τ прогрева низа", "tau_heat_bottom_h", 2.0, 1, unit="ч")
        walls.add_field("τ прогрева верха", "tau_heat_top_h", 14.0, 2, unit="ч")
        walls.add_field("Профиль β", "tau_profile_beta", 2.5, 3)
        self.param_frames['walls'] = walls

        mats = ParameterFrame(parent, "Свойства материалов"); mats.pack(fill=tk.X, padx=5, pady=5)
        mats.add_field("Плотность кокса", "rho_coke_bulk", 950, 0, unit="кг/м³")
        mats.add_field("Плотность паров", "rho_dist_vap", 2.5, 1, unit="кг/м³")
        mats.add_field("Мин. пористость", "porosity_min", 0.30, 2)
        self.param_frames['materials'] = mats

        timef = ParameterFrame(parent, "Параметры расчёта"); timef.pack(fill=tk.X, padx=5, pady=5)
        timef.add_field("Время расчёта", "total_hours", 12.0, 0, unit="ч")
        timef.add_field("Шаг по времени", "dt", 0.05, 1, unit="с")
        self.stop_on_full_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(timef, text="Стоп при заполнении (aC > порог)", variable=self.stop_on_full_var)\
            .grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        ttk.Label(timef, text="Порог фронта").grid(row=2, column=2, sticky="e", padx=5)
        self.front_thr_var = tk.DoubleVar(value=10.0)
        front_thr_entry = ttk.Entry(timef, textvariable=self.front_thr_var, width=6)
        front_thr_entry.grid(row=2, column=3, sticky="w", padx=5)
        timef.entries["front_thr_pct"] = front_thr_entry

        info = ttk.Frame(timef); info.grid(row=3, column=0, columnspan=4, sticky="ew", padx=5, pady=5)
        ttk.Label(info, text="📊 О фронте коксового слоя:", font=("Arial", 10, "bold")).pack(anchor="w")
        ttk.Label(info, text="• Фронт — верхняя граница слоя (где aC > порог)", font=("Arial", 9), foreground="gray").pack(anchor="w")
        ttk.Label(info, text="• Эквивалентная высота — суммарный объём кокса", font=("Arial", 9), foreground="gray").pack(anchor="w")

        self.param_frames['time'] = timef

        kin = ParameterFrame(parent, "Кинетика"); kin.pack(fill=tk.X, padx=5, pady=5)
        kin.add_field("T1 (переход 1→1.5)", "T1_C", 487.8, 0, unit="°C")
        kin.add_field("T2 (переход 1.5→2)", "T2_C", 570.1, 1, unit="°C")
        kin.add_field("Масштаб дистиллятов", "scale_dist", 0.002, 2)
        kin.add_field("Масштаб кокса", "scale_coke", 0.50, 3)
        self.param_frames['kinetics'] = kin

        # пакетные tooltips
        TOOLTIPS = {
          "geometry": {
            "H": "Высота, м. Влияет на максимум H_eq и Δz (через NZ).",
            "D": "Диаметр, м. Через площадь A влияет на скорости и перевод массы в высоту.",
          },
          "inlet": {
            "T_in_C": "Температура сырья на входе. Выше → короче индукционный участок.",
            "rho_vr": "Плотность VR (кг/м³). Влияет на баланс масс и конверсии.",
            "v_gas_base_factor": "Фактор газа: усиливает прогрев и вынос. 1 — нейтрально; 2–10 — быстрее.",
          },
          "walls": {
            "T_wall_C": "Температура стенки (°C) — целевая температура прогрева.",
            "tau_heat_bottom_h": "Характерное время прогрева у низа (ч).",
            "tau_heat_top_h": "Характерное время прогрева у верха (ч). Главный рычаг длительности.",
            "tau_profile_beta": "Форма τ(z): s=(z/H)^β. 2–3 — обычно достаточно.",
          },
          "materials": {
            "rho_coke_bulk": "Плотность кокса (кг/м³). Чем выше, тем меньше H_eq при той же массе.",
            "rho_dist_vap": "Плотность паров (кг/м³). На % влияет слабо (остаточная величина).",
            "porosity_min": "Мин. пористость ε_min. Ограничивает максимум aC=1−ε_min.",
          },
          "time": {
            "total_hours": "Длительность прогона (ч). При автостопе может завершиться раньше.",
            "dt": "Шаг по времени (с). Держи v·dt/Δz ≤ 1. Обычно 0.05–0.10 с.",
            "front_thr_pct": "Порог aC для красного «фронта», %. Диагностика, не критерий высоты.",
          },
          "kinetics": {
            "T1_C": "Порог T1 (°C) — переход порядка 1→1.5.",
            "T2_C": "Порог T2 (°C) — переход порядка 1.5→2.",
            "scale_dist": "Масштаб ветви «дистилляты». Грубая калибровка.",
            "scale_coke": "Масштаб ветви «кокс». ↑ быстрее рост кокса и H_eq.",
          },
        }
        for block, tips in TOOLTIPS.items():
            frame = self.param_frames.get(block)
            if not frame: continue
            for var_name, text in tips.items():
                if var_name in frame.entries:
                    frame._create_tooltip(frame.entries[var_name], text)

    def create_progress_tab(self):
        self.progress_var = tk.DoubleVar()
        ttk.Progressbar(self.progress_frame, variable=self.progress_var, maximum=100)\
            .pack(fill=tk.X, padx=10, pady=10)

        log_frame = ttk.LabelFrame(self.progress_frame, text="Лог расчёта"); log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD)
        log_scroll = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.fig_realtime = Figure(figsize=(8, 4), dpi=80)
        self.ax_realtime = self.fig_realtime.add_subplot(111)
        self.ax_realtime.set_xlabel("Время (ч)"); self.ax_realtime.set_ylabel("Высота слоя (см)"); self.ax_realtime.grid(True, alpha=0.3)
        self.canvas_realtime = FigureCanvasTkAgg(self.fig_realtime, self.progress_frame)
        self.canvas_realtime.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

    # ---------- params ----------
    def get_parameters(self):
        g = self.param_frames['geometry'].get_values()
        geom = Geometry(H=g['H'], D=g['D'], NZ=int(g['NZ']))
        i = self.param_frames['inlet'].get_values()
        inlet = Inlet(T_in_C=i['T_in_C'], m_dot_kg_s=i['m_dot_g_min']/60.0/1000.0, rho_vr=i['rho_vr'], v_gas_base_factor=i['v_gas_base_factor'])
        w = self.param_frames['walls'].get_values()
        walls = Walls(T_wall_C=w['T_wall_C'], tau_heat_bottom_s=w['tau_heat_bottom_h']*3600.0, tau_heat_top_s=w['tau_heat_top_h']*3600.0, tau_profile_beta=w['tau_profile_beta'])
        m = self.param_frames['materials'].get_values()
        mats = Materials(rho_coke_bulk=m['rho_coke_bulk'], rho_dist_vap=m['rho_dist_vap'], porosity_min=m['porosity_min'])
        t = self.param_frames['time'].get_values()
        tcfg = TimeSetup(total_hours=t['total_hours'], dt=t['dt'], snapshots_h=tuple(np.linspace(0, t['total_hours'], 7)), contour_every_s=60.0)
        k = self.param_frames['kinetics'].get_values()
        kin = VR3Kinetics(T1_C=k['T1_C'], T2_C=k['T2_C'], scale_dist=k['scale_dist'], scale_coke=k['scale_coke'])
        options = {"stop_on_full": bool(self.stop_on_full_var.get()), "front_thr": float(self.front_thr_var.get())/100.0}
        return geom, inlet, walls, mats, tcfg, kin, options

    # ---------- run ----------
    def run_simulation(self):
        if self.calculation_thread and self.calculation_thread.is_alive():
            messagebox.showwarning("Предупреждение", "Расчёт уже выполняется!"); return
        self._stop_flag = False
        self.log_text.delete(1.0, tk.END)
        self.ax_realtime.clear(); self.ax_realtime.set_xlabel("Время (ч)"); self.ax_realtime.set_ylabel("Высота слоя (см)"); self.ax_realtime.grid(True, alpha=0.3)
        self.canvas_realtime.draw()
        self.run_button.config(state=tk.DISABLED); self.stop_button.config(state=tk.NORMAL)
        self.status_var.set("Выполняется расчёт...")
        self.root.update()
        self.calculation_thread = threading.Thread(target=self._run_calculation, daemon=True); self.calculation_thread.start()

    def _collect_results_from_solver(self, solver: Coking1DSolver):
        res = {
            "H_bed_m": solver.bed_height_equiv(),
            "H_front_m": solver.bed_height_front(),
            "yield_pct": solver.coke_yield_pct_feed(),
            "T_avg_C": float(np.mean(solver.T)),
            "porosity_avg": solver.porosity_avg,
            "final": {"T": solver.T.copy(), "aR": solver.aR.copy(), "aD": solver.aD.copy(), "aC": solver.aC.copy()},
            "z": solver.g.z.copy(),
            "snapshots": {
                "t_h": np.array(solver.snapshots["t_h"], dtype=float),
                "T":  np.stack(solver.snapshots["T"],  axis=0) if solver.snapshots["T"]  else np.empty((0,)),
                "aR": np.stack(solver.snapshots["aR"], axis=0) if solver.snapshots["aR"] else np.empty((0,)),
                "aD": np.stack(solver.snapshots["aD"], axis=0) if solver.snapshots["aD"] else np.empty((0,)),
                "aC": np.stack(solver.snapshots["aC"], axis=0) if solver.snapshots["aC"] else np.empty((0,)),
            },
            "growth":    {"t_h": np.array(solver.time_h_hist,      dtype=float),
                          "H_cm": np.array(solver.bed_eq_cm_hist,   dtype=float),
                          "H_front_cm": np.array(solver.bed_front_cm_hist, dtype=float)},
            "contours":  {"t_s": np.array(solver.contour_t, dtype=float),
                          "T":  np.stack(solver.contour_T,  axis=0) if solver.contour_T  else np.empty((0,)),
                          "aR": np.stack(solver.contour_aR, axis=0) if solver.contour_aR else np.empty((0,)),
                          "aD": np.stack(solver.contour_aD, axis=0) if solver.contour_aD else np.empty((0,)),
                          "aC": np.stack(solver.contour_aC, axis=0) if solver.contour_aC else np.empty((0,))},
            "meta": {"A_m2": float(solver.g.A), "m_dot_kg_s": float(solver.inlet.m_dot_kg_s),
                     "rho_vr": float(solver.inlet.rho_vr), "rho_dist": float(solver.mats.rho_dist_vap),
                     "rho_coke": float(solver.mats.rho_coke_bulk), "m_vr0": float(solver.m_vr0)}
        }
        return res

    def _run_calculation(self):
        try:
            import time
            geom, inlet, walls, mats, tcfg, kin, opts = self.get_parameters()
            self.solver = Coking1DSolver(geom, inlet, walls, mats, tcfg, kin)

            steps = int(tcfg.total_hours * 3600.0 / tcfg.dt)
            next_hour_s, next_growth_s = 3600.0, 300.0
            H_reactor_cm = float(geom.H * 100.0)
            front_thr = float(opts.get("front_thr", 0.10))

            last_ui, MIN_DT = time.perf_counter(), 0.20

            self.solver.time_h_hist, self.solver.bed_eq_cm_hist, self.solver.bed_front_cm_hist = [], [], []
            t_hist, h_hist = [], []

            for step_i in range(steps):
                if self._stop_flag:
                    self.update_queue.put(('stopped', None))
                    return

                self.solver.step()

                if step_i % max(100, int(1.0 / tcfg.dt)) == 0:
                    time.sleep(0.001)
                    self.update_queue.put(('heartbeat', None))

                H_eq_cm = self.solver.bed_height_equiv() * 100.0

                now = time.perf_counter()
                if now - last_ui >= MIN_DT:
                    fill_pct = min(100.0, 100.0 * H_eq_cm / H_reactor_cm)
                    self.update_queue.put(('progress', fill_pct))
                    last_ui = now

                if self.solver.time_s >= next_growth_s - 1e-12:
                    t_h = self.solver.time_s / 3600.0
                    H_front_cm = self.solver.bed_height_front(thr=front_thr) * 100.0

                    if not t_hist and H_eq_cm <= 1e-9:
                        next_growth_s += 300.0
                    else:
                        self.solver.time_h_hist.append(t_h)
                        self.solver.bed_eq_cm_hist.append(H_eq_cm)
                        self.solver.bed_front_cm_hist.append(H_front_cm)

                        t_hist.append(t_h)
                        h_hist.append(H_eq_cm)
                        self.update_queue.put(('plot', (t_hist.copy(), h_hist.copy())))
                        next_growth_s += 300.0

                if self.solver.time_s >= next_hour_s - 1e-12:
                    t_h = self.solver.time_s / 3600.0
                    y_feed = self.solver.coke_yield_pct_feed()
                    y_bal = self.solver.coke_yield_pct_balance()
                    self.update_queue.put(('log',
                                           f"t = {t_h:5.1f} ч | H_eq = {H_eq_cm:.1f} см | Yбал = {y_bal:.2f}% | Y = {y_feed:.2f}%\n"))
                    next_hour_s += 3600.0

                self.solver._maybe_take_snapshot()
                self.solver._maybe_take_contour()

                if opts.get("stop_on_full", False) and H_eq_cm >= H_reactor_cm - 1e-6:
                    t_fill_h = self.solver.time_s / 3600.0
                    Y_at_fill = self.solver.coke_yield_pct_feed()
                    self.update_queue.put(('log',
                                           f"[fill] t_fill={t_fill_h:.2f} ч | Y_fill={Y_at_fill:.2f}% | H_eq={H_eq_cm:.1f} см\n"))
                    results = self._collect_results_from_solver(self.solver)
                    results.setdefault("extra", {}).update({
                        "t_fill_h": t_fill_h, "Y_at_fill_pct": Y_at_fill, "front_thr": front_thr
                    })
                    self.update_queue.put(('finished', results))
                    return

            results = self._collect_results_from_solver(self.solver)
            results.setdefault("extra", {}).update({
                "t_fill_h": None, "Y_at_fill_pct": None, "front_thr": front_thr
            })
            self.update_queue.put(('finished', results))

        except Exception as e:
            import traceback
            self.update_queue.put(('error', f"{str(e)}\n\n{traceback.format_exc()}"))

    def stop_simulation(self): self._stop_flag = True; self.stop_button.config(state=tk.DISABLED)

    # ---------- messaging ----------
    def process_queue(self):
        try:
            while True:
                msg_type, data = self.update_queue.get_nowait()
                if msg_type == 'progress': self.progress_var.set(float(data))
                elif msg_type == 'log': self.log_text.insert(tk.END, data); self.log_text.see(tk.END)
                elif msg_type == 'heartbeat': pass
                elif msg_type == 'plot':
                    t_hist, h_hist = data
                    self.ax_realtime.clear()
                    self.ax_realtime.plot(t_hist, h_hist, 'b-', linewidth=2)
                    self.ax_realtime.set_xlabel("Время (ч)"); self.ax_realtime.set_ylabel("Высота слоя (см)")
                    self.ax_realtime.grid(True, alpha=0.3); self.canvas_realtime.draw()
                elif msg_type == 'finished':
                    self.run_button.config(state=tk.NORMAL); self.stop_button.config(state=tk.DISABLED)
                    self.status_var.set("Расчёт завершён"); self._stop_flag = False
                    self.results = data; self.display_results(self.results)
                    messagebox.showinfo("Готово", "Расчёт успешно завершён!")
                elif msg_type == 'stopped':
                    self.run_button.config(state=tk.NORMAL); self.stop_button.config(state=tk.DISABLED)
                    self.status_var.set("Расчёт остановлен"); self._stop_flag = False
                elif msg_type == 'error':
                    self.run_button.config(state=tk.NORMAL); self.stop_button.config(state=tk.DISABLED)
                    self.status_var.set("Ошибка расчёта"); messagebox.showerror("Ошибка", f"Ошибка при расчёте:\n{data}")
        except Exception:
            pass
        self.root.after(100, self.process_queue)

    # ---------- results ----------
    def display_results(self, results: dict):
        self.notebook.select(self.plots_frame)
        for w in self.plots_frame.winfo_children(): w.destroy()
        tabs = ttk.Notebook(self.plots_frame); tabs.pack(fill=tk.BOTH, expand=True)
        self.create_result_plots_tabbed(tabs, results)
        self.update_results_tab(results)

    def create_result_plots_tabbed(self, notebook: ttk.Notebook, results: dict):
        import numpy as np
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        z_m  = np.asarray(results['z'], dtype=float); z_cm = z_m * 100.0
        snaps = results.get('snapshots', {}); cont = results.get('contours', {}); meta = results.get('meta', {})

        def _m_vr0_from_any() -> float:
            A = float(meta["A_m2"]); rho_vr = float(meta["rho_vr"])
            aR0 = None
            sr = snaps.get("aR", None)
            if sr is not None and np.size(sr) > 0:
                aR0 = np.asarray(sr); aR0 = aR0[0] if aR0.ndim == 2 else aR0
            if aR0 is None:
                ar = cont.get("aR", None)
                if ar is not None and np.size(ar) > 0:
                    aR0 = np.asarray(ar); aR0 = aR0[0] if aR0.ndim == 2 else aR0
            return float(A * np.trapz(aR0 * rho_vr, z_m)) if aR0 is not None else 0.0

        def _phase_series(start_frac: float = 0.02):
            A = float(meta["A_m2"]); m_dot = float(meta["m_dot_kg_s"])
            rho_vr = float(meta["rho_vr"]); rho_coke = float(meta["rho_coke"])
            t_s = np.asarray(cont.get("t_s", []), dtype=float)
            aR  = np.asarray(cont.get("aR",  [])); aC = np.asarray(cont.get("aC",  []))
            if t_s.size == 0 or aR.size == 0 or aC.size == 0: raise ValueError("Недостаточно данных контуров")
            m_vr   = A * np.trapz(aR * rho_vr,   z_m, axis=1)
            m_coke = A * np.trapz(aC * rho_coke, z_m, axis=1)
            M_feed  = m_dot * t_s; m_vr0 = _m_vr0_from_any()
            M_total = np.maximum(M_feed + m_vr0, 1e-12)
            Y_vr = 100.0 * m_vr   / M_total; Y_ck = 100.0 * m_coke / M_total
            Y_ds = 100.0 * np.maximum(M_total - m_vr - m_coke, 0.0) / M_total
            mask = M_feed >= (start_frac * m_vr0)
            return (t_s[mask] / 3600.0), Y_vr[mask], Y_ds[mask], Y_ck[mask]

        # 1) Профили
        tab1 = ttk.Frame(notebook); notebook.add(tab1, text="Профили")
        fig1 = Figure(figsize=(7,10), dpi=80)
        ax_vr = fig1.add_subplot(3,1,1); ax_ds = fig1.add_subplot(3,1,2, sharey=ax_vr); ax_ck = fig1.add_subplot(3,1,3, sharey=ax_vr)
        panels = [(ax_vr,"VR","aR"),(ax_ds,"Дистилляты","aD"),(ax_ck,"Кокс","aC")]
        times_h = snaps.get('t_h', [])
        if len(times_h) > 0:
            step = max(1, len(times_h)//6)
            for ax, title, key in panels:
                arr = snaps.get(key, []);
                if np.size(arr)==0: continue
                for j in range(0, len(times_h), step): ax.plot(np.asarray(arr)[j], z_cm, label=f"{times_h[j]:.1f} ч")
                ax.set_title(title); ax.set_xlabel("Доля объёма"); ax.grid(True, alpha=0.3)
            ax_vr.set_ylabel("Высота (см)"); ax_vr.legend(fontsize=8, loc='best')
        fig1.tight_layout(); FigureCanvasTkAgg(fig1, tab1).get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 2) Температура
        tabT = ttk.Frame(notebook); notebook.add(tabT, text="Температура")
        figT = Figure(figsize=(8,5), dpi=80); axT = figT.add_subplot(111)
        if len(times_h)>0 and np.size(snaps.get("T",[]))>0:
            step=max(1,len(times_h)//6)
            for j in range(0,len(times_h),step): axT.plot(np.asarray(snaps["T"])[j], z_cm, label=f"{times_h[j]:.1f} ч")
            axT.set_title("Эволюция температурного профиля"); axT.set_xlabel("T (°C)"); axT.set_ylabel("Высота (см)")
            axT.grid(True, alpha=0.3); axT.legend(fontsize=8, loc="best")
        else: axT.text(0.5,0.5,"Недостаточно данных",ha="center",va="center",transform=axT.transAxes)
        figT.tight_layout(); FigureCanvasTkAgg(figT, tabT).get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 3) Пористость
        tabP = ttk.Frame(notebook); notebook.add(tabP, text="Пористость")
        figP = Figure(figsize=(8,5), dpi=80); axP = figP.add_subplot(111)
        if len(times_h)>0 and np.size(snaps.get("aC",[]))>0:
            step=max(1,len(times_h)//6)
            for j in range(0,len(times_h),step): axP.plot(1.0-np.asarray(snaps["aC"])[j], z_cm, label=f"{times_h[j]:.1f} ч")
            axP.set_title("Эволюция пористости (γ = 1 − aC)"); axP.set_xlabel("Пористость γ"); axP.set_xlim(0,1)
            axP.set_ylabel("Высота (см)"); axP.grid(True, alpha=0.3); axP.legend(fontsize=8, loc="best")
        else: axP.text(0.5,0.5,"Недостаточно данных",ha="center",va="center",transform=axP.transAxes)
        figP.tight_layout(); FigureCanvasTkAgg(figP, tabP).get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 4) Рост
        tab2 = ttk.Frame(notebook); notebook.add(tab2, text="Рост")
        fig2 = Figure(figsize=(8,4), dpi=80); ax2 = fig2.add_subplot(111)
        ax2.plot(results['growth'].get('t_h', []), results['growth'].get('H_cm', []), 'b-', linewidth=2, label='Эквивалентная')
        thr = float(results.get('extra', {}).get('front_thr', 0.10))
        try:
            t_h = np.asarray(cont["t_s"], dtype=float)/3600.0; aC_hist = np.asarray(cont["aC"])
            dz = float(z_m[1]-z_m[0]) if len(z_m)>1 else 0.0
            H_front=[]
            for aC in aC_hist:
                idx=np.where(aC>thr)[0]; H=((idx[-1]+1)*dz if idx.size else 0.0)*100.0; H_front.append(H)
            ax2.plot(t_h, H_front, 'r--', linewidth=1.5, label=f'Фронт ({int(thr*100)}%)')
        except Exception: pass
        ax2.set_title("Рост коксового слоя"); ax2.set_xlabel("Время (ч)"); ax2.set_ylabel("Высота (см)")
        ax2.grid(True, alpha=0.3); ax2.legend()
        fig2.tight_layout(); FigureCanvasTkAgg(fig2, tab2).get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 5) Выходы фаз
        tab3 = ttk.Frame(notebook); notebook.add(tab3, text="Выходы фаз")
        fig3 = Figure(figsize=(8,4), dpi=80); ax3 = fig3.add_subplot(111)
        try:
            th, Y_vr, Y_d, Y_c = _phase_series(start_frac=0.02)
            ax3.plot(th, Y_vr, label="VR (остаток)"); ax3.plot(th, Y_d, label="Дистилляты (накоплено)"); ax3.plot(th, Y_c, label="Кокс"); ax3.legend()
        except Exception: ax3.text(0.5,0.5,"Недостаточно данных контуров",ha="center",va="center",transform=ax3.transAxes)
        ax3.set_xlabel("Время (ч)"); ax3.set_ylabel("Выход, % (баланс масс)"); ax3.set_title("Выходы фаз во времени"); ax3.grid(True, alpha=0.3)
        fig3.tight_layout(); FigureCanvasTkAgg(fig3, tab3).get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 6) Тепловые карты
        tab4 = ttk.Frame(notebook); notebook.add(tab4, text="Тепловые карты")
        fig4 = Figure(figsize=(10,8), dpi=80); axs=[fig4.add_subplot(2,2,i+1) for i in range(4)]
        try:
            t_h = np.asarray(cont["t_s"], dtype=float)/3600.0
            aR=np.asarray(cont["aR"]); aD=np.asarray(cont["aD"]); aC=np.asarray(cont["aC"]); TT=np.asarray(cont["T"])
            ex=[t_h.min(), t_h.max(), z_cm.min(), z_cm.max()]
            for ax,data,title in zip(axs,[aR,aD,aC,TT],["VR","Дистилляты","Кокс","T (°C)"]):
                im=ax.imshow(data.T, origin="lower", aspect="auto", extent=ex, cmap="viridis")
                ax.set_title(title); ax.set_xlabel("Время (ч)"); fig4.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            axs[0].set_ylabel("Высота (см)"); axs[2].set_ylabel("Высота (см)")
        except Exception: axs[0].text(0.5,0.5,"Недостаточно данных",ha="center",va="center",transform=axs[0].transAxes)
        fig4.tight_layout(); FigureCanvasTkAgg(fig4, tab4).get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 7) Финальные выходы
        tab5 = ttk.Frame(notebook); notebook.add(tab5, text="Финальные выходы")
        fig5 = Figure(figsize=(6,4), dpi=80); ax5 = fig5.add_subplot(111)
        try:
            _, Y_vr, Y_ds, Y_c = _phase_series(start_frac=0.02)
            vals=[float(Y_vr[-1]), float(Y_ds[-1]), float(Y_c[-1])]
            bars=ax5.bar(["VR","Дистилляты","Кокс"], vals)
            for b,v in zip(bars,vals): ax5.text(b.get_x()+b.get_width()/2, v+0.5, f"{v:.2f}%", ha="center")
            ax5.set_ylabel("% (баланс масс)"); ax5.set_title("Финальные выходы фаз"); ax5.set_ylim(0,105); ax5.grid(axis="y", alpha=0.2)
        except Exception as e: ax5.text(0.5,0.5,f"Недостаточно данных\n{str(e)}",ha="center",va="center",transform=ax5.transAxes)
        fig5.tight_layout(); FigureCanvasTkAgg(fig5, tab5).get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def update_results_tab(self, results):
        import numpy as np
        for w in self.results_frame.winfo_children(): w.destroy()

        text = tk.Text(self.results_frame, wrap=tk.WORD, font=("Courier", 10))
        scroll = ttk.Scrollbar(self.results_frame, command=text.yview)
        text.configure(yscrollcommand=scroll.set)

        H_cm = float(results['H_bed_m'] * 100.0);
        H_front_cm = float(results['H_front_m'] * 100.0)
        Y_feed = float(results.get('yield_pct', 0.0))
        T_avg = float(results.get('T_avg_C', 0.0));
        por_avg = float(results.get('porosity_avg', 0.0))

        report = "=" * 60 + "\nРЕЗУЛЬТАТЫ СИМУЛЯЦИИ\n" + "=" * 60 + "\n\n"
        report += (f"Финальные показатели:\n"
                   f"  Высота слоя (экв.):  {H_cm:.2f} см\n"
                   f"  Высота фронта:       {H_front_cm:.2f} см\n"
                   f"  Выход кокса (по подаче): {Y_feed:.2f} %\n"
                   f"  Средняя температура: {T_avg:.1f} °C\n"
                   f"  Средняя пористость:  {por_avg:.3f}\n")

        # ---- Балансный выход + массы (кг) ----
        try:
            z_m = np.asarray(results['z'], dtype=float)
            meta = results.get("meta", {})
            A = float(meta.get("A_m2", 0.0))
            m_dot = float(meta.get("m_dot_kg_s", 0.0))
            rho_vr = float(meta.get("rho_vr", 0.0))
            rho_c = float(meta.get("rho_coke", 0.0))

            final = results.get("final", {})
            aR_fin = np.asarray(final.get("aR", []))
            aC_fin = np.asarray(final.get("aC", []))

            # время окончания (сек): сначала контуры, затем рост, затем снимки
            cont = results.get("contours", {})
            t_s = np.asarray(cont.get("t_s", []), dtype=float)
            if t_s.size == 0:
                t_h_g = np.asarray(results.get("growth", {}).get("t_h", []), dtype=float)
                if t_h_g.size > 0: t_s = np.array([t_h_g[-1] * 3600.0])
            if t_s.size == 0:
                t_h_sn = np.asarray(results.get("snapshots", {}).get("t_h", []), dtype=float)
                if t_h_sn.size > 0: t_s = np.array([t_h_sn[-1] * 3600.0])
            t_end_s = float(t_s[-1]) if t_s.size else 0.0

            # начальная масса VR (берём из meta если есть; иначе восстанавливаем по первому снимку/контуру)
            m_vr0 = float(meta.get("m_vr0", 0.0))
            if m_vr0 <= 0.0:
                aR0 = None
                sr = results.get("snapshots", {}).get("aR", [])
                if np.size(sr) > 0:
                    sr = np.asarray(sr)
                    aR0 = sr[0] if sr.ndim == 2 else None
                if aR0 is None:
                    ar = cont.get("aR", None)
                    if ar is not None and np.size(ar) > 0:
                        ar = np.asarray(ar)
                        aR0 = ar[0] if ar.ndim == 2 else None
                if aR0 is not None:
                    m_vr0 = float(A * np.trapz(aR0 * rho_vr, z_m))

            # массы на конце
            M_feed = m_dot * t_end_s
            M_vr_end = float(A * np.trapz(aR_fin * rho_vr, z_m)) if aR_fin.size else 0.0
            M_coke = float(A * np.trapz(aC_fin * rho_c, z_m)) if aC_fin.size else 0.0
            M_dist = max(M_feed + m_vr0 - M_vr_end - M_coke, 0.0)

            # балансный выход по финальным массам
            Y_c_bal = 100.0 * M_coke / max(M_feed + m_vr0, 1e-12)
            report += f"  Балансный выход кокса: {Y_c_bal:.2f} %\n"

            # печать баланса масс
            report += ("\nМассовый баланс (кг):\n"
                       f"  VR(начало)   = {m_vr0:.3f}\n"
                       f"  Подача       = {M_feed:.3f}\n"
                       f"  VR(конец)    = {M_vr_end:.3f}\n"
                       f"  Кокс         = {M_coke:.3f}\n"
                       f"  Дистилляты   = {M_dist:.3f}\n")
        except Exception:
            pass

        report += "\n"
        extra = results.get("extra", {})
        if extra and extra.get("t_fill_h") is not None:
            report += ("Момент заполнения барабана:\n"
                       f"  t_fill = {extra['t_fill_h']:.2f} ч\n"
                       f"  Выход к t_fill:      {extra['Y_at_fill_pct']:.2f} % "
                       f"(порог aC>{extra.get('front_thr', 0.10) * 100:.0f}%)\n\n")

        final = results.get('final', {})
        if isinstance(final, dict) and len(final) > 0:
            z = np.asarray(results['z'], dtype=float) * 100.0
            report += "Финальный профиль (выборочные точки):\n"
            report += f"{'z, см':>8} {'T, °C':>8} {'α_VR':>8} {'α_coke':>8} {'ε':>8}\n" + "-" * 44 + "\n"
            for i in [0, len(z) // 4, len(z) // 2, 3 * len(z) // 4, len(z) - 1]:
                if 0 <= i < len(z):
                    eps = 1.0 - float(final['aC'][i])
                    report += f"{z[i]:>8.1f} {float(final['T'][i]):>8.1f} {float(final['aR'][i]):>8.3f} {float(final['aC'][i]):>8.3f} {eps:>8.3f}\n"

        text.insert(tk.END, report);
        text.config(state=tk.DISABLED)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5);
        scroll.pack(side=tk.RIGHT, fill=tk.Y)

    # ---------- save/load/export ----------
    def save_config(self):
        filename = filedialog.asksaveasfilename(title="Сохранить конфигурацию", defaultextension=".json",
                                                filetypes=[("JSON files","*.json"),("All files","*.*")])
        if not filename: return
        config = {name: frame.get_values() for name, frame in self.param_frames.items()}
        config.setdefault("time", {}).update({"stop_on_full": bool(self.stop_on_full_var.get()),
                                              "front_thr_pct": float(self.front_thr_var.get())})
        with open(filename,'w',encoding='utf-8') as f: json.dump(config,f,indent=2,ensure_ascii=False)
        self.status_var.set(f"Конфигурация сохранена: {Path(filename).name}")

    def load_config(self):
        filename = filedialog.askopenfilename(title="Загрузить конфигурацию",
                                              filetypes=[("JSON files","*.json"),("All files","*.*")])
        if not filename: return
        try:
            with open(filename,'r',encoding='utf-8') as f: config=json.load(f)
            for name, values in config.items():
                if name in self.param_frames and isinstance(values, dict): self.param_frames[name].set_values(values)
            t = config.get("time", {})
            if "stop_on_full" in t: self.stop_on_full_var.set(bool(t["stop_on_full"]))
            if "front_thr_pct" in t: self.front_thr_var.set(float(t["front_thr_pct"]))
            self.status_var.set(f"Конфигурация загружена: {Path(filename).name}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка загрузки конфигурации:\n{e}")

    def export_results(self):
        if not self.results:
            messagebox.showwarning("Предупреждение", "Нет результатов для экспорта")
            return

        folder = filedialog.askdirectory(title="Выберите папку для экспорта")
        if not folder:
            return

        try:
            from src.visualization import render_all_ru
            output_dir = Path(folder)
            render_all_ru(self.results, output_dir, exp_h_cm=48.34, exp_y_pct=36.57)
            messagebox.showinfo("Готово", f"Результаты экспортированы в:\n{output_dir}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка экспорта:\n{e}")

    def show_about(self):
        messagebox.showinfo("О программе",
                            "Симулятор замедленного коксования v1.0\n\n"
                            "Моделирование коксования вакуумного остатка в периодическом реакторе.\n© 2025")

    # ---------- Руководство ----------
    def open_user_guide(self):
        win = tk.Toplevel(self.root)
        win.title("Руководство пользователя")
        win.geometry("900x650")

        nb = ttk.Notebook(win); nb.pack(fill=tk.BOTH, expand=True)

        def _add_tab(title, text):
            fr = ttk.Frame(nb); nb.add(fr, text=title)
            txt = tk.Text(fr, wrap=tk.WORD, font=("Segoe UI", 10), padx=8, pady=8)
            txt.insert(tk.END, text.strip()+"\n"); txt.config(state=tk.DISABLED)
            y = ttk.Scrollbar(fr, command=txt.yview); txt.configure(yscrollcommand=y.set)
            txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); y.pack(side=tk.RIGHT, fill=tk.Y)

        _add_tab("Быстрый старт", USER_GUIDE_SECTIONS["quickstart"])
        _add_tab("Параметры входа", USER_GUIDE_SECTIONS["inlet"])
        _add_tab("Теплообмен", USER_GUIDE_SECTIONS["walls"])
        _add_tab("Материалы", USER_GUIDE_SECTIONS["materials"])
        _add_tab("Кинетика", USER_GUIDE_SECTIONS["kinetics"])
        _add_tab("Расчёт / стоп", USER_GUIDE_SECTIONS["run_stop"])
        _add_tab("Графики и метрики", USER_GUIDE_SECTIONS["plots"])
        _add_tab("Типовые режимы", USER_GUIDE_SECTIONS["presets"])
        _add_tab("Советы", USER_GUIDE_SECTIONS["tips"])
        _add_tab("Ограничения", USER_GUIDE_SECTIONS["limits"])
        _add_tab("Как устроена модель", USER_GUIDE_SECTIONS["theory"])
        _add_tab("Устойчивость и скорость", USER_GUIDE_SECTIONS["stability"])
        _add_tab("Калибровка по данным", USER_GUIDE_SECTIONS["calibration"])
        _add_tab("Диагностика", USER_GUIDE_SECTIONS["troubleshooting"])
        _add_tab("FAQ", USER_GUIDE_SECTIONS["faq"])


# ------------------------------ entry ------------------------------
def main():
    root = tk.Tk()
    app = CokingSimulatorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
