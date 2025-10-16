#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Точка входа для GUI приложения симулятора коксования

Запуск:
    python run_gui.py

или в PyCharm:
    - Правый клик -> Run 'run_gui'
"""

import sys
from pathlib import Path

# Добавляем корень проекта в путь (если запускаем не из корня)
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Импортируем и запускаем GUI
from gui.app import main

if __name__ == "__main__":
    print("Запуск GUI симулятора коксования...")
    main()