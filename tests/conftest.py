"""Pytest session setup.

macOS/arm64 gotcha: torch and xgboost each bundle their own OpenMP runtime. When both are used in
one process their thread pools clash, producing segfaults or deadlocks. Forcing OpenMP to a single
thread (and allowing the duplicate runtime) before either library is imported sidesteps the clash.
This must run before any test module imports torch/xgboost — a root conftest is imported first, so
this is the right place. The same guard is applied at the top of ``scripts/train_baseline.py``,
which also uses both libraries in one process.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
