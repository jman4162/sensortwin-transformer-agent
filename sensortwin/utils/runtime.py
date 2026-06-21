"""Runtime environment helpers.

The torch + xgboost OpenMP clash (both bundle an OpenMP runtime; their thread pools segfault or
deadlock when used in one process) is **macOS-specific**. Pinning ``OMP_NUM_THREADS=1`` everywhere
also single-threads NumPy/SciPy/sklearn feature extraction, which needlessly slows Linux/Colab runs.
So we gate the workaround to Darwin and leave other platforms with their default thread pools.

``configure_omp`` only touches ``os.environ`` and imports nothing heavy, so it is safe to call at
the very top of a script *before* importing torch or xgboost (the env vars must be set first).
"""

from __future__ import annotations

import os
import platform


def configure_omp() -> None:
    """On macOS only, set the single-threaded OpenMP guard before torch/xgboost import."""
    if platform.system() == "Darwin":
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
