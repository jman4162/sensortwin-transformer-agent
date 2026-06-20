"""Models (roadmap v0.2-v0.3).

v0.2 ships the baselines: classical (LogReg/RandomForest/XGBoost + IsolationForest anomaly) on
engineered features in ``baselines.py``, plus deep ``SensorCNN`` (``cnn.py``) and ``SensorLSTM``
(``lstm.py``) on raw ``[B, C, T]`` signals.

Planned (v0.3+): ``transformer.py`` (``SensorPatchTST``: per-channel patching + channel embeddings
+ transformer encoder + attention pooling), ``heads.py``. See CLAUDE.md for the architecture
contract (input ``[B, C, T]``, ``d_model=128``, ``num_layers=4``, ``num_heads=4``).

Note: ``baselines.py``/``cnn.py``/``lstm.py`` require the ``ml`` extra (sklearn/xgboost/torch) and
are intentionally not imported here, so the core package imports without those dependencies.
"""
