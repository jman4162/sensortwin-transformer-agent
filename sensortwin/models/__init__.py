"""Models (roadmap v0.2-v0.3).

v0.2 ships the baselines: classical (LogReg/RandomForest/XGBoost + IsolationForest anomaly) on
engineered features in ``baselines.py``, plus deep ``SensorCNN`` (``cnn.py``) and ``SensorLSTM``
(``lstm.py``) on raw ``[B, C, T]`` signals.

v0.3 adds ``transformer.py`` (``SensorPatchTST``: per-channel patching + learned channel
embeddings + pre-norm transformer encoder + attention pooling; input ``[B, C, T]``,
``d_model=128``, ``num_layers=4``, ``num_heads=4``). Planned (v0.4+): ``heads.py`` (aux heads).

Note: ``baselines.py``/``cnn.py``/``lstm.py`` require the ``ml`` extra (sklearn/xgboost/torch) and
are intentionally not imported here, so the core package imports without those dependencies.
"""
