"""Models (roadmap v0.2-v0.3).

Planned: ``baselines.py`` (LogReg/RF/GBM on features), ``cnn.py``, ``lstm.py``,
``transformer.py`` (``SensorPatchTST``: per-channel patching + channel embeddings + transformer
encoder + attention pooling), ``heads.py`` (classification / masked-reconstruction / severity).
Not yet implemented — see CLAUDE.md for the architecture contract (input ``[B, C, T]``,
``d_model=128``, ``num_layers=4``, ``num_heads=4``).
"""
