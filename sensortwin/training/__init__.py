"""Training and self-supervised pretraining (roadmap v0.2-v0.4).

v0.2 ships ``loop.py``: a minimal reusable supervised loop (``train_model`` with early stopping on
val macro-F1, inverse-frequency ``class_weights``, ``predict_proba``) used by the deep baselines and
reused for the transformer in v0.3.

Planned (v0.4): ``pretrain.py`` (masked-patch reconstruction), ``losses.py`` (focal), schedulers.
``loop.py`` requires the ``ml`` extra (torch/sklearn) and is not imported here.
"""
