"""Training and self-supervised pretraining (roadmap v0.2-v0.4).

v0.2 ships ``loop.py``: a minimal reusable supervised loop (``train_model`` with early stopping on
val macro-F1, inverse-frequency ``class_weights``, ``predict_proba``) used by the deep baselines and
reused for the transformer in v0.3.

v0.4 adds ``pretrain.py``: masked-patch self-supervised pretraining (``pretrain_model``, MSE on
masked patches) plus ``freeze_encoder``/``transfer_encoder`` for the label-efficiency fine-tuning
arms. Submodules require the ``ml`` extra (torch/sklearn) and are not imported here.
"""
