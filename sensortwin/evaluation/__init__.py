"""Evaluation: metrics, calibration, robustness, plots (roadmap v0.2-v0.5).

v0.2 ships: ``metrics.py`` (accuracy, macro/weighted-F1, per-class P/R, one-vs-rest AUROC,
confusion matrix), ``calibration.py`` (ECE, multiclass Brier, reliability curve), ``robustness.py``
(missing-channel probe), ``plots.py`` (confusion / reliability / per-class-F1 figures).
Macro-F1 is the headline metric (classes are imbalanced).

Planned (v0.5): full robustness suite (noise / domain-shift / short-window / rare-event),
calibration study, and ``interpretability.py`` (attention / occlusion / saliency).

Submodules require the ``ml`` extra (sklearn/matplotlib) and are not imported here so the core
package stays importable without them.
"""
