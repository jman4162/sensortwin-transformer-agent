"""Evaluation: metrics, calibration, robustness, plots (roadmap v0.2-v0.5).

v0.2 ships: ``metrics.py`` (accuracy, macro/weighted-F1, per-class P/R, one-vs-rest AUROC,
confusion matrix), ``calibration.py`` (ECE, multiclass Brier, reliability curve), ``robustness.py``
(missing-channel probe), ``plots.py`` (confusion / reliability / per-class-F1 figures).
Macro-F1 is the headline metric (classes are imbalanced).

v0.5 adds: noise + short-window severity sweeps in ``robustness.py`` (domain shift is generation-
based), temperature scaling in ``calibration.py``, and ``interpretability.py`` (attention map,
occlusion, integrated gradients, and a ``localization_score`` faithfulness check vs the generator's
ground-truth event region). Attention is a diagnostic, not an explanation.

Submodules require the ``ml`` extra (sklearn/matplotlib/scipy) and are not imported here so the core
package stays importable without them.
"""
