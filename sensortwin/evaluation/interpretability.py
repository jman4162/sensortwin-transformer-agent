"""Interpretability diagnostics for SensorPatchTST (roadmap v0.5, spec §12.5).

Three lenses, in increasing order of faithfulness:
  * **attention map** — the pooling weights over (channel, patch). A *diagnostic only*: attention is
    not explanation (Jain & Wallace 2019; Wiegreffe & Pinter 2019). Not causal evidence.
  * **occlusion** — zero a channel / time-window and measure the drop in the true-class probability.
    Perturbation-based, so more faithful than attention.
  * **integrated gradients** — axiomatic gradient attribution (Sundararajan et al. 2017).

Because the generator records each event's ``affected_channels`` and window, we can do something a
real benchmark cannot: *quantitatively* check whether the saliency lands on the known event region
(``localization_score``), against a random-saliency baseline.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from sensortwin.models.transformer import SensorPatchTST


def attention_map(model: SensorPatchTST, X: np.ndarray, device: str = "cpu") -> np.ndarray | None:
    """Per-(channel, patch) attention weights ``[B, C, N]``; None if the model mean-pools."""
    pool = model.pool
    if pool is None:
        return None
    model = model.to(device).eval()
    x = torch.from_numpy(np.ascontiguousarray(X)).float().to(device)
    with torch.no_grad():
        model(x)
    weights = pool.last_weights  # [B, C*N], channel-major / patch-minor
    if weights is None:
        return None
    n_patches = (X.shape[2] - model.patch_len) // model.stride + 1
    return weights.cpu().numpy().reshape(X.shape[0], model.in_channels, n_patches)


def attention_to_time(attn: np.ndarray, T: int, patch_len: int, stride: int) -> np.ndarray:
    """Broadcast a per-patch attention map ``[C, N]`` onto a per-timestep map ``[C, T]``.

    Each patch contributes its weight to the timesteps it covers; overlaps are averaged.
    """
    C, n_patches = attn.shape
    out = np.zeros((C, T), dtype=np.float64)
    cover = np.zeros(T, dtype=np.float64)
    for p in range(n_patches):
        s, e = p * stride, min(p * stride + patch_len, T)
        out[:, s:e] += attn[:, p : p + 1]
        cover[s:e] += 1
    cover[cover == 0] = 1.0
    return out / cover


def occlusion_importance(
    predict_proba_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    targets: np.ndarray,
    *,
    n_windows: int = 8,
) -> dict[str, np.ndarray]:
    """Mean drop in the true-class probability when each channel / time-window is zeroed.

    Higher drop = more important to the prediction. Faithful (the model is actually re-run), unlike
    attention. Returns ``{"channel_drop": [C], "window_drop": [n_windows]}``.
    """
    n, C, T = X.shape
    idx = np.arange(n)
    base = predict_proba_fn(X)[idx, targets]

    channel_drop = np.empty(C)
    for c in range(C):
        Xc = X.copy()
        Xc[:, c, :] = 0.0
        channel_drop[c] = float((base - predict_proba_fn(Xc)[idx, targets]).mean())

    window_drop = np.empty(n_windows)
    edges = np.linspace(0, T, n_windows + 1, dtype=int)
    for w in range(n_windows):
        Xw = X.copy()
        Xw[:, :, edges[w] : edges[w + 1]] = 0.0
        window_drop[w] = float((base - predict_proba_fn(Xw)[idx, targets]).mean())
    return {"channel_drop": channel_drop, "window_drop": window_drop}


def integrated_gradients(
    model: SensorPatchTST,
    x: np.ndarray,
    target: int,
    *,
    steps: int = 32,
    baseline: float = 0.0,
    device: str = "cpu",
) -> np.ndarray:
    """Integrated Gradients attribution ``[C, T]`` for one sample ``x [C, T]`` and a target class.

    Attribution_i = (x_i - baseline_i) * mean_k d f_target / d x  along the straight-line path from
    baseline to x (Sundararajan et al. 2017). Returns signed attribution per (channel, timestep).
    """
    model = model.to(device).eval()
    x_t = torch.from_numpy(np.ascontiguousarray(x)).float().to(device)
    base = torch.full_like(x_t, float(baseline))
    grads = torch.zeros_like(x_t)
    for k in range(1, steps + 1):
        point = (base + (k / steps) * (x_t - base)).unsqueeze(0).requires_grad_(True)
        logit = model(point)[0, target]
        (grad,) = torch.autograd.grad(logit, point)
        grads += grad.squeeze(0)
    attribution = (x_t - base) * grads / steps
    return attribution.detach().cpu().numpy()


def localization_score(saliency: np.ndarray, event_meta: dict[str, Any]) -> float | None:
    """Fraction of total saliency magnitude that falls inside the known event region.

    The event region is ``affected_channels`` x ``[start, start+duration]`` from the generator
    metadata. Returns None for events with no localized region (e.g. ``normal``, ``compound_fault``,
    or whole-window degradations). Compare to the random baseline ``region_area / total_area``.
    """
    channels = event_meta.get("affected_channels") or []
    start, duration = event_meta.get("start"), event_meta.get("duration")
    C, T = saliency.shape
    if not channels or start is None or duration is None or duration >= T:
        return None
    mass = np.abs(saliency)
    total = mass.sum()
    if total <= 0:
        return None
    region = mass[np.ix_(channels, range(start, min(start + duration, T)))].sum()
    return float(region / total)


def random_localization_baseline(
    event_meta: dict[str, Any], n_channels: int, T: int
) -> float | None:
    """Expected ``localization_score`` for uniform-random saliency = region area / total area."""
    channels = event_meta.get("affected_channels") or []
    start, duration = event_meta.get("start"), event_meta.get("duration")
    if not channels or start is None or duration is None or duration >= T:
        return None
    region_area = len(channels) * min(duration, T - start)
    return float(region_area / (n_channels * T))
