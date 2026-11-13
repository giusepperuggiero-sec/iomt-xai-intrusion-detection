"""
xai_metrics.py — Explainability Metrics and Algorithms for IoMT NIDS
===================================================================
Implements the quantitative metrics used in the XAI evaluation pipeline:
  1. field_shapley: exact interventional Shapley values at the header-field level (F=6)
  2. compute_temporal_alignment: temporal agreement between SHAP and intrinsic attention
     (Spearman rank correlation rho and Cosine Similarity in R^10)
  3. compute_audc: Area Under the Degradation Curve (AUDC) for progressive
     MoRF vs. LeRF curves on frozen models
"""

from math import factorial
import warnings
from typing import Callable, Tuple

import numpy as np
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# 1. Exact field-level SHAP (treating the 6 header fields as players)
# ---------------------------------------------------------------------------

def field_shapley(
    model_fn: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    x_bg: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact interventional SHAP values treating header fields (F columns) as players.

    With F=6 there are 2^6 = 64 coalitions, allowing exact combinatorial computation.
    A field absent from a coalition takes values from background flows.
    Real packets never contain zero across valid features (0 = semantic padding); each
    flow x uses background flows at least as long as x to preserve padding structure.

    Args:
        model_fn: Callable mapping (M, T, F) -> probabilities / logits (M, C).
        x:        Flows to explain (N, T, F).
        x_bg:     Background flows (B, T, F).

    Returns:
        (phi, base): phi (N, F, C), base (N, C), satisfying phi.sum(1) == f(x) - base.
    """
    N, T, F_dim = x.shape
    real = np.any(x != 0, axis=-1)                   # (N, T)
    bg_real = np.any(x_bg != 0, axis=-1)             # (B, T)
    wgt = (bg_real.sum(1)[None] >= real.sum(1)[:, None]).astype(np.float64)  # (N, B)
    longest = (bg_real.sum(1) == bg_real.sum(1).max()).astype(np.float64)
    wgt[wgt.sum(1) == 0] = longest
    wgt /= wgt.sum(1, keepdims=True)
    fill = real[:, None, :, None] & bg_real[None, :, :, None]  # (N, B, T, 1)
    v = []
    for k in range(1 << F_dim):
        keep = np.array([(k >> f) & 1 for f in range(F_dim)], dtype=bool)
        xm = np.where(keep | ~fill, x[:, None], x_bg[None])  # (N, B, T, F)
        p = model_fn(xm.reshape(-1, T, F_dim).astype(np.float32))
        v.append(np.einsum("nbc,nb->nc", p.reshape(N, len(x_bg), -1), wgt))
    v = np.stack(v, axis=1)  # (N, 2^F, C)

    w = [factorial(s) * factorial(F_dim - s - 1) / factorial(F_dim) for s in range(F_dim)]
    phi = np.zeros((N, F_dim, v.shape[-1]))
    for f in range(F_dim):
        for k in range(1 << F_dim):
            if not (k >> f) & 1:
                phi[:, f] += w[bin(k).count("1")] * (v[:, k | (1 << f)] - v[:, k])
    return phi, v[:, 0]


# ---------------------------------------------------------------------------
# 2. Temporal Agreement and AUDC (imported from util.xai_utils for consistency)
# ---------------------------------------------------------------------------
from util.xai_utils import compute_temporal_alignment, compute_audc

