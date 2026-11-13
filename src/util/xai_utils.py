"""
xai_utils.py — XAI Utilities and Helper Functions for IoMT NIDS
==============================================================
Implements the formal explainability framework of Alfredo Nascita et al.
(IEEE COMST 2025 & IEEE TNSM 2021):
  - normalize_shap_by_instance: Eq. (5) instance-wise normalization preserving algebraic sign
  - pool_afa_multiscale: multi-granularity AFA aggregation (Cell, Field, Packet)
  - compute_temporal_alignment: temporal agreement (Spearman rho and Cosine Similarity)
  - compute_audc: Area Under the Degradation Curve for MoRF / LeRF
"""

import warnings
from typing import Tuple, Dict

import numpy as np
from scipy.stats import spearmanr


def normalize_shap_by_instance(shap_values: np.ndarray) -> np.ndarray:
    """Normalize local SHAP explanations instance-wise while preserving algebraic sign (+/-),
    following Eq. (5) from Alfredo Nascita et al. (IEEE TNSM 2021):
        \\tilde{\\phi}_{t,f}(x) = \\phi_{t,f}(x) / \\sum_{t'=1}^{10} \\sum_{f'=1}^{6} |\\phi_{t',f'}(x)|

    The sum of absolute values for each flow equals 1.0.

    Args:
        shap_values: ndarray of shape (N, T, F) or (T, F) containing signed Shapley values.

    Returns:
        Normalized ndarray with the same shape as the input.
    """
    is_2d = False
    arr = np.asarray(shap_values, dtype=np.float64)
    if arr.ndim == 2:
        is_2d = True
        arr = arr[np.newaxis, ...]

    denom = np.sum(np.abs(arr), axis=(1, 2), keepdims=True)
    denom = np.where(denom == 0, 1e-12, denom)
    normalized = arr / denom

    if is_2d:
        return normalized[0]
    return normalized


def pool_afa_multiscale(shap_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Additive Feature Attribution (AFA) decomposition across 3 granularities:
    1. Cell: [N, T, F] (default 10x6) - full spatio-temporal attribution map
    2. Field: [N, F] (summed across T packets)
    3. Packet: [N, T] (summed across 6 header fields)

    Args:
        shap_norm: instance-normalized ndarray of shape (N, T, F).

    Returns:
        tuple (cell_view, field_view, packet_view)
    """
    arr = np.asarray(shap_norm)
    phi_field = np.sum(arr, axis=1)  # [N, F]
    phi_pkt = np.sum(arr, axis=2)    # [N, T]
    return arr, phi_field, phi_pkt


def compute_temporal_alignment(shap_pkt: np.ndarray, attention_pkt: np.ndarray, valid: np.ndarray = None) -> Dict[str, float]:
    """Compute agreement metrics between the SHAP temporal profile (T=10)
    and intrinsic attention weights (Attention Rollout or MHAN):
    - Spearman rho (rank agreement, [-1, 1])
    - Cosine Similarity (geometric alignment in R^10, [-1, 1])

    Args:
        shap_pkt: ndarray [N, T] or [T] with SHAP packet importances.
        attention_pkt: ndarray [N, T] or [T] with intrinsic attention weights.
        valid: boolean ndarray [N, T] marking real packets (semantic padding where
               torch.all(x == 0, dim=-1) is excluded from comparison). None = all valid.

    Returns:
        dict containing mean and standard deviation of spearman_rho and cosine_similarity.
    """
    s = np.asarray(shap_pkt)
    a = np.asarray(attention_pkt)

    if s.ndim == 1:
        s = s[np.newaxis, :]
    if a.ndim == 1:
        a = a[np.newaxis, :]

    s_mag = np.abs(s)
    a_mag = np.abs(a)
    v_all = np.ones(s_mag.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool).reshape(s_mag.shape)

    rhos = []
    cosines = []

    for i in range(len(s)):
        si, ai = s_mag[i][v_all[i]], a_mag[i][v_all[i]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rho, _ = spearmanr(si, ai)
            if np.isnan(rho):
                rho = 0.0
            rhos.append(float(rho))

        denom = float(np.linalg.norm(si) * np.linalg.norm(ai))
        if denom > 1e-12:
            cos = float(np.dot(si, ai) / denom)
        else:
            cos = 0.0
        cosines.append(cos)

    return {
        'spearman_rho_mean': float(np.mean(rhos)),
        'spearman_rho_std': float(np.std(rhos)),
        'cosine_similarity_mean': float(np.mean(cosines)),
        'cosine_similarity_std': float(np.std(cosines))
    }


def compute_audc(steps: np.ndarray, metric_values: np.ndarray) -> float:
    """Compute the Area Under the Degradation Curve (AUDC) for MoRF or LeRF curves.

    Args:
        steps: array of removed field fractions or counts (e.g., [0, 1, 2, 3, 4, 5]).
        metric_values: array of metric values (e.g., Macro-F1 or Balanced Accuracy).

    Returns:
        Normalized scalar area computed using the trapezoidal rule.
    """
    x = np.asarray(steps, dtype=np.float64)
    y = np.asarray(metric_values, dtype=np.float64)
    if x.max() > 1.0:
        x = (x - x.min()) / (x.max() - x.min())
    if hasattr(np, 'trapezoid'):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))
