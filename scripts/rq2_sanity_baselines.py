#!/usr/bin/env python3
"""
RQ2 sanity baselines on the frozen checkpoints and the N explained flows of xai_explanations.npz.

1. Temporal Masking (same protocol as notebook 04: top-k valid packets imputed with 0.5) driven by
   - post-hoc SHAP |phi_pkt| and intrinsic attention (Rollout / MHAN),
   - a uniform profile (random ranking of the valid packets, averaged over N_RANDOM draws),
   - a positional profile (first-k valid packets).
2. Padding-aware concordance with SHAP (Spearman rho, Cosine Similarity) of
   intrinsic attention, the uniform profile and the positional profile 1/t.

Outputs (results/metrics/):
    rq2_temporal_masking_curves.csv   Delta Macro-F1 [p.p.] for k = 1..5
    rq2_alignment_baselines.csv       mean Spearman rho / Cosine Similarity vs SHAP

Usage: python scripts/rq2_sanity_baselines.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils import seed_everything  # noqa: E402
from util.xai_utils import compute_temporal_alignment  # noqa: E402
from xai.model_loader import load_checkpoint  # noqa: E402

RESULTS_DIR = ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
K_STEPS = [1, 2, 3, 4, 5]
N_RANDOM = 20  # random rankings averaged for the uniform baseline

RUNS = {  # network: (run id, npz key, intrinsic attention name or None)
    "Lopez17CNN": ("S2_R01_Lopez17CNN_multi_10pkts_seed0", "lopez", None),
    "BiflowTransformer": ("S2_R02_BiflowTransformer_multi_10pkts_seed0", "biflow", "Rollout"),
    "MedDefenderTransformer": ("S2_R03_MedDefender_multi_10pkts_seed0", "meddef", "MHAN"),
}


def mask_top_k_packets(x, ranking, k, mask_val=0.5):
    """Identical to notebook 04: impute all fields of the top-k packets; padding stays 0.0."""
    x_masked = x.clone()
    for i in range(x.size(0)):
        for t in np.argsort(-ranking[i])[:k]:
            if x_masked[i, 0, t].abs().sum() > 0:
                x_masked[i, 0, t, :] = mask_val
    return x_masked


def main():
    seed_everything(0)
    data = np.load(METRICS_DIR / "xai_explanations.npz", allow_pickle=True)
    x = torch.tensor(data["x_xai"])
    y = data["y_xai"]
    valid = ~np.all(data["x_xai"].squeeze(1) == 0, axis=-1)  # [N, T]
    only_valid = lambda r: np.where(valid, r, -np.inf)  # padding never enters the top-k
    t_idx = np.arange(1, valid.shape[1] + 1, dtype=float)
    positional = np.where(valid, 1.0 / t_idx, 0.0)
    uniform = valid.astype(float)
    rng = np.random.default_rng(0)

    curves, align = [], []
    for net, (run, key, attn_name) in RUNS.items():
        model = load_checkpoint(exp_name=f"cic-iomt_scratch_{run}", results_dir=str(RESULTS_DIR),
                                network=net, device="cpu")[0]
        model.eval()

        def macro_f1(xx):
            with torch.no_grad():
                out = model(xx)
                out = out[0] if isinstance(out, (list, tuple)) else out
            return f1_score(y, out.argmax(dim=-1).numpy(), average="macro") * 100

        base = macro_f1(x)
        shap_pkt = np.abs(data[f"shap_{key}_pkt"])
        rankings = {"SHAP": only_valid(shap_pkt), "Positional (first-k)": only_valid(positional)}
        profiles = [("Uniform", uniform), ("Positional (1/t)", positional)]
        if attn_name:
            attn_pkt = data[f"attn_{key}_pkt"].astype(float)
            rankings[attn_name] = only_valid(attn_pkt)
            profiles.insert(0, (attn_name, attn_pkt))
        for name, r in rankings.items():
            d = [macro_f1(mask_top_k_packets(x, r, k)) - base for k in K_STEPS]
            curves.append({"Model": net, "Ranking": name, **{f"k={k}": v for k, v in zip(K_STEPS, d)},
                           **{f"std_k={k}": 0.0 for k in K_STEPS}})

        rand = np.array([[macro_f1(mask_top_k_packets(x, only_valid(rng.random(valid.shape)), k)) - base
                          for k in K_STEPS] for _ in range(N_RANDOM)])
        curves.append({"Model": net, "Ranking": "Uniform (random)",
                       **{f"k={k}": v for k, v in zip(K_STEPS, rand.mean(0))},
                       **{f"std_k={k}": v for k, v in zip(K_STEPS, rand.std(0))}})

        for name, prof in profiles:
            a = compute_temporal_alignment(shap_pkt, prof, valid)
            align.append({"Model": net, "Profile": name,
                          # a constant profile has no ranking: Spearman is undefined
                          "Spearman_rho": np.nan if name == "Uniform" else a["spearman_rho_mean"],
                          "Cosine_Similarity": a["cosine_similarity_mean"]})
        print(f"[OK] {net}: baseline Macro-F1 on the {len(y)} explained flows = {base:.2f}%")

    df_c, df_a = pd.DataFrame(curves), pd.DataFrame(align)
    df_c.to_csv(METRICS_DIR / "rq2_temporal_masking_curves.csv", index=False)
    df_a.to_csv(METRICS_DIR / "rq2_alignment_baselines.csv", index=False)
    print(df_c.round(2).to_string(index=False))
    print(df_a.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
