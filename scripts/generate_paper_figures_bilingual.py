#!/usr/bin/env python3
"""
Generates publication-grade IEEE vector PDF and 300-DPI PNG figures in both English and Italian
directly from the certified experimental outputs in results/metrics/ and results/*/results/.

Every plot is an autonomous single-column figure (3.5 in wide).

Usage:
    python scripts/generate_paper_figures_bilingual.py                       # default documents/Paper dirs
    python scripts/generate_paper_figures_bilingual.py --en DIR --it DIR     # custom output dirs
"""

import argparse
import glob
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix

# Set global IEEE style
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = ROOT / "results" / "metrics"
ORIG_FIG_DIR = ROOT / "results" / "figures"

DIR_EN = ROOT / "documents" / "Paper" / "Paper inglese" / "figures"
DIR_IT = ROOT / "documents" / "Paper" / "Paper italiano" / "figures"
DIR_MAIN = ROOT / "documents" / "Paper" / "Paper" / "figures"

COL_W = 3.5  # IEEE single-column width [in]

field_names = ["PL", "IAT", "DIR", "WIN", "TTL", "FLG"]
pkt_indices = np.arange(1, 11)
_classes_candidates = [
    ROOT / "data" / "classes.txt",
    ROOT.parent / "Dataset" / "classes.txt",
]
_classes_file = next((p for p in _classes_candidates if p.exists()), None)
class_names = (
    _classes_file.read_text(encoding="utf-8").split()
    if _classes_file is not None
    else [
        "benign", "dos-icmp", "dos-mqtt-conn", "dos-mqtt-malf", "dos-mqtt-publ", "dos-tcp", "dos-tcp-syn",
        "dos-udp", "infog-scan-host", "infog-scan-os", "infog-scan-port", "infog-scan-vuln", "mitm-arpspoof",
    ]
)

MODELS = [
    # key, run id, display name, color
    ("lopez", "S2_R01", "Lopez17CNN", "#d95f02"),
    ("biflow", "S2_R02", "BiflowTransformer", "#7570b3"),
    ("meddef", "S2_R03", "MedDefenderTransformer", "#1b9e77"),
]

K_COLS = [f"k={k}" for k in range(1, 6)]
npz_data = None
df_morf = None
cells = {}
cell_vmax = 0.0
field_med = {}
valid_pkt = None
t_shap = {}
t_attn = {}
df_tm = None


def share(v):
    m = v.mean(axis=0)
    return m / m.sum() * 100.0


def load_experiment_data():
    """Load precomputed XAI artifacts lazily so importing the script never fails without outputs."""
    global npz_data, df_morf, cells, cell_vmax, field_med, valid_pkt, t_shap, t_attn, df_tm
    npz_data = np.load(METRICS_DIR / "xai_explanations.npz", allow_pickle=True)
    df_morf = pd.read_csv(METRICS_DIR / "morf_lerf_curves.csv")

    # Cell heatmaps: median signed attribution across flows, shape 10x6
    cells = {k: np.median(npz_data[f"shap_{k}_cell"], axis=0) for k, *_ in MODELS}
    cell_vmax = max(np.abs(c).max() for c in cells.values())

    # Global field medians, shape (6,)
    field_med = {k: np.median(np.abs(npz_data[f"shap_{k}_field"]), axis=0) for k, *_ in MODELS}

    # Temporal relevance share [%]: mean over flows on valid (non-padded) packets only
    valid_pkt = ~np.all(npz_data["x_xai"].squeeze(1) == 0, axis=-1)
    t_shap = {k: share(np.abs(npz_data[f"shap_{k}_pkt"]) * valid_pkt) for k, *_ in MODELS}
    t_attn = {k: share(npz_data[f"attn_{k}_pkt"] * valid_pkt) for k in ("biflow", "meddef")}

    # Temporal Masking curves (Delta Macro-F1 [p.p.], k=1..5) from scripts/rq2_sanity_baselines.py
    df_tm = pd.read_csv(METRICS_DIR / "rq2_temporal_masking_curves.csv").set_index(["Model", "Ranking"])


def L(lang, en, it):
    return en if lang == "en" else it


def save(fig, out_dir, name):
    for ext in ["pdf", "png"]:
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def build_confusion_matrices(lang, out_dir):
    """One autonomous figure per model: class indices on both axes, own colorbar, index->name legend."""
    for key, run_id, name, _ in MODELS:
        p_file = glob.glob(str(ROOT / "results" / f"*{run_id}*" / "results" / "outputs_targets*.parquet"))[0]
        df_p = pd.read_parquet(p_file)
        scores = np.vstack(df_p["Scores"].iloc[0][0])
        targets = np.asarray(df_p["Targets"].iloc[0][0], dtype=int)
        cm = confusion_matrix(targets, scores.argmax(axis=1), labels=list(range(13)), normalize="true") * 100.0

        fig, ax = plt.subplots(figsize=(COL_W, 4.1))
        sns.heatmap(cm, ax=ax, cmap="Blues", vmin=0.0, vmax=100.0, square=True,
                    xticklabels=list(range(13)), yticklabels=list(range(13)),
                    cbar_kws={"label": L(lang, "Classification Rate [%]", "Tasso di Classificazione [%]"),
                              "shrink": 0.72, "pad": 0.03})
        ax.tick_params(axis="y", rotation=0)
        ax.set_xlabel(L(lang, r"Predicted Class, $\hat{y}$", r"Classe Predetta, $\hat{y}$"))
        ax.set_ylabel(L(lang, "True Class, $y$", "Classe Reale, $y$"))
        handles = [Patch(color="none", label=f"{i}: {c}") for i, c in enumerate(class_names)]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, -0.15), ncol=3,
                  fontsize=6.3, frameon=True, handlelength=0, handletextpad=0, columnspacing=1.0,
                  title=L(lang, "Class index: name", "Indice classe: nome"), title_fontsize=6.5)
        save(fig, out_dir, f"fig_cm_{key}")


def build_rq1(lang, out_dir):
    """One cell heatmap per model (shared colour scale, own colorbar) + global field bar chart."""
    for key, _, name, _ in MODELS:
        fig, ax = plt.subplots(figsize=(COL_W, 2.9))
        sns.heatmap(cells[key], ax=ax, cmap="coolwarm", center=0.0, vmin=-cell_vmax, vmax=cell_vmax,
                    xticklabels=field_names, yticklabels=list(pkt_indices), linewidths=0.4, linecolor="white",
                    cbar_kws={"label": L(lang, r"Signed SHAP $\tilde{\phi}_{t,f}$ [a.u.]",
                                         r"SHAP con Segno $\tilde{\phi}_{t,f}$ [a.u.]")})
        ax.tick_params(axis="y", rotation=0)
        ax.set_xlabel(L(lang, "Header Field, $f$", "Campo di Header, $f$"))
        ax.set_ylabel(L(lang, "Packet Index, $t$", "Indice Pacchetto, $t$"))
        save(fig, out_dir, f"fig_rq1_heatmap_{key}")

    fig, ax = plt.subplots(figsize=(COL_W, 2.4))
    x = np.arange(len(field_names))
    w = 0.26
    for off, (key, _, name, color) in zip((-w, 0, w), MODELS):
        ax.bar(x + off, field_med[key], width=w, label=name, color=color, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(field_names)
    ax.set_xlabel(L(lang, "Header Field, $f$", "Campo di Header, $f$"))
    ax.set_ylabel(L(lang, r"Median $|\tilde{\phi}_{\mathrm{field}}(f)|$ [a.u.]",
                    r"Mediana $|\tilde{\phi}_{\mathrm{field}}(f)|$ [a.u.]"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(frameon=True, loc="upper left")
    save(fig, out_dir, "fig_rq1_field_importance")


def build_rq2(lang, out_dir):
    # Temporal alignment
    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    ax.plot(pkt_indices, t_shap["lopez"], "o--", color="#d95f02", markersize=3.5, linewidth=1.0, label="Lopez17CNN SHAP")
    ax.plot(pkt_indices, t_shap["biflow"], "s-", color="#7570b3", markersize=3.5, linewidth=1.1, label="BiflowTransformer SHAP")
    ax.plot(pkt_indices, t_attn["biflow"], "^--", color="#e7298a", markersize=3.5, linewidth=1.1, label="BiflowTransformer Rollout")
    ax.plot(pkt_indices, t_shap["meddef"], "D-", color="#1b9e77", markersize=3.5, linewidth=1.1, label="MedDefender SHAP")
    ax.plot(pkt_indices, t_attn["meddef"], "v--", color="#66a61e", markersize=3.5, linewidth=1.1, label="MedDefender MHAN")
    ax.set_xticks(pkt_indices)
    ax.set_xlabel(L(lang, "Packet Index, $t$", "Indice Pacchetto, $t$"))
    ax.set_ylabel(L(lang, "Mean Relevance Share [%]", "Quota Media di Rilevanza [%]"))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", frameon=True)
    save(fig, out_dir, "fig_rq2_temporal_alignment")

    # Temporal masking
    k = np.arange(1, 6)
    fig, ax = plt.subplots(figsize=(COL_W, 2.7))
    series = [  # model, ranking, label suffix, style, color
        ("BiflowTransformer", "SHAP", "SHAP", "s-", "#7570b3"),
        ("BiflowTransformer", "Rollout", "Rollout", "^--", "#e7298a"),
        ("BiflowTransformer", "Uniform (random)", L(lang, "Random", "Casuale"), ":", "#7570b3"),
        ("MedDefenderTransformer", "SHAP", "SHAP", "D-", "#1b9e77"),
        ("MedDefenderTransformer", "MHAN", "MHAN", "v--", "#66a61e"),
        ("MedDefenderTransformer", "Uniform (random)", L(lang, "Random", "Casuale"), ":", "#1b9e77"),
    ]
    for model, ranking, suffix, style, color in series:
        row = df_tm.loc[(model, ranking)]
        mean = row[K_COLS].to_numpy(float)
        short = "Biflow" if model.startswith("Biflow") else "MedDef"
        ax.plot(k, mean, style, color=color, markersize=3.5, linewidth=1.1, label=f"{short} ({suffix})")
        if ranking.startswith("Uniform"):  # mean +/- std over the random draws
            std = row[[f"std_{c}" for c in K_COLS]].to_numpy(float)
            ax.fill_between(k, mean - std, mean + std, color=color, alpha=0.2, linewidth=0)
    ax.set_xticks(k)
    ax.set_xlabel(L(lang, "Masked Top-$k$ Packets", "Top-$k$ Pacchetti Mascherati"))
    ax.set_ylabel(r"$\Delta$ Macro-F1 [p.p.]")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", frameon=True)
    save(fig, out_dir, "fig_rq2_temporal_masking")


def build_rq3(lang, out_dir):
    colors = {m[2]: m[3] for m in MODELS}
    short = {"Lopez17CNN": "Lopez17CNN", "BiflowTransformer": "BiflowTransf.", "MedDefenderTransformer": "MedDefender"}
    m_steps = np.arange(0, 6)

    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    for _, row in df_morf.iterrows():
        vals = [row[f"m={i}"] for i in range(6)]
        if row["Curve"] == "MoRF_F1":
            ax.plot(m_steps, vals, "o-", color=colors[row["Model"]], markersize=3.5, linewidth=1.2,
                    label=f"{short[row['Model']]} MoRF")
        elif row["Curve"] == "LeRF_F1":
            ax.plot(m_steps, vals, "s--", color=colors[row["Model"]], markersize=3, linewidth=1.0, alpha=0.75,
                    label=f"{short[row['Model']]} LeRF")
    ax.axvline(1, color="gray", linestyle=":", linewidth=1.0)
    ax.set_xticks(m_steps)
    ax.set_xlabel(L(lang, "Masked Fields, $m$", "Campi Mascherati, $m$"))
    ax.set_ylabel("Macro-F1 [%]")
    ax.grid(True, linestyle="--", alpha=0.4)
    # Single-column legend in the upper right: for m >= 3 all curves stay below ~10%, so it covers no data
    ax.legend(loc="upper right", frameon=True, ncol=1, fontsize=6.3)
    save(fig, out_dir, "fig_rq3_morf_lerf")

    # AUDC from the same curves (trapezoidal rule, Eq. AUDC)
    audc = {(r["Model"], r["Curve"]): np.trapezoid([r[f"m={i}"] for i in range(6)]) / 5 for _, r in df_morf.iterrows()}
    names = [m[2] for m in MODELS]
    x = np.arange(3)
    w = 0.34
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    ax.bar(x - w / 2, [audc[(n, "MoRF_F1")] for n in names], width=w, color="#e41a1c", edgecolor="black",
           linewidth=0.4, label=r"MoRF $\downarrow$")
    ax.bar(x + w / 2, [audc[(n, "LeRF_F1")] for n in names], width=w, color="#4daf4a", edgecolor="black",
           linewidth=0.4, label=r"LeRF $\uparrow$")
    ax.set_xticks(x)
    ax.set_xticklabels([short[n] for n in names])
    ax.set_xlabel(L(lang, "Architecture", "Architettura"))
    ax.set_ylabel("AUDC [p.p.]")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", frameon=True)
    save(fig, out_dir, "fig_rq3_audc")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--en", type=Path, help="English figures output dir")
    ap.add_argument("--it", type=Path, help="Italian figures output dir")
    args = ap.parse_args()
    load_experiment_data()
    targets = ([("en", args.en)] if args.en else []) + ([("it", args.it)] if args.it else [])
    if not targets:
        if (ROOT / "documents" / "Paper").exists():
            targets = [("en", DIR_EN), ("it", DIR_IT), ("en", DIR_MAIN)]
            # Copy notebook figures as backup assets into the default paper dirs
            for _, d in targets:
                d.mkdir(parents=True, exist_ok=True)
                for f in ORIG_FIG_DIR.glob("*.*"):
                    if f.name != ".gitkeep":
                        shutil.copy2(f, d / f.name)
        else:
            targets = [("en", ORIG_FIG_DIR)]
    for lang, d in targets:
        d.mkdir(parents=True, exist_ok=True)
        build_confusion_matrices(lang, d)
        build_rq1(lang, d)
        build_rq2(lang, d)
        build_rq3(lang, d)
    print("Successfully generated bilingual IEEE figures!")
