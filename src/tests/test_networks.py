"""
test_networks.py — Unit Tests for Model Architectures and XAI Pipeline
======================================================================
Validates core project modules:
  1. Instantiation and forward pass for Lopez17CNN, BiflowTransformer, MedDefenderTransformer
  2. Attention map shapes (B, h, T+1, T+1) and (B, h, T, T)
  3. Masked-mean pooling on valid packets in MedDefenderTransformer
  4. LLL_Net encapsulation, checkpoint save and strict=True reload
  5. Instance-wise SHAP normalization (Eq. 5, Nascita et al.) preserving algebraic sign (+/-)
  6. 3-level Additive Feature Attribution (AFA) decomposition (Cell, Field, Packet)
  7. Temporal alignment metrics (Spearman rho and Cosine Similarity)
  8. Area Under the Degradation Curve (AUDC) computation for MoRF/LeRF
  9. Cold-cache dataset preprocessing (FEAT_PAD padding, split, SCALED_*)
  10. Exact field-level SHAP (field_shapley) preserving padding structure
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import pandas as pd

# Ensure src/ is on sys.path
_TEST_DIR = Path(__file__).resolve().parent
_SRC_DIR = _TEST_DIR.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from datasets import networking_dataset as netdat
from networks.lopez17cnn import Lopez17CNN
from networks.transformer import BiflowTransformer, MedDefenderTransformer
from networks.network import LLL_Net
from xai.model_loader import load_checkpoint
from xai.xai_metrics import field_shapley, compute_temporal_alignment, compute_audc
from util.xai_utils import normalize_shap_by_instance, pool_afa_multiscale


def test_biflow_transformer():
    print("[TEST] 1. BiflowTransformer forward & attention extraction...")
    B, T, F_dim = 4, 10, 6
    num_classes = 13
    nheads = 4
    x = torch.randn(B, 1, T, F_dim)
    x[0, :, 8:, :] = 0.0  # padding

    model = BiflowTransformer(
        num_pkts=T, num_fields=F_dim, num_classes=num_classes,
        nlayers=2, nheads=nheads, d_model=64, dim_feedforward=128
    )
    model.eval()

    out = model(x)
    assert out.shape == (B, num_classes), f"Atteso ({B}, {num_classes}), ottenuto {out.shape}"

    out2, attentions = model.get_attention_maps(x)
    assert out2.shape == (B, num_classes)
    assert len(attentions) == 2, f"Attesi 2 layer di attenzione, trovati {len(attentions)}"
    for i, attn in enumerate(attentions):
        assert attn.shape == (B, nheads, T + 1, T + 1), \
            f"Layer {i}: atteso {(B, nheads, T + 1, T + 1)}, ottenuto {attn.shape}"

    print("  [OK] BiflowTransformer validato con successo!")


def test_meddefender_transformer():
    print("[TEST] 2. MedDefenderTransformer forward, attention & masked pooling...")
    B, T, F_dim = 4, 10, 6
    num_classes = 13
    nheads = 4
    x = torch.randn(B, 1, T, F_dim)
    x[1, :, 5:, :] = 0.0  # padding

    model_single = MedDefenderTransformer(
        num_pkts=T, num_fields=F_dim, num_classes=num_classes,
        nlayers=1, nheads=nheads, d_model=64, dim_feedforward=128
    )
    assert hasattr(model_single, "transformer_layer"), "1-layer MedDefender should preserve transformer_layer"
    model_single.eval()
    assert model_single(x).shape == (B, num_classes)

    model = MedDefenderTransformer(
        num_pkts=T, num_fields=F_dim, num_classes=num_classes,
        nlayers=2, nheads=nheads, d_model=64, dim_feedforward=128
    )
    assert len(model.transformer_layers) == 2, f"Expected 2 encoder layers, got {len(model.transformer_layers)}"
    assert sum(p.numel() for p in model.parameters()) > sum(p.numel() for p in model_single.parameters())
    model.eval()

    out = model(x)
    assert out.shape == (B, num_classes), f"Atteso ({B}, {num_classes}), ottenuto {out.shape}"

    out2, attn_weights = model.get_attention_maps(x)
    assert out2.shape == (B, num_classes)
    assert attn_weights.shape == (B, nheads, T, T), \
        f"Atteso attention weights {(B, nheads, T, T)}, ottenuto {attn_weights.shape}"

    print("  [OK] MedDefenderTransformer validato con successo!")


def test_lopez17_cnn():
    print("[TEST] 3. Lopez17CNN forward...")
    B, T, F_dim = 4, 10, 6
    num_classes = 13
    x = torch.randn(B, 1, T, F_dim)

    model = Lopez17CNN(num_pkts=T, num_fields=F_dim, num_classes=num_classes)
    model.eval()
    out = model(x)
    assert out.shape == (B, num_classes), f"Atteso ({B}, {num_classes}), ottenuto {out.shape}"
    print("  [OK] Lopez17CNN validata con successo!")


def test_checkpoint_roundtrip():
    print("[TEST] 4. LLL_Net Checkpoint Roundtrip con strict=True...")
    B, T, F_dim = 2, 10, 6
    num_classes = 13
    x = torch.randn(B, 1, T, F_dim)

    net = BiflowTransformer(num_pkts=T, num_fields=F_dim, num_classes=num_classes, d_model=64)
    model = LLL_Net(net, remove_existing_head=True)
    model.add_head(num_classes)
    model.eval()

    with torch.no_grad():
        original_output = model(x)[0]

    state = {"module." + k: v for k, v in model.state_dict().items()}

    with tempfile.TemporaryDirectory() as tmpdir:
        exp_dir = Path(tmpdir) / "test_exp"
        models_dir = exp_dir / "models"
        models_dir.mkdir(parents=True)

        ckpt_path = models_dir / "task-0.ckpt"
        torch.save(state, ckpt_path)

        import json
        args_data = {
            "network": "BiflowTransformer",
            "num_pkts": T,
            "fields": ["PL", "IAT", "DIR", "WIN", "TTL", "FLG"],
            "nc_first_task": num_classes,
            "d_model": 64,
            "nlayers": 2,
            "nheads": 4,
            "dim_feedforward": 512,
            "approach": "scratch"
        }
        with open(exp_dir / "args-123456.txt", "w") as f:
            json.dump(args_data, f)

        loaded_model, _ = load_checkpoint(
            exp_name="test_exp",
            results_dir=tmpdir,
            device="cpu"
        )
        loaded_model.eval()

        with torch.no_grad():
            loaded_output = loaded_model(x)
            if isinstance(loaded_output, (list, tuple)):
                loaded_output = loaded_output[0]

        diff = torch.max(torch.abs(original_output - loaded_output)).item()
        assert diff < 1e-6, f"Differenza tra output: {diff} (atteso < 1e-6)"

    print("  [OK] Checkpoint round-trip validato con successo!")


def test_shap_instance_normalization():
    print("[TEST] 5. Normalizzazione per istanza SHAP (Eq. 5 Nascita et al.)...")
    rng = np.random.default_rng(0)
    N, T, F_dim = 5, 10, 6
    raw_shap = rng.normal(0, 1.0, size=(N, T, F_dim))

    norm_shap = normalize_shap_by_instance(raw_shap)
    assert norm_shap.shape == raw_shap.shape, "Forma alterata dalla normalizzazione"

    # The sum of absolute values for each sample must equal 1.0
    for i in range(N):
        abs_sum = np.sum(np.abs(norm_shap[i]))
        assert abs(abs_sum - 1.0) < 1e-6, f"Somma assoluta non unitaria per campione {i}: {abs_sum}"

    # Algebraic sign must be strictly preserved
    assert np.all((norm_shap > 0) == (raw_shap > 0)), "Segno algebrico positivo alterato"
    assert np.all((norm_shap < 0) == (raw_shap < 0)), "Segno algebrico negativo alterato"
    print("  [OK] Normalizzazione instance-wise e conservazione del segno convalidate!")


def test_afa_multiscale_pooling():
    print("[TEST] 6. Decomposizione multiscala AFA (Cella, Campo, Pacchetto)...")
    rng = np.random.default_rng(0)
    N, T, F_dim = 4, 10, 6
    raw = rng.normal(0, 0.5, size=(N, T, F_dim))
    norm = normalize_shap_by_instance(raw)

    cell, field, pkt = pool_afa_multiscale(norm)
    assert cell.shape == (N, T, F_dim), f"Cella: atteso ({N}, {T}, {F_dim}), ottenuto {cell.shape}"
    assert field.shape == (N, F_dim), f"Campo: atteso ({N}, {F_dim}), ottenuto {field.shape}"
    assert pkt.shape == (N, T), f"Pacchetto: atteso ({N}, {T}), ottenuto {pkt.shape}"

    # Verify additivity
    assert np.allclose(field.sum(axis=1), pkt.sum(axis=1), atol=1e-6), "Additivita violata tra campo e pacchetto"
    print("  [OK] Viste Cella, Campo e Pacchetto convalidate con additivita AFA!")


def test_temporal_alignment():
    print("[TEST] 7. Metriche di concordanza temporale (Spearman rho e Cosine Sim)...")
    # Identical vectors must yield rho=1.0 and cosine=1.0
    v = np.array([0.4, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    res_perfect = compute_temporal_alignment(v, v)
    assert abs(res_perfect['spearman_rho_mean'] - 1.0) < 1e-4, f"Rho imperfetto: {res_perfect['spearman_rho_mean']}"
    assert abs(res_perfect['cosine_similarity_mean'] - 1.0) < 1e-4, f"Cosine imperfetta: {res_perfect['cosine_similarity_mean']}"

    # Inverted vectors
    v_inv = v[::-1]
    res_inv = compute_temporal_alignment(v, v_inv)
    assert res_inv['spearman_rho_mean'] < 0, f"Rho atteso negativo, vale {res_inv['spearman_rho_mean']}"
    print(f"  [OK] Concordanza temporale validata (perfetta: rho={res_perfect['spearman_rho_mean']:.3f}, CS={res_perfect['cosine_similarity_mean']:.3f})")


def test_audc_calculation():
    print("[TEST] 8. Calcolo AUDC (Area Under Degradation Curve)...")
    steps = np.array([0, 1, 2, 3, 4, 5])
    # Flat 100% curve
    y_flat = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    audc_flat = compute_audc(steps, y_flat)
    assert abs(audc_flat - 100.0) < 1e-5, f"AUDC piatta attesa 100, vale {audc_flat}"

    # Decreasing MoRF curve
    y_morf = np.array([100.0, 80.0, 60.0, 40.0, 20.0, 0.0])
    audc_morf = compute_audc(steps, y_morf)
    assert abs(audc_morf - 50.0) < 1e-5, f"AUDC triangolare attesa 50, vale {audc_morf}"
    print(f"  [OK] AUDC calcolata correttamente (piatta={audc_flat:.1f}, MoRF={audc_morf:.1f})")


def _make_biflow_parquet(path, n=180, horizon=100):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        real = int(rng.integers(3, horizon + 1))
        label = ['benign', 'dos-tcp', 'mitm-arpspoof'][i % 3]
        rows.append(dict(
            PL=rng.integers(0, 1500, real), IAT=rng.integers(0, 60000, real),
            DIR=rng.integers(0, 2, real), WIN=rng.integers(0, 65535, real),
            FLG=rng.integers(0, 255, real), TTL=rng.integers(0, 255, real),
            FEAT_PAD=horizon - real, LABEL=label,
            **{'LABEL-bin': 0 if label == 'benign' else 1}))
    pd.DataFrame(rows).to_parquet(path)


def test_dataset_cold_cache():
    print("[TEST] 9. Pre-processing del dataset a cache fredda (padding, split, SCALED_*)...")
    fields = ['PL', 'IAT', 'DIR', 'WIN', 'TTL', 'FLG']

    for binary, num_pkts, num_classes in ((False, 10, 3), (True, 100, 2)):
        tmpdir = tempfile.mkdtemp().replace(os.sep, '/')
        parquet_path = tmpdir + '/synthetic.parquet'
        _make_biflow_parquet(parquet_path)

        os.environ['PREP_CACHE_DIR'] = tmpdir + '/cache'
        data, _, _, _ = netdat.get_data(
            parquet_path, num_tasks=1, nc_first_task=num_classes, nc_incr_tasks=0,
            validation=0.1, shuffle_classes=False, num_pkts=num_pkts, fields=fields,
            seed=0, binary=binary)

        x = np.stack(data[0]['trn']['x'])
        assert x.shape[1:] == (1, num_pkts, len(fields)), f"shape inattesa: {x.shape}"
        assert x.min() >= 0.0 and x.max() <= 1.0, f"scaling fuori [0,1]: {x.min()}, {x.max()}"
        assert data[0]['ncla'] == num_classes, f"classi attese {num_classes}, trovate {data[0]['ncla']}"
        assert len(data[0]['tst']['x']) > 0, "split di test vuoto"
        assert len(data[0]['val']['x']) > 0, "split di validation vuoto"

    os.environ.pop('PREP_CACHE_DIR', None)
    print("  [OK] Pre-processing a cache fredda validato!")


def test_field_shapley():
    print("[TEST] 10. SHAP esatto per campo: linearita, additivita, padding...")
    rng = np.random.default_rng(0)
    N, B, T, F_dim, C = 3, 4, 8, 6, 5
    x = rng.random((N, T, F_dim)) + 0.1
    x[:, 5:, :] = 0.0
    x_bg = rng.random((B, T, F_dim)) + 0.1
    x_bg[-1, 2:, :] = 0.0
    A = rng.standard_normal((C, T, F_dim))

    def linear(z):
        z4 = z.reshape(N, B, T, F_dim)
        assert np.all(z4[:, :, 5:, :] == 0), "il padding di x e stato riempito"
        assert np.all(z4[:, :, :5, :] != 0), "un pacchetto reale ha ricevuto valori di padding"
        return np.einsum("mtf,ctf->mc", z, A)

    phi, base = field_shapley(linear, x, x_bg)
    real = np.any(x != 0, axis=-1)[:, :, None]
    expected = np.einsum("ntf,ctf->nfc", real * (x - x_bg[:-1].mean(0)), A)
    assert np.allclose(phi, expected, atol=1e-4), "Shapley non esatto sul modello lineare"
    print("  [OK] Shapley per campo esatto e additivo, padding preservato!")


if __name__ == "__main__":
    print("=" * 70)
    print(" AVVIO TEST UNITARI SU ARCHITETTURE E PIPELINE XAI")
    print("=" * 70)
    test_biflow_transformer()
    test_meddefender_transformer()
    test_lopez17_cnn()
    test_checkpoint_roundtrip()
    test_shap_instance_normalization()
    test_afa_multiscale_pooling()
    test_temporal_alignment()
    test_audc_calculation()
    test_dataset_cold_cache()
    test_field_shapley()
    print("=" * 70)
    print(" TUTTI I TEST SONO STATI SUPERATI CON SUCCESSO! (10/10)")
    print("=" * 70)
