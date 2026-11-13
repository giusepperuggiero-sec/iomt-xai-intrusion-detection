**Project**: Explainable AI (XAI) for Early Intrusion Detection in IoMT Networks (DIETI — University of Naples Federico II)  
**Dataset**: CICIoMT2024 (13-class multiclass traffic, $T=10$ packets, $F=6$ header fields, `seed=0`)

---

## 1. Benchmark Classification Performance & Single-Thread Edge CPU Efficiency
| Run ID | Model | Macro-F1 (%) | Balanced Acc (%) | Best Epoch | Val Loss Min | Training Time (min) | Params | Size (MB) | Latency CPU (ms) | Latency P95 (ms) | Throughput (fps) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `S2_R01` | `Lopez17CNN` | 84.76 | 86.07 | 50 | 0.0607 | 100.3 | 787,741 | 3.00 | 1.03 | 1.63 | 975.2 |
| `S2_R02` | `BiflowTransformer` | 74.40 | 76.91 | 48 | 0.0993 | 124.8 | 425,981 | 1.62 | 2.48 | 4.40 | 404.0 |
| `S2_R03` | `MedDefenderTransformer` | 74.46 | 76.91 | 34 | 0.0997 | 107.8 | 353,085 | 1.35 | 2.10 | 3.40 | 475.7 |

---

## 2. Formal XAI Audit: By-Design Attention vs. Post-Hoc GradientSHAP ($N=957$ Explained Flows)
| Model | Explanation Method | Spearman $\rho$ (vs SHAP) | Cosine Sim (vs SHAP) | Latency per Flow | Edge Computational Cost | Temporal Masking (Top-3 $\Delta$ F1) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Lopez17CNN` | GradientSHAP (Post-Hoc) | N/A (Baseline) | N/A (Baseline) | 1.16 s | Prohibitive ($O(M \cdot \text{samples})$) | -76.3% |
| `BiflowTransformer` | GradientSHAP (Post-Hoc) | 1.000 | 1.000 | 1.40 s | Prohibitive ($O(M \cdot \text{samples})$) | -68.7% |
| `BiflowTransformer` | Attention Rollout (By-Design) | 0.222 | 0.876 | 2.48 ms | Zero-cost (embedded in forward pass) | -67.2% |
| `MedDefenderTransformer` | GradientSHAP (Post-Hoc) | 1.000 | 1.000 | 1.32 s | Prohibitive ($O(M \cdot \text{samples})$) | -59.0% |
| `MedDefenderTransformer` | MHAN Attention (By-Design) | 0.290 | 0.771 | 2.10 ms | Zero-cost (embedded in forward pass) | -55.0% |

---

## 3. Causal Validation on Frozen Models (MoRF vs. LeRF Perturbation)
| Metric / Step | Description | `Lopez17CNN` | `BiflowTransformer` | `MedDefenderTransformer` |
| :--- | :--- | :--- | :--- | :--- |
| MoRF Step $k=1$ (top field) | Inference-time occlusion of top field (`Lopez17CNN`: `DIR`, `BiflowTransformer`: `TTL`, `MedDefenderTransformer`: `TTL`) | 64.59% ($\Delta$ F1 -20.17%, $\Delta$ BA -17.80%) | 2.13% ($\Delta$ F1 -72.28%, $\Delta$ BA -69.06%) | 26.64% ($\Delta$ F1 -47.83%, $\Delta$ BA -39.71%) |
| MoRF Step $k=2$ | Joint neutralization of top 2 header fields | 1.40% ($\Delta$ F1 -83.36%, $\Delta$ BA -79.74%) | 4.41% ($\Delta$ F1 -70.00%, $\Delta$ BA -69.12%) | 11.18% ($\Delta$ F1 -63.29%, $\Delta$ BA -56.27%) |
| MoRF Step $k=3$ | Joint neutralization of top 3 header fields | 1.46% ($\Delta$ F1 -83.30%, $\Delta$ BA -78.96%) | 0.22% ($\Delta$ F1 -74.18%, $\Delta$ BA -69.22%) | 0.62% ($\Delta$ F1 -73.84%, $\Delta$ BA -73.44%) |
| MoRF Step $k=5$ | Joint neutralization of 5 out of 6 header fields | 1.36% ($\Delta$ F1 -83.40%, $\Delta$ BA -79.74%) | 0.22% ($\Delta$ F1 -74.19%, $\Delta$ BA -69.22%) | 1.00% ($\Delta$ F1 -73.46%, $\Delta$ BA -70.31%) |
| **AUDC MoRF ($\downarrow$)** | Area Under Degradation Curve (Most Relevant First) | 22.44 | 8.86 | 15.43 |
| **AUDC LeRF ($\uparrow$)** | Area Under Degradation Curve (Least Relevant First) | 21.29 | 9.18 | 14.44 |

