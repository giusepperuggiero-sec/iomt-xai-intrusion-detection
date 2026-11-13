#!/usr/bin/env python3
"""
scripts/profile_latency.py — Single-Thread CPU Inference Profiling for Edge IoMT
================================================================================
Measures per-flow inference latency, throughput, parameter count, and estimated
checkpoint footprint on single-thread CPU across the project architectures.
Loads trained checkpoints strictly via xai.model_loader.load_checkpoint; raises
FileNotFoundError if missing. Passing --no-checkpoint profiles only the untrained
architecture.

Usage:
    python scripts/profile_latency.py --all
    python scripts/profile_latency.py --network BiflowTransformer --num-pkts 10
    python scripts/profile_latency.py --network Lopez17CNN --num-pkts 10
    python scripts/profile_latency.py --network MedDefenderTransformer --num-pkts 10
    python scripts/profile_latency.py --all --no-checkpoint
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_SRC_DIR = _REPO_ROOT / "src"
_RESULTS_DIR = _REPO_ROOT / "results"
_METRICS_DIR = _RESULTS_DIR / "metrics"

RUN_IDS = {
    "Lopez17CNN": "S2_R01_Lopez17CNN_multi_10pkts_seed0",
    "BiflowTransformer": "S2_R02_BiflowTransformer_multi_10pkts_seed0",
    "MedDefenderTransformer": "S2_R03_MedDefender_multi_10pkts_seed0",
}

for d in [_RESULTS_DIR, _METRICS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

for p in [str(_SRC_DIR), str(_REPO_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from networks.lopez17cnn import Lopez17CNN
from networks.transformer import BiflowTransformer, MedDefenderTransformer
from xai.model_loader import load_checkpoint


def get_backbone(network_name, num_pkts, num_fields, num_classes, nlayers, nheads, d_model, dim_feedforward):
    """Instantiate the specified backbone architecture."""
    if network_name == "Lopez17CNN":
        return Lopez17CNN(num_pkts=num_pkts, num_fields=num_fields, num_classes=num_classes)
    elif network_name == "BiflowTransformer":
        return BiflowTransformer(
            num_pkts=num_pkts,
            num_fields=num_fields,
            num_classes=num_classes,
            nlayers=nlayers,
            nheads=nheads,
            d_model=d_model,
            dim_feedforward=dim_feedforward,
        )
    elif network_name == "MedDefenderTransformer":
        return MedDefenderTransformer(
            num_pkts=num_pkts,
            num_fields=num_fields,
            num_classes=num_classes,
            nlayers=nlayers,
            nheads=nheads,
            d_model=d_model,
            dim_feedforward=dim_feedforward,
        )
    else:
        raise ValueError(f"Unrecognized architecture: {network_name}")


def load_model_for_inference(network_name, args):
    """Load a trained model via load_checkpoint; --no-checkpoint profiles the untrained architecture only."""
    if not args.load_checkpoint:
        print("[INFO] --no-checkpoint active: profiling untrained architecture with random initial weights.")
        backbone = get_backbone(
            network_name=network_name,
            num_pkts=args.num_pkts,
            num_fields=args.num_fields,
            num_classes=args.num_classes,
            nlayers=args.nlayers,
            nheads=args.nheads,
            d_model=args.d_model,
            dim_feedforward=args.dim_feedforward,
        )
        return backbone, "None (Initial weights)"

    # Raise FileNotFoundError if training is incomplete (no silent fallback)
    exp_name = f"cic-iomt_scratch_{RUN_IDS[network_name]}"
    model, _ = load_checkpoint(exp_name=exp_name, results_dir=str(_RESULTS_DIR),
                               network=network_name, device="cpu")
    return model, exp_name


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def benchmark_latency(model, input_tensor, warmup=100, repetitions=1000):
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(warmup):
            _ = model(input_tensor)

        # Precise single-thread latency measurement
        latencies = []
        for _ in range(repetitions):
            t0 = time.perf_counter()
            _ = model(input_tensor)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

    latencies = np.array(latencies)
    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "median_ms": float(np.median(latencies)),
        "p90_ms": float(np.percentile(latencies, 90)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "throughput_fps": float(1000.0 / np.median(latencies)),
    }


def profile_network(network_name, args):
    model, ckpt_info = load_model_for_inference(network_name, args)

    total_params, _ = count_parameters(model)
    size_mb = (total_params * 4) / (1024 * 1024)

    dummy_input = torch.randn(1, 1, args.num_pkts, args.num_fields)

    print("======================================================================")
    print(f" CPU INFERENCE PROFILING ({args.threads} core/thread): {network_name}")
    print(f" Config:     T={args.num_pkts}, fields={args.num_fields}, classes={args.num_classes}, N={args.nlayers}, h={args.nheads}")
    print(f" Checkpoint: {ckpt_info}")
    print(f" Params:     {total_params:,} ({size_mb:.2f} MB float32)")
    print("======================================================================")

    res = benchmark_latency(model, dummy_input, warmup=args.warmup, repetitions=args.repetitions)

    print(f" Median Latency:  {res['median_ms']:.2f} ms")
    print(f" Mean ± StdDev:   {res['mean_ms']:.2f} ± {res['std_ms']:.2f} ms")
    print(f" 90th Percentile: {res['p90_ms']:.2f} ms")
    print(f" 95th Percentile: {res['p95_ms']:.2f} ms")
    print(f" 99th Percentile: {res['p99_ms']:.2f} ms")
    print(f" Min / Max:       {res['min_ms']:.2f} ms / {res['max_ms']:.2f} ms")
    print(f" Throughput:      {res['throughput_fps']:.1f} flows/sec")
    print("======================================================================")

    return {
        "Network": network_name,
        "Params": total_params,
        "Size_MB": round(size_mb, 2),
        "Latency_Median_ms": round(res["median_ms"], 2),
        "Latency_Mean_ms": round(res["mean_ms"], 2),
        "Latency_Std_ms": round(res["std_ms"], 2),
        "Latency_P90_ms": round(res["p90_ms"], 2),
        "Latency_P95_ms": round(res["p95_ms"], 2),
        "Latency_P99_ms": round(res["p99_ms"], 2),
        "Throughput_fps": round(res["throughput_fps"], 1),
        "Checkpoint": ckpt_info,
    }


def main():
    parser = argparse.ArgumentParser(description="Edge IoMT single-thread CPU inference profiling")
    parser.add_argument("--network", type=str, default="BiflowTransformer",
                        choices=["BiflowTransformer", "Lopez17CNN", "MedDefenderTransformer"])
    parser.add_argument("--all", action="store_true", help="Run profiling across all 3 models")
    parser.add_argument("--no-checkpoint", dest="load_checkpoint", action="store_false",
                        help="Skip checkpoint loading and profile only the initialized architecture")
    parser.set_defaults(load_checkpoint=True)
    parser.add_argument("--num-pkts", type=int, default=10, help="Packet horizon T (default 10)")
    parser.add_argument("--num-fields", type=int, default=6)
    parser.add_argument("--num-classes", type=int, default=13, help="13 = CICIoMT2024 multiclass split")
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--nheads", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--threads", type=int, default=1, help="CPU thread count (default 1 for edge)")
    args = parser.parse_args()

    # Enforce single-thread CPU execution to profile the edge deployment constraint
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.threads)

    targets = ["Lopez17CNN", "BiflowTransformer", "MedDefenderTransformer"] if args.all else [args.network]
    records = []
    for net in targets:
        rec = profile_network(net, args)
        records.append(rec)

    df = pd.DataFrame(records)
    csv_out = _METRICS_DIR / "edge_latency_profiling.csv"
    df.to_csv(csv_out, index=False)
    print(f"\n[OK] Results saved to: {csv_out}")


if __name__ == "__main__":
    main()
