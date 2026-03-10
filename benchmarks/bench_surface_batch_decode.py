#!/usr/bin/env python3
"""Benchmark grouped surface-code DEM-building + matching decode throughput."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_SRC = REPO_ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

import qerasure as qe


def run_benchmark(
    distance: int,
    rounds: int,
    shots: int,
    seed: int,
    p_tqe: float,
    p_fn: float,
    p_fp: float,
    max_persistence: int,
    reset_failure_prob: float,
    erasable_qubits: str,
    num_threads: int,
    decode_threads: int,
    max_batch_bytes: int,
) -> dict:
    circuit = qe.SurfaceCodeRotated(distance).build_circuit(
        rounds=rounds,
        erasure_prob=p_tqe,
        erasable_qubits=erasable_qubits,
        reset_failure_prob=reset_failure_prob,
    )

    model = qe.ErasureModel(
        max_persistence,
        qe.PauliChannel(0.25, 0.25, 0.25),
        qe.PauliChannel(0.25, 0.25, 0.25),
        qe.TQGSpreadModel(
            qe.PauliChannel(0.25, 0.25, 0.25),
            qe.PauliChannel(0.25, 0.25, 0.25),
        ),
    )
    model.check_false_negative_prob = p_fn
    model.check_false_positive_prob = p_fp

    compiled = qe.CompiledErasureProgram(circuit, model)
    sampler = qe.StreamSampler(compiled)
    dem_builder = qe.SurfDemBuilder(compiled)
    grouped_decoder = qe.SurfaceCodeBatchDecoder(
        compiled,
        dem_builder=dem_builder,
        max_batch_bytes=max_batch_bytes,
    )

    t0 = time.perf_counter()
    dets, obs, checks = sampler.sample(num_shots=shots, seed=seed, num_threads=num_threads)
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    predictions = grouped_decoder.decode_batch(dets, checks, num_threads=decode_threads)
    t3 = time.perf_counter()

    truths = obs if obs.ndim == 2 else obs[:, None]
    n_obs = min(truths.shape[1], predictions.shape[1])
    mismatches = np.any(predictions[:, :n_obs] != truths[:, :n_obs], axis=1)
    ler = float(np.mean(mismatches)) if len(mismatches) else 0.0

    packed = np.packbits(checks, axis=1, bitorder="little")
    unique_groups = int(np.unique(packed, axis=0).shape[0])

    sample_s = t1 - t0
    decode_s = t3 - t2
    total_s = t3 - t0

    return {
        "distance": distance,
        "rounds": rounds,
        "shots": shots,
        "seed": seed,
        "erasable_qubits": erasable_qubits,
        "p_tqe": p_tqe,
        "p_fn": p_fn,
        "p_fp": p_fp,
        "max_persistence": max_persistence,
        "reset_failure_prob": reset_failure_prob,
        "num_threads": num_threads,
        "decode_threads": decode_threads,
        "max_batch_bytes": max_batch_bytes,
        "sample_time_s": sample_s,
        "decode_time_s": decode_s,
        "total_time_s": total_s,
        "sample_throughput_shots_per_s": (shots / sample_s) if sample_s > 0 else None,
        "decode_throughput_shots_per_s": (shots / decode_s) if decode_s > 0 else None,
        "logical_error_rate": ler,
        "detector_shape": list(dets.shape),
        "observable_shape": list(obs.shape),
        "check_shape": list(checks.shape),
        "prediction_shape": list(predictions.shape),
        "unique_check_patterns": unique_groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Grouped surface-code batch decode benchmark.")
    parser.add_argument("--distance", type=int, default=15)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--p-tqe", type=float, default=0.01)
    parser.add_argument("--p-fn", type=float, default=0.01)
    parser.add_argument("--p-fp", type=float, default=0.0)
    parser.add_argument("--max-persistence", type=int, default=2)
    parser.add_argument("--reset-failure-prob", type=float, default=0.0)
    parser.add_argument("--erasable-qubits", type=str, default="ALL")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--decode-threads", type=int, default=None)
    parser.add_argument("--max-batch-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / "surface_batch_decode_d15_r15_10k.json",
    )
    args = parser.parse_args()

    result = run_benchmark(
        distance=args.distance,
        rounds=args.rounds,
        shots=args.shots,
        seed=args.seed,
        p_tqe=args.p_tqe,
        p_fn=args.p_fn,
        p_fp=args.p_fp,
        max_persistence=args.max_persistence,
        reset_failure_prob=args.reset_failure_prob,
        erasable_qubits=args.erasable_qubits,
        num_threads=args.num_threads,
        decode_threads=args.decode_threads if args.decode_threads is not None else args.num_threads,
        max_batch_bytes=args.max_batch_bytes,
    )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
