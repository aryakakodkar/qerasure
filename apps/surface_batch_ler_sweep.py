#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_SRC = REPO_ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

import qerasure as qe


def parse_configs(configs_text: str) -> list[tuple[int, int]]:
    configs: list[tuple[int, int]] = []
    for raw in configs_text.split(";"):
        raw = raw.strip()
        if not raw:
            continue
        pieces = [p.strip() for p in raw.split(",")]
        if len(pieces) != 2:
            raise ValueError(f"Invalid config '{raw}', expected 'distance,rounds'.")
        distance = int(pieces[0])
        rounds = int(pieces[1])
        if distance <= 0 or rounds <= 0:
            raise ValueError(f"Invalid config '{raw}', distance and rounds must be > 0.")
        configs.append((distance, rounds))
    if not configs:
        raise ValueError("No valid configs parsed from --configs.")
    return configs


def make_p_values(
    p_values_text: str | None,
    p_min: float,
    p_max: float,
    points: int,
) -> list[float]:
    if p_values_text is not None and p_values_text.strip():
        values = [float(v.strip()) for v in p_values_text.split(",") if v.strip()]
        if not values:
            raise ValueError("--p-values provided, but no values were parsed.")
        if any(v <= 0 for v in values):
            raise ValueError("All p values must be > 0.")
        return values

    if p_min <= 0 or p_max <= 0 or p_min >= p_max:
        raise ValueError("Expected 0 < p-min < p-max for log sweep.")
    if points < 2:
        raise ValueError("--points must be >= 2.")
    return list(np.logspace(math.log10(p_min), math.log10(p_max), points))


def bernoulli_per_round(logical_error_rate: float, rounds: int) -> float:
    # p_round = 1 - (1 - P_logical)^(1/rounds)
    p = float(np.clip(logical_error_rate, 0.0, 1.0))
    return 1.0 - (1.0 - p) ** (1.0 / float(rounds))


def run_single_point(
    distance: int,
    rounds: int,
    shots: int,
    p_tqe: float,
    seed: int,
    sample_threads: int,
    decode_threads: int,
    max_batch_bytes: int,
    single_qubit_errors: bool,
) -> dict:
    circuit = qe.SurfaceCodeRotated(distance).build_circuit(
        rounds=rounds,
        erasure_prob=p_tqe,
        erasable_qubits="ALL",
        reset_failure_prob=0.0,
        single_qubit_errors=single_qubit_errors,
        post_clifford_pauli_prob = 0.00
    )

    model = qe.ErasureModel(
        2,
        qe.PauliChannel(0.25, 0.25, 0.25),
        qe.PauliChannel(0.25, 0.25, 0.25),
        qe.TQGSpreadModel(
            qe.PauliChannel(0.25, 0.25, 0.25),
            qe.PauliChannel(0.25, 0.25, 0.25),
        ),
    )
    model.check_false_negative_prob = 0.0
    model.check_false_positive_prob = 0.0

    compiled = qe.CompiledErasureProgram(circuit, model)
    sampler = qe.StreamSampler(compiled)
    dem_builder = qe.SurfDemBuilder(compiled)
    grouped_decoder = qe.SurfaceCodeBatchDecoder(
        compiled,
        dem_builder=dem_builder,
        max_batch_bytes=max_batch_bytes,
    )

    t0 = time.perf_counter()
    dets, obs, checks = sampler.sample(num_shots=shots, seed=seed, num_threads=sample_threads)
    t1 = time.perf_counter()
    predictions = grouped_decoder.decode_batch(dets, checks, num_threads=decode_threads)
    t2 = time.perf_counter()

    truths = obs if obs.ndim == 2 else obs[:, None]
    n_obs = min(truths.shape[1], predictions.shape[1])
    mismatches = np.any(predictions[:, :n_obs] != truths[:, :n_obs], axis=1)
    ler = float(np.mean(mismatches)) if len(mismatches) else 0.0

    return {
        "distance": distance,
        "qec_rounds": rounds,
        "shots": shots,
        "seed": seed,
        "p_two_qubit_erasure": p_tqe,
        "logical_error_rate": ler,
        "logical_error_rate_per_round": bernoulli_per_round(ler, rounds),
        "timing_seconds": {
            "sample": t1 - t0,
            "decode": t2 - t1,
            "total": t2 - t0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep two-qubit erasure probability for multiple (distance,rounds) "
            "surface-code configs using grouped batch decoding."
        )
    )
    parser.add_argument("--configs", type=str, default="3,3;5,5;7,7")
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=random.randint(0, 2**32 - 1))
    parser.add_argument("--points", type=int, default=10)
    parser.add_argument("--p-min", type=float, default=1e-3)
    parser.add_argument("--p-max", type=float, default=1e-2)
    parser.add_argument("--p-values", type=str, default=None)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--decode-threads", type=int, default=None)
    parser.add_argument("--max-batch-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument(
        "--single-qubit-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include single-qubit erasure onsets after H and ECR operations.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / "apps" / "results" / "surface_batch_ler_sweep.json",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=REPO_ROOT / "apps" / "results" / "surface_batch_ler_sweep.png",
    )
    args = parser.parse_args()

    configs = parse_configs(args.configs)
    p_values = make_p_values(args.p_values, args.p_min, args.p_max, args.points)
    decode_threads = args.decode_threads if args.decode_threads is not None else args.num_threads

    rows: list[dict] = []
    t0 = time.perf_counter()
    for cfg_idx, (distance, rounds) in enumerate(configs):
        for p_idx, p_tqe in enumerate(p_values):
            seed = args.seed + cfg_idx * 1_000_000 + p_idx
            row = run_single_point(
                distance=distance,
                rounds=rounds,
                shots=args.shots,
                p_tqe=float(p_tqe),
                seed=seed,
                sample_threads=args.num_threads,
                decode_threads=decode_threads,
                max_batch_bytes=args.max_batch_bytes,
                single_qubit_errors=args.single_qubit_errors,
            )
            rows.append(row)
            print(
                f"[cfg {cfg_idx + 1}/{len(configs)} | p {p_idx + 1}/{len(p_values)}] "
                f"(d={distance},r={rounds}) p={p_tqe:.6g} "
                f"LER={row['logical_error_rate']:.6g} "
                f"LER/round={row['logical_error_rate_per_round']:.6g}"
            )
    elapsed = time.perf_counter() - t0

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "configs": [{"distance": d, "qec_rounds": r} for d, r in configs],
        "shots_per_point": args.shots,
        "p_values": [float(p) for p in p_values],
        "elapsed_seconds": elapsed,
        "rows": rows,
    }
    args.json_out.write_text(json.dumps(payload, indent=2))

    plt.figure(figsize=(8.4, 5.2))
    for distance, rounds in configs:
        curve = [
            row for row in rows if row["distance"] == distance and row["qec_rounds"] == rounds
        ]
        curve.sort(key=lambda r: r["p_two_qubit_erasure"])
        x = np.array([row["p_two_qubit_erasure"] for row in curve], dtype=float)
        y = np.array([row["logical_error_rate_per_round"] for row in curve], dtype=float)
        plt.plot(x, y, marker="o", linewidth=1.5, label=f"d={distance}, r={rounds}")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Two-qubit erasure probability p")
    plt.ylabel("Logical error rate per round (Bernoulli-normalized)")
    plt.title("Surface-Code Logical Error Rate Per Round vs Erasure Probability")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    args.plot_out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.plot_out, dpi=220)
    plt.close()

    print(f"\nSaved JSON: {args.json_out}")
    print(f"Saved Plot: {args.plot_out}")
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
