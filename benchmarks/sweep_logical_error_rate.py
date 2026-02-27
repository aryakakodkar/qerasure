#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bench_virtual_decode import REPO_ROOT, run_benchmark


def make_logspace(min_p: float, max_p: float, points: int) -> list[float]:
    if min_p <= 0.0 or max_p <= 0.0:
        raise ValueError("min_p and max_p must be > 0 for log spacing")
    if min_p >= max_p:
        raise ValueError("min_p must be < max_p")
    if points < 2:
        raise ValueError("points must be >= 2")
    return list(np.logspace(math.log10(min_p), math.log10(max_p), points))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep two-qubit erasure probability and plot logical error rate."
    )
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--qec-rounds", type=int, default=3)
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--points", type=int, default=15)
    parser.add_argument("--p-min", type=float, default=1e-2)
    parser.add_argument("--p-max", type=float, default=3e-1)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / "logical_error_sweep_d3.json",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / "logical_error_sweep_d3.png",
    )
    args = parser.parse_args()

    p_values = make_logspace(args.p_min, args.p_max, args.points)
    rows: list[dict] = []
    t0 = time.perf_counter()

    for i, p_tqe in enumerate(p_values):
        result = run_benchmark(
            distance=args.distance,
            qec_rounds=args.qec_rounds,
            shots=args.shots,
            p_tqe=float(p_tqe),
            seed=args.seed + i,
        )
        rows.append(
            {
                "p_two_qubit_erasure": float(p_tqe),
                "logical_error_rate": 1 - (1 - result["logical_error_rate"])**(1/args.qec_rounds) if result["logical_error_rate"] is not None else None,
                "shots_attempted": result["shots_attempted"],
                "shots_failed": result["shots_failed"],
                "throughput_shots_per_sec": result["throughput_shots_per_sec"],
                "timing_seconds": result["timing_seconds"],
            }
        )
        print(
            f"[{i + 1:02d}/{len(p_values):02d}] "
            f"p={p_tqe:.6g} ler={result['logical_error_rate']} "
            f"attempted={result['shots_attempted']} failed={result['shots_failed']}"
        )

    total_time = time.perf_counter() - t0
    payload = {
        "distance": args.distance,
        "qec_rounds": args.qec_rounds,
        "shots_per_point": args.shots,
        "seed_start": args.seed,
        "points": len(rows),
        "p_min": args.p_min,
        "p_max": args.p_max,
        "elapsed_seconds": total_time,
        "rows": rows,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2))

    x = np.array([row["p_two_qubit_erasure"] for row in rows], dtype=float)
    y = np.array(
        [
            np.nan if row["logical_error_rate"] is None else float(row["logical_error_rate"])
            for row in rows
        ],
        dtype=float,
    )

    plt.figure(figsize=(7.2, 4.8))
    plt.plot(x, y, marker="o", linewidth=1.5)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Two-qubit erasure probability p")
    plt.ylabel("Logical error rate")
    plt.title(
        f"Logical Error Sweep (d={args.distance}, rounds={args.qec_rounds}, shots={args.shots})"
    )
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    args.plot_out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.plot_out, dpi=200)
    plt.close()

    print(f"\nSaved data: {args.json_out}")
    print(f"Saved plot: {args.plot_out}")
    print(f"Elapsed: {total_time:.2f}s")


if __name__ == "__main__":
    main()
