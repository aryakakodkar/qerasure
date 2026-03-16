#!/usr/bin/env python3
"""Benchmark grouped surface-code batch decode throughput for d=15, mp=3."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_IMPL = REPO_ROOT / "benchmarks" / "bench_surface_batch_decode.py"


def _load_benchmark_impl():
    spec = importlib.util.spec_from_file_location("bench_surface_batch_decode_impl", BENCH_IMPL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load benchmark implementation from {BENCH_IMPL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grouped surface-code batch decode benchmark for d=15, r=15, 10k shots, mp=3."
    )
    parser.add_argument("--distance", type=int, default=15)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--shots", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--p-tqe", type=float, default=0.01)
    parser.add_argument("--p-fn", type=float, default=0.01)
    parser.add_argument("--p-fp", type=float, default=0.0)
    parser.add_argument("--reset-failure-prob", type=float, default=0.0)
    parser.add_argument("--erasable-qubits", type=str, default="ALL")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--decode-threads", type=int, default=None)
    parser.add_argument("--max-batch-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / "surface_batch_decode_mp3_d15_r15_10k.json",
    )
    args = parser.parse_args()

    bench_impl = _load_benchmark_impl()
    result = bench_impl.run_benchmark(
        distance=args.distance,
        rounds=args.rounds,
        shots=args.shots,
        seed=args.seed,
        p_tqe=args.p_tqe,
        p_fn=args.p_fn,
        p_fp=args.p_fp,
        max_persistence=3,
        reset_failure_prob=args.reset_failure_prob,
        erasable_qubits=args.erasable_qubits,
        num_threads=args.num_threads,
        decode_threads=args.decode_threads if args.decode_threads is not None else args.num_threads,
        max_batch_bytes=args.max_batch_bytes,
    )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(__import__("json").dumps(result, indent=2))
    print(__import__("json").dumps(result, indent=2))


if __name__ == "__main__":
    main()
