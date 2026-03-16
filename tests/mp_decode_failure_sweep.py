#!/usr/bin/env python3
"""Run a fixed d=7 decode-failure sweep over multiple max-persistence values."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

QP_PAIRS = [
    # Fill with (q_check_error, p_pauli_error) pairs before running remotely.
    # Example: (0.02, 0.0),
]

QS = [0.0, 0.005, 0.01]
PS = [0.0, 0.0025, 0.005, 0.0075, 0.01]

for q in QS:
    for p in PS:
        QP_PAIRS.append((q, p))

def _import_qerasure():
    cwd = Path.cwd().resolve()
    repo_root = None
    for candidate in [cwd, *cwd.parents]:
        if (candidate / "python" / "qerasure").exists():
            repo_root = candidate
            break
    if repo_root is None:
        raise RuntimeError("Could not find repository root containing python/qerasure")
    sys.path.insert(0, str(repo_root / "python"))
    import stim  # noqa: F401  pylint: disable=import-error,unused-import
    import qerasure as qe  # pylint: disable=import-error

    return qe


def _build_program(
    qe,
    *,
    distance: int,
    rounds: int,
    erasure_prob: float,
    check_error_prob: float,
    pauli_error_prob: float,
    max_persistence: int,
    single_qubit_errors: bool,
):
    circuit = qe.SurfaceCodeRotated(distance).build_circuit(
        rounds=rounds,
        erasure_prob=erasure_prob,
        erasable_qubits="ALL",
        reset_failure_prob=0.0,
        single_qubit_errors=single_qubit_errors,
        post_clifford_pauli_prob=pauli_error_prob,
    )
    model = qe.ErasureModel(
        int(max_persistence),
        qe.PauliChannel(0.25, 0.25, 0.25),
        qe.PauliChannel(0.25, 0.25, 0.25),
        qe.TQGSpreadModel(
            qe.PauliChannel(0.25, 0.25, 0.25),
            qe.PauliChannel(0.25, 0.25, 0.25),
        ),
    )
    model.check_false_negative_prob = float(check_error_prob)
    model.check_false_positive_prob = float(check_error_prob)
    return qe.CompiledErasureProgram(circuit, model)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run 100k-shot d=7 grouped-decoder sweeps for max_persistence in {2,3,4} "
            "over an in-file array of (q, p) pairs and report whether any decode failures occurred."
        )
    )
    parser.add_argument("--distance", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=920004)
    parser.add_argument("--erasure-prob", type=float, default=0.00172034)
    parser.add_argument("--decode-threads", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument(
        "--single-qubit-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    qe = _import_qerasure()
    if not QP_PAIRS:
        raise RuntimeError("QP_PAIRS is empty. Add one or more (q, p) pairs before running.")
    results = []
    for pair_index, (q_check_error, p_pauli_error) in enumerate(QP_PAIRS):
        for max_persistence in (2, 3, 4):
            program = _build_program(
                qe,
                distance=int(args.distance),
                rounds=int(args.rounds),
                erasure_prob=float(args.erasure_prob),
                check_error_prob=float(q_check_error),
                pauli_error_prob=float(p_pauli_error),
                max_persistence=max_persistence,
                single_qubit_errors=bool(args.single_qubit_errors),
            )
            sampler = qe.StreamSampler(program)
            dem_builder = qe.SurfDemBuilder(program)
            decoder = qe.SurfaceCodeBatchDecoder(program, dem_builder=dem_builder)

            dets, _obs, checks = sampler.sample(
                num_shots=int(args.shots),
                seed=int(args.seed) + pair_index * 100 + max_persistence,
                num_threads=1,
            )

            decode_failed = False
            error_text = ""
            try:
                decoder.decode_batch(dets, checks, num_threads=int(args.decode_threads))
            except Exception as exc:  # pylint: disable=broad-except
                decode_failed = True
                error_text = f"{type(exc).__name__}: {exc}"

            row = {
                "distance": int(args.distance),
                "rounds": int(args.rounds),
                "shots": int(args.shots),
                "seed": int(args.seed) + pair_index * 100 + max_persistence,
                "erasure_prob": float(args.erasure_prob),
                "check_error_prob": float(q_check_error),
                "pauli_error_prob": float(p_pauli_error),
                "single_qubit_errors": bool(args.single_qubit_errors),
                "max_persistence": max_persistence,
                "decode_failed": decode_failed,
                "error": error_text,
            }
            results.append(row)
            status = "FAIL" if decode_failed else "OK"
            print(
                f"q={q_check_error} p={p_pauli_error} mp={max_persistence} status={status} "
                f"distance={args.distance} rounds={args.rounds} shots={args.shots} "
                f"decode_threads={args.decode_threads}"
            )
            if error_text:
                print(f"  error={error_text}")

    print(json.dumps({"results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
