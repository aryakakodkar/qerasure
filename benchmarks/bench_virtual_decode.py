#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pymatching as pm

# Local package import (repo-root/python)
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_SRC = REPO_ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

import qerasure


def make_lowering_params() -> qerasure.LoweringParams:
    program = qerasure.SpreadProgram()
    program.append("Z_ERROR(0.5) X_1; Z_ERROR(0.5) X_2")
    program.append("COND_X_ERROR(0.5) Z_1; ELSE_X_ERROR(1.0) Z_2")

    reset = qerasure.LoweredErrorParams(qerasure.PauliError.DEPOLARIZE, 1.0)
    return qerasure.LoweringParams(program, reset)


def write_failure_artifacts(
    out_dir: Path,
    shot_index: int,
    err: Exception,
    erasure_results,
    lowering_result,
    logical_circuit,
    virtual_circuit,
) -> None:
    shot_dir = out_dir / f"shot_{shot_index:05d}"
    shot_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "shot_index": shot_index,
        "error_type": type(err).__name__,
        "error_message": str(err),
    }
    (shot_dir / "error.json").write_text(json.dumps(meta, indent=2))

    # Save per-shot sparse erasure and lowering data for debug reproducibility.
    debug = {
        "erasure_timestep_offsets": erasure_results.erasure_timestep_offsets[shot_index],
        "sparse_erasures": [
            {"qubit_idx": int(e.qubit_idx), "event_type": int(e.event_type)}
            for e in erasure_results.sparse_erasures[shot_index]
        ],
        "lowering_timestep_offsets": lowering_result.clifford_timestep_offsets[shot_index],
        "sparse_cliffords": [
            {
                "qubit_idx": int(e.qubit_idx),
                "error_type": int(e.error_type),
                "origin": int(e.origin),
            }
            for e in lowering_result.sparse_cliffords[shot_index]
        ],
        "check_error_round_flags": list(lowering_result.check_error_round_flags[shot_index])
        if shot_index < len(lowering_result.check_error_round_flags)
        else [],
        "erasure_round_flags": list(lowering_result.erasure_round_flags[shot_index])
        if shot_index < len(lowering_result.erasure_round_flags)
        else [],
        "reset_round_qubits": [
            [int(r), int(q)] for (r, q) in lowering_result.reset_round_qubits[shot_index]
        ]
        if shot_index < len(lowering_result.reset_round_qubits)
        else [],
    }
    (shot_dir / "shot_debug.json").write_text(json.dumps(debug, indent=2))

    if logical_circuit is not None:
        (shot_dir / "logical_circuit.stim").write_text(str(logical_circuit))
    if virtual_circuit is not None:
        (shot_dir / "virtual_circuit.stim").write_text(str(virtual_circuit))


def run_benchmark(distance: int, qec_rounds: int, shots: int, p_tqe: float, seed: int) -> dict:
    code = qerasure.RotatedSurfaceCode(distance)

    noise = qerasure.NoiseParams()
    noise.set(qerasure.NoiseChannel.TWO_QUBIT_ERASURE, p_tqe)

    sim_params = qerasure.ErasureSimParams(
        code=code,
        noise=noise,
        qec_rounds=qec_rounds,
        shots=shots,
        seed=seed,
        erasure_selection=qerasure.ErasureQubitSelection.DATA_QUBITS,
    )

    lowering_params = make_lowering_params()

    t0 = time.perf_counter()
    erasure_results = qerasure.ErasureSimulator(sim_params).simulate()
    t1 = time.perf_counter()
    lowering_result = qerasure.Lowerer(code, lowering_params).lower(erasure_results)
    t2 = time.perf_counter()

    failure_root = REPO_ROOT / "benchmarks" / "failures" / time.strftime("%Y%m%d_%H%M%S")
    failures = 0
    mismatches = 0
    attempted = 0

    for shot_index in range(shots):
        logical_circuit = None
        virtual_circuit = None
        try:
            logical_circuit = qerasure.build_logical_stabilizer_circuit_object(
                code=code, lowering_result=lowering_result, shot_index=shot_index
            )
            virtual_circuit = qerasure.build_virtual_decoder_stim_circuit_object(
                code=code,
                qec_rounds=qec_rounds,
                lowering_params=lowering_params,
                lowering_result=lowering_result,
                shot_index=shot_index,
                two_qubit_erasure_probability=p_tqe,
                condition_on_erasure_in_round=True,
            )

            # Sample one detector/observable record from the logical-circuit shot.
            sampler = logical_circuit.compile_detector_sampler()
            detector_sample, observable_flip = sampler.sample(
                shots=1, separate_observables=True
            )

            virtual_dem = virtual_circuit.detector_error_model(
                decompose_errors=True,
                approximate_disjoint_errors=True,
            )
            matching = pm.Matching.from_detector_error_model(virtual_dem)

            syndrome = np.asarray(detector_sample, dtype=np.uint8)
            if syndrome.ndim == 1:
                syndrome = syndrome[None, :]
            truth = np.asarray(observable_flip, dtype=np.uint8)
            if truth.ndim == 1:
                truth = truth[:, None]

            if hasattr(matching, "decode_batch"):
                pred = np.asarray(matching.decode_batch(syndrome), dtype=np.uint8)
            else:
                pred = np.asarray([matching.decode(s) for s in syndrome], dtype=np.uint8)
            if pred.ndim == 1:
                pred = pred[:, None]

            n_obs = min(pred.shape[1], truth.shape[1])
            mismatch = bool(np.any((pred[:, :n_obs] ^ truth[:, :n_obs]) != 0))
            mismatches += int(mismatch)
            attempted += 1

            # Explicitly discard per-shot heavy objects on success.
            del logical_circuit
            del virtual_circuit
            del sampler
            del virtual_dem
            del matching
            del detector_sample
            del observable_flip
            del syndrome
            del truth
            del pred

        except Exception as err:  # noqa: BLE001
            failures += 1
            write_failure_artifacts(
                out_dir=failure_root,
                shot_index=shot_index,
                err=err,
                erasure_results=erasure_results,
                lowering_result=lowering_result,
                logical_circuit=logical_circuit,
                virtual_circuit=virtual_circuit,
            )

    t3 = time.perf_counter()

    return {
        "distance": distance,
        "qec_rounds": qec_rounds,
        "shots_requested": shots,
        "shots_attempted": attempted,
        "shots_failed": failures,
        "two_qubit_erasure_probability": p_tqe,
        "seed": seed,
        "logical_error_rate": (mismatches / attempted) if attempted else None,
        "timing_seconds": {
            "simulate": t1 - t0,
            "lower": t2 - t1,
            "build_sample_decode_loop": t3 - t2,
            "total": t3 - t0,
        },
        "throughput_shots_per_sec": (attempted / (t3 - t2)) if (t3 - t2) > 0 else None,
        "failure_artifacts_dir": str(failure_root) if failures else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Virtual-decoder benchmark on lowered erasure shots.")
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--qec-rounds", type=int, default=3)
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--p-tqe", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "results" / "virtual_decode_d3_10k.json",
    )
    args = parser.parse_args()

    result = run_benchmark(
        distance=args.distance,
        qec_rounds=args.qec_rounds,
        shots=args.shots,
        p_tqe=args.p_tqe,
        seed=args.seed,
    )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2))

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
