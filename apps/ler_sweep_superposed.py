#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pymatching as pm

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


def parse_configs(configs_text: str) -> list[tuple[int, int]]:
    configs: list[tuple[int, int]] = []
    for raw in configs_text.split(";"):
        raw = raw.strip()
        if not raw:
            continue
        pieces = [p.strip() for p in raw.split(",")]
        if len(pieces) != 2:
            raise ValueError(f"Invalid config '{raw}', expected 'distance,rounds'")
        distance = int(pieces[0])
        rounds = int(pieces[1])
        if distance <= 0 or rounds <= 0:
            raise ValueError(f"Invalid config '{raw}', distance and rounds must be > 0")
        configs.append((distance, rounds))
    if not configs:
        raise ValueError("No valid configs provided")
    return configs


def make_p_values(
    p_values_text: str | None, p_min: float, p_max: float, points: int
) -> list[float]:
    if p_values_text is not None and p_values_text.strip():
        values = [float(x.strip()) for x in p_values_text.split(",") if x.strip()]
        if not values:
            raise ValueError("--p-values was provided but no valid values were parsed")
        if any(v <= 0 for v in values):
            raise ValueError("All p-values must be > 0")
        return values
    if p_min <= 0 or p_max <= 0 or p_min >= p_max:
        raise ValueError("Expected 0 < p-min < p-max for log spacing")
    if points < 2:
        raise ValueError("--points must be >= 2")
    return list(np.logspace(math.log10(p_min), math.log10(p_max), points))


def normalize_bernoulli_per_round(logical_error_rate: float | None, rounds: int) -> float | None:
    if logical_error_rate is None:
        return None
    p = float(np.clip(logical_error_rate, 0.0, 1.0))
    if rounds <= 0:
        raise ValueError("rounds must be > 0 for Bernoulli normalization")
    return 1.0 - (1.0 - p) ** (1.0 / float(rounds))


def run_single_point(
    distance: int, qec_rounds: int, shots: int, p_tqe: float, seed: int
) -> dict:
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

    erasure_results = qerasure.ErasureSimulator(sim_params).simulate()
    lowering_result = qerasure.Lowerer(code, lowering_params).lower(erasure_results)

    mismatches = 0
    attempted = 0
    failures = 0
    for shot_index in range(shots):
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

            sampler = logical_circuit.compile_detector_sampler()
            detector_sample, observable_flip = sampler.sample(
                shots=1, separate_observables=True
            )
            virtual_dem = virtual_circuit.detector_error_model(
                decompose_errors=True, approximate_disjoint_errors=True
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
        except Exception:
            failures += 1

    ler = (mismatches / attempted) if attempted else None
    return {
        "distance": distance,
        "qec_rounds": qec_rounds,
        "shots_requested": shots,
        "shots_attempted": attempted,
        "shots_failed": failures,
        "p_two_qubit_erasure": p_tqe,
        "logical_error_rate": ler,
        "logical_error_rate_per_round": normalize_bernoulli_per_round(ler, qec_rounds),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LER sweeps for multiple (distance,rounds) configs and superpose curves."
    )
    parser.add_argument(
        "--configs",
        type=str,
        default="3,3;5,5;7,7",
        help="Semicolon-separated tuples: 'd,r;d,r;...'",
    )
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--points", type=int, default=15)
    parser.add_argument("--p-min", type=float, default=1e-2)
    parser.add_argument("--p-max", type=float, default=3e-1)
    parser.add_argument(
        "--p-values",
        type=str,
        default=None,
        help="Optional comma-separated p values. If set, overrides logspace args.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / "apps" / "results" / "ler_superposed_d3_d5_d7.json",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=REPO_ROOT / "apps" / "results" / "ler_superposed_d3_d5_d7.png",
    )
    args = parser.parse_args()

    configs = parse_configs(args.configs)
    p_values = make_p_values(args.p_values, args.p_min, args.p_max, args.points)

    all_rows: list[dict] = []
    t0 = time.perf_counter()
    for d_idx, (distance, rounds) in enumerate(configs):
        for p_idx, p_tqe in enumerate(p_values):
            seed = args.seed + d_idx * 1_000_000 + p_idx
            result = run_single_point(
                distance=distance,
                qec_rounds=rounds,
                shots=args.shots,
                p_tqe=float(p_tqe),
                seed=seed,
            )
            all_rows.append(result)
            print(
                f"[cfg {d_idx + 1}/{len(configs)} | p {p_idx + 1}/{len(p_values)}] "
                f"(d={distance},r={rounds}) p={p_tqe:.6g} "
                f"LER={result['logical_error_rate']} "
                f"LER/round={result['logical_error_rate_per_round']} "
                f"attempted={result['shots_attempted']} failed={result['shots_failed']}"
            )

    elapsed = time.perf_counter() - t0

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "configs": [{"distance": d, "qec_rounds": r} for d, r in configs],
        "shots_per_point": args.shots,
        "p_values": [float(p) for p in p_values],
        "elapsed_seconds": elapsed,
        "rows": all_rows,
    }
    args.json_out.write_text(json.dumps(payload, indent=2))

    plt.figure(figsize=(8.2, 5.2))
    for distance, rounds in configs:
        rows = [
            row
            for row in all_rows
            if row["distance"] == distance and row["qec_rounds"] == rounds
        ]
        rows.sort(key=lambda r: r["p_two_qubit_erasure"])
        x = np.array([row["p_two_qubit_erasure"] for row in rows], dtype=float)
        y = np.array(
            [
                np.nan
                if row["logical_error_rate_per_round"] is None
                else float(row["logical_error_rate_per_round"])
                for row in rows
            ],
            dtype=float,
        )
        plt.plot(x, y, marker="o", linewidth=1.5, label=f"d={distance}, r={rounds}")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Two-qubit erasure probability p")
    plt.ylabel("Logical error rate per round (Bernoulli-normalized)")
    plt.title("Logical Error Rate Per Round vs Two-Qubit Erasure Probability")
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
