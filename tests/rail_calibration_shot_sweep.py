#!/usr/bin/env python3
"""Sweep rail calibration shot count and measure decode LER on fixed rail-sampled data."""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import time
from collections import defaultdict
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

import qerasure as qe


def bernoulli_per_round(logical_error_rate: float, rounds: int) -> float:
    p = float(np.clip(logical_error_rate, 0.0, 1.0))
    return 1.0 - (1.0 - p) ** (1.0 / float(rounds))


def parse_int_list(values_text: str, name: str) -> list[int]:
    values = [int(v.strip()) for v in values_text.split(",") if v.strip()]
    if not values:
        raise ValueError(f"No values parsed for {name}.")
    if any(v <= 0 for v in values):
        raise ValueError(f"{name} values must be positive.")
    return values


def schedule_bucket(schedule_type: int) -> int:
    if schedule_type == 1:
        return 0
    if schedule_type == 2:
        return 1
    return -1


def pair_inconsistency_in_round(
    rail_program: qe.RailSurfaceCompiledProgram,
    det_row: np.ndarray,
    data_qubit: int,
    round_index: int,
) -> bool:
    if round_index < 0:
        return False
    slot0, slot1 = rail_program.data_z_ancilla_slots(data_qubit)
    if slot0 < 0 or slot1 < 0:
        return False
    d0 = rail_program.round_detector_index(int(round_index), int(slot0))
    d1 = rail_program.round_detector_index(int(round_index), int(slot1))
    if d0 < 0 or d1 < 0:
        return False
    return int(det_row[d0]) != int(det_row[d1])


def condition_bucket(
    rail_program: qe.RailSurfaceCompiledProgram,
    det_row: np.ndarray,
    data_qubit: int,
    check_round: int,
) -> int:
    inc_round1 = pair_inconsistency_in_round(rail_program, det_row, data_qubit, int(check_round) - 1)
    inc_round2 = pair_inconsistency_in_round(rail_program, det_row, data_qubit, int(check_round))
    if inc_round1 and inc_round2:
        return 3
    if inc_round1:
        return 1
    if inc_round2:
        return 2
    return 0


def map_onset_op_to_bin(
    rail_program: qe.RailSurfaceCompiledProgram,
    check_round: int,
    event_rows: list[dict],
    true_onset_op: int,
) -> int:
    prev_round = int(check_round) - 1
    curr_round = int(check_round)
    prev_rows = [r for r in event_rows if rail_program.op_round(int(r["onset_op_index"])) == prev_round]
    curr_rows = [r for r in event_rows if rail_program.op_round(int(r["onset_op_index"])) == curr_round]
    if len(prev_rows) < 4 or len(curr_rows) < 4:
        return -1
    prev_rows.sort(key=lambda r: int(r["onset_op_index"]))
    curr_rows.sort(key=lambda r: int(r["onset_op_index"]))
    onset_to_bin: dict[int, int] = {}
    for j, row in enumerate(prev_rows[:4]):
        onset_to_bin[int(row["onset_op_index"])] = j
    for j, row in enumerate(curr_rows[:4]):
        onset_to_bin[int(row["onset_op_index"])] = 4 + j
    return onset_to_bin.get(int(true_onset_op), -1)


def build_circuit(
    *,
    distance: int,
    rounds: int,
    erasure_prob: float,
    rounds_per_check: int,
    single_qubit_errors: bool,
    pauli_prob: float,
):
    return qe.SurfaceCodeRotated(distance).build_circuit(
        rounds=rounds,
        erasure_prob=float(erasure_prob),
        erasable_qubits="ALL",
        reset_failure_prob=0.0,
        single_qubit_errors=bool(single_qubit_errors),
        post_clifford_pauli_prob=float(pauli_prob),
        rounds_per_check=int(rounds_per_check),
    )


def build_model(check_prob: float):
    model = qe.ErasureModel(
        2,
        qe.PauliChannel(0.25, 0.25, 0.25),
        qe.PauliChannel(0.25, 0.25, 0.25),
        qe.TQGSpreadModel(
            qe.PauliChannel(0.0, 0.0, 0.0),
            qe.PauliChannel(0.0, 0.0, 0.5),
        ),
    )
    model.check_false_negative_prob = float(check_prob)
    model.check_false_positive_prob = float(check_prob)
    return model


def calibrate_onset_posteriors(
    *,
    rail_program: qe.RailSurfaceCompiledProgram,
    shots: int,
    seed: int,
    final_round_only: bool,
) -> tuple[list[list[list[float]]], dict]:
    sampler = qe.RailCalibrationSampler(rail_program)
    dem_builder = qe.RailSurfaceDemBuilder(rail_program)
    dets, _obs, checks, onset_ops = sampler.sample(
        num_shots=int(shots),
        seed=int(seed),
        num_threads=1,
    )

    counts = np.zeros((2, 4, 8), dtype=np.int64)
    events_per_bucket = np.zeros((2, 4), dtype=np.int64)
    for shot in range(int(dets.shape[0])):
        check_row = checks[shot].tolist()
        det_row = dets[shot]
        onset_row = onset_ops[shot]
        rows = dem_builder.calibration_rows(check_row, det_row.tolist())
        if not rows:
            continue
        grouped: dict[tuple[int, int, int, int, bool], list[dict]] = defaultdict(list)
        for row in rows:
            key = (
                int(row["check_event_index"]),
                int(row["data_qubit"]),
                int(row["check_round"]),
                int(row["schedule_type"]),
                bool(row["boundary_data_qubit"]),
            )
            grouped[key].append(row)
        for (check_event_index, data_qubit, check_round, schedule_type, is_boundary), event_rows in grouped.items():
            if is_boundary:
                continue
            if not rail_program.data_qubit_is_full_interior(int(data_qubit)):
                continue
            if int(check_round) <= 0:
                continue
            is_final_round_check = int(check_round) == int(rail_program.rounds) - 1
            if final_round_only and not is_final_round_check:
                continue
            if not final_round_only and is_final_round_check:
                continue
            s_bucket = schedule_bucket(int(schedule_type))
            if s_bucket < 0:
                continue
            c_bucket = condition_bucket(
                rail_program=rail_program,
                det_row=det_row,
                data_qubit=int(data_qubit),
                check_round=int(check_round),
            )
            true_onset_op = int(onset_row[int(check_event_index)])
            bin_index = map_onset_op_to_bin(
                rail_program=rail_program,
                check_round=int(check_round),
                event_rows=event_rows,
                true_onset_op=true_onset_op,
            )
            if bin_index < 0:
                continue
            counts[s_bucket, c_bucket, bin_index] += 1
            events_per_bucket[s_bucket, c_bucket] += 1

    posteriors = np.zeros((2, 4, 8), dtype=float)
    for s_bucket in range(2):
        for c_bucket in range(4):
            total = int(events_per_bucket[s_bucket, c_bucket])
            if total <= 0:
                posteriors[s_bucket, c_bucket, :] = 1.0 / 8.0
            else:
                posteriors[s_bucket, c_bucket, :] = counts[s_bucket, c_bucket, :] / float(total)

    summary = {
        "scope": "final_round_only" if final_round_only else "non_final_rounds",
        "counts": counts.tolist(),
        "events_per_bucket": events_per_bucket.tolist(),
    }
    return posteriors.tolist(), summary


def decode_with_rail_from_fixed_samples(
    *,
    rail_program: qe.RailSurfaceCompiledProgram,
    dets: np.ndarray,
    obs: np.ndarray,
    checks: np.ndarray,
    rounds: int,
    calibration_posteriors: list[list[list[float]]],
    final_round_calibration_posteriors: list[list[list[float]]],
    calibration_erasure_prob: float,
) -> dict:
    dem_builder = qe.RailSurfaceDemBuilder(rail_program)
    dem_builder.set_calibrated_onset_posteriors(
        erasure_probability=float(calibration_erasure_prob),
        posteriors=calibration_posteriors,
        boost_nonzero_with_pe2=True,
    )
    dem_builder.set_final_round_calibrated_onset_posteriors(
        erasure_probability=float(calibration_erasure_prob),
        posteriors=final_round_calibration_posteriors,
        boost_nonzero_with_pe2=True,
    )

    t0 = time.perf_counter()
    pred_rows = []
    decode_failures = 0
    for shot in range(int(dets.shape[0])):
        try:
            decoded_circuit = dem_builder.build_decoded_circuit(
                checks[shot].tolist(),
                dets[shot].tolist(),
                verbose=False,
            )
            decoded_dem = decoded_circuit.detector_error_model(
                decompose_errors=True,
                approximate_disjoint_errors=True,
            )
            matching = pm.Matching.from_detector_error_model(decoded_dem)
            pred = np.asarray(matching.decode(dets[shot]), dtype=np.uint8)
            if pred.ndim == 0:
                pred = pred.reshape(1)
            pred_rows.append(pred)
        except Exception:
            decode_failures += 1
            pred_rows.append(np.zeros((1,), dtype=np.uint8))
    t1 = time.perf_counter()

    width = max((int(row.shape[0]) for row in pred_rows), default=1)
    predictions = np.zeros((len(pred_rows), width), dtype=np.uint8)
    for i, row in enumerate(pred_rows):
        n = min(width, int(row.shape[0]))
        predictions[i, :n] = row[:n]

    truths = obs if obs.ndim == 2 else obs[:, None]
    n_obs = min(int(truths.shape[1]), int(predictions.shape[1]))
    mismatches = np.any(predictions[:, :n_obs] != truths[:, :n_obs], axis=1)
    ler = float(np.mean(mismatches)) if len(mismatches) else 0.0
    return {
        "logical_error_rate": ler,
        "logical_error_rate_per_round": bernoulli_per_round(ler, rounds),
        "decode_failures": int(decode_failures),
        "decode_seconds": float(t1 - t0),
    }


def decode_with_standard_from_fixed_samples(
    *,
    compiled: qe.CompiledErasureProgram,
    dets: np.ndarray,
    obs: np.ndarray,
    checks: np.ndarray,
    rounds: int,
) -> dict:
    dem_builder = qe.SurfDemBuilder(compiled)
    decoder = qe.SurfaceCodeBatchDecoder(compiled, dem_builder=dem_builder)
    t0 = time.perf_counter()
    predictions = np.asarray(decoder.decode_batch(dets, checks, num_threads=1), dtype=np.uint8)
    if predictions.ndim == 1:
        predictions = predictions[:, None]
    t1 = time.perf_counter()

    truths = obs if obs.ndim == 2 else obs[:, None]
    n_obs = min(int(truths.shape[1]), int(predictions.shape[1]))
    mismatches = np.any(predictions[:, :n_obs] != truths[:, :n_obs], axis=1)
    ler = float(np.mean(mismatches)) if len(mismatches) else 0.0
    return {
        "logical_error_rate": ler,
        "logical_error_rate_per_round": bernoulli_per_round(ler, rounds),
        "decode_seconds": float(t1 - t0),
    }


def plot_results(
    *,
    rows: list[dict],
    standard_ler_per_round: float,
    out_png: Path,
) -> None:
    x = np.asarray([int(r["calibration_shots"]) for r in rows], dtype=float)
    y = np.asarray([float(r["rail_ler_per_round"]) for r in rows], dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(x, y, marker="o", linestyle="-", linewidth=1.8, markersize=5, label="rail_calibrated")
    ax.axhline(
        float(standard_ler_per_round),
        linestyle="--",
        linewidth=1.5,
        label=f"standard_dem baseline ({standard_ler_per_round:.6g})",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Calibration shots")
    ax.set_ylabel("Logical error rate / round")
    ax.set_title("d=7, rounds=7, q=0, p=0, e=0.01")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep calibration-shot count and measure rail decoder LER on fixed rail-sampled data."
    )
    parser.add_argument("--distance", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--erasure-prob", type=float, default=0.01)
    parser.add_argument("--check-prob", type=float, default=0.0)
    parser.add_argument("--pauli-prob", type=float, default=0.0)
    parser.add_argument("--rounds-per-check", type=int, default=2)
    parser.add_argument("--single-qubit-errors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval-shots", type=int, default=10000)
    parser.add_argument(
        "--calibration-shot-values",
        type=str,
        default="3000,10000,30000,100000,200000,300000",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed. If omitted, a random 32-bit seed is used.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / "tests" / "artifacts" / "rail_calibration_shot_sweep.json",
    )
    parser.add_argument(
        "--png-out",
        type=Path,
        default=REPO_ROOT / "tests" / "artifacts" / "rail_calibration_shot_sweep.png",
    )
    args = parser.parse_args()

    if args.distance <= 0 or args.rounds <= 0:
        raise ValueError("distance and rounds must be positive")
    if args.eval_shots <= 0:
        raise ValueError("--eval-shots must be positive")
    if args.rounds_per_check <= 0:
        raise ValueError("--rounds-per-check must be positive")
    calibration_shot_values = parse_int_list(args.calibration_shot_values, "--calibration-shot-values")

    base_seed = int(args.seed) if args.seed is not None else int(secrets.randbits(32))
    print(f"base_seed={base_seed}", flush=True)

    circuit = build_circuit(
        distance=int(args.distance),
        rounds=int(args.rounds),
        erasure_prob=float(args.erasure_prob),
        rounds_per_check=int(args.rounds_per_check),
        single_qubit_errors=bool(args.single_qubit_errors),
        pauli_prob=float(args.pauli_prob),
    )
    model = build_model(float(args.check_prob))
    rail_program = qe.RailSurfaceCompiledProgram(
        circuit=circuit,
        model=model,
        distance=int(args.distance),
        rounds=int(args.rounds),
    )
    compiled = qe.CompiledErasureProgram(circuit, model)

    sampler = qe.RailStreamSampler(rail_program)
    print(f"sampling fixed evaluation dataset: {args.eval_shots} shots", flush=True)
    t_sample_0 = time.perf_counter()
    dets, obs, checks = sampler.sample(
        num_shots=int(args.eval_shots),
        seed=int((base_seed ^ 0xA1B2C3D4) & 0xFFFFFFFF),
        num_threads=1,
    )
    t_sample_1 = time.perf_counter()

    standard = decode_with_standard_from_fixed_samples(
        compiled=compiled,
        dets=dets,
        obs=obs,
        checks=checks,
        rounds=int(args.rounds),
    )
    print(
        f"standard baseline: LER/round={standard['logical_error_rate_per_round']:.6g} "
        f"(decode_s={standard['decode_seconds']:.2f})",
        flush=True,
    )

    rows: list[dict] = []
    sweep_start = time.perf_counter()
    total = len(calibration_shot_values)
    for i, cal_shots in enumerate(calibration_shot_values, start=1):
        cal_seed = (base_seed + i * 1000003) & 0xFFFFFFFF
        cal_nonfinal, summary_nonfinal = calibrate_onset_posteriors(
            rail_program=rail_program,
            shots=int(cal_shots),
            seed=int(cal_seed),
            final_round_only=False,
        )
        cal_final, summary_final = calibrate_onset_posteriors(
            rail_program=rail_program,
            shots=int(cal_shots),
            seed=int((cal_seed ^ 0x6C6C6C6C) & 0xFFFFFFFF),
            final_round_only=True,
        )
        rail = decode_with_rail_from_fixed_samples(
            rail_program=rail_program,
            dets=dets,
            obs=obs,
            checks=checks,
            rounds=int(args.rounds),
            calibration_posteriors=cal_nonfinal,
            final_round_calibration_posteriors=cal_final,
            calibration_erasure_prob=float(args.erasure_prob),
        )
        elapsed = time.perf_counter() - sweep_start
        print(
            f"[{i}/{total}] cal_shots={cal_shots} elapsed={elapsed:.1f}s "
            f"rail LER/round={rail['logical_error_rate_per_round']:.6g} "
            f"decode_failures={rail['decode_failures']}",
            flush=True,
        )
        rows.append(
            {
                "calibration_shots": int(cal_shots),
                "rail_ler": float(rail["logical_error_rate"]),
                "rail_ler_per_round": float(rail["logical_error_rate_per_round"]),
                "rail_decode_failures": int(rail["decode_failures"]),
                "rail_decode_seconds": float(rail["decode_seconds"]),
                "calibration_summary_non_final": summary_nonfinal,
                "calibration_summary_final_round": summary_final,
            }
        )

    plot_results(
        rows=rows,
        standard_ler_per_round=float(standard["logical_error_rate_per_round"]),
        out_png=args.png_out,
    )

    payload = {
        "distance": int(args.distance),
        "rounds": int(args.rounds),
        "erasure_prob": float(args.erasure_prob),
        "check_prob": float(args.check_prob),
        "pauli_prob": float(args.pauli_prob),
        "rounds_per_check": int(args.rounds_per_check),
        "single_qubit_errors": bool(args.single_qubit_errors),
        "base_seed": int(base_seed),
        "eval_shots": int(args.eval_shots),
        "eval_sample_seconds": float(t_sample_1 - t_sample_0),
        "calibration_shot_values": [int(v) for v in calibration_shot_values],
        "standard_baseline": standard,
        "rows": rows,
        "png_path": str(args.png_out),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2))

    print(f"saved json: {args.json_out}", flush=True)
    print(f"saved plot: {args.png_out}", flush=True)


if __name__ == "__main__":
    main()
