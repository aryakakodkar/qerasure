#!/usr/bin/env python3
"""Compare rail-calibrated decode vs standard decode across (e, p, q) sweeps.

For each sweep point (distance, rounds, e, p, q), this script:
1. Runs calibration using latent onset labels.
2. Runs rail-calibrated decoding.
3. Runs standard decoding.
4. Records LER metrics and writes overlay plots.
"""

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


def parse_configs(configs_text: str) -> list[tuple[int, int]]:
    configs: list[tuple[int, int]] = []
    for raw in configs_text.split(";"):
        raw = raw.strip()
        if not raw:
            continue
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid config '{raw}', expected 'distance,rounds'.")
        distance = int(parts[0])
        rounds = int(parts[1])
        if distance <= 0 or rounds <= 0:
            raise ValueError(f"Invalid config '{raw}', distance and rounds must be > 0.")
        configs.append((distance, rounds))
    if not configs:
        raise ValueError("No valid configs parsed from --configs.")
    return configs


def parse_float_list(values_text: str, name: str) -> list[float]:
    values = [float(v.strip()) for v in values_text.split(",") if v.strip()]
    if not values:
        raise ValueError(f"No values parsed for {name}.")
    return values


def parse_logspace_e_values(e_min: float, e_max: float, e_count: int) -> list[float]:
    if e_count <= 0:
        raise ValueError("--e-count must be positive")
    if e_min <= 0.0 or e_max <= 0.0:
        raise ValueError("--e-min and --e-max must be > 0 for log spacing")
    if e_max < e_min:
        raise ValueError("--e-max must be >= --e-min")
    if e_count == 1:
        return [float(e_min)]
    return [float(v) for v in np.logspace(np.log10(e_min), np.log10(e_max), int(e_count))]


def bernoulli_per_round(logical_error_rate: float, rounds: int) -> float:
    p = float(np.clip(logical_error_rate, 0.0, 1.0))
    return 1.0 - (1.0 - p) ** (1.0 / float(rounds))


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


def build_rail_model(check_prob: float):
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


def build_standard_model(check_prob: float):
    model = qe.ErasureModel(
        2,
        qe.PauliChannel(0.25, 0.25, 0.25),
        qe.PauliChannel(0.25, 0.25, 0.25),
        qe.TQGSpreadModel(
            qe.PauliChannel(0.5, 0.0, 0.0),
            qe.PauliChannel(0.0, 0.0, 0.5),
        ),
    )
    model.check_false_negative_prob = float(check_prob)
    model.check_false_positive_prob = float(check_prob)
    return model


def calibrate_onset_posteriors(
    *,
    distance: int,
    rounds: int,
    shots: int,
    seed: int,
    erasure_prob: float,
    check_prob: float,
    rounds_per_check: int,
    single_qubit_errors: bool,
    pauli_prob: float,
    final_round_only: bool,
) -> tuple[list[list[list[float]]], dict]:
    circuit = build_circuit(
        distance=distance,
        rounds=rounds,
        erasure_prob=erasure_prob,
        rounds_per_check=rounds_per_check,
        single_qubit_errors=single_qubit_errors,
        pauli_prob=pauli_prob,
    )
    model = build_rail_model(check_prob=check_prob)
    rail_program = qe.RailSurfaceCompiledProgram(
        circuit=circuit,
        model=model,
        distance=distance,
        rounds=rounds,
    )
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
            is_final_round_check = int(check_round) == int(rounds) - 1
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


def decode_with_rail(
    *,
    distance: int,
    rounds: int,
    shots: int,
    seed: int,
    erasure_prob: float,
    check_prob: float,
    pauli_prob: float,
    rounds_per_check: int,
    single_qubit_errors: bool,
    calibration_posteriors: list[list[list[float]]],
    final_round_calibration_posteriors: list[list[list[float]]],
    calibration_erasure_prob: float,
) -> dict:
    circuit = build_circuit(
        distance=distance,
        rounds=rounds,
        erasure_prob=erasure_prob,
        rounds_per_check=rounds_per_check,
        single_qubit_errors=single_qubit_errors,
        pauli_prob=pauli_prob,
    )
    model = build_rail_model(check_prob=check_prob)
    rail_program = qe.RailSurfaceCompiledProgram(
        circuit=circuit,
        model=model,
        distance=distance,
        rounds=rounds,
    )
    sampler = qe.RailStreamSampler(rail_program)
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
    dets, obs, checks = sampler.sample(num_shots=int(shots), seed=int(seed), num_threads=1)
    t1 = time.perf_counter()

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
    t2 = time.perf_counter()

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
        "timing_seconds": {
            "sample": float(t1 - t0),
            "decode": float(t2 - t1),
            "total": float(t2 - t0),
        },
    }


def decode_with_normal(
    *,
    distance: int,
    rounds: int,
    shots: int,
    seed: int,
    erasure_prob: float,
    check_prob: float,
    pauli_prob: float,
    rounds_per_check: int,
    single_qubit_errors: bool,
) -> dict:
    circuit = build_circuit(
        distance=distance,
        rounds=rounds,
        erasure_prob=erasure_prob,
        rounds_per_check=rounds_per_check,
        single_qubit_errors=single_qubit_errors,
        pauli_prob=pauli_prob,
    )
    model = build_standard_model(check_prob=check_prob)
    compiled = qe.CompiledErasureProgram(circuit, model)
    sampler = qe.StreamSampler(compiled)
    dem_builder = qe.SurfDemBuilder(compiled)
    decoder = qe.SurfaceCodeBatchDecoder(compiled, dem_builder=dem_builder)

    t0 = time.perf_counter()
    dets, obs, checks = sampler.sample(num_shots=int(shots), seed=int(seed), num_threads=1)
    t1 = time.perf_counter()
    predictions = np.asarray(decoder.decode_batch(dets, checks, num_threads=1), dtype=np.uint8)
    if predictions.ndim == 1:
        predictions = predictions[:, None]
    t2 = time.perf_counter()

    truths = obs if obs.ndim == 2 else obs[:, None]
    n_obs = min(int(truths.shape[1]), int(predictions.shape[1]))
    mismatches = np.any(predictions[:, :n_obs] != truths[:, :n_obs], axis=1)
    ler = float(np.mean(mismatches)) if len(mismatches) else 0.0
    return {
        "logical_error_rate": ler,
        "logical_error_rate_per_round": bernoulli_per_round(ler, rounds),
        "decode_failures": 0,
        "timing_seconds": {
            "sample": float(t1 - t0),
            "decode": float(t2 - t1),
            "total": float(t2 - t0),
        },
    }


def plot_overlays(
    rows: list[dict],
    configs: list[tuple[int, int]],
    p_values: list[float],
    q_values: list[float],
    plot_dir: Path,
) -> list[str]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for q in q_values:
        for p in p_values:
            subset = [
                row
                for row in rows
                if abs(float(row["check_prob"]) - float(q)) < 1e-15
                and abs(float(row["pauli_prob"]) - float(p)) < 1e-15
            ]
            if not subset:
                continue
            fig, ax = plt.subplots(figsize=(8.6, 5.2))
            colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(configs))))
            for cfg_idx, (distance, rounds) in enumerate(configs):
                for scheme, marker, linestyle in (
                    ("normal", "o", "-"),
                    ("rail_calibrated", "s", "--"),
                ):
                    curve = [
                        row
                        for row in subset
                        if int(row["distance"]) == int(distance)
                        and int(row["rounds"]) == int(rounds)
                        and row["scheme"] == scheme
                    ]
                    if not curve:
                        continue
                    curve.sort(key=lambda r: float(r["erasure_prob"]))
                    x = np.asarray([float(r["erasure_prob"]) for r in curve], dtype=float)
                    y = np.asarray([float(r["logical_error_rate_per_round"]) for r in curve], dtype=float)
                    ax.plot(
                        x,
                        y,
                        marker=marker,
                        linestyle=linestyle,
                        color=colors[cfg_idx],
                        linewidth=1.6,
                        markersize=4.5,
                        label=f"{scheme}, d={distance}, r={rounds}",
                    )
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel("Erasure probability e")
            ax.set_ylabel("Logical error rate / round")
            ax.set_title(f"Rail vs Standard Decode (q={q:.6g}, p={p:.6g})")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(fontsize=8)
            fig.tight_layout()
            out_path = plot_dir / f"rail_vs_standard_q{q:.6g}_p{p:.6g}.png"
            fig.savefig(out_path, dpi=220)
            plt.close(fig)
            saved.append(str(out_path))
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep and compare rail-calibrated decode vs standard decode. "
            "Runs calibration before every (distance, rounds, e, p, q) point."
        )
    )
    parser.add_argument("--configs", type=str, default="3,3;5,5;7,7")
    parser.add_argument(
        "--e-values",
        type=str,
        default="",
        help="Optional explicit comma-separated e values. If omitted, uses log spacing from --e-min/--e-max/--e-count.",
    )
    parser.add_argument("--e-min", type=float, default=0.001)
    parser.add_argument("--e-max", type=float, default=0.005)
    parser.add_argument("--e-count", type=int, default=5)
    parser.add_argument("--p-values", type=str, default="0.0")
    parser.add_argument("--q-values", type=str, default="0.0,0.005")
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--calibration-shots", type=int, default=30000)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base RNG seed. If omitted, a fresh random 32-bit seed is used each run.",
    )
    parser.add_argument(
        "--rounds-per-check",
        type=int,
        default=2,
        help="Shared fallback rounds-per-check when scheme-specific values are not provided.",
    )
    parser.add_argument(
        "--rail-rounds-per-check",
        type=int,
        default=None,
        help="Rounds-per-check for rail sampling/calibration/decode. Defaults to --rounds-per-check.",
    )
    parser.add_argument(
        "--normal-rounds-per-check",
        type=int,
        default=None,
        help="Rounds-per-check for normal (standard) sampling/decode. Defaults to --rounds-per-check.",
    )
    parser.add_argument(
        "--single-qubit-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / "tests" / "artifacts" / "rail_vs_standard_calibrated_sweep.json",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=REPO_ROOT / "tests" / "artifacts",
    )
    args = parser.parse_args()

    if args.shots <= 0:
        raise ValueError("--shots must be positive")
    if args.calibration_shots <= 0:
        raise ValueError("--calibration-shots must be positive")
    rail_rounds_per_check = (
        int(args.rail_rounds_per_check)
        if args.rail_rounds_per_check is not None
        else int(args.rounds_per_check)
    )
    normal_rounds_per_check = (
        int(args.normal_rounds_per_check)
        if args.normal_rounds_per_check is not None
        else int(args.rounds_per_check)
    )
    if rail_rounds_per_check <= 0:
        raise ValueError("--rail-rounds-per-check must be positive")
    if normal_rounds_per_check <= 0:
        raise ValueError("--normal-rounds-per-check must be positive")

    configs = parse_configs(args.configs)
    if args.e_values.strip():
        e_values = parse_float_list(args.e_values, "--e-values")
    else:
        e_values = parse_logspace_e_values(
            e_min=float(args.e_min),
            e_max=float(args.e_max),
            e_count=int(args.e_count),
        )
    p_values = parse_float_list(args.p_values, "--p-values")
    q_values = parse_float_list(args.q_values, "--q-values")

    total_points = len(configs) * len(e_values) * len(p_values) * len(q_values)
    if total_points == 0:
        raise RuntimeError("No sweep points requested.")

    base_seed = int(args.seed) if args.seed is not None else int(secrets.randbits(32))
    print(f"base_seed={base_seed}", flush=True)

    rows: list[dict] = []
    calibration_summary_by_point: dict[str, dict] = {}
    point_counter = 0
    sweep_start = time.perf_counter()

    for q_idx, q in enumerate(q_values):
        for p_idx, p in enumerate(p_values):
            for cfg_idx, (distance, rounds) in enumerate(configs):
                for e_idx, e in enumerate(e_values):
                    point_counter += 1
                    point_seed = (
                        base_seed + q_idx * 10_000_000 + p_idx * 1_000_000 + cfg_idx * 10_000 + e_idx
                    ) & 0xFFFFFFFF
                    calib_seed = (point_seed ^ 0xA5A5A5A5) & 0xFFFFFFFF
                    point_label = (
                        f"d={distance},r={rounds},e={float(e):.12g},"
                        f"p={float(p):.12g},q={float(q):.12g}"
                    )
                    calibration_posteriors, calibration_summary = calibrate_onset_posteriors(
                        distance=int(distance),
                        rounds=int(rounds),
                        shots=int(args.calibration_shots),
                        seed=int(calib_seed),
                        erasure_prob=float(e),
                        check_prob=float(q),
                        rounds_per_check=int(rail_rounds_per_check),
                        single_qubit_errors=bool(args.single_qubit_errors),
                        pauli_prob=float(p),
                        final_round_only=False,
                    )
                    final_round_calibration_posteriors, final_round_calibration_summary = calibrate_onset_posteriors(
                        distance=int(distance),
                        rounds=int(rounds),
                        shots=int(args.calibration_shots),
                        seed=int((calib_seed ^ 0x6C6C6C6C) & 0xFFFFFFFF),
                        erasure_prob=float(e),
                        check_prob=float(q),
                        rounds_per_check=int(rail_rounds_per_check),
                        single_qubit_errors=bool(args.single_qubit_errors),
                        pauli_prob=float(p),
                        final_round_only=True,
                    )
                    calibration_summary_by_point[point_label] = {
                        "non_final": calibration_summary,
                        "final_round": final_round_calibration_summary,
                    }

                    rail_row = decode_with_rail(
                        distance=int(distance),
                        rounds=int(rounds),
                        shots=int(args.shots),
                        seed=int(point_seed),
                        erasure_prob=float(e),
                        check_prob=float(q),
                        pauli_prob=float(p),
                        rounds_per_check=int(rail_rounds_per_check),
                        single_qubit_errors=bool(args.single_qubit_errors),
                        calibration_posteriors=calibration_posteriors,
                        final_round_calibration_posteriors=final_round_calibration_posteriors,
                        calibration_erasure_prob=float(e),
                    )
                    rail_row.update(
                        {
                            "scheme": "rail_calibrated",
                            "sampling_model": "rail",
                            "distance": int(distance),
                            "rounds": int(rounds),
                            "erasure_prob": float(e),
                            "pauli_prob": float(p),
                            "check_prob": float(q),
                            "shots": int(args.shots),
                            "seed": int(point_seed),
                            "rounds_per_check": int(rail_rounds_per_check),
                        }
                    )
                    rows.append(rail_row)

                    normal_row = decode_with_normal(
                        distance=int(distance),
                        rounds=int(rounds),
                        shots=int(args.shots),
                        seed=int(point_seed),
                        erasure_prob=float(e),
                        check_prob=float(q),
                        pauli_prob=float(p),
                        rounds_per_check=int(normal_rounds_per_check),
                        single_qubit_errors=bool(args.single_qubit_errors),
                    )
                    normal_row.update(
                        {
                            "scheme": "normal",
                            "sampling_model": "standard",
                            "distance": int(distance),
                            "rounds": int(rounds),
                            "erasure_prob": float(e),
                            "pauli_prob": float(p),
                            "check_prob": float(q),
                            "shots": int(args.shots),
                            "seed": int(point_seed),
                            "rounds_per_check": int(normal_rounds_per_check),
                        }
                    )
                    rows.append(normal_row)

                    elapsed = time.perf_counter() - sweep_start
                    print(
                        f"[{point_counter}/{total_points}] {point_label} elapsed={elapsed:.1f}s "
                        f"rail LER/round={rail_row['logical_error_rate_per_round']:.6g} "
                        f"decode_failures={rail_row['decode_failures']} | "
                        f"normal LER/round={normal_row['logical_error_rate_per_round']:.6g}",
                        flush=True,
                    )

    plot_paths = plot_overlays(
        rows=rows,
        configs=configs,
        p_values=p_values,
        q_values=q_values,
        plot_dir=args.plot_dir,
    )
    elapsed_total = time.perf_counter() - sweep_start

    payload = {
        "configs": [{"distance": d, "rounds": r} for d, r in configs],
        "e_values": [float(v) for v in e_values],
        "p_values": [float(v) for v in p_values],
        "q_values": [float(v) for v in q_values],
        "base_seed": int(base_seed),
        "shots_per_point": int(args.shots),
        "calibration_shots_per_point": int(args.calibration_shots),
        "calibration_strategy": "per-point non-final + per-point final-round tables",
        "rail_noise_model": {
            "control_spread": {"p_x": 0.0, "p_y": 0.0, "p_z": 0.0},
            "target_spread": {"p_x": 0.0, "p_y": 0.0, "p_z": 0.5},
        },
        "standard_noise_model": {
            "control_spread": {"p_x": 0.5, "p_y": 0.0, "p_z": 0.0},
            "target_spread": {"p_x": 0.0, "p_y": 0.0, "p_z": 0.5},
        },
        "shared_rounds_per_check_fallback": int(args.rounds_per_check),
        "rail_rounds_per_check": int(rail_rounds_per_check),
        "normal_rounds_per_check": int(normal_rounds_per_check),
        "single_qubit_errors": bool(args.single_qubit_errors),
        "elapsed_seconds": float(elapsed_total),
        "calibration_summary_by_point": calibration_summary_by_point,
        "rows": rows,
        "plot_paths": plot_paths,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2))
    print(f"saved json: {args.json_out}", flush=True)
    for path in plot_paths:
        print(f"saved plot: {path}", flush=True)


if __name__ == "__main__":
    main()
