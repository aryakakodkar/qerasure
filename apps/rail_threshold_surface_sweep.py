#!/usr/bin/env python3
"""Build threshold surfaces over (p, q) for rail and standard decoders."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import secrets
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pymatching as pm
from matplotlib.ticker import LogLocator

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
    inc_round1 = pair_inconsistency_in_round(
        rail_program, det_row, data_qubit, int(check_round) - 1
    )
    inc_round2 = pair_inconsistency_in_round(
        rail_program, det_row, data_qubit, int(check_round)
    )
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
    first_decode_error: str | None = None
    first_decode_traceback: str | None = None
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
        except Exception as exc:
            decode_failures += 1
            if first_decode_error is None:
                first_decode_error = f"{type(exc).__name__}: {exc}"
                first_decode_traceback = traceback.format_exc()
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
        "first_decode_error": first_decode_error,
        "first_decode_traceback": first_decode_traceback,
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


def _crossings_linear(
    curve_small: list[tuple[float, float]],
    curve_large: list[tuple[float, float]],
) -> list[float]:
    roots: list[float] = []
    if len(curve_small) != len(curve_large):
        return roots
    for i in range(len(curve_small) - 1):
        p0, y0s = curve_small[i]
        p1, y1s = curve_small[i + 1]
        q0, y0l = curve_large[i]
        q1, y1l = curve_large[i + 1]
        if abs(p0 - q0) > 1e-15 or abs(p1 - q1) > 1e-15:
            return []
        d0 = y0s - y0l
        d1 = y1s - y1l
        if d0 * d1 < 0.0:
            frac = -d0 / (d1 - d0)
            roots.append(p0 + frac * (p1 - p0))
    unique: list[float] = []
    for x in sorted(roots):
        if not unique or abs(x - unique[-1]) > 1e-12:
            unique.append(x)
    return unique


def estimate_threshold_for_slice(
    rows: list[dict],
    scheme: str,
    q_value: float,
    p_value: float,
    metric: str = "logical_error_rate_per_round",
) -> dict:
    subset = [
        r
        for r in rows
        if str(r["scheme"]) == scheme
        and abs(float(r["check_prob"]) - float(q_value)) < 1e-15
        and abs(float(r["pauli_prob"]) - float(p_value)) < 1e-15
    ]
    curves: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for row in subset:
        curves[int(row["distance"])].append((float(row["erasure_prob"]), float(row[metric])))
    for d in curves:
        curves[d].sort(key=lambda t: t[0])
    distances = sorted(curves.keys())
    pair_crossings: list[dict] = []
    for i in range(len(distances) - 1):
        d_small = distances[i]
        d_large = distances[i + 1]
        roots = _crossings_linear(curves[d_small], curves[d_large])
        pair_crossings.append({"pair": [d_small, d_large], "crossings": roots})

    threshold_estimate = None
    status = "unresolved"
    if distances == [3, 5, 7]:
        roots_35 = pair_crossings[0]["crossings"] if pair_crossings else []
        roots_57 = pair_crossings[1]["crossings"] if len(pair_crossings) > 1 else []
        best = None
        for r35 in roots_35:
            for r57 in roots_57:
                gap = abs(float(r35) - float(r57))
                mid = 0.5 * (float(r35) + float(r57))
                if best is None or gap < best[0]:
                    best = (gap, mid)
        if best is not None:
            threshold_estimate = float(best[1])
            status = "resolved"
    else:
        selected = []
        for row in pair_crossings:
            roots = row["crossings"]
            if roots:
                selected.append(float(roots[0]))
        if selected:
            threshold_estimate = float(np.mean(np.asarray(selected, dtype=float)))
            status = "resolved_partial"

    return {
        "scheme": scheme,
        "check_prob": float(q_value),
        "pauli_prob": float(p_value),
        "metric": metric,
        "distances": distances,
        "pair_crossings": pair_crossings,
        "threshold_estimate": threshold_estimate,
        "status": status,
    }


def summarize_decode_cycle_seconds(rows: list[dict], q_value: float, p_value: float) -> dict:
    subset = [
        r
        for r in rows
        if abs(float(r["check_prob"]) - float(q_value)) < 1e-15
        and abs(float(r["pauli_prob"]) - float(p_value)) < 1e-15
    ]
    rail_total = float(
        np.sum(
            [
                float(r["timing_seconds"]["total"])
                for r in subset
                if str(r["scheme"]) == "rail_calibrated"
            ],
            dtype=float,
        )
    )
    normal_total = float(
        np.sum(
            [
                float(r["timing_seconds"]["total"])
                for r in subset
                if str(r["scheme"]) == "normal"
            ],
            dtype=float,
        )
    )
    return {
        "check_prob": float(q_value),
        "pauli_prob": float(p_value),
        "rail_decode_cycle_seconds": rail_total,
        "normal_decode_cycle_seconds": normal_total,
        "combined_decode_cycle_seconds": rail_total + normal_total,
    }


def run_single_point(task: dict) -> dict:
    distance = int(task["distance"])
    rounds = int(task["rounds"])
    erasure_prob = float(task["erasure_prob"])
    pauli_prob = float(task["pauli_prob"])
    check_prob = float(task["check_prob"])
    point_seed = int(task["point_seed"])
    calib_seed = int(task["calib_seed"])
    rail_rounds_per_check = int(task["rail_rounds_per_check"])
    normal_rounds_per_check = int(task["normal_rounds_per_check"])
    single_qubit_errors = bool(task["single_qubit_errors"])
    calibration_shots = int(task["calibration_shots"])
    shots = int(task["shots"])

    calibration_posteriors, calibration_summary = calibrate_onset_posteriors(
        distance=distance,
        rounds=rounds,
        shots=calibration_shots,
        seed=calib_seed,
        erasure_prob=erasure_prob,
        check_prob=check_prob,
        rounds_per_check=rail_rounds_per_check,
        single_qubit_errors=single_qubit_errors,
        pauli_prob=pauli_prob,
        final_round_only=False,
    )
    final_round_calibration_posteriors, final_round_calibration_summary = calibrate_onset_posteriors(
        distance=distance,
        rounds=rounds,
        shots=calibration_shots,
        seed=int((calib_seed ^ 0x6C6C6C6C) & 0xFFFFFFFF),
        erasure_prob=erasure_prob,
        check_prob=check_prob,
        rounds_per_check=rail_rounds_per_check,
        single_qubit_errors=single_qubit_errors,
        pauli_prob=pauli_prob,
        final_round_only=True,
    )

    rail_row = decode_with_rail(
        distance=distance,
        rounds=rounds,
        shots=shots,
        seed=point_seed,
        erasure_prob=erasure_prob,
        check_prob=check_prob,
        pauli_prob=pauli_prob,
        rounds_per_check=rail_rounds_per_check,
        single_qubit_errors=single_qubit_errors,
        calibration_posteriors=calibration_posteriors,
        final_round_calibration_posteriors=final_round_calibration_posteriors,
        calibration_erasure_prob=erasure_prob,
    )
    rail_row.update(
        {
            "scheme": "rail_calibrated",
            "sampling_model": "rail",
            "distance": distance,
            "rounds": rounds,
            "erasure_prob": erasure_prob,
            "pauli_prob": pauli_prob,
            "check_prob": check_prob,
            "shots": shots,
            "seed": point_seed,
            "rounds_per_check": rail_rounds_per_check,
        }
    )

    normal_row = decode_with_normal(
        distance=distance,
        rounds=rounds,
        shots=shots,
        seed=point_seed,
        erasure_prob=erasure_prob,
        check_prob=check_prob,
        pauli_prob=pauli_prob,
        rounds_per_check=normal_rounds_per_check,
        single_qubit_errors=single_qubit_errors,
    )
    normal_row.update(
        {
            "scheme": "normal",
            "sampling_model": "standard",
            "distance": distance,
            "rounds": rounds,
            "erasure_prob": erasure_prob,
            "pauli_prob": pauli_prob,
            "check_prob": check_prob,
            "shots": shots,
            "seed": point_seed,
            "rounds_per_check": normal_rounds_per_check,
        }
    )

    return {
        "task_id": int(task["task_id"]),
        "point_label": str(task["point_label"]),
        "calibration_summary": {
            "non_final": calibration_summary,
            "final_round": final_round_calibration_summary,
        },
        "rail_row": rail_row,
        "normal_row": normal_row,
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
            fig, ax = plt.subplots(figsize=(8.8, 5.4))
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
            ax.set_title(f"Threshold Curves (q={q:.6g}, p={p:.6g})")
            ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
            ax.grid(True, which="major", alpha=0.35)
            ax.grid(True, which="minor", axis="x", alpha=0.22)
            ax.legend(fontsize=8)
            fig.tight_layout()
            out_path = plot_dir / f"threshold_curves_q{q:.6g}_p{p:.6g}.png"
            fig.savefig(out_path, dpi=220)
            plt.close(fig)
            saved.append(str(out_path))
    return saved


def plot_threshold_surfaces(
    threshold_rows: list[dict],
    p_values: list[float],
    q_values: list[float],
    plot_dir: Path,
) -> list[str]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    schemes = ("normal", "rail_calibrated")
    for scheme in schemes:
        grid = np.full((len(q_values), len(p_values)), np.nan, dtype=float)
        for qi, q in enumerate(q_values):
            for pi, p in enumerate(p_values):
                row = next(
                    (
                        r
                        for r in threshold_rows
                        if r["scheme"] == scheme
                        and abs(float(r["check_prob"]) - float(q)) < 1e-15
                        and abs(float(r["pauli_prob"]) - float(p)) < 1e-15
                    ),
                    None,
                )
                if row is None:
                    continue
                value = row.get("threshold_estimate")
                if value is None:
                    continue
                grid[qi, pi] = float(value)

        fig, ax = plt.subplots(figsize=(8.2, 4.6))
        masked = np.ma.masked_invalid(grid)
        im = ax.imshow(masked, origin="lower", aspect="auto", cmap="viridis")
        ax.set_xticks(np.arange(len(p_values)))
        ax.set_xticklabels([f"{p:.4g}" for p in p_values], rotation=0)
        ax.set_yticks(np.arange(len(q_values)))
        ax.set_yticklabels([f"{q:.4g}" for q in q_values])
        ax.set_xlabel("Pauli probability p")
        ax.set_ylabel("Check error probability q")
        ax.set_title(f"Estimated Threshold Surface ({scheme})")
        for qi in range(len(q_values)):
            for pi in range(len(p_values)):
                value = grid[qi, pi]
                text = "NA" if not math.isfinite(value) else f"{value:.3g}"
                ax.text(pi, qi, text, ha="center", va="center", color="white", fontsize=8)
        cbar = fig.colorbar(im, ax=ax, fraction=0.048, pad=0.03)
        cbar.set_label("Estimated e threshold")
        fig.tight_layout()
        out_path = plot_dir / f"threshold_surface_{scheme}.png"
        fig.savefig(out_path, dpi=220)
        plt.close(fig)
        saved.append(str(out_path))
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate threshold surfaces over (p, q) for rail-calibrated and "
            "standard decoding by sweeping e and extracting d=3/5/7 crossings."
        )
    )
    parser.add_argument("--configs", type=str, default="3,3;5,5;7,7")
    parser.add_argument(
        "--e-values",
        type=str,
        default="",
        help="Optional explicit comma-separated e values. If omitted, uses log spacing.",
    )
    parser.add_argument("--e-min", type=float, default=1e-3)
    parser.add_argument("--e-max", type=float, default=4e-2)
    parser.add_argument("--e-count", type=int, default=12)
    parser.add_argument("--p-values", type=str, default="0,0.0033,0.0066,0.01")
    parser.add_argument("--q-values", type=str, default="0,0.005,0.01")
    parser.add_argument("--shots", type=int, default=100000)
    parser.add_argument("--calibration-shots", type=int, default=30000)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base RNG seed. If omitted, a fresh random 32-bit seed is used each run.",
    )
    parser.add_argument("--rail-rounds-per-check", type=int, default=2)
    parser.add_argument("--normal-rounds-per-check", type=int, default=1)
    parser.add_argument(
        "--sweep-workers",
        type=int,
        default=1,
        help="Number of parallel worker processes over sweep points.",
    )
    parser.add_argument(
        "--single-qubit-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / "apps" / "results" / "rail_threshold_surface_sweep.json",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=REPO_ROOT / "apps" / "results" / "rail_threshold_surface",
    )
    args = parser.parse_args()

    if args.shots <= 0:
        raise ValueError("--shots must be positive")
    if args.calibration_shots <= 0:
        raise ValueError("--calibration-shots must be positive")
    if int(args.rail_rounds_per_check) <= 0:
        raise ValueError("--rail-rounds-per-check must be positive")
    if int(args.normal_rounds_per_check) <= 0:
        raise ValueError("--normal-rounds-per-check must be positive")
    if int(args.sweep_workers) <= 0:
        raise ValueError("--sweep-workers must be positive")

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
    print(
        f"configs={len(configs)} e={len(e_values)} p={len(p_values)} q={len(q_values)} "
        f"total_points={total_points} sweep_workers={int(args.sweep_workers)}",
        flush=True,
    )

    rows: list[dict] = []
    calibration_summary_by_point: dict[str, dict] = {}
    pair_decode_cycle_seconds: list[dict] = []
    threshold_surface_rows: list[dict] = []
    point_tasks: list[dict] = []
    sweep_start = time.perf_counter()
    task_id = 0
    for q_idx, q in enumerate(q_values):
        for p_idx, p in enumerate(p_values):
            for cfg_idx, (distance, rounds) in enumerate(configs):
                for e_idx, e in enumerate(e_values):
                    task_id += 1
                    point_seed = (
                        base_seed + q_idx * 10_000_000 + p_idx * 1_000_000 + cfg_idx * 10_000 + e_idx
                    ) & 0xFFFFFFFF
                    calib_seed = (point_seed ^ 0xA5A5A5A5) & 0xFFFFFFFF
                    point_label = (
                        f"d={distance},r={rounds},e={float(e):.12g},"
                        f"p={float(p):.12g},q={float(q):.12g}"
                    )
                    point_tasks.append(
                        {
                            "task_id": int(task_id),
                            "point_label": point_label,
                            "distance": int(distance),
                            "rounds": int(rounds),
                            "erasure_prob": float(e),
                            "pauli_prob": float(p),
                            "check_prob": float(q),
                            "point_seed": int(point_seed),
                            "calib_seed": int(calib_seed),
                            "rail_rounds_per_check": int(args.rail_rounds_per_check),
                            "normal_rounds_per_check": int(args.normal_rounds_per_check),
                            "single_qubit_errors": bool(args.single_qubit_errors),
                            "calibration_shots": int(args.calibration_shots),
                            "shots": int(args.shots),
                        }
                    )

    completed = 0

    def handle_point_result(result: dict) -> None:
        nonlocal completed
        completed += 1
        point_label = str(result["point_label"])
        calibration_summary_by_point[point_label] = result["calibration_summary"]
        rail_row = result["rail_row"]
        normal_row = result["normal_row"]
        rows.append(rail_row)
        rows.append(normal_row)
        elapsed = time.perf_counter() - sweep_start
        print(
            f"[{completed}/{total_points}] {point_label} elapsed={elapsed:.1f}s "
            f"rail LER/round={rail_row['logical_error_rate_per_round']:.6g} "
            f"decode_failures={rail_row['decode_failures']} | "
            f"normal LER/round={normal_row['logical_error_rate_per_round']:.6g}",
            flush=True,
        )
        if int(rail_row.get("decode_failures", 0)) > 0:
            print(
                f"  [rail decode failure] first_decode_error: "
                f"{rail_row.get('first_decode_error', '<none captured>')}",
                flush=True,
            )
            if rail_row.get("first_decode_traceback"):
                print(rail_row["first_decode_traceback"], flush=True)

    if int(args.sweep_workers) == 1:
        for task in point_tasks:
            handle_point_result(run_single_point(task))
    else:
        def consume_futures(executor: cf.Executor) -> None:
            future_to_task = {
                executor.submit(run_single_point, task): task for task in point_tasks
            }
            for future in cf.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"sweep worker failed for point {task['point_label']}"
                    ) from exc
                handle_point_result(result)

        try:
            with cf.ProcessPoolExecutor(max_workers=int(args.sweep_workers)) as executor:
                consume_futures(executor)
        except Exception as exc:
            if "Operation not permitted" not in str(exc):
                raise
            print(
                "warning: process sweep workers unavailable; falling back to serial execution",
                flush=True,
            )
            for task in point_tasks:
                handle_point_result(run_single_point(task))

    for q in q_values:
        for p in p_values:
            pair_decode_summary = summarize_decode_cycle_seconds(
                rows=rows,
                q_value=float(q),
                p_value=float(p),
            )
            pair_decode_cycle_seconds.append(pair_decode_summary)

            rail_threshold = estimate_threshold_for_slice(
                rows=rows,
                scheme="rail_calibrated",
                q_value=float(q),
                p_value=float(p),
            )
            normal_threshold = estimate_threshold_for_slice(
                rows=rows,
                scheme="normal",
                q_value=float(q),
                p_value=float(p),
            )
            threshold_surface_rows.append(rail_threshold)
            threshold_surface_rows.append(normal_threshold)
            print(
                f"[pair complete] p={float(p):.6g} q={float(q):.6g} "
                f"decode_cycle_s rail={pair_decode_summary['rail_decode_cycle_seconds']:.2f} "
                f"normal={pair_decode_summary['normal_decode_cycle_seconds']:.2f} "
                f"combined={pair_decode_summary['combined_decode_cycle_seconds']:.2f} | "
                f"e_th rail={rail_threshold['threshold_estimate']} normal={normal_threshold['threshold_estimate']}",
                flush=True,
            )

    overlay_paths = plot_overlays(
        rows=rows,
        configs=configs,
        p_values=p_values,
        q_values=q_values,
        plot_dir=args.plot_dir,
    )
    surface_paths = plot_threshold_surfaces(
        threshold_rows=threshold_surface_rows,
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
        "rail_noise_model": {
            "control_spread": {"p_x": 0.0, "p_y": 0.0, "p_z": 0.0},
            "target_spread": {"p_x": 0.0, "p_y": 0.0, "p_z": 0.5},
        },
        "standard_noise_model": {
            "control_spread": {"p_x": 0.5, "p_y": 0.0, "p_z": 0.0},
            "target_spread": {"p_x": 0.0, "p_y": 0.0, "p_z": 0.5},
        },
        "rail_rounds_per_check": int(args.rail_rounds_per_check),
        "normal_rounds_per_check": int(args.normal_rounds_per_check),
        "sweep_workers": int(args.sweep_workers),
        "single_qubit_errors": bool(args.single_qubit_errors),
        "elapsed_seconds": float(elapsed_total),
        "calibration_summary_by_point": calibration_summary_by_point,
        "pair_decode_cycle_seconds": pair_decode_cycle_seconds,
        "threshold_surface": threshold_surface_rows,
        "rows": rows,
        "overlay_plot_paths": overlay_paths,
        "surface_plot_paths": surface_paths,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2))
    print(f"saved json: {args.json_out}", flush=True)
    for path in overlay_paths:
        print(f"saved overlay plot: {path}", flush=True)
    for path in surface_paths:
        print(f"saved surface plot: {path}", flush=True)


if __name__ == "__main__":
    main()
