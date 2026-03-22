#!/usr/bin/env python3
"""Locate a rail-calibrated decode failure and dump reproducible artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pymatching as pm


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

    return repo_root, qe


def _build_circuit(
    qe,
    *,
    distance: int,
    rounds: int,
    erasure_prob: float,
    rounds_per_check: int,
):
    return qe.SurfaceCodeRotated(distance).build_circuit(
        rounds=rounds,
        erasure_prob=erasure_prob,
        erasable_qubits="ALL",
        reset_failure_prob=0.0,
        single_qubit_errors=True,
        post_clifford_pauli_prob=0.0,
        rounds_per_check=rounds_per_check,
    )


def _build_model(q: float):
    import qerasure as qe  # pylint: disable=import-error

    model = qe.ErasureModel(
        2,
        qe.PauliChannel(0.25, 0.25, 0.25),
        qe.PauliChannel(0.25, 0.25, 0.25),
        qe.TQGSpreadModel(
            qe.PauliChannel(0.0, 0.0, 0.0),
            qe.PauliChannel(0.0, 0.0, 0.5),
        ),
    )
    model.check_false_negative_prob = float(q)
    model.check_false_positive_prob = float(q)
    return model


def _schedule_bucket(schedule_type: int) -> int:
    if schedule_type == 1:
        return 0
    if schedule_type == 2:
        return 1
    return -1


def _pair_inconsistency_in_round(rail_program, det_row: np.ndarray, data_qubit: int, round_index: int) -> bool:
    if round_index < 0:
        return False
    slot0, slot1 = rail_program.data_z_ancilla_slots(data_qubit)
    if slot0 < 0 or slot1 < 0:
        return False
    d0 = rail_program.round_detector_index(round_index, slot0)
    d1 = rail_program.round_detector_index(round_index, slot1)
    if d0 < 0 or d1 < 0:
        return False
    return int(det_row[d0]) != int(det_row[d1])


def _condition_bucket(rail_program, det_row: np.ndarray, data_qubit: int, check_round: int) -> int:
    inc_round1 = _pair_inconsistency_in_round(rail_program, det_row, data_qubit, check_round - 1)
    inc_round2 = _pair_inconsistency_in_round(rail_program, det_row, data_qubit, check_round)
    if inc_round1 and inc_round2:
        return 3
    if inc_round1:
        return 1
    if inc_round2:
        return 2
    return 0


def _map_onset_op_to_bin(rail_program, check_round: int, event_rows: list[dict], true_onset_op: int) -> int:
    prev_round = check_round - 1
    curr_round = check_round
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


def _calibrate_posteriors(
    qe,
    *,
    distance: int,
    rounds: int,
    shots: int,
    seed: int,
    erasure_prob: float,
    check_prob: float,
    rounds_per_check: int,
) -> tuple[list[list[list[float]]], dict]:
    circuit = _build_circuit(
        qe,
        distance=distance,
        rounds=rounds,
        erasure_prob=erasure_prob,
        rounds_per_check=rounds_per_check,
    )
    model = _build_model(check_prob)
    rail_program = qe.RailSurfaceCompiledProgram(circuit, model, distance, rounds)
    sampler = qe.RailCalibrationSampler(rail_program)
    dem_builder = qe.RailSurfaceDemBuilder(rail_program)
    dets, _obs, checks, onset_ops = sampler.sample(shots, seed, num_threads=1)

    counts = np.zeros((2, 4, 8), dtype=np.int64)
    events_per_bucket = np.zeros((2, 4), dtype=np.int64)
    for shot_idx in range(int(dets.shape[0])):
        check_row = checks[shot_idx].tolist()
        det_row = dets[shot_idx]
        onset_row = onset_ops[shot_idx]
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
            if not rail_program.data_qubit_is_full_interior(data_qubit):
                continue
            if check_round <= 0:
                continue
            s_bucket = _schedule_bucket(schedule_type)
            if s_bucket < 0:
                continue
            c_bucket = _condition_bucket(rail_program, det_row, data_qubit, check_round)
            onset_bin = _map_onset_op_to_bin(
                rail_program,
                check_round,
                event_rows,
                int(onset_row[int(check_event_index)]),
            )
            if onset_bin < 0:
                continue
            counts[s_bucket, c_bucket, onset_bin] += 1
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
        "events_per_bucket": events_per_bucket.tolist(),
        "counts": counts.tolist(),
    }
    return posteriors.tolist(), summary


def _try_decode_rail_shot(dem_builder, det_row: np.ndarray, check_row: np.ndarray) -> dict | None:
    check_list = np.asarray(check_row, dtype=np.uint8).tolist()
    det_list = np.asarray(det_row, dtype=np.uint8).tolist()
    try:
        decoded_circuit = dem_builder.build_decoded_circuit(check_list, det_list, verbose=False)
    except Exception as exc:  # pylint: disable=broad-except
        return {
            "stage": "build_decoded_circuit",
            "error": f"{type(exc).__name__}: {exc}",
            "virtual_circuit_text": None,
            "virtual_dem_text": None,
        }

    virtual_circuit_text = str(decoded_circuit)
    try:
        decoded_dem = decoded_circuit.detector_error_model(
            decompose_errors=True,
            approximate_disjoint_errors=True,
        )
    except Exception as exc:  # pylint: disable=broad-except
        return {
            "stage": "detector_error_model",
            "error": f"{type(exc).__name__}: {exc}",
            "virtual_circuit_text": virtual_circuit_text,
            "virtual_dem_text": None,
        }

    virtual_dem_text = str(decoded_dem)
    try:
        matching = pm.Matching.from_detector_error_model(decoded_dem)
    except Exception as exc:  # pylint: disable=broad-except
        return {
            "stage": "matching_from_dem",
            "error": f"{type(exc).__name__}: {exc}",
            "virtual_circuit_text": virtual_circuit_text,
            "virtual_dem_text": virtual_dem_text,
        }

    try:
        _ = matching.decode(np.asarray(det_row, dtype=np.uint8))
    except Exception as exc:  # pylint: disable=broad-except
        return {
            "stage": "matching_decode",
            "error": f"{type(exc).__name__}: {exc}",
            "virtual_circuit_text": virtual_circuit_text,
            "virtual_dem_text": virtual_dem_text,
        }

    return None


def _save_artifacts(
    output_dir: Path,
    *,
    erasure_circuit_text: str,
    logical_circuit_text: str,
    virtual_circuit_text: str | None,
    virtual_dem_text: str | None,
    virtual_dem_error: str | None,
    metadata: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "erasure_circuit.qer").write_text(erasure_circuit_text)
    (output_dir / "logical_circuit.stim").write_text(logical_circuit_text)
    if virtual_circuit_text is not None:
        (output_dir / "virtual_circuit.stim").write_text(virtual_circuit_text)
    if virtual_dem_text is not None:
        (output_dir / "virtual_dem.dem").write_text(virtual_dem_text)
    else:
        message = "# DEM generation failed.\n"
        if virtual_dem_error:
            message += f"# {virtual_dem_error}\n"
        (output_dir / "virtual_dem.dem").write_text(message)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Locate a rail-calibrated decode failure and dump artifacts."
    )
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--erasure-prob", type=float, default=0.004)
    parser.add_argument("--check-prob", type=float, default=0.005)
    parser.add_argument("--rounds-per-check", type=int, default=2)
    parser.add_argument("--shots", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=920004)
    parser.add_argument("--calibration-distance", type=int, default=7)
    parser.add_argument("--calibration-rounds", type=int, default=7)
    parser.add_argument("--calibration-erasure-prob", type=float, default=0.001)
    parser.add_argument("--calibration-shots", type=int, default=6000)
    parser.add_argument(
        "--artifact-subdir",
        type=str,
        default="rail_decode_failure",
        help="Subdirectory under tests/artifacts where artifacts are written.",
    )
    args = parser.parse_args()

    repo_root, qe = _import_qerasure()
    target = {
        "distance": int(args.distance),
        "rounds": int(args.rounds),
        "erasure_prob": float(args.erasure_prob),
        "check_prob": float(args.check_prob),
        "rounds_per_check": int(args.rounds_per_check),
    }
    calibration = {
        "distance": int(args.calibration_distance),
        "rounds": int(args.calibration_rounds),
        "erasure_prob": float(args.calibration_erasure_prob),
        "shots": int(args.calibration_shots),
    }
    shots = int(args.shots)
    seed = int(args.seed)

    artifacts_dir = repo_root / "tests" / "artifacts" / str(args.artifact_subdir)

    calibration_posteriors, calibration_summary = _calibrate_posteriors(
        qe,
        distance=int(calibration["distance"]),
        rounds=int(calibration["rounds"]),
        shots=int(calibration["shots"]),
        seed=int(seed) ^ 0xA5A5A5A5,
        erasure_prob=float(calibration["erasure_prob"]),
        check_prob=float(target["check_prob"]),
        rounds_per_check=int(target["rounds_per_check"]),
    )

    target_circuit = _build_circuit(
        qe,
        distance=int(target["distance"]),
        rounds=int(target["rounds"]),
        erasure_prob=float(target["erasure_prob"]),
        rounds_per_check=int(target["rounds_per_check"]),
    )
    target_model = _build_model(float(target["check_prob"]))
    rail_program = qe.RailSurfaceCompiledProgram(
        target_circuit,
        target_model,
        int(target["distance"]),
        int(target["rounds"]),
    )
    sampler = qe.RailStreamSampler(rail_program)
    dem_builder = qe.RailSurfaceDemBuilder(rail_program)
    dem_builder.set_calibrated_onset_posteriors(
        float(calibration["erasure_prob"]),
        calibration_posteriors,
        True,
    )

    dets, _obs, checks = sampler.sample(shots, seed, num_threads=1)
    failure = None
    for shot in range(int(dets.shape[0])):
        failure = _try_decode_rail_shot(dem_builder, dets[shot], checks[shot])
        if failure is not None:
            failure["failing_shot"] = int(shot)
            failure["check_row"] = np.asarray(checks[shot], dtype=np.uint8).tolist()
            failure["detector_row"] = np.asarray(dets[shot], dtype=np.uint8).tolist()
            break

    if failure is None:
        raise RuntimeError(
            "No rail decode failure was found with d=3, rounds=3 under current parameters."
        )

    logical_circuit_text, replayed_check_row = sampler.sample_exact_shot(seed, int(failure["failing_shot"]))
    if not np.array_equal(np.asarray(replayed_check_row, dtype=np.uint8), np.asarray(failure["check_row"], dtype=np.uint8)):
        raise RuntimeError(
            "Replayed check bits do not match failing shot; refusing to save mismatched artifacts."
        )

    metadata = {
        "target": target,
        "calibration": calibration,
        "seed": int(seed),
        "shots": int(shots),
        "failure_stage": failure["stage"],
        "failure_error": failure["error"],
        "failing_shot": int(failure["failing_shot"]),
        "check_row": [int(v) for v in failure["check_row"]],
        "replayed_check_row": [int(v) for v in np.asarray(replayed_check_row, dtype=np.uint8).tolist()],
        "detector_row": [int(v) for v in failure["detector_row"]],
        "calibration_events_per_bucket": calibration_summary["events_per_bucket"],
    }
    _save_artifacts(
        artifacts_dir,
        erasure_circuit_text=target_circuit.to_string(),
        logical_circuit_text=logical_circuit_text,
        virtual_circuit_text=failure.get("virtual_circuit_text"),
        virtual_dem_text=failure.get("virtual_dem_text"),
        virtual_dem_error=failure.get("error"),
        metadata=metadata,
    )

    print("python_rail_decode_failure_artifact_test")
    print(f"status: saved failure artifacts to {artifacts_dir}")
    print(f"failing_shot: {failure['failing_shot']}")
    print(f"failure_stage: {failure['stage']}")
    print(f"error: {failure['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
