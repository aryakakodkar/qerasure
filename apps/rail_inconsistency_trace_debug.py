#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_SRC = REPO_ROOT / "python"

import sys

if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

import qerasure as qe


def schedule_label(schedule_type: int) -> str:
    if schedule_type == 1:
        return "XZZX_interior"
    if schedule_type == 2:
        return "ZXXZ_interior"
    return "UNKNOWN_interior"


def pauli_label(code: int) -> str:
    if code == 0:
        return "X"
    if code == 1:
        return "Y"
    if code == 2:
        return "Z"
    if code == 3:
        return "I"
    return "NONE"


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
    d0 = rail_program.round_detector_index(int(round_index), slot0)
    d1 = rail_program.round_detector_index(int(round_index), slot1)
    if d0 < 0 or d1 < 0:
        return False
    return int(det_row[d0]) != int(det_row[d1])


def two_round_state(
    rail_program: qe.RailSurfaceCompiledProgram,
    det_row: np.ndarray,
    data_qubit: int,
    check_round: int,
) -> str:
    inc_round1 = pair_inconsistency_in_round(
        rail_program, det_row, data_qubit, int(check_round) - 1
    )
    inc_round2 = pair_inconsistency_in_round(
        rail_program, det_row, data_qubit, int(check_round)
    )
    if inc_round1 and inc_round2:
        return "11"
    if inc_round1 and not inc_round2:
        return "10"
    if (not inc_round1) and inc_round2:
        return "01"
    return "00"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace rail inconsistency states with latent onset-pair debug metadata."
    )
    parser.add_argument("--distance", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--rounds-per-check", type=int, default=2)
    parser.add_argument("--shots", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=920004)
    parser.add_argument("--erasure-prob", type=float, default=0.001)
    parser.add_argument("--check-fn", type=float, default=0.0)
    parser.add_argument("--check-fp", type=float, default=0.0)
    parser.add_argument(
        "--exclude-final-check-round",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--full-interior-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--max-example-rows",
        type=int,
        default=30,
        help="Max retained per-bin 11-event examples.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT
        / "apps"
        / "results"
        / "rail_inconsistency_trace_debug_q000_pe001.json",
    )
    args = parser.parse_args()

    circuit = qe.build_surface_code_erasure_circuit(
        distance=args.distance,
        rounds=args.rounds,
        erasure_prob=float(args.erasure_prob),
        erasable_qubits="ALL",
        reset_failure_prob=0.0,
        ecr_after_each_step=False,
        single_qubit_errors=False,
        post_clifford_pauli_prob=0.0,
        rounds_per_check=args.rounds_per_check,
    )
    model = qe.ErasureModel(
        2,
        qe.PauliChannel(0.25, 0.25, 0.25),
        qe.PauliChannel(0.0, 0.0, 0.0),
        qe.TQGSpreadModel(
            qe.PauliChannel(0.0, 0.0, 0.0),
            qe.PauliChannel(0.0, 0.0, 0.0),
        ),
    )
    model.check_false_negative_prob = float(args.check_fn)
    model.check_false_positive_prob = float(args.check_fp)

    rail_program = qe.RailSurfaceCompiledProgram(
        circuit=circuit,
        model=model,
        distance=args.distance,
        rounds=args.rounds,
    )
    dem_builder = qe.RailSurfaceDemBuilder(rail_program)
    sampler = qe.RailCalibrationSampler(rail_program)

    (
        dets,
        _obs,
        checks,
        onset_ops,
        onset_is_pair,
        onset_companion_qubit,
        onset_companion_pauli,
        erasure_age,
        chosen_z_rail,
    ) = sampler.sample_debug(args.shots, args.seed, num_threads=1)

    labels = [
        "round(n-1)_step1",
        "round(n-1)_step2",
        "round(n-1)_step3",
        "round(n-1)_step4",
        "round(n)_step1",
        "round(n)_step2",
        "round(n)_step3",
        "round(n)_step4",
        "no_erasure_or_older",
    ]

    # stats[qtype][onset_label] -> aggregate counters.
    stats = defaultdict(
        lambda: defaultdict(
            lambda: {
                "total": 0,
                "case_counts": {"00": 0, "10": 0, "01": 0, "11": 0},
                "pair_total": 0,
                "pair_case_counts": {"00": 0, "10": 0, "01": 0, "11": 0},
                "pair_non_i_total": 0,
                "pair_non_i_case_counts": {"00": 0, "10": 0, "01": 0, "11": 0},
                "companion_pauli_counts": {"X": 0, "Y": 0, "Z": 0, "I": 0, "NONE": 0},
                "case_counts_by_companion_pauli": {
                    "X": {"00": 0, "10": 0, "01": 0, "11": 0},
                    "Y": {"00": 0, "10": 0, "01": 0, "11": 0},
                    "Z": {"00": 0, "10": 0, "01": 0, "11": 0},
                    "I": {"00": 0, "10": 0, "01": 0, "11": 0},
                    "NONE": {"00": 0, "10": 0, "01": 0, "11": 0},
                },
            }
        )
    )
    examples = defaultdict(
        lambda: defaultdict(list)
    )  # examples[qtype][onset_label] -> list of 11 rows

    for shot in range(args.shots):
        check_row = checks[shot].tolist()
        det_row = dets[shot]
        rows = dem_builder.calibration_rows(check_row, det_row.tolist())
        if not rows:
            continue

        grouped = defaultdict(list)
        for row in rows:
            key = (
                int(row["check_event_index"]),
                int(row["data_qubit"]),
                int(row["check_round"]),
                int(row["schedule_type"]),
                bool(row["boundary_data_qubit"]),
            )
            grouped[key].append(row)

        for key, event_rows in grouped.items():
            check_event_index, data_qubit, check_round, schedule_type, is_boundary = key
            if args.exclude_final_check_round and int(check_round) == int(args.rounds) - 1:
                continue
            if args.full_interior_only:
                if not rail_program.data_qubit_is_full_interior(data_qubit):
                    continue
            elif is_boundary:
                continue

            onset_op = int(onset_ops[shot, check_event_index])
            prev_round = int(check_round) - 1
            curr_round = int(check_round)
            rows_by_round: dict[int, list[dict]] = {prev_round: [], curr_round: []}
            for row in event_rows:
                onset_round = rail_program.op_round(int(row["onset_op_index"]))
                if onset_round in rows_by_round:
                    rows_by_round[onset_round].append(row)
            for rr in rows_by_round:
                rows_by_round[rr].sort(key=lambda row: int(row["onset_op_index"]))

            onset_to_bin: dict[int, int] = {}
            for j, row in enumerate(rows_by_round[prev_round][:4]):
                onset_to_bin[int(row["onset_op_index"])] = j
            for j, row in enumerate(rows_by_round[curr_round][:4]):
                onset_to_bin[int(row["onset_op_index"])] = 4 + j
            mapped_index = onset_to_bin.get(onset_op, -1)
            onset_label = labels[mapped_index] if mapped_index >= 0 else "no_erasure_or_older"

            state = two_round_state(
                rail_program=rail_program,
                det_row=det_row,
                data_qubit=data_qubit,
                check_round=check_round,
            )
            qtype = schedule_label(schedule_type)
            item = stats[qtype][onset_label]
            item["total"] += 1
            item["case_counts"][state] += 1

            pair_flag = int(onset_is_pair[shot, check_event_index]) == 1
            comp_qubit = int(onset_companion_qubit[shot, check_event_index])
            comp_pauli_code = int(onset_companion_pauli[shot, check_event_index])
            comp_pauli = pauli_label(comp_pauli_code)
            item["companion_pauli_counts"][comp_pauli] += 1
            item["case_counts_by_companion_pauli"][comp_pauli][state] += 1
            if pair_flag:
                item["pair_total"] += 1
                item["pair_case_counts"][state] += 1
                if comp_pauli in ("X", "Y", "Z"):
                    item["pair_non_i_total"] += 1
                    item["pair_non_i_case_counts"][state] += 1

            if state == "11" and len(examples[qtype][onset_label]) < args.max_example_rows:
                examples[qtype][onset_label].append(
                    {
                        "shot": int(shot),
                        "check_event_index": int(check_event_index),
                        "data_qubit": int(data_qubit),
                        "check_round": int(check_round),
                        "onset_op_index": int(onset_op),
                        "onset_is_pair": int(pair_flag),
                        "companion_qubit": int(comp_qubit),
                        "companion_pauli": comp_pauli,
                        "erasure_age_at_check": int(erasure_age[shot, check_event_index]),
                        "chosen_z_rail": int(chosen_z_rail[shot, check_event_index]),
                    }
                )

    summary = defaultdict(dict)
    for qtype, bins in stats.items():
        for onset_label, agg in bins.items():
            total = int(agg["total"])
            if total <= 0:
                continue
            pair_total = int(agg["pair_total"])
            non_i_total = int(agg["pair_non_i_total"])
            p11 = float(agg["case_counts"]["11"]) / float(total)
            p11_pair = (
                float(agg["pair_case_counts"]["11"]) / float(pair_total)
                if pair_total > 0
                else 0.0
            )
            p11_pair_non_i = (
                float(agg["pair_non_i_case_counts"]["11"]) / float(non_i_total)
                if non_i_total > 0
                else 0.0
            )
            summary[qtype][onset_label] = {
                "total": total,
                "p11": p11,
                "case_counts": agg["case_counts"],
                "pair_total": pair_total,
                "pair_fraction": float(pair_total) / float(total),
                "p11_given_pair": p11_pair,
                "pair_non_i_total": non_i_total,
                "p11_given_pair_non_i": p11_pair_non_i,
                "companion_pauli_counts": agg["companion_pauli_counts"],
                "case_counts_by_companion_pauli": agg["case_counts_by_companion_pauli"],
            }

    payload = {
        "metadata": {
            "distance": int(args.distance),
            "rounds": int(args.rounds),
            "rounds_per_check": int(args.rounds_per_check),
            "shots": int(args.shots),
            "seed": int(args.seed),
            "erasure_prob": float(args.erasure_prob),
            "check_false_negative_prob": float(args.check_fn),
            "check_false_positive_prob": float(args.check_fp),
            "exclude_final_check_round": bool(args.exclude_final_check_round),
            "full_interior_only": bool(args.full_interior_only),
        },
        "summary": summary,
        "examples_11": examples,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2))

    focus_bins = ["round(n-1)_step1", "round(n-1)_step2", "round(n-1)_step3", "round(n-1)_step4"]
    for qtype in ("XZZX_interior", "ZXXZ_interior"):
        print(f"\n{qtype}:")
        qsum = summary.get(qtype, {})
        for b in focus_bins:
            if b not in qsum:
                continue
            row = qsum[b]
            print(
                f"  {b}: p11={row['p11']:.4f}, pair_frac={row['pair_fraction']:.4f}, "
                f"p11|pair={row['p11_given_pair']:.4f}, "
                f"p11|pair_nonI={row['p11_given_pair_non_i']:.4f}, "
                f"N={row['total']}"
            )
    print(f"\nSaved: {args.json_out}")


if __name__ == "__main__":
    main()
