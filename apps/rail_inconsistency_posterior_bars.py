#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def has_pair_inconsistency(
    rail_program: qe.RailSurfaceCompiledProgram,
    det_row: np.ndarray,
    data_qubit: int,
    check_round: int,
    window: str,
) -> bool:
    slot0, slot1 = rail_program.data_z_ancilla_slots(data_qubit)
    if slot0 < 0 or slot1 < 0:
        return False
    if window == "current_round":
        start_round = check_round
    elif window == "lookback_1":
        start_round = max(0, check_round - 1)
    else:
        raise ValueError(f"Unsupported inconsistency window '{window}'")
    for r in range(start_round, check_round + 1):
        if pair_inconsistency_in_round(rail_program, det_row, data_qubit, r):
            return True
    return False


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


def classify_two_round_case(
    rail_program: qe.RailSurfaceCompiledProgram,
    det_row: np.ndarray,
    data_qubit: int,
    check_round: int,
    include_round1: bool = True,
) -> str:
    # round1 := check_round-1, round2 := check_round
    # where round2 is the round immediately preceding the flagged check.
    inc_round1 = (
        pair_inconsistency_in_round(
            rail_program, det_row, data_qubit, int(check_round) - 1
        )
        if include_round1
        else False
    )
    inc_round2 = pair_inconsistency_in_round(
        rail_program, det_row, data_qubit, int(check_round)
    )
    if inc_round1 and inc_round2:
        return "inc_both"
    if inc_round1:
        return "inc_round1_only"
    if inc_round2:
        return "inc_round2_only"
    return "consistent_both"


def event_matches_condition(
    condition: str,
    rail_program: qe.RailSurfaceCompiledProgram,
    det_row: np.ndarray,
    data_qubit: int,
    check_round: int,
    inconsistency_window: str,
) -> bool:
    inconsistent = has_pair_inconsistency(
        rail_program,
        det_row,
        data_qubit,
        check_round,
        inconsistency_window,
    )
    if condition == "inconsistency":
        return inconsistent
    if condition == "no_inconsistency":
        return not inconsistent
    raise ValueError(f"Unsupported condition '{condition}'")


def z_steps_for_schedule_type(schedule_type: int) -> tuple[int, int]:
    # Rail pathway is surface-code specific:
    # XZZX interior interacts with Z ancillas at steps 2 and 3,
    # ZXXZ interior interacts with Z ancillas at steps 1 and 4.
    if schedule_type == 1:
        return (2, 3)
    if schedule_type == 2:
        return (1, 4)
    raise ValueError(f"Unsupported interior schedule_type={schedule_type}")


def branch_inconsistency_likelihood(
    schedule_type: int,
    onset_step: int,
    onset_flip_probability: float,
) -> float:
    z_step_a, z_step_b = z_steps_for_schedule_type(schedule_type)
    z_steps = (z_step_a, z_step_b)
    if onset_step < 1 or onset_step > 4:
        raise ValueError(f"onset_step must be in [1,4], got {onset_step}")
    partner_index = -1
    if onset_step == z_step_a:
        partner_index = 0
    elif onset_step == z_step_b:
        partner_index = 1

    total = 0.0
    for chosen_index in (0, 1):
        det = [0, 0]
        if onset_step < z_steps[chosen_index]:
            det[chosen_index] = 1
        if partner_index < 0:
            incons = 1.0 if det[0] != det[1] else 0.0
        else:
            other = 1 - partner_index
            base = det[partner_index] ^ det[other]
            incons = (1.0 - onset_flip_probability) if base == 1 else onset_flip_probability
        total += 0.5 * incons
    return float(total)


def branch_two_round_case_likelihood(
    schedule_type: int,
    onset_round_offset: int,
    onset_step: int,
    onset_flip_probability: float,
    case_key: str,
) -> float:
    z_step_a, z_step_b = z_steps_for_schedule_type(schedule_type)
    z_steps = (z_step_a, z_step_b)
    partner_index = -1
    if onset_step == z_step_a:
        partner_index = 0
    elif onset_step == z_step_b:
        partner_index = 1

    total = 0.0
    for chosen_index in (0, 1):
        for onset_flip_bit, p_flip in (
            (0, 1.0 - onset_flip_probability),
            (1, onset_flip_probability),
        ):
            if partner_index < 0 and onset_flip_bit == 1:
                continue
            if partner_index < 0:
                p_flip = 1.0
            det = [[0, 0], [0, 0]]  # [round1, round2] x [z0, z1]
            for r in (0, 1):
                for anc in (0, 1):
                    erasure_bit = 0
                    if r > onset_round_offset:
                        erasure_bit = 1 if chosen_index == anc else 0
                    elif r == onset_round_offset:
                        erasure_bit = 1 if (chosen_index == anc and onset_step < z_steps[anc]) else 0
                    onset_flip_here = (
                        1
                        if (
                            onset_flip_bit == 1
                            and r == onset_round_offset
                            and partner_index == anc
                        )
                        else 0
                    )
                    det[r][anc] = erasure_bit ^ onset_flip_here
            inc_round1 = (det[0][0] ^ det[0][1]) == 1
            inc_round2 = (det[1][0] ^ det[1][1]) == 1
            if case_key == "inc_round1_only":
                match = inc_round1 and (not inc_round2)
            elif case_key == "inc_round2_only":
                match = (not inc_round1) and inc_round2
            elif case_key == "inc_both":
                match = inc_round1 and inc_round2
            elif case_key == "consistent_both":
                match = (not inc_round1) and (not inc_round2)
            else:
                raise ValueError(f"Unsupported case key '{case_key}'")
            if match:
                total += 0.5 * p_flip
    return float(total)


def condition_display_label(condition_key: str) -> str:
    if condition_key == "inconsistency":
        return "inconsistency"
    if condition_key == "no_inconsistency":
        return "consistency"
    if condition_key == "inc_round1_only":
        return "inc only in round1 (10)"
    if condition_key == "inc_round2_only":
        return "inc only in round2 (01)"
    if condition_key == "inc_both":
        return "inc in both rounds (11)"
    if condition_key == "consistent_both":
        return "consistent in both (00)"
    return condition_key


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot 9-way conditional posterior probabilities "
            "(8 onset spots + no-erasure) given Z-ancilla inconsistency."
        )
    )
    parser.add_argument("--distance", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--shots", type=int, default=10000)
    parser.add_argument(
        "--adaptive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Adaptively increase shots until error bars are below target relative threshold.",
    )
    parser.add_argument(
        "--chunk-shots",
        type=int,
        default=5000,
        help="Additional shots sampled per adaptive iteration.",
    )
    parser.add_argument(
        "--max-shots",
        type=int,
        default=200000,
        help="Maximum total shots when adaptive mode is enabled.",
    )
    parser.add_argument(
        "--target-rel-error",
        type=float,
        default=0.2,
        help="Stop when all nonzero bins satisfy error_bar <= target_rel_error * mean.",
    )
    parser.add_argument(
        "--min-events-per-type",
        type=int,
        default=200,
        help="Minimum qualifying events per qubit type before convergence can be declared.",
    )
    parser.add_argument(
        "--isolate-single-flagged-data-check",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Only use shots with exactly one flagged data-qubit erasure check in the whole shot. "
            "This reduces multi-erasure contamination in conditioning statistics."
        ),
    )
    parser.add_argument(
        "--require-prev-check-unflagged",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep only flagged checks whose immediately previous check on the same qubit "
            "exists and is unflagged."
        ),
    )
    parser.add_argument(
        "--truncate-lookback-on-consecutive-flag",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For --condition three_case, if previous same-qubit check is flagged, "
            "only use round(n) (drop round(n-1)) to avoid double counting."
        ),
    )
    parser.add_argument(
        "--full-interior-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Restrict to data qubits with both X-ancilla partners and both Z-ancilla partners. "
            "For rounds_per_check=1 this yields the expected four onset opportunities."
        ),
    )
    parser.add_argument(
        "--exclude-final-check-round",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Exclude flagged checks that occur in the final circuit round from calibration stats."
        ),
    )
    parser.add_argument("--seed", type=int, default=920004)
    parser.add_argument("--erasure-prob", type=float, default=0.02)
    parser.add_argument("--check-fn", type=float, default=0.0)
    parser.add_argument("--check-fp", type=float, default=0.0)
    parser.add_argument("--rounds-per-check", type=int, default=2)
    parser.add_argument(
        "--condition",
        choices=["inconsistency", "no_inconsistency", "both", "three_case"],
        default="both",
        help="Conditioning event(s) for flagged data checks.",
    )
    parser.add_argument(
        "--canonical-step-bins",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For rounds_per_check=1, map onset hypotheses to per-round canonical step bins "
            "(step1..step4) instead of raw spot indices."
        ),
    )
    parser.add_argument(
        "--inconsistency-window",
        choices=["current_round", "lookback_1"],
        default="current_round",
        help=(
            "Which rounds are used when detecting Z-ancilla inconsistency for conditioning. "
            "'current_round' aligns with 4-step onset opportunities at rounds_per_check=1."
        ),
    )
    parser.add_argument(
        "--calibration-source",
        choices=["latent", "inferred"],
        default="latent",
        help=(
            "How onset posteriors are obtained. "
            "'latent' uses true onset locations from a calibration-only sampler; "
            "'inferred' uses DEM-builder posterior inference."
        ),
    )
    parser.add_argument(
        "--conditioning-signal",
        choices=["pair_parity", "full_detectors"],
        default="pair_parity",
        help=(
            "How to compute onset posterior under the selected condition. "
            "'pair_parity' conditions only on Z-ancilla parity event; "
            "'full_detectors' uses full detector pattern likelihood from DEM calibration rows."
        ),
    )
    parser.add_argument(
        "--full-lookback-bins",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For rounds_per_check=1, show full max-persistence lookback hypotheses as "
            "round(n-1)_step1..4 + round(n)_step1..4 (+ no_erasure if enabled)."
        ),
    )
    parser.add_argument(
        "--include-no-erasure",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include explicit no_erasure bar. When false, omit it from plots/JSON.",
    )
    parser.add_argument(
        "--png-out",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--inverse-png-out",
        type=Path,
        default=None,
        help="Output PNG for inverse posterior plot P(condition | onset).",
    )
    parser.add_argument(
        "--inverse-json-out",
        type=Path,
        default=None,
        help="Output JSON for inverse posterior summary P(condition | onset).",
    )
    args = parser.parse_args()
    if args.png_out is None:
        if args.condition == "three_case":
            suffix = "three_case_4x2"
        elif args.condition == "both":
            suffix = "inconsistency_consistency_4x4"
        else:
            suffix = "inconsistency" if args.condition == "inconsistency" else "no_inconsistency"
        args.png_out = REPO_ROOT / "apps" / "results" / f"rail_{suffix}_posterior_bars.png"
    if args.json_out is None:
        if args.condition == "three_case":
            suffix = "three_case_4x2"
        elif args.condition == "both":
            suffix = "inconsistency_consistency_4x4"
        else:
            suffix = "inconsistency" if args.condition == "inconsistency" else "no_inconsistency"
        args.json_out = REPO_ROOT / "apps" / "results" / f"rail_{suffix}_posterior_bars.json"
    if args.inverse_png_out is None:
        args.inverse_png_out = args.png_out.with_name(
            f"{args.png_out.stem}_condition_given_onset{args.png_out.suffix}"
        )
    if args.inverse_json_out is None:
        args.inverse_json_out = args.json_out.with_name(
            f"{args.json_out.stem}_condition_given_onset{args.json_out.suffix}"
        )
    if args.shots <= 0:
        raise ValueError("--shots must be positive")
    if args.chunk_shots <= 0:
        raise ValueError("--chunk-shots must be positive")
    if args.max_shots <= 0:
        raise ValueError("--max-shots must be positive")
    if args.target_rel_error <= 0.0:
        raise ValueError("--target-rel-error must be > 0")
    if args.min_events_per_type <= 0:
        raise ValueError("--min-events-per-type must be positive")
    if args.condition == "three_case" and not (
        args.canonical_step_bins and args.full_lookback_bins
    ):
        raise ValueError(
            "--condition three_case requires "
            "--canonical-step-bins and --full-lookback-bins."
        )

    circuit = qe.build_surface_code_erasure_circuit(
        distance=args.distance,
        rounds=args.rounds,
        erasure_prob=float(args.erasure_prob),
        erasable_qubits="ALL",
        reset_failure_prob=0.0,
        ecr_after_each_step=False,
        single_qubit_errors=True,
        post_clifford_pauli_prob=0.0,
        rounds_per_check=args.rounds_per_check,
    )

    # Calibration model aligned with the real sweep calibrator.
    model = qe.ErasureModel(
        2,
        qe.PauliChannel(0.25, 0.25, 0.25),
        qe.PauliChannel(0.25, 0.25, 0.25),
        qe.TQGSpreadModel(
            qe.PauliChannel(0.0, 0.0, 0.0),
            qe.PauliChannel(0.0, 0.0, 0.5),
        ),
    )
    onset_flip_probability = float(model.onset.p_x + model.onset.p_y)
    model.check_false_negative_prob = float(args.check_fn)
    model.check_false_positive_prob = float(args.check_fp)
    three_case_mismatch_floor = max(1e-12, float(args.check_fn) * float(args.check_fp))

    rail_program = qe.RailSurfaceCompiledProgram(
        circuit=circuit,
        model=model,
        distance=args.distance,
        rounds=args.rounds,
    )
    if args.calibration_source == "latent":
        sampler = qe.RailCalibrationSampler(rail_program)
    else:
        sampler = qe.RailStreamSampler(rail_program)
    dem_builder = qe.RailSurfaceDemBuilder(rail_program)

    use_canonical_step_bins = bool(args.canonical_step_bins)
    use_full_lookback_bins = bool(use_canonical_step_bins and args.full_lookback_bins)
    if use_full_lookback_bins:
        num_hypotheses = 9 if args.include_no_erasure else 8
    elif use_canonical_step_bins:
        num_hypotheses = 4
    else:
        num_hypotheses = 9 if args.include_no_erasure else 8
    qtypes = ["XZZX_interior", "ZXXZ_interior"]
    if args.condition == "both":
        condition_keys = ["inconsistency", "no_inconsistency"]
    elif args.condition == "three_case":
        condition_keys = [
            "consistent_both",
            "inc_round1_only",
            "inc_round2_only",
            "inc_both",
        ]
    else:
        condition_keys = [args.condition]
    sum_by_condition_type = {
        cond: {qt: np.zeros(num_hypotheses, dtype=float) for qt in qtypes}
        for cond in condition_keys
    }
    sumsq_by_condition_type = {
        cond: {qt: np.zeros(num_hypotheses, dtype=float) for qt in qtypes}
        for cond in condition_keys
    }
    count_by_condition_type = {
        cond: {qt: 0 for qt in qtypes}
        for cond in condition_keys
    }
    used_events_by_condition_type = {
        cond: {qt: 0 for qt in qtypes}
        for cond in condition_keys
    }
    total_shots_sampled = 0
    next_seed = int(args.seed)
    check_event_qubits = [int(q) for q in rail_program.check_event_to_qubit]
    prev_check_event_index = [-1] * len(check_event_qubits)
    last_seen_check_for_qubit: dict[int, int] = {}
    for check_event_index, qubit in enumerate(check_event_qubits):
        prev_check_event_index[check_event_index] = last_seen_check_for_qubit.get(qubit, -1)
        last_seen_check_for_qubit[qubit] = check_event_index

    while True:
        shots_this_iter = args.chunk_shots if args.adaptive else args.shots
        remaining = args.max_shots - total_shots_sampled
        if remaining <= 0:
            break
        shots_this_iter = min(shots_this_iter, remaining)
        if args.calibration_source == "latent":
            dets, _, checks, latent_onset_ops = sampler.sample(
                shots_this_iter,
                next_seed,
                num_threads=1,
            )
        else:
            dets, _, checks = sampler.sample(shots_this_iter, next_seed, num_threads=1)
            latent_onset_ops = None
        total_shots_sampled += shots_this_iter
        next_seed += shots_this_iter

        for shot in range(shots_this_iter):
            check_row = checks[shot].tolist()
            det_row = dets[shot]
            onset_row = None if latent_onset_ops is None else latent_onset_ops[shot]
            unique_flagged_data_idx = None
            if args.isolate_single_flagged_data_check:
                flagged_indices = [idx for idx, bit in enumerate(check_row) if bit == 1]
                if len(flagged_indices) != 1:
                    continue
                only_idx = flagged_indices[0]
                q = check_event_qubits[only_idx]
                if q >= rail_program.num_data_qubits:
                    continue
                unique_flagged_data_idx = only_idx

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
                if args.isolate_single_flagged_data_check:
                    if unique_flagged_data_idx is None:
                        continue
                    if int(check_event_index) != int(unique_flagged_data_idx):
                        continue
                prev_idx = prev_check_event_index[int(check_event_index)]
                prev_flagged_same_qubit = (
                    prev_idx >= 0 and int(check_row[prev_idx]) == 1
                )
                if args.require_prev_check_unflagged:
                    if prev_idx < 0:
                        continue
                    if int(check_row[prev_idx]) != 0:
                        continue
                use_round1 = True
                if (
                    args.condition == "three_case"
                    and args.truncate_lookback_on_consecutive_flag
                    and prev_flagged_same_qubit
                ):
                    use_round1 = False
                if args.condition == "three_case":
                    condition_key = classify_two_round_case(
                        rail_program,
                        det_row,
                        data_qubit,
                        check_round,
                        include_round1=use_round1,
                    )
                else:
                    inconsistent = has_pair_inconsistency(
                        rail_program,
                        det_row,
                        data_qubit,
                        check_round,
                        args.inconsistency_window,
                    )
                    condition_key = "inconsistency" if inconsistent else "no_inconsistency"
                if condition_key not in condition_keys:
                    continue

                event_rows.sort(key=lambda r: int(r["onset_op_index"]))
                if args.calibration_source == "latent":
                    true_onset_op = int(onset_row[int(check_event_index)]) if onset_row is not None else -1
                    vec = np.zeros(num_hypotheses, dtype=float)
                    mapped_index = -1
                    if use_full_lookback_bins:
                        prev_round = int(check_round) - 1
                        curr_round = int(check_round)
                        rows_by_round = {
                            prev_round: [],
                            curr_round: [],
                        }
                        for row in event_rows:
                            onset_round = rail_program.op_round(int(row["onset_op_index"]))
                            if onset_round in rows_by_round:
                                rows_by_round[onset_round].append(row)
                        for rr in rows_by_round:
                            rows_by_round[rr].sort(key=lambda row: int(row["onset_op_index"]))
                        onset_to_bin: dict[int, int] = {}
                        if use_round1:
                            for j, row in enumerate(rows_by_round[prev_round][:4]):
                                onset_to_bin[int(row["onset_op_index"])] = j
                        for j, row in enumerate(rows_by_round[curr_round][:4]):
                            onset_to_bin[int(row["onset_op_index"])] = 4 + j
                        mapped_index = onset_to_bin.get(true_onset_op, -1)
                    elif use_canonical_step_bins:
                        current_round_rows = [
                            row
                            for row in event_rows
                            if rail_program.op_round(int(row["onset_op_index"])) == int(check_round)
                        ]
                        if len(current_round_rows) != 4:
                            continue
                        current_round_rows.sort(key=lambda row: int(row["onset_op_index"]))
                        onset_to_bin = {
                            int(row["onset_op_index"]): j for j, row in enumerate(current_round_rows[:4])
                        }
                        mapped_index = onset_to_bin.get(true_onset_op, -1)
                    else:
                        if len(event_rows) < 8:
                            continue
                        onset_to_bin = {
                            int(row["onset_op_index"]): j for j, row in enumerate(event_rows[:8])
                        }
                        mapped_index = onset_to_bin.get(true_onset_op, -1)

                    if mapped_index >= 0:
                        vec[mapped_index] = 1.0
                    elif args.include_no_erasure:
                        vec[-1] = 1.0
                else:
                    if use_full_lookback_bins:
                        prev_round = int(check_round) - 1
                        curr_round = int(check_round)
                        rows_by_round = {
                            prev_round: [],
                            curr_round: [],
                        }
                        for r in event_rows:
                            onset_round = rail_program.op_round(int(r["onset_op_index"]))
                            if onset_round in rows_by_round:
                                rows_by_round[onset_round].append(r)
                        for rr in rows_by_round:
                            rows_by_round[rr].sort(key=lambda r: int(r["onset_op_index"]))

                        onset_values = [0.0] * 8
                        onset_priors = [0.0] * 8
                        onset_event_likelihood = [0.0] * 8

                        def fill_round_bins(round_index: int, base_offset: int) -> None:
                            local_rows = rows_by_round[round_index]
                            for j, row in enumerate(local_rows[:4]):
                                onset_values[base_offset + j] = float(row["posterior_mass"])
                                onset_priors[base_offset + j] = max(0.0, float(row["prior_mass"]))

                        if use_round1:
                            fill_round_bins(prev_round, 0)
                        fill_round_bins(curr_round, 4)

                        if args.conditioning_signal == "full_detectors":
                            onset_probs = onset_values
                            if args.include_no_erasure:
                                no_erasure = max(0.0, 1.0 - float(sum(onset_probs)))
                                vec = np.asarray(onset_probs + [no_erasure], dtype=float)
                            else:
                                vec = np.asarray(onset_probs, dtype=float)
                        else:
                            if args.condition == "three_case":
                                for step_idx in range(1, 5):
                                    p_prev = 0.0
                                    if use_round1:
                                        p_prev = branch_two_round_case_likelihood(
                                            int(schedule_type), 0, step_idx, onset_flip_probability, condition_key
                                        )
                                    p_curr = branch_two_round_case_likelihood(
                                        int(schedule_type), 1, step_idx, onset_flip_probability, condition_key
                                    )
                                    onset_event_likelihood[step_idx - 1] = max(
                                        three_case_mismatch_floor, p_prev
                                    )
                                    onset_event_likelihood[4 + step_idx - 1] = max(
                                        three_case_mismatch_floor, p_curr
                                    )
                            else:
                                for i in range(4):
                                    if prev_round >= 0:
                                        p_incons = 1.0
                                    else:
                                        p_incons = 0.0
                                    onset_event_likelihood[i] = (
                                        p_incons if condition_key == "inconsistency" else (1.0 - p_incons)
                                    )
                                current_round_rows = rows_by_round[curr_round]
                                for j in range(min(4, len(current_round_rows))):
                                    step_idx = j + 1
                                    p_incons = branch_inconsistency_likelihood(
                                        int(schedule_type),
                                        step_idx,
                                        onset_flip_probability,
                                    )
                                    onset_event_likelihood[4 + j] = (
                                        p_incons if condition_key == "inconsistency" else (1.0 - p_incons)
                                    )
                            no_erasure_posterior = max(0.0, 1.0 - float(sum(onset_values)))
                            unnorm = [
                                onset_priors[i] * max(0.0, onset_event_likelihood[i]) for i in range(8)
                            ]
                            total_onset = float(sum(unnorm))
                            if total_onset <= 0.0:
                                continue
                            visible_mass = max(0.0, 1.0 - no_erasure_posterior)
                            onset_posterior = [(x / total_onset) * visible_mass for x in unnorm]
                            if args.include_no_erasure:
                                vec = np.asarray(
                                    onset_posterior + [no_erasure_posterior],
                                    dtype=float,
                                )
                            else:
                                vec = np.asarray(onset_posterior, dtype=float)
                    elif use_canonical_step_bins:
                        current_round_rows = [
                            r for r in event_rows if rail_program.op_round(int(r["onset_op_index"])) == int(check_round)
                        ]
                        if len(current_round_rows) != 4:
                            continue
                        current_round_rows.sort(key=lambda r: int(r["onset_op_index"]))
                        if args.conditioning_signal == "full_detectors":
                            step_probs = [float(r["posterior_mass"]) for r in current_round_rows]
                            total = float(sum(step_probs))
                            if total <= 0.0:
                                continue
                            step_probs = [x / total for x in step_probs]
                        else:
                            priors = [max(0.0, float(r["prior_mass"])) for r in current_round_rows]
                            unnorm = []
                            for step_idx, prior in enumerate(priors, start=1):
                                p_incons = branch_inconsistency_likelihood(
                                    int(schedule_type),
                                    step_idx,
                                    onset_flip_probability,
                                )
                                p_event = p_incons if condition_key == "inconsistency" else (1.0 - p_incons)
                                unnorm.append(prior * max(0.0, p_event))
                            total = float(sum(unnorm))
                            if total <= 0.0:
                                continue
                            step_probs = [x / total for x in unnorm]
                        vec = np.asarray(step_probs, dtype=float)
                    else:
                        spot_probs = [float(r["posterior_mass"]) for r in event_rows]
                        if len(spot_probs) != 8:
                            continue
                        if args.include_no_erasure:
                            no_erasure = max(0.0, 1.0 - float(sum(spot_probs)))
                            vec = np.asarray(spot_probs + [no_erasure], dtype=float)
                        else:
                            vec = np.asarray(spot_probs, dtype=float)
                label = schedule_label(schedule_type)
                sum_by_condition_type[condition_key][label] += vec
                sumsq_by_condition_type[condition_key][label] += vec * vec
                count_by_condition_type[condition_key][label] += 1
                used_events_by_condition_type[condition_key][label] += 1

        if not args.adaptive:
            break

        converged = True
        for cond in condition_keys:
            for qtype in qtypes:
                n = count_by_condition_type[cond][qtype]
                if n < args.min_events_per_type:
                    converged = False
                    break
                mean = sum_by_condition_type[cond][qtype] / float(n)
                var = np.maximum(
                    0.0,
                    sumsq_by_condition_type[cond][qtype] / float(n) - mean * mean,
                )
                std = np.sqrt(var)
                sem = std / np.sqrt(float(n))
                mask = mean > 1e-8
                if np.any(mask) and np.any(sem[mask] > (args.target_rel_error * mean[mask])):
                    converged = False
                    break
            if not converged:
                break
        if converged:
            break

    labels = (
        (
            [
                "round(n-1)_step1",
                "round(n-1)_step2",
                "round(n-1)_step3",
                "round(n-1)_step4",
                "round(n)_step1",
                "round(n)_step2",
                "round(n)_step3",
                "round(n)_step4",
                "no_erasure",
            ]
            if args.include_no_erasure
            else [
                "round(n-1)_step1",
                "round(n-1)_step2",
                "round(n-1)_step3",
                "round(n-1)_step4",
                "round(n)_step1",
                "round(n)_step2",
                "round(n)_step3",
                "round(n)_step4",
            ]
        )
        if use_full_lookback_bins
        else (
            ["step1", "step2", "step3", "step4"]
            if use_canonical_step_bins
            else (
                ["spot1", "spot2", "spot3", "spot4", "spot5", "spot6", "spot7", "spot8", "no_erasure"]
                if args.include_no_erasure
                else ["spot1", "spot2", "spot3", "spot4", "spot5", "spot6", "spot7", "spot8"]
            )
        )
    )
    stats_by_condition_type = {}
    y_top_by_condition = {cond: 0.0 for cond in condition_keys}
    for cond in condition_keys:
        stats_by_condition_type[cond] = {}
        for qtype in qtypes:
            n = count_by_condition_type[cond][qtype]
            if n > 0:
                mean = sum_by_condition_type[cond][qtype] / float(n)
                var = np.maximum(
                    0.0,
                    sumsq_by_condition_type[cond][qtype] / float(n) - mean * mean,
                )
                std = np.sqrt(var)
                err = std / np.sqrt(float(n))
                y_top_by_condition[cond] = max(
                    y_top_by_condition[cond], float(np.max(mean + err))
                )
            else:
                mean = np.zeros(num_hypotheses, dtype=float)
                std = np.zeros(num_hypotheses, dtype=float)
                err = np.zeros(num_hypotheses, dtype=float)
            stats_by_condition_type[cond][qtype] = (n, mean, std, err)

    y_lim_top_by_condition = {
        cond: min(1.0, max(0.02, y_top_by_condition[cond] * 1.15))
        for cond in condition_keys
    }
    summary = {cond: {} for cond in condition_keys}

    if len(condition_keys) > 1:
        n_rows = len(condition_keys)
        fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4.8 * n_rows), sharey="row")
        for row, cond in enumerate(condition_keys):
            for col, qtype in enumerate(qtypes):
                ax = axes[row, col]
                n, mean, std, err = stats_by_condition_type[cond][qtype]
                x = np.arange(len(labels))
                ax.bar(x, mean, color="#2c7fb8", alpha=0.85)
                ax.errorbar(
                    x,
                    mean,
                    yerr=err,
                    fmt="none",
                    ecolor="black",
                    elinewidth=1.0,
                    capsize=3,
                )
                cond_label = condition_display_label(cond)
                ax.set_title(f"{qtype} | {cond_label} (n={n})")
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=30, ha="right")
                ax.set_ylim(0.0, y_lim_top_by_condition[cond])
                ax.grid(axis="y", alpha=0.3)
                summary[cond][qtype] = {
                    "count": n,
                    "mean": mean.tolist(),
                    "mean_sum": float(np.sum(mean)),
                    "bin_weight_sum": sum_by_condition_type[cond][qtype].tolist(),
                    "std": std.tolist(),
                    "error_bar": err.tolist(),
                    "labels": labels,
                }
            axes[row, 0].set_ylabel("P(onset | condition)")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        cond = condition_keys[0]
        for i, qtype in enumerate(qtypes):
            ax = axes[i]
            n, mean, std, err = stats_by_condition_type[cond][qtype]
            x = np.arange(len(labels))
            ax.bar(x, mean, color="#2c7fb8", alpha=0.85)
            ax.errorbar(
                x,
                mean,
                yerr=err,
                fmt="none",
                ecolor="black",
                elinewidth=1.0,
                capsize=3,
            )
            cond_label = condition_display_label(cond)
            ax.set_title(f"{qtype} | {cond_label} (n={n})")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_ylim(0.0, y_lim_top_by_condition[cond])
            ax.grid(axis="y", alpha=0.3)
            summary[cond][qtype] = {
                "count": n,
                "mean": mean.tolist(),
                "mean_sum": float(np.sum(mean)),
                "bin_weight_sum": sum_by_condition_type[cond][qtype].tolist(),
                "std": std.tolist(),
                "error_bar": err.tolist(),
                "labels": labels,
            }
        axes[0].set_ylabel("P(onset step | flagged check)")

    if args.condition == "both":
        cond_descriptor = "inconsistency+consistency"
    elif args.condition == "three_case":
        cond_descriptor = "00|10|01|11"
    else:
        cond_descriptor = args.condition
    final_check_note = "excluding final-check round" if args.exclude_final_check_round else "including final-check round"
    fig.suptitle(
        f"Rail Calibration Posterior | d={args.distance}, r={args.rounds}, shots={total_shots_sampled}, "
        f"condition={cond_descriptor}, source={args.calibration_source} ({final_check_note})",
        y=0.995,
    )
    layout_top = 0.92 if len(condition_keys) > 1 else 0.90
    fig.tight_layout(rect=(0.0, 0.0, 1.0, layout_top))

    args.png_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.inverse_png_out.parent.mkdir(parents=True, exist_ok=True)
    args.inverse_json_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.png_out, dpi=220)
    plt.close(fig)

    inverse_summary = {cond: {} for cond in condition_keys}
    inverse_prob_by_condition_type = {
        cond: {qt: np.zeros(num_hypotheses, dtype=float) for qt in qtypes}
        for cond in condition_keys
    }
    inverse_y_top_by_condition = {cond: 0.0 for cond in condition_keys}
    for qtype in qtypes:
        denom = np.zeros(num_hypotheses, dtype=float)
        for cond in condition_keys:
            denom += sum_by_condition_type[cond][qtype]
        for cond in condition_keys:
            numer = sum_by_condition_type[cond][qtype]
            prob = np.divide(
                numer,
                denom,
                out=np.zeros_like(numer, dtype=float),
                where=denom > 0,
            )
            inverse_prob_by_condition_type[cond][qtype] = prob
            inverse_y_top_by_condition[cond] = max(
                inverse_y_top_by_condition[cond], float(np.max(prob))
            )
            inverse_summary[cond][qtype] = {
                "prob": prob.tolist(),
                "prob_sum": float(np.sum(prob)),
                "bin_weight_sum": numer.tolist(),
                "onset_bin_total_weight": denom.tolist(),
                "labels": labels,
            }

    inverse_y_lim_top_by_condition = {
        cond: min(1.0, max(0.02, inverse_y_top_by_condition[cond] * 1.15))
        for cond in condition_keys
    }
    if len(condition_keys) > 1:
        n_rows = len(condition_keys)
        inv_fig, inv_axes = plt.subplots(n_rows, 2, figsize=(14, 4.8 * n_rows), sharey="row")
        for row, cond in enumerate(condition_keys):
            for col, qtype in enumerate(qtypes):
                ax = inv_axes[row, col]
                prob = inverse_prob_by_condition_type[cond][qtype]
                x = np.arange(len(labels))
                ax.bar(x, prob, color="#1b9e77", alpha=0.9)
                cond_label = condition_display_label(cond)
                ax.set_title(f"{qtype} | {cond_label}")
                ax.set_xticks(x)
                ax.set_xticklabels(labels, rotation=30, ha="right")
                ax.set_ylim(0.0, inverse_y_lim_top_by_condition[cond])
                ax.grid(axis="y", alpha=0.3)
            inv_axes[row, 0].set_ylabel("P(condition | onset)")
    else:
        inv_fig, inv_axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        cond = condition_keys[0]
        for i, qtype in enumerate(qtypes):
            ax = inv_axes[i]
            prob = inverse_prob_by_condition_type[cond][qtype]
            x = np.arange(len(labels))
            ax.bar(x, prob, color="#1b9e77", alpha=0.9)
            cond_label = condition_display_label(cond)
            ax.set_title(f"{qtype} | {cond_label}")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_ylim(0.0, inverse_y_lim_top_by_condition[cond])
            ax.grid(axis="y", alpha=0.3)
        inv_axes[0].set_ylabel("P(condition | onset)")

    inv_fig.suptitle(
        f"Rail Calibration Inverse Posterior | d={args.distance}, r={args.rounds}, shots={total_shots_sampled}, "
        f"condition={cond_descriptor}, source={args.calibration_source} ({final_check_note})",
        y=0.995,
    )
    inv_layout_top = 0.92 if len(condition_keys) > 1 else 0.90
    inv_fig.tight_layout(rect=(0.0, 0.0, 1.0, inv_layout_top))
    inv_fig.savefig(args.inverse_png_out, dpi=220)
    plt.close(inv_fig)

    payload = {
        "distance": args.distance,
        "rounds": args.rounds,
        "shots_requested": args.shots,
        "shots_sampled": total_shots_sampled,
        "seed": args.seed,
        "erasure_prob": float(args.erasure_prob),
        "check_false_negative_prob": float(args.check_fn),
        "check_false_positive_prob": float(args.check_fp),
        "rounds_per_check": args.rounds_per_check,
        "perfect_checks": bool(args.check_fn == 0.0 and args.check_fp == 0.0),
        "single_qubit_errors": False,
        "condition": args.condition,
        "conditions": condition_keys,
        "inconsistency_window": args.inconsistency_window,
        "calibration_source": args.calibration_source,
        "conditioning_signal": args.conditioning_signal,
        "isolate_single_flagged_data_check": bool(args.isolate_single_flagged_data_check),
        "require_prev_check_unflagged": bool(args.require_prev_check_unflagged),
        "truncate_lookback_on_consecutive_flag": bool(
            args.truncate_lookback_on_consecutive_flag
        ),
        "full_interior_only": bool(args.full_interior_only),
        "exclude_final_check_round": bool(args.exclude_final_check_round),
        "canonical_step_bins": bool(use_canonical_step_bins),
        "full_lookback_bins": bool(use_full_lookback_bins),
        "include_no_erasure": bool(args.include_no_erasure),
        "adaptive": bool(args.adaptive),
        "chunk_shots": int(args.chunk_shots),
        "max_shots": int(args.max_shots),
        "target_rel_error": float(args.target_rel_error),
        "min_events_per_type": int(args.min_events_per_type),
        "hypothesis_count": num_hypotheses,
        "hypotheses": labels,
        "metadata": {
            "distance": int(args.distance),
            "rounds": int(args.rounds),
            "shots_requested": int(args.shots),
            "shots_sampled": int(total_shots_sampled),
            "seed": int(args.seed),
            "erasure_prob": float(args.erasure_prob),
            "check_false_negative_prob": float(args.check_fn),
            "check_false_positive_prob": float(args.check_fp),
            "rounds_per_check": int(args.rounds_per_check),
            "condition": args.condition,
            "conditions": condition_keys,
            "calibration_source": args.calibration_source,
            "inconsistency_window": args.inconsistency_window,
            "full_lookback_bins": bool(use_full_lookback_bins),
            "include_no_erasure": bool(args.include_no_erasure),
            "exclude_final_check_round": bool(args.exclude_final_check_round),
            "final_check_round_note": (
                "Final-round checks excluded from calibration statistics."
                if args.exclude_final_check_round
                else "Final-round checks included in calibration statistics."
            ),
        },
        "summary": summary,
    }
    args.json_out.write_text(json.dumps(payload, indent=2))

    inverse_payload = {
        "distance": args.distance,
        "rounds": args.rounds,
        "shots_requested": args.shots,
        "shots_sampled": total_shots_sampled,
        "seed": args.seed,
        "erasure_prob": float(args.erasure_prob),
        "check_false_negative_prob": float(args.check_fn),
        "check_false_positive_prob": float(args.check_fp),
        "rounds_per_check": args.rounds_per_check,
        "condition": args.condition,
        "conditions": condition_keys,
        "hypothesis_count": num_hypotheses,
        "hypotheses": labels,
        "calibration_source": args.calibration_source,
        "exclude_final_check_round": bool(args.exclude_final_check_round),
        "definition": "P(condition | onset_bin)",
        "metadata": {
            "distance": int(args.distance),
            "rounds": int(args.rounds),
            "shots_requested": int(args.shots),
            "shots_sampled": int(total_shots_sampled),
            "seed": int(args.seed),
            "erasure_prob": float(args.erasure_prob),
            "check_false_negative_prob": float(args.check_fn),
            "check_false_positive_prob": float(args.check_fp),
            "rounds_per_check": int(args.rounds_per_check),
            "condition": args.condition,
            "conditions": condition_keys,
            "calibration_source": args.calibration_source,
            "include_no_erasure": bool(args.include_no_erasure),
            "exclude_final_check_round": bool(args.exclude_final_check_round),
            "final_check_round_note": (
                "Final-round checks excluded from calibration statistics."
                if args.exclude_final_check_round
                else "Final-round checks included in calibration statistics."
            ),
        },
        "summary": inverse_summary,
    }
    args.inverse_json_out.write_text(json.dumps(inverse_payload, indent=2))

    print(f"Saved PNG: {args.png_out}")
    print(f"Saved JSON: {args.json_out}")
    print(f"Saved inverse PNG: {args.inverse_png_out}")
    print(f"Saved inverse JSON: {args.inverse_json_out}")
    for cond in condition_keys:
        for qtype in qtypes:
            cond_label = condition_display_label(cond)
            print(f"{qtype} ({cond_label}) events: {used_events_by_condition_type[cond][qtype]}")


if __name__ == "__main__":
    main()
