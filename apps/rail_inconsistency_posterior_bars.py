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
) -> bool:
    slot0, slot1 = rail_program.data_z_ancilla_slots(data_qubit)
    if slot0 < 0 or slot1 < 0:
        return False
    start_round = max(0, check_round - 1)
    for r in range(start_round, check_round + 1):
        d0 = rail_program.round_detector_index(r, slot0)
        d1 = rail_program.round_detector_index(r, slot1)
        if d0 < 0 or d1 < 0:
            continue
        if int(det_row[d0]) != int(det_row[d1]):
            return True
    return False


def event_matches_condition(
    condition: str,
    rail_program: qe.RailSurfaceCompiledProgram,
    det_row: np.ndarray,
    data_qubit: int,
    check_round: int,
) -> bool:
    inconsistent = has_pair_inconsistency(rail_program, det_row, data_qubit, check_round)
    if condition == "inconsistency":
        return inconsistent
    if condition == "no_inconsistency":
        return not inconsistent
    raise ValueError(f"Unsupported condition '{condition}'")


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
    parser.add_argument("--seed", type=int, default=920004)
    parser.add_argument("--erasure-prob", type=float, default=0.02)
    parser.add_argument("--rounds-per-check", type=int, default=2)
    parser.add_argument(
        "--condition",
        choices=["inconsistency", "no_inconsistency"],
        default="inconsistency",
        help="Conditioning event for flagged data checks.",
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
    args = parser.parse_args()
    if args.png_out is None:
        suffix = "inconsistency" if args.condition == "inconsistency" else "no_inconsistency"
        args.png_out = REPO_ROOT / "apps" / "results" / f"rail_{suffix}_posterior_bars.png"
    if args.json_out is None:
        suffix = "inconsistency" if args.condition == "inconsistency" else "no_inconsistency"
        args.json_out = REPO_ROOT / "apps" / "results" / f"rail_{suffix}_posterior_bars.json"

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

    # Erasure-only + perfect checks baseline.
    model = qe.ErasureModel(
        2,
        qe.PauliChannel(0.0, 0.0, 0.0),
        qe.PauliChannel(0.0, 0.0, 0.0),
        qe.TQGSpreadModel(
            qe.PauliChannel(0.0, 0.0, 0.0),
            qe.PauliChannel(0.0, 0.0, 0.0),
        ),
    )
    model.check_false_negative_prob = 0.0
    model.check_false_positive_prob = 0.0

    rail_program = qe.RailSurfaceCompiledProgram(
        circuit=circuit,
        model=model,
        distance=args.distance,
        rounds=args.rounds,
    )
    sampler = qe.RailStreamSampler(rail_program)
    dem_builder = qe.RailSurfaceDemBuilder(rail_program)

    dets, _, checks = sampler.sample(args.shots, args.seed, num_threads=1)

    # Aggregation by schedule type; each sample is length-9 [spot1..spot8, no_erasure].
    vectors_by_type: dict[str, list[np.ndarray]] = defaultdict(list)
    used_events_by_type: dict[str, int] = defaultdict(int)

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
            _, data_qubit, check_round, schedule_type, is_boundary = key
            if is_boundary:
                continue
            if not event_matches_condition(
                args.condition, rail_program, det_row, data_qubit, check_round
            ):
                continue

            # Sort onset candidates by operation index to define 8 spot positions.
            event_rows.sort(key=lambda r: int(r["onset_op_index"]))
            spot_probs = [float(r["posterior_mass"]) for r in event_rows]
            if len(spot_probs) != 8:
                # Keep strict 9-hypothesis view requested by user.
                continue
            no_erasure = max(0.0, 1.0 - float(sum(spot_probs)))
            vec = np.asarray(spot_probs + [no_erasure], dtype=float)
            label = schedule_label(schedule_type)
            vectors_by_type[label].append(vec)
            used_events_by_type[label] += 1

    labels = ["spot1", "spot2", "spot3", "spot4", "spot5", "spot6", "spot7", "spot8", "no_erasure"]
    summary = {}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for i, qtype in enumerate(["XZZX_interior", "ZXXZ_interior"]):
        ax = axes[i]
        vectors = vectors_by_type.get(qtype, [])
        if vectors:
            stack = np.vstack(vectors)
            mean = np.mean(stack, axis=0)
            std = np.std(stack, axis=0)
        else:
            mean = np.zeros(9, dtype=float)
            std = np.zeros(9, dtype=float)

        x = np.arange(len(labels))
        ax.bar(x, mean, color="#2c7fb8", alpha=0.85)
        ax.errorbar(x, mean, yerr=std, fmt="none", ecolor="black", elinewidth=1.0, capsize=3)
        ax.set_title(f"{qtype} (n={len(vectors)})")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylim(0.0, min(1.0, max(0.25, float(np.max(mean + std) * 1.15) if vectors else 0.25)))
        ax.grid(axis="y", alpha=0.3)

        summary[qtype] = {
            "count": len(vectors),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "labels": labels,
        }

    axes[0].set_ylabel(f"P(onset spot | flagged check + {args.condition})")
    fig.suptitle(
        f"9-way Rail Calibration Posterior (d={args.distance}, rounds={args.rounds}, "
        f"shots={args.shots}, p_e={args.erasure_prob}, perfect checks, mp=2, "
        f"condition={args.condition})"
    )
    fig.tight_layout()

    args.png_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.png_out, dpi=220)
    plt.close(fig)

    payload = {
        "distance": args.distance,
        "rounds": args.rounds,
        "shots": args.shots,
        "seed": args.seed,
        "erasure_prob": float(args.erasure_prob),
        "rounds_per_check": args.rounds_per_check,
        "perfect_checks": True,
        "single_qubit_errors": False,
        "condition": args.condition,
        "hypothesis_count": 9,
        "hypotheses": labels,
        "summary": summary,
    }
    args.json_out.write_text(json.dumps(payload, indent=2))

    print(f"Saved PNG: {args.png_out}")
    print(f"Saved JSON: {args.json_out}")
    for qtype in ["XZZX_interior", "ZXXZ_interior"]:
        print(f"{qtype} events: {used_events_by_type.get(qtype, 0)}")


if __name__ == "__main__":
    main()
