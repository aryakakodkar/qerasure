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


def schedule_label(schedule_type: int, boundary: bool) -> str:
    if schedule_type == 1:
        base = "XZZX"
    elif schedule_type == 2:
        base = "ZXXZ"
    else:
        base = "UNKNOWN"
    return f"{base}_{'boundary' if boundary else 'interior'}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect rail-decoder onset calibration posteriors and plot by data-qubit type."
    )
    parser.add_argument("--distance", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--shots", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=920004)
    parser.add_argument("--erasure-prob", type=float, default=0.02)
    parser.add_argument("--rounds-per-check", type=int, default=2)
    parser.add_argument(
        "--png-out",
        type=Path,
        default=REPO_ROOT / "apps" / "results" / "rail_calibration_stats.png",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / "apps" / "results" / "rail_calibration_stats.json",
    )
    args = parser.parse_args()

    if args.distance < 3 or args.distance % 2 == 0:
        raise ValueError("--distance must be odd and >= 3")
    if args.rounds <= 0:
        raise ValueError("--rounds must be > 0")
    if args.shots <= 0:
        raise ValueError("--shots must be > 0")
    if args.rounds_per_check <= 0:
        raise ValueError("--rounds-per-check must be > 0")

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

    # "Just erasure errors": no Pauli channels aside from rail-induced logic.
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

    posterior_by_type_and_delta: dict[tuple[str, int], list[float]] = defaultdict(list)
    prior_by_type_and_delta: dict[tuple[str, int], list[float]] = defaultdict(list)
    total_rows = 0
    flagged_data_checks = 0

    for shot in range(args.shots):
        check_row = checks[shot].tolist()
        det_row = dets[shot].tolist()
        rows = dem_builder.calibration_rows(check_row, det_row)
        if not rows:
            continue
        total_rows += len(rows)
        flagged_data_checks += len({int(r["check_event_index"]) for r in rows})
        for row in rows:
            qtype = schedule_label(
                int(row["schedule_type"]),
                bool(row["boundary_data_qubit"]),
            )
            onset_delta = int(row["check_op_index"]) - int(row["onset_op_index"])
            key = (qtype, onset_delta)
            posterior_by_type_and_delta[key].append(float(row["posterior_mass"]))
            prior_by_type_and_delta[key].append(float(row["prior_mass"]))

    grouped = defaultdict(list)
    for (qtype, onset_delta), values in posterior_by_type_and_delta.items():
        mean_post = float(np.mean(values))
        std_post = float(np.std(values))
        mean_prior = float(np.mean(prior_by_type_and_delta[(qtype, onset_delta)]))
        grouped[qtype].append(
            {
                "onset_delta_ops": onset_delta,
                "mean_posterior": mean_post,
                "std_posterior": std_post,
                "mean_prior": mean_prior,
                "count": len(values),
            }
        )

    for qtype in grouped:
        grouped[qtype].sort(key=lambda x: x["onset_delta_ops"])

    args.png_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for qtype in sorted(grouped.keys()):
        points = grouped[qtype]
        x = np.array([p["onset_delta_ops"] for p in points], dtype=float)
        y = np.array([p["mean_posterior"] for p in points], dtype=float)
        yerr = np.array([p["std_posterior"] for p in points], dtype=float)
        plt.plot(x, y, marker="o", linewidth=1.5, label=qtype)
        plt.fill_between(x, np.clip(y - yerr, 0.0, 1.0), np.clip(y + yerr, 0.0, 1.0), alpha=0.15)

    plt.xlabel("Onset Spot Offset (check_op_index - onset_op_index)")
    plt.ylabel("Conditional Onset Probability")
    plt.title(
        f"Rail Calibration Posteriors (d={args.distance}, rounds={args.rounds}, shots={args.shots}, "
        f"p_e={args.erasure_prob}, perfect checks, mp=2)"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.png_out, dpi=220)
    plt.close()

    payload = {
        "distance": args.distance,
        "rounds": args.rounds,
        "shots": args.shots,
        "seed": args.seed,
        "erasure_prob": float(args.erasure_prob),
        "rounds_per_check": args.rounds_per_check,
        "model": {
            "max_persistence": 2,
            "check_false_negative_prob": 0.0,
            "check_false_positive_prob": 0.0,
            "onset_channel": [0.0, 0.0, 0.0],
            "reset_channel": [0.0, 0.0, 0.0],
            "spread_control_channel": [0.0, 0.0, 0.0],
            "spread_target_channel": [0.0, 0.0, 0.0],
        },
        "total_calibration_rows": total_rows,
        "flagged_data_checks": flagged_data_checks,
        "by_qubit_type": grouped,
    }
    args.json_out.write_text(json.dumps(payload, indent=2))

    print(f"Saved PNG: {args.png_out}")
    print(f"Saved JSON: {args.json_out}")
    print(f"Total calibration rows: {total_rows}")
    print(f"Flagged data-check events: {flagged_data_checks}")


if __name__ == "__main__":
    main()
