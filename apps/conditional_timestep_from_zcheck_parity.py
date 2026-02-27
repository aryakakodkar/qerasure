#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


def normalize_rows(counts: np.ndarray) -> np.ndarray:
    probs = np.zeros_like(counts, dtype=float)
    for i in range(counts.shape[0]):
        total = int(np.sum(counts[i]))
        if total > 0:
            probs[i] = counts[i] / total
    return probs


def summarize_from_counts(counts: np.ndarray) -> dict:
    # P(step | class)
    p_step_given_class = normalize_rows(counts)

    # P(class | step): transpose view and normalize each step row.
    counts_by_step = counts.T.copy()  # [step, class]
    p_class_given_step = normalize_rows(counts_by_step)

    p0_consistent = float(p_step_given_class[0, 0])
    p0_inconsistent = float(p_step_given_class[1, 0])
    step0_ratio = (
        float("inf")
        if p0_consistent == 0.0 and p0_inconsistent > 0.0
        else (p0_inconsistent / p0_consistent if p0_consistent > 0.0 else float("nan"))
    )

    return {
        "counts_parity_by_step": {
            "consistent": counts[0].tolist(),
            "inconsistent": counts[1].tolist(),
        },
        "p_step_given_consistency": {
            "consistent": p_step_given_class[0].tolist(),
            "inconsistent": p_step_given_class[1].tolist(),
        },
        "p_consistency_given_step": {
            "step_0": {
                "consistent": float(p_class_given_step[0, 0]),
                "inconsistent": float(p_class_given_step[0, 1]),
            },
            "step_1": {
                "consistent": float(p_class_given_step[1, 0]),
                "inconsistent": float(p_class_given_step[1, 1]),
            },
            "step_2": {
                "consistent": float(p_class_given_step[2, 0]),
                "inconsistent": float(p_class_given_step[2, 1]),
            },
            "step_3": {
                "consistent": float(p_class_given_step[3, 0]),
                "inconsistent": float(p_class_given_step[3, 1]),
            },
        },
        "step0_likelihood_ratio_inconsistent_over_consistent": step0_ratio,
    }


def compute_conditionals(
    distance: int,
    qec_rounds: int,
    shots: int,
    p_two_qubit_erasure: float,
    seed: int,
    strict_filter: bool,
    consistency_source: str,
) -> dict:
    if distance < 5 or distance % 2 == 0:
        raise ValueError("distance must be odd and >= 5")
    if qec_rounds <= 0 or shots <= 0:
        raise ValueError("qec_rounds and shots must be > 0")
    if consistency_source not in {"measurement", "detector"}:
        raise ValueError("consistency_source must be 'measurement' or 'detector'")

    code = qerasure.RotatedSurfaceCode(distance)
    noise = qerasure.NoiseParams()
    noise.set(qerasure.NoiseChannel.TWO_QUBIT_ERASURE, p_two_qubit_erasure)
    sim_params = qerasure.ErasureSimParams(
        code=code,
        noise=noise,
        qec_rounds=qec_rounds,
        shots=shots,
        seed=seed,
        erasure_selection=qerasure.ErasureQubitSelection.DATA_QUBITS,
    )

    erasure_results = qerasure.ErasureSimulator(sim_params).simulate()
    lowering_result = qerasure.Lowerer(code, make_lowering_params()).lower(erasure_results)

    no_partner = (1 << 64) - 1
    num_data = code.x_anc_offset
    z_anc_offset = code.z_anc_offset
    num_x_anc = code.z_anc_offset - code.x_anc_offset
    num_z_anc = code.num_qubits - z_anc_offset
    num_anc = num_x_anc + num_z_anc
    data_to_z = code.data_to_z_ancilla_slots
    partner_map = code.partner_map

    # Start from data qubits with two Z neighbors.
    non_boundary_data = [
        q for q, (z1, z2) in enumerate(data_to_z) if z1 != no_partner and z2 != no_partner
    ]

    # Classify non-boundary data qubits by X/Z interaction schedule pattern.
    schedule_label_by_data = {}
    schedule_qubit_counts = defaultdict(int)
    for q in non_boundary_data:
        step_types = []
        for step in range(4):
            partner = int(partner_map[step * code.num_qubits + q])
            if partner == no_partner:
                step_types.append("N")
            elif code.x_anc_offset <= partner < code.z_anc_offset:
                step_types.append("X")
            elif partner >= code.z_anc_offset:
                step_types.append("Z")
            else:
                step_types.append("?")
        pattern = "".join(step_types)
        if pattern == "XZZX":
            label = "XZZX"
        elif pattern == "ZXXZ":
            label = "ZXXZ"
        else:
            label = f"OTHER:{pattern}"
        schedule_label_by_data[q] = label
        schedule_qubit_counts[label] += 1

    # Restrict analysis to qubits with one of the two full interior schedules.
    non_boundary_data = [
        q for q in non_boundary_data if schedule_label_by_data[q] in {"XZZX", "ZXXZ"}
    ]
    non_boundary_set = set(non_boundary_data)

    # z ancilla -> supporting data qubits, used for confounding filter.
    z_to_data = defaultdict(set)
    for q in range(num_data):
        z1, z2 = data_to_z[q]
        if z1 != no_partner:
            z_to_data[int(z1)].add(q)
        if z2 != no_partner:
            z_to_data[int(z2)].add(q)

    # Earliest erasure step per (shot, round, q), and all erased data per (shot, round).
    erasure_step = {}
    erased_data_by_shot_round = defaultdict(set)
    for shot in range(shots):
        events = erasure_results.sparse_erasures[shot]
        offsets = erasure_results.erasure_timestep_offsets[shot]
        for r in range(qec_rounds):
            for s in range(4):
                t = r * 4 + s
                for k in range(offsets[t], offsets[t + 1]):
                    ev = events[k]
                    if ev.event_type != qerasure.EventType.ERASURE:
                        continue
                    q = int(ev.qubit_idx)
                    if q >= num_data:
                        continue
                    erased_data_by_shot_round[(shot, r)].add(q)
                    if q in non_boundary_set:
                        key = (shot, r, q)
                        if key not in erasure_step:
                            erasure_step[key] = s

    strict_keep = set(erasure_step.keys())
    if strict_filter:
        strict_keep.clear()
        for (shot, r, q), _ in erasure_step.items():
            z1, z2 = data_to_z[q]
            support = set(z_to_data[int(z1)]) | set(z_to_data[int(z2)])
            others = erased_data_by_shot_round[(shot, r)] - {q}
            confounded = any(q2 in support for q2 in others)
            if consistency_source == "detector" and r > 0:
                # Detector bits are temporal parities (round r vs r-1). If any
                # support-overlapping data qubit was erased in r-1, it can
                # influence same-round detector parity and confound attribution.
                confounded = confounded or any(
                    q2 in support for q2 in erased_data_by_shot_round[(shot, r - 1)]
                )
            if not confounded:
                strict_keep.add((shot, r, q))

    # counts[parity_class, step]
    # parity_class: 0=consistent, 1=inconsistent
    counts = np.zeros((2, 4), dtype=np.int64)
    counts_by_schedule = {
        "XZZX": np.zeros((2, 4), dtype=np.int64),
        "ZXXZ": np.zeros((2, 4), dtype=np.int64),
    }
    for shot in range(shots):
        circuit = qerasure.build_logical_stabilizer_circuit_object(
            code=code, lowering_result=lowering_result, shot_index=shot
        )
        if consistency_source == "measurement":
            meas_sample = circuit.compile_sampler().sample(shots=1)
            rec = np.asarray(meas_sample, dtype=np.uint8)
        else:
            det_sample, _ = circuit.compile_detector_sampler().sample(
                shots=1, separate_observables=True
            )
            rec = np.asarray(det_sample, dtype=np.uint8)
        if rec.ndim == 2:
            rec = rec[0]

        for r in range(qec_rounds):
            round_offset = r * num_z_anc
            for q in non_boundary_data:
                key = (shot, r, q)
                if key not in erasure_step:
                    continue
                if key not in strict_keep:
                    continue

                step = erasure_step[key]
                z1, z2 = data_to_z[q]
                zi1 = int(z1 - z_anc_offset)
                zi2 = int(z2 - z_anc_offset)
                if consistency_source == "measurement":
                    # MR order per round is [all X ancillas, then all Z ancillas].
                    i1 = r * num_anc + num_x_anc + zi1
                    i2 = r * num_anc + num_x_anc + zi2
                else:
                    # Detector order per round is all Z checks.
                    i1 = round_offset + zi1
                    i2 = round_offset + zi2
                parity = int(rec[i1] ^ rec[i2])  # 0 consistent, 1 inconsistent
                counts[parity, step] += 1
                label = schedule_label_by_data.get(q, "")
                if label in counts_by_schedule:
                    counts_by_schedule[label][parity, step] += 1

    schedule_total_instances = defaultdict(int)
    schedule_used_instances = defaultdict(int)
    for shot, r, q in erasure_step.keys():
        label = schedule_label_by_data.get(q, "")
        if label in counts_by_schedule:
            schedule_total_instances[label] += 1
    for shot, r, q in strict_keep:
        label = schedule_label_by_data.get(q, "")
        if label in counts_by_schedule:
            schedule_used_instances[label] += 1

    result = {
        "distance": distance,
        "qec_rounds": qec_rounds,
        "shots": shots,
        "p_two_qubit_erasure": p_two_qubit_erasure,
        "seed": seed,
        "strict_filter": strict_filter,
        "consistency_source": consistency_source,
        "instances_total_non_boundary": len(erasure_step),
        "instances_used_after_filter": len(strict_keep),
        "schedule_qubit_counts": dict(schedule_qubit_counts),
        "by_schedule": {},
    }
    result.update(summarize_from_counts(counts))
    for label in ("XZZX", "ZXXZ"):
        section = summarize_from_counts(counts_by_schedule[label])
        section["instances_total_non_boundary"] = int(schedule_total_instances[label])
        section["instances_used_after_filter"] = int(schedule_used_instances[label])
        result["by_schedule"][label] = section
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute conditional erasure-timestep probabilities from parity consistency "
            "of the two associated same-round Z-check detectors."
        )
    )
    parser.add_argument("--distance", type=int, default=5)
    parser.add_argument("--qec-rounds", type=int, default=5)
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--p-tqe", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--no-strict-filter",
        action="store_true",
        help="Disable confounding filter that removes rounds where other erased data touch either Z check.",
    )
    parser.add_argument(
        "--consistency-source",
        type=str,
        default="detector",
        choices=["measurement", "detector"],
        help=(
            "Use same-round Z-ancilla measurements ('measurement') or Z-check detector bits "
            "('detector') when defining consistency."
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / "apps" / "results" / "conditional_timestep_zcheck_parity.json",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=REPO_ROOT / "apps" / "results" / "conditional_timestep_zcheck_parity.png",
    )
    args = parser.parse_args()

    result = compute_conditionals(
        distance=args.distance,
        qec_rounds=args.qec_rounds,
        shots=args.shots,
        p_two_qubit_erasure=args.p_tqe,
        seed=args.seed,
        strict_filter=not args.no_strict_filter,
        consistency_source=args.consistency_source,
    )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2))

    def extract_plot_arrays(section: dict):
        p_step_given_consistent = np.array(
            section["p_step_given_consistency"]["consistent"], dtype=float
        )
        p_step_given_inconsistent = np.array(
            section["p_step_given_consistency"]["inconsistent"], dtype=float
        )
        p_consistent_given_step = np.array(
            [
                section["p_consistency_given_step"]["step_0"]["consistent"],
                section["p_consistency_given_step"]["step_1"]["consistent"],
                section["p_consistency_given_step"]["step_2"]["consistent"],
                section["p_consistency_given_step"]["step_3"]["consistent"],
            ],
            dtype=float,
        )
        p_inconsistent_given_step = np.array(
            [
                section["p_consistency_given_step"]["step_0"]["inconsistent"],
                section["p_consistency_given_step"]["step_1"]["inconsistent"],
                section["p_consistency_given_step"]["step_2"]["inconsistent"],
                section["p_consistency_given_step"]["step_3"]["inconsistent"],
            ],
            dtype=float,
        )
        return (
            p_step_given_consistent,
            p_step_given_inconsistent,
            p_consistent_given_step,
            p_inconsistent_given_step,
        )

    def save_conditional_figure(section: dict, out_path: Path, title_suffix: str) -> None:
        (
            p_step_given_consistent,
            p_step_given_inconsistent,
            p_consistent_given_step,
            p_inconsistent_given_step,
        ) = extract_plot_arrays(section)
        steps = np.arange(4)
        fig, axs = plt.subplots(2, 2, figsize=(10, 7))

        axs[0, 0].bar(steps, p_step_given_consistent)
        axs[0, 0].set_title("P(step=s | consistent)")
        axs[0, 0].set_xlabel("step s")
        axs[0, 0].set_ylabel("probability")
        axs[0, 0].set_ylim(0.0, 1.0)
        axs[0, 0].set_xticks(steps)
        axs[0, 0].grid(alpha=0.25, axis="y")

        axs[0, 1].bar(steps, p_step_given_inconsistent)
        axs[0, 1].set_title("P(step=s | inconsistent)")
        axs[0, 1].set_xlabel("step s")
        axs[0, 1].set_ylabel("probability")
        axs[0, 1].set_ylim(0.0, 1.0)
        axs[0, 1].set_xticks(steps)
        axs[0, 1].grid(alpha=0.25, axis="y")

        axs[1, 0].bar(steps, p_consistent_given_step)
        axs[1, 0].set_title("P(consistent | step=s)")
        axs[1, 0].set_xlabel("step s")
        axs[1, 0].set_ylabel("probability")
        axs[1, 0].set_ylim(0.0, 1.0)
        axs[1, 0].set_xticks(steps)
        axs[1, 0].grid(alpha=0.25, axis="y")

        axs[1, 1].bar(steps, p_inconsistent_given_step)
        axs[1, 1].set_title("P(inconsistent | step=s)")
        axs[1, 1].set_xlabel("step s")
        axs[1, 1].set_ylabel("probability")
        axs[1, 1].set_ylim(0.0, 1.0)
        axs[1, 1].set_xticks(steps)
        axs[1, 1].grid(alpha=0.25, axis="y")

        fig.suptitle(
            f"d={result['distance']}, rounds={result['qec_rounds']}, shots={result['shots']}, "
            f"strict_filter={result['strict_filter']}, {title_suffix}",
            fontsize=10,
        )
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=220)
        plt.close(fig)

    save_conditional_figure(result, args.plot_out, "ALL schedules")
    xzzx_out = args.plot_out.with_name(args.plot_out.stem + "_XZZX" + args.plot_out.suffix)
    zxxz_out = args.plot_out.with_name(args.plot_out.stem + "_ZXXZ" + args.plot_out.suffix)
    save_conditional_figure(result["by_schedule"]["XZZX"], xzzx_out, "schedule XZZX")
    save_conditional_figure(result["by_schedule"]["ZXXZ"], zxxz_out, "schedule ZXXZ")

    print(json.dumps(result, indent=2))
    print(f"Saved plot: {args.plot_out}")
    print(f"Saved plot: {xzzx_out}")
    print(f"Saved plot: {zxxz_out}")


if __name__ == "__main__":
    main()
