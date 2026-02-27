#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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

import qerasure


NO_PARTNER = (1 << 64) - 1


def make_lowering_params() -> qerasure.LoweringParams:
    program = qerasure.SpreadProgram()
    program.append("Z_ERROR(0.5) X_1; Z_ERROR(0.5) X_2")
    program.append("COND_X_ERROR(0.5) Z_1; ELSE_X_ERROR(1.0) Z_2")
    reset = qerasure.LoweredErrorParams(qerasure.PauliError.DEPOLARIZE, 1.0)
    return qerasure.LoweringParams(program, reset)


def bernoulli_per_round(logical_error_rate: float | None, rounds: int) -> float | None:
    if logical_error_rate is None:
        return None
    p = float(np.clip(logical_error_rate, 0.0, 1.0))
    return 1.0 - (1.0 - p) ** (1.0 / float(rounds))


def normalize_step_given_consistency(
    counts: np.ndarray, floor_probability: float
) -> dict[str, list[float]]:
    # counts[parity, step], parity 0=consistent, 1=inconsistent
    probs = np.zeros((2, 4), dtype=float)
    for parity in (0, 1):
        total = int(np.sum(counts[parity]))
        if total > 0:
            probs[parity] = counts[parity] / total
        # Avoid exact zeros for unseen-but-possible events.
        probs[parity] = np.where(probs[parity] <= 0.0, floor_probability, probs[parity])
        norm = float(np.sum(probs[parity]))
        if norm > 0.0:
            probs[parity] /= norm
    return {
        "consistent": probs[0].tolist(),
        "inconsistent": probs[1].tolist(),
    }


def infer_schedule_labels(code: qerasure.RotatedSurfaceCode) -> dict[int, str]:
    partner_map = code.partner_map
    num_qubits = code.num_qubits
    labels: dict[int, str] = {}
    for q in range(code.x_anc_offset):
        xz = []
        for step in range(4):
            partner = int(partner_map[step * num_qubits + q])
            if partner == NO_PARTNER:
                xz.append("N")
            elif code.x_anc_offset <= partner < code.z_anc_offset:
                xz.append("X")
            else:
                xz.append("Z")
        pattern = "".join(xz)
        if pattern == "XZZX":
            labels[q] = "XZZX"
        elif pattern == "ZXXZ":
            labels[q] = "ZXXZ"
        else:
            labels[q] = f"OTHER:{pattern}"
    return labels


def simulate_and_lower(
    code: qerasure.RotatedSurfaceCode,
    qec_rounds: int,
    shots: int,
    p_tqe: float,
    seed: int,
    lowering_params: qerasure.LoweringParams,
):
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
    erasure_result = qerasure.ErasureSimulator(sim_params).simulate()
    lowering_result = qerasure.Lowerer(code, lowering_params).lower(erasure_result)
    return sim_params, erasure_result, lowering_result


def calibrate_conditionals(
    code: qerasure.RotatedSurfaceCode,
    qec_rounds: int,
    p_tqe: float,
    calibration_shots: int,
    seed: int,
    lowering_params: qerasure.LoweringParams,
) -> dict:
    _, erasure_result, lowering_result = simulate_and_lower(
        code=code,
        qec_rounds=qec_rounds,
        shots=calibration_shots,
        p_tqe=p_tqe,
        seed=seed,
        lowering_params=lowering_params,
    )

    schedule_label_by_data = infer_schedule_labels(code)
    data_to_z = code.data_to_z_ancilla_slots
    num_data = code.x_anc_offset
    num_z_anc = code.num_qubits - code.z_anc_offset

    # Non-boundary qubits in the two target schedules only.
    eligible_data = [
        q
        for q in range(num_data)
        if data_to_z[q][0] != NO_PARTNER
        and data_to_z[q][1] != NO_PARTNER
        and schedule_label_by_data[q] in {"XZZX", "ZXXZ"}
    ]
    eligible_set = set(eligible_data)

    z_to_data = defaultdict(set)
    for q in range(num_data):
        z1, z2 = data_to_z[q]
        if z1 != NO_PARTNER:
            z_to_data[int(z1)].add(q)
        if z2 != NO_PARTNER:
            z_to_data[int(z2)].add(q)

    erasure_step = {}
    erased_data_by_shot_round = defaultdict(set)
    for shot in range(calibration_shots):
        events = erasure_result.sparse_erasures[shot]
        offsets = erasure_result.erasure_timestep_offsets[shot]
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
                    if q in eligible_set:
                        key = (shot, r, q)
                        if key not in erasure_step:
                            erasure_step[key] = s

    # Strict confound filter for detector parity:
    # remove same-round and previous-round support-overlapping erasures.
    strict_keep = set()
    for (shot, r, q), _ in erasure_step.items():
        z1, z2 = data_to_z[q]
        support = set(z_to_data[int(z1)]) | set(z_to_data[int(z2)])
        others = erased_data_by_shot_round[(shot, r)] - {q}
        confounded = any(q2 in support for q2 in others)
        if r > 0:
            confounded = confounded or any(
                q2 in support for q2 in erased_data_by_shot_round[(shot, r - 1)]
            )
        if not confounded:
            strict_keep.add((shot, r, q))

    entries_by_shot = defaultdict(list)
    for shot, r, q in strict_keep:
        step = erasure_step[(shot, r, q)]
        z1, z2 = data_to_z[q]
        zi1 = int(z1 - code.z_anc_offset)
        zi2 = int(z2 - code.z_anc_offset)
        label = schedule_label_by_data[q]
        entries_by_shot[shot].append((r, step, zi1, zi2, label))

    counts_by_schedule = {
        "XZZX": np.zeros((2, 4), dtype=np.int64),
        "ZXXZ": np.zeros((2, 4), dtype=np.int64),
    }

    for shot in range(calibration_shots):
        if shot not in entries_by_shot:
            continue
        logical = qerasure.build_logical_stabilizer_circuit_object(code, lowering_result, shot)
        det, _ = logical.compile_detector_sampler().sample(
            shots=1, separate_observables=True
        )
        det_bits = np.asarray(det, dtype=np.uint8)
        if det_bits.ndim == 2:
            det_bits = det_bits[0]
        z_det = det_bits[: qec_rounds * num_z_anc]
        for r, step, zi1, zi2, label in entries_by_shot[shot]:
            parity = int(z_det[r * num_z_anc + zi1] ^ z_det[r * num_z_anc + zi2])
            counts_by_schedule[label][parity, step] += 1

    floor_probability = float(p_tqe * p_tqe)
    p_xzzx = normalize_step_given_consistency(
        counts_by_schedule["XZZX"], floor_probability=floor_probability
    )
    p_zxxz = normalize_step_given_consistency(
        counts_by_schedule["ZXXZ"], floor_probability=floor_probability
    )
    return {
        "p_step_given_consistent_xzzx": p_xzzx["consistent"],
        "p_step_given_inconsistent_xzzx": p_xzzx["inconsistent"],
        "p_step_given_consistent_zxxz": p_zxxz["consistent"],
        "p_step_given_inconsistent_zxxz": p_zxxz["inconsistent"],
        "zero_probability_floor": floor_probability,
        "counts_by_schedule": {
            "XZZX": counts_by_schedule["XZZX"].tolist(),
            "ZXXZ": counts_by_schedule["ZXXZ"].tolist(),
        },
        "instances_used_after_filter": int(len(strict_keep)),
    }


def evaluate_conditioned_decoder(
    code: qerasure.RotatedSurfaceCode,
    qec_rounds: int,
    p_tqe: float,
    eval_shots: int,
    seed: int,
    lowering_params: qerasure.LoweringParams,
    conditionals: dict,
    failure_root: Path,
) -> dict:
    sim_params, erasure_result, lowering_result = simulate_and_lower(
        code=code,
        qec_rounds=qec_rounds,
        shots=eval_shots,
        p_tqe=p_tqe,
        seed=seed,
        lowering_params=lowering_params,
    )

    num_z_anc = code.num_qubits - code.z_anc_offset
    mismatches = 0
    attempted = 0
    failures = 0
    t0 = time.perf_counter()

    for shot in range(eval_shots):
        logical = None
        virtual = None
        try:
            logical = qerasure.build_logical_stabilizer_circuit_object(
                code=code, lowering_result=lowering_result, shot_index=shot
            )
            det_sample, obs = logical.compile_detector_sampler().sample(
                shots=1, separate_observables=True
            )
            det = np.asarray(det_sample, dtype=np.uint8)
            if det.ndim == 1:
                det = det[None, :]
            truth = np.asarray(obs, dtype=np.uint8)
            if truth.ndim == 1:
                truth = truth[:, None]

            z_bits = det[0, : qec_rounds * num_z_anc]
            virtual = qerasure.build_virtual_decoder_stim_circuit_conditioned_object(
                code=code,
                qec_rounds=qec_rounds,
                lowering_params=lowering_params,
                lowering_result=lowering_result,
                shot_index=shot,
                two_qubit_erasure_probability=p_tqe,
                z_detector_syndrome_bits=[int(x) for x in z_bits],
                p_step_given_consistent_xzzx=conditionals["p_step_given_consistent_xzzx"],
                p_step_given_inconsistent_xzzx=conditionals[
                    "p_step_given_inconsistent_xzzx"
                ],
                p_step_given_consistent_zxxz=conditionals["p_step_given_consistent_zxxz"],
                p_step_given_inconsistent_zxxz=conditionals[
                    "p_step_given_inconsistent_zxxz"
                ],
                condition_on_erasure_in_round=True,
            )
            dem = virtual.detector_error_model(
                decompose_errors=True,
                approximate_disjoint_errors=True,
            )
            matching = pm.Matching.from_detector_error_model(dem)
            if hasattr(matching, "decode_batch"):
                pred = np.asarray(matching.decode_batch(det), dtype=np.uint8)
            else:
                pred = np.asarray([matching.decode(s) for s in det], dtype=np.uint8)
            if pred.ndim == 1:
                pred = pred[:, None]
            n_obs = min(pred.shape[1], truth.shape[1])
            mismatch = bool(np.any((pred[:, :n_obs] ^ truth[:, :n_obs]) != 0))
            mismatches += int(mismatch)
            attempted += 1
        except Exception as exc:
            failures += 1
            fail_dir = failure_root / f"shot_{shot:05d}"
            fail_dir.mkdir(parents=True, exist_ok=True)

            erasure_qubits = [
                int(ev.qubit_idx) for ev in erasure_result.sparse_erasures[shot]
            ]
            lowering_qubits = [
                int(ev.qubit_idx) for ev in lowering_result.sparse_cliffords[shot]
            ]

            # Save circuits for direct inspection.
            if logical is not None:
                (fail_dir / "logical_circuit.stim").write_text(str(logical))
            if virtual is not None:
                (fail_dir / "virtual_circuit.stim").write_text(str(virtual))

            # Save erasure/lowering visualizations.
            er_fig, _ = qerasure.visualize_erasures(
                erasure_result, sim_params, shot_idx=shot
            )
            er_fig.savefig(fail_dir / "erasure_timeline.png", dpi=220)
            plt.close(er_fig)

            low_fig, _ = qerasure.visualize_lowering(
                lowering_result, sim_params, shot_idx=shot
            )
            low_fig.savefig(fail_dir / "lowering_timeline.png", dpi=220)
            plt.close(low_fig)

            meta = {
                "shot_index": shot,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "distance": code.distance,
                "qec_rounds": qec_rounds,
                "p_two_qubit_erasure": p_tqe,
                "seed": seed,
                "conditionals": conditionals,
                "erasure_qubit_indices": erasure_qubits,
                "lowering_qubit_indices": lowering_qubits,
            }
            (fail_dir / "failure.json").write_text(json.dumps(meta, indent=2))
            print(f"erasure_qubit_indices: {erasure_qubits}")
            print(f"lowering_qubit_indices: {lowering_qubits}")
            print(
                f"Failure at shot {shot}; artifacts saved to {fail_dir}. "
                "Exiting immediately."
            )
            raise SystemExit(1) from exc

    elapsed = time.perf_counter() - t0
    ler = (mismatches / attempted) if attempted else None
    return {
        "logical_error_rate": ler,
        "logical_error_rate_per_round": bernoulli_per_round(ler, qec_rounds),
        "shots_attempted": attempted,
        "shots_failed": failures,
        "throughput_shots_per_sec": (attempted / elapsed) if elapsed > 0 else None,
        "decode_loop_seconds": elapsed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Superposed LER sweep using syndrome-conditioned virtual decoder "
            "(calibrate conditionals, then decode with conditioned priors)."
        )
    )
    parser.add_argument("--calibration-shots", type=int, default=10000)
    parser.add_argument("--eval-shots", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--points", type=int, default=10)
    parser.add_argument("--p-min", type=float, default=1e-2)
    parser.add_argument("--p-max", type=float, default=1e-1)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / "apps" / "results" / "ler_superposed_conditioned_d3_d5_d7.json",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=REPO_ROOT / "apps" / "results" / "ler_superposed_conditioned_d3_d5_d7.png",
    )
    parser.add_argument(
        "--failure-root",
        type=Path,
        default=REPO_ROOT / "apps" / "failures_conditioned",
    )
    args = parser.parse_args()

    configs = [(3,3), (5, 5), (7, 7)]
    p_values = list(np.logspace(math.log10(args.p_min), math.log10(args.p_max), args.points))
    lowering_params = make_lowering_params()

    rows = []
    t_start = time.perf_counter()

    for cfg_idx, (distance, rounds) in enumerate(configs):
        code = qerasure.RotatedSurfaceCode(distance)
        for p_idx, p_tqe in enumerate(p_values):
            base_seed = args.seed + cfg_idx * 1_000_000 + p_idx * 10_000
            t0 = time.perf_counter()
            cond = calibrate_conditionals(
                code=code,
                qec_rounds=rounds,
                p_tqe=float(p_tqe),
                calibration_shots=args.calibration_shots,
                seed=base_seed,
                lowering_params=lowering_params,
            )
            t1 = time.perf_counter()
            eval_result = evaluate_conditioned_decoder(
                code=code,
                qec_rounds=rounds,
                p_tqe=float(p_tqe),
                eval_shots=args.eval_shots,
                seed=base_seed + 1,
                lowering_params=lowering_params,
                conditionals=cond,
                failure_root=(
                    args.failure_root
                    / f"d{distance}_r{rounds}_p{float(p_tqe):.6g}_seed{base_seed+1}"
                ),
            )
            t2 = time.perf_counter()
            row = {
                "distance": distance,
                "qec_rounds": rounds,
                "p_two_qubit_erasure": float(p_tqe),
                "calibration_shots": args.calibration_shots,
                "eval_shots": args.eval_shots,
                "conditionals": cond,
                "evaluation": eval_result,
                "timing_seconds": {
                    "calibration": t1 - t0,
                    "evaluation": t2 - t1,
                    "total_point": t2 - t0,
                },
            }
            rows.append(row)
            print(
                f"[cfg {cfg_idx+1}/{len(configs)} | p {p_idx+1}/{len(p_values)}] "
                f"d={distance} r={rounds} p={p_tqe:.4g} "
                f"ler={eval_result['logical_error_rate']} "
                f"ler/round={eval_result['logical_error_rate_per_round']} "
                f"attempted={eval_result['shots_attempted']} failed={eval_result['shots_failed']}"
            )

    elapsed = time.perf_counter() - t_start
    payload = {
        "configs": [{"distance": d, "qec_rounds": r} for d, r in configs],
        "p_values": [float(x) for x in p_values],
        "calibration_shots": args.calibration_shots,
        "eval_shots": args.eval_shots,
        "seed": args.seed,
        "elapsed_seconds": elapsed,
        "rows": rows,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2))

    plt.figure(figsize=(8.2, 5.2))
    for distance, rounds in configs:
        cfg_rows = [
            row
            for row in rows
            if row["distance"] == distance and row["qec_rounds"] == rounds
        ]
        cfg_rows.sort(key=lambda r: r["p_two_qubit_erasure"])
        x = np.array([r["p_two_qubit_erasure"] for r in cfg_rows], dtype=float)
        y = np.array(
            [
                np.nan
                if r["evaluation"]["logical_error_rate_per_round"] is None
                else float(r["evaluation"]["logical_error_rate_per_round"])
                for r in cfg_rows
            ],
            dtype=float,
        )
        plt.plot(x, y, marker="o", linewidth=1.5, label=f"d={distance}, r={rounds}")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Two-qubit erasure probability p")
    plt.ylabel("Logical error rate per round")
    plt.title("Conditioned Decoder: LER/round vs p")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    args.plot_out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.plot_out, dpi=220)
    plt.close()

    print(f"\nSaved JSON: {args.json_out}")
    print(f"Saved plot: {args.plot_out}")
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
