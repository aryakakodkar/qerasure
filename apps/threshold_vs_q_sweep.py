#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_SRC = REPO_ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
    sys.path.insert(0, str(PYTHON_SRC))

# Load Stim bindings first so pybind cross-module cast of stim::Circuit works.
import stim  # noqa: F401
import qerasure as qe

_PM_MODULE = None


def _log(message: str) -> None:
    """Emit progress logs immediately, even when stdout is block-buffered."""
    print(message, flush=True)


def _to_u32_seed(value: int) -> int:
    """Normalize arbitrary Python ints into the uint32 seed range."""
    return int(value) & 0xFFFFFFFF


def get_pymatching():
    """Import pymatching once per process."""
    global _PM_MODULE
    if _PM_MODULE is None:
        import pymatching as pm

        _PM_MODULE = pm
    return _PM_MODULE


def parse_configs(configs_text: str) -> list[tuple[int, int]]:
    configs: list[tuple[int, int]] = []
    for raw in configs_text.split(";"):
        raw = raw.strip()
        if not raw:
            continue
        pieces = [p.strip() for p in raw.split(",")]
        if len(pieces) != 2:
            raise ValueError(f"Invalid config '{raw}', expected 'distance,rounds'.")
        distance = int(pieces[0])
        rounds = int(pieces[1])
        if distance <= 0 or rounds <= 0:
            raise ValueError(f"Invalid config '{raw}', distance and rounds must be > 0.")
        configs.append((distance, rounds))
    if not configs:
        raise ValueError("No valid configs parsed from --configs.")
    return configs


def parse_float_list(text: str, name: str) -> list[float]:
    values = [float(v.strip()) for v in text.split(",") if v.strip()]
    if not values:
        raise ValueError(f"No values parsed for {name}.")
    return values


def resolve_e_values(e_values_text: str, e_interior_points: int) -> list[float]:
    """Resolve erasure sweep values.

    If exactly two e-values are provided, interpolate a log-spaced grid with
    `e_interior_points` strictly between them (plus both endpoints).
    """
    raw_values = parse_float_list(e_values_text, "--e-values")
    if len(raw_values) != 2:
        return raw_values

    interior = int(e_interior_points)
    if interior < 0:
        raise ValueError("--e-interior-points must be >= 0.")

    e0 = float(raw_values[0])
    e1 = float(raw_values[1])
    if e0 <= 0.0 or e1 <= 0.0:
        raise ValueError(
            "When exactly two --e-values are provided, both must be > 0 for log interpolation."
        )
    if abs(e0 - e1) < 1e-18:
        return [e0]

    e_min = min(e0, e1)
    e_max = max(e0, e1)
    total_points = interior + 2
    return [float(v) for v in np.logspace(np.log10(e_min), np.log10(e_max), total_points)]


def float_tag(value: float) -> str:
    tag = format(float(value), ".12g")
    tag = tag.replace("-", "m").replace("+", "")
    tag = tag.replace(".", "p")
    return tag


def bernoulli_per_round(logical_error_rate: float, rounds: int) -> float:
    p = float(np.clip(logical_error_rate, 0.0, 1.0))
    return 1.0 - (1.0 - p) ** (1.0 / float(rounds))


def _ensure_prediction_width(predictions, shots: int, required_width: int):
    import numpy as np

    width = max(0, int(required_width))
    if predictions is None:
        return np.zeros((shots, width), dtype=np.uint8)
    if width <= predictions.shape[1]:
        return predictions
    grown = np.zeros((shots, width), dtype=np.uint8)
    grown[:, : predictions.shape[1]] = predictions
    return grown


def decode_batch_allow_failures(
    grouped_decoder: qe.SurfaceCodeBatchDecoder,
    dem_builder: qe.SurfDemBuilder,
    detector_samples,
    check_flags,
    num_threads: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode shots while tolerating rare pymatching failures.

    Returns:
      - predictions: uint8 array [num_shots, num_observables]
      - failed_mask: bool array [num_shots], true where decode failed
    """
    import numpy as np
    pm = get_pymatching()

    dets = np.asarray(detector_samples, dtype=np.uint8)
    checks = np.asarray(check_flags, dtype=np.uint8)
    shots = int(dets.shape[0])

    failed_mask = np.zeros((shots,), dtype=bool)
    if shots == 0:
        return np.zeros((0, 0), dtype=np.uint8), failed_mask

    try:
        preds = grouped_decoder.decode_batch(dets, checks, num_threads=int(num_threads))
        preds = np.asarray(preds, dtype=np.uint8)
        if preds.ndim == 1:
            preds = preds[:, None]
        return preds, failed_mask
    except Exception as exc:
        _log(
            "decode_batch failed, retrying per-check-pattern with failure-tolerant path:"
            f" {type(exc).__name__}: {exc}"
        )

    packed_checks = np.packbits(checks, axis=1, bitorder="little")
    groups: dict[bytes, list[int]] = {}
    for shot_idx in range(shots):
        key = packed_checks[shot_idx].tobytes()
        groups.setdefault(key, []).append(shot_idx)

    predictions = None
    for key, shot_indices in groups.items():
        if checks.shape[1] == 0:
            check_row = np.zeros((0,), dtype=np.uint8)
        else:
            packed_row = np.frombuffer(key, dtype=np.uint8)
            unpacked = np.unpackbits(packed_row, bitorder="little")
            check_row = unpacked[: checks.shape[1]].astype(np.uint8, copy=False)

        try:
            decoded_circuit = dem_builder.build_decoded_circuit(check_row, verbose=False)
            decoded_dem = decoded_circuit.detector_error_model(
                decompose_errors=True,
                approximate_disjoint_errors=True,
            )
            matching = pm.Matching.from_detector_error_model(decoded_dem)
        except Exception:
            failed_mask[np.asarray(shot_indices, dtype=np.int64)] = True
            continue

        group_detectors = dets[np.asarray(shot_indices, dtype=np.int64)]
        try:
            group_preds = np.asarray(matching.decode_batch(group_detectors), dtype=np.uint8)
            if group_preds.ndim == 1:
                group_preds = group_preds[:, None]
            predictions = _ensure_prediction_width(predictions, shots, int(group_preds.shape[1]))
            n = min(predictions.shape[1], group_preds.shape[1])
            for group_pos, shot_idx in enumerate(shot_indices):
                predictions[shot_idx, :n] = group_preds[group_pos, :n]
            continue
        except Exception:
            pass

        for group_pos, shot_idx in enumerate(shot_indices):
            try:
                pred = np.asarray(matching.decode(group_detectors[group_pos]), dtype=np.uint8)
                if pred.ndim == 0:
                    pred = pred.reshape(1)
            except Exception:
                failed_mask[shot_idx] = True
                continue
            predictions = _ensure_prediction_width(predictions, shots, int(pred.shape[0]))
            n = min(predictions.shape[1], pred.shape[0])
            predictions[shot_idx, :n] = pred[:n]

    if predictions is None:
        predictions = np.zeros((shots, 0), dtype=np.uint8)
    return predictions, failed_mask


def run_single_point(
    distance: int,
    rounds: int,
    shots: int,
    erasure_prob: float,
    check_prob: float,
    pauli_prob: float,
    seed: int,
    sample_threads: int,
    decode_threads: int,
    max_batch_bytes: int,
    single_qubit_errors: bool,
) -> dict:
    circuit = qe.SurfaceCodeRotated(distance).build_circuit(
        rounds=rounds,
        erasure_prob=erasure_prob,
        erasable_qubits="ALL",
        reset_failure_prob=0.0,
        single_qubit_errors=single_qubit_errors,
        post_clifford_pauli_prob=pauli_prob,
    )

    model = qe.ErasureModel(
        2,
        qe.PauliChannel(0.25, 0.25, 0.25),
        qe.PauliChannel(0.25, 0.25, 0.25),
        qe.TQGSpreadModel(
            qe.PauliChannel(0.25, 0.25, 0.25),
            qe.PauliChannel(0.25, 0.25, 0.25),
        ),
    )
    model.check_false_negative_prob = check_prob
    model.check_false_positive_prob = check_prob

    compiled = qe.CompiledErasureProgram(circuit, model)
    sampler = qe.StreamSampler(compiled)
    dem_builder = qe.SurfDemBuilder(compiled)
    grouped_decoder = qe.SurfaceCodeBatchDecoder(
        compiled,
        dem_builder=dem_builder,
        max_batch_bytes=max_batch_bytes,
    )

    t0 = time.perf_counter()
    dets, obs, checks = sampler.sample(num_shots=shots, seed=seed, num_threads=sample_threads)
    t1 = time.perf_counter()
    predictions, failed_mask = decode_batch_allow_failures(
        grouped_decoder,
        dem_builder,
        dets,
        checks,
        num_threads=decode_threads,
    )
    t2 = time.perf_counter()

    truths = obs if obs.ndim == 2 else obs[:, None]
    n_obs = min(truths.shape[1], predictions.shape[1])

    valid_mask = ~failed_mask
    decoded_shots = int(np.sum(valid_mask))
    decode_failures = int(np.sum(failed_mask))

    if decoded_shots > 0 and n_obs > 0:
        mismatches = np.any(
            predictions[valid_mask, :n_obs] != truths[valid_mask, :n_obs],
            axis=1,
        )
        decoded_error_count = int(np.sum(mismatches))
    else:
        decoded_error_count = 0

    ler_decoded = float(decoded_error_count / decoded_shots) if decoded_shots > 0 else 0.0
    ler_all = float((decoded_error_count + decode_failures) / max(1, shots))

    return {
        "distance": distance,
        "qec_rounds": rounds,
        "shots": shots,
        "seed": seed,
        "single_qubit_errors": single_qubit_errors,
        "q_check": check_prob,
        "p_pauli": pauli_prob,
        "e_erasure": erasure_prob,
        "logical_error_rate": ler_decoded,
        "logical_error_rate_per_round": bernoulli_per_round(ler_decoded, rounds),
        "logical_error_rate_all_shots": ler_all,
        "logical_error_rate_per_round_all_shots": bernoulli_per_round(ler_all, rounds),
        "decoded_shots": decoded_shots,
        "decode_failures": decode_failures,
        "decode_failure_rate": float(decode_failures / max(1, shots)),
        "timing_seconds": {
            "sample": t1 - t0,
            "decode": t2 - t1,
            "total": t2 - t0,
        },
    }


def run_single_point_job(job: dict) -> tuple[dict, dict]:
    """Worker entrypoint for parallel sweep execution."""
    row = run_single_point(
        distance=int(job["distance"]),
        rounds=int(job["rounds"]),
        shots=int(job["shots"]),
        erasure_prob=float(job["e_erasure"]),
        check_prob=float(job["q_check"]),
        pauli_prob=float(job["p_pauli"]),
        seed=int(job["seed"]),
        sample_threads=int(job["sample_threads"]),
        decode_threads=1,
        max_batch_bytes=int(job["max_batch_bytes"]),
        single_qubit_errors=bool(job["single_qubit_errors"]),
    )
    row["case"] = str(job["case_label"])
    return job, row


def crossings_linear(
    curve_small: list[tuple[float, float]],
    curve_large: list[tuple[float, float]],
) -> list[float]:
    if len(curve_small) != len(curve_large):
        raise ValueError("Distance curves have different point counts.")

    crossings: list[float] = []
    for i in range(len(curve_small) - 1):
        x0, y0s = curve_small[i]
        x1, y1s = curve_small[i + 1]
        q0, y0l = curve_large[i]
        q1, y1l = curve_large[i + 1]
        if abs(x0 - q0) > 1e-15 or abs(x1 - q1) > 1e-15:
            raise ValueError("Distance curves use different x grids.")

        d0 = y0s - y0l
        d1 = y1s - y1l

        if d0 == 0.0:
            crossings.append(x0)
            continue
        if d0 * d1 < 0.0:
            frac = -d0 / (d1 - d0)
            crossings.append(x0 + frac * (x1 - x0))
        elif d1 == 0.0:
            crossings.append(x1)

    unique: list[float] = []
    for x in crossings:
        if not unique or abs(x - unique[-1]) > 1e-12:
            unique.append(x)
    return unique


def estimate_threshold_for_slice(rows: list[dict]) -> dict:
    curves: dict[int, list[tuple[float, float]]] = {}
    total_failures = 0
    total_shots = 0
    for row in rows:
        distance = int(row["distance"])
        erasure = float(row["e_erasure"])
        y = float(row["logical_error_rate_per_round"])
        curves.setdefault(distance, []).append((erasure, y))
        total_failures += int(row.get("decode_failures", 0))
        total_shots += int(row.get("shots", 0))

    distances = sorted(curves.keys())
    for distance in distances:
        curves[distance].sort(key=lambda t: t[0])

    pairwise: list[dict] = []
    selected_crossings: list[float] = []
    for i in range(len(distances) - 1):
        d_small = distances[i]
        d_large = distances[i + 1]
        cross = crossings_linear(curves[d_small], curves[d_large])
        selected = min(cross) if cross else None
        pairwise.append(
            {
                "pair": [d_small, d_large],
                "crossings": cross,
                "selected": selected,
            }
        )
        if selected is not None:
            selected_crossings.append(selected)

    threshold = float(np.mean(selected_crossings)) if selected_crossings else None
    spread = float(np.std(selected_crossings)) if len(selected_crossings) > 1 else 0.0
    return {
        "pairwise_crossings": pairwise,
        "threshold_estimate": threshold,
        "threshold_spread": spread,
        "num_pairs_with_crossing": len(selected_crossings),
        "decode_failures": total_failures,
        "decode_failure_rate": float(total_failures / max(1, total_shots)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "For each (p_pauli, q_check) pair, sweep e_erasure values and estimate the "
            "erasure threshold for both two-qubit-only and single-qubit-enabled builders."
        )
    )
    parser.add_argument("--configs", type=str, default="3,3;5,5;7,7")
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=random.randint(0, 2**32 - 1))
    parser.add_argument(
        "--e-values",
        type=str,
        required=True,
        help=(
            "Erasure probabilities. If 2 values are given, script builds a log-spaced grid "
            "with --e-interior-points values between endpoints."
        ),
    )
    parser.add_argument(
        "--e-interior-points",
        type=int,
        default=10,
        help="Number of interior log-spaced points to add when exactly 2 e-values are provided.",
    )
    parser.add_argument(
        "--q-values",
        type=str,
        required=True,
        help="Comma-separated check-error probabilities q (used for FN and FP).",
    )
    parser.add_argument(
        "--p-values",
        type=str,
        required=True,
        help="Comma-separated post-clifford Pauli probabilities p.",
    )
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument(
        "--sweep-threads",
        type=int,
        default=1,
        help=(
            "Number of outer sweep workers (parallel sweep points). "
            "Decoder always uses 1 thread."
        ),
    )
    parser.add_argument(
        "--sweep-backend",
        type=str,
        default="process",
        choices=["process", "thread"],
        help="Parallel backend for outer sweep workers.",
    )
    parser.add_argument(
        "--decode-threads",
        type=int,
        default=None,
        help="Deprecated: decoder is forced to 1 thread in this script.",
    )
    parser.add_argument("--max-batch-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / "apps" / "results" / "threshold_vs_q_sweep_summary.json",
    )
    parser.add_argument(
        "--pair-json-dir",
        type=Path,
        default=REPO_ROOT / "apps" / "results" / "threshold_vs_q_pairs",
    )
    parser.add_argument(
        "--plot-out",
        type=Path,
        default=REPO_ROOT / "apps" / "results" / "threshold_vs_q_sweep.png",
    )
    args = parser.parse_args()
    base_seed = _to_u32_seed(args.seed)

    configs = parse_configs(args.configs)
    e_values = resolve_e_values(args.e_values, args.e_interior_points)
    q_values = parse_float_list(args.q_values, "--q-values")
    p_values = parse_float_list(args.p_values, "--p-values")

    if any(e < 0.0 or e > 1.0 for e in e_values):
        raise ValueError("All e-values must be in [0, 1].")
    if any(q < 0.0 or q > 1.0 for q in q_values):
        raise ValueError("All q-values must be in [0, 1].")
    if any(p < 0.0 or p > 1.0 for p in p_values):
        raise ValueError("All p-values must be in [0, 1].")

    if args.sweep_threads <= 0:
        raise ValueError("--sweep-threads must be positive.")
    if args.decode_threads is not None and int(args.decode_threads) != 1:
        _log(
            f"warning: forcing decoder threads to 1 (received --decode-threads={args.decode_threads})."
        )

    # two_qubit_only=False means single_qubit_errors=True in builder.
    cases = [
        {"label": "two_qubit_only", "single_qubit_errors": False},
        {"label": "single_qubit_enabled", "single_qubit_errors": True},
    ]

    args.pair_json_dir.mkdir(parents=True, exist_ok=True)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)

    total_jobs = len(p_values) * len(q_values) * len(cases) * len(configs) * len(e_values)
    job_idx = 0
    t0 = time.perf_counter()

    pair_summaries: list[dict] = []
    for p_i, p_pauli in enumerate(p_values):
        for q_i, q_check in enumerate(q_values):
            pair_t0 = time.perf_counter()
            pair_rows: list[dict] = []
            pair_thresholds: list[dict] = []

            jobs = []
            for case_i, case in enumerate(cases):
                for cfg_i, (distance, rounds) in enumerate(configs):
                    for e_i, e_erasure in enumerate(e_values):
                        seed = _to_u32_seed(
                            base_seed
                            + p_i * 1_000_000_000
                            + q_i * 100_000_000
                            + case_i * 10_000_000
                            + cfg_i * 100_000
                            + e_i
                        )
                        jobs.append(
                            {
                                "case_label": case["label"],
                                "single_qubit_errors": bool(case["single_qubit_errors"]),
                                "distance": int(distance),
                                "rounds": int(rounds),
                                "e_erasure": float(e_erasure),
                                "q_check": float(q_check),
                                "p_pauli": float(p_pauli),
                                "shots": int(args.shots),
                                "seed": int(seed),
                                "sample_threads": int(args.num_threads),
                                "max_batch_bytes": int(args.max_batch_bytes),
                            }
                        )

            case_rows_map: dict[str, list[dict]] = {case["label"]: [] for case in cases}
            max_workers = min(int(args.sweep_threads), max(1, len(jobs)))
            executor_cls = ProcessPoolExecutor if args.sweep_backend == "process" else ThreadPoolExecutor

            def record_result(job: dict, row: dict) -> None:
                nonlocal job_idx
                pair_rows.append(row)
                case_rows_map[job["case_label"]].append(row)
                job_idx += 1
                _log(
                    f"[{job_idx}/{total_jobs}] "
                    f"case={job['case_label']} p={p_pauli:.6g} q={q_check:.6g} "
                    f"(d={job['distance']},r={job['rounds']}) e={job['e_erasure']:.6g} "
                    f"LER/round={row['logical_error_rate_per_round']:.6g} "
                    f"decode_failures={row['decode_failures']}"
                )

            if max_workers <= 1:
                for job in jobs:
                    completed_job, row = run_single_point_job(job)
                    record_result(completed_job, row)
            else:
                try:
                    executor_kwargs = {"max_workers": max_workers}
                    # For process backend, prefer `fork` so workers inherit already-imported
                    # pymatching/scipy state and avoid slow spawn-time imports.
                    if executor_cls is ProcessPoolExecutor:
                        _ = get_pymatching()
                        try:
                            executor_kwargs["mp_context"] = mp.get_context("fork")
                        except ValueError:
                            pass
                    with executor_cls(**executor_kwargs) as pool:
                        future_to_job = {
                            pool.submit(run_single_point_job, job): job
                            for job in jobs
                        }
                        for future in as_completed(future_to_job):
                            completed_job, row = future.result()
                            record_result(completed_job, row)
                except (PermissionError, OSError) as exc:
                    _log(
                        f"warning: failed to start parallel sweep backend '{args.sweep_backend}'"
                        f" ({type(exc).__name__}: {exc}); falling back to sequential."
                    )
                    for job in jobs:
                        completed_job, row = run_single_point_job(job)
                        record_result(completed_job, row)

            for case in cases:
                case_rows = case_rows_map[case["label"]]
                est = estimate_threshold_for_slice(case_rows)
                threshold_row = {
                    "case": case["label"],
                    "p_pauli": float(p_pauli),
                    "q_check": float(q_check),
                    **est,
                }
                pair_thresholds.append(threshold_row)
                _log(
                    f"threshold case={case['label']} p={p_pauli:.6g} q={q_check:.6g}: "
                    f"e_th={est['threshold_estimate']} "
                    f"(pairs={est['num_pairs_with_crossing']}, decode_failures={est['decode_failures']})"
                )

            pair_elapsed = time.perf_counter() - pair_t0
            pair_payload = {
                "configs": [{"distance": d, "qec_rounds": r} for d, r in configs],
                "shots_per_point": args.shots,
                "p_pauli": float(p_pauli),
                "q_check": float(q_check),
                "e_values": [float(e) for e in e_values],
                "cases": cases,
                "elapsed_seconds": pair_elapsed,
                "rows": pair_rows,
                "thresholds": pair_thresholds,
            }

            pair_path = args.pair_json_dir / (
                f"threshold_sweep_p_{float_tag(p_pauli)}_q_{float_tag(q_check)}.json"
            )
            pair_path.write_text(json.dumps(pair_payload, indent=2))

            pair_summary = {
                "p_pauli": float(p_pauli),
                "q_check": float(q_check),
                "elapsed_seconds": pair_elapsed,
                "pair_json": str(pair_path),
                "thresholds": pair_thresholds,
            }
            pair_summaries.append(pair_summary)

            elapsed = time.perf_counter() - t0
            summary_payload = {
                "configs": [{"distance": d, "qec_rounds": r} for d, r in configs],
                "shots_per_point": args.shots,
                "e_values": [float(e) for e in e_values],
                "q_values": [float(q) for q in q_values],
                "p_values": [float(p) for p in p_values],
                "cases": cases,
                "elapsed_seconds": elapsed,
                "pairs": pair_summaries,
            }
            args.json_out.write_text(json.dumps(summary_payload, indent=2))

            _log(
                f"saved pair json: {pair_path} "
                f"(p={p_pauli:.6g}, q={q_check:.6g}, elapsed={pair_elapsed:.2f}s)"
            )

    case_to_label = {
        "two_qubit_only": "Two-qubit erasure only",
        "single_qubit_enabled": "Single-qubit erasure enabled",
    }

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9.4, 5.8))
    has_curve = False
    for case in ["two_qubit_only", "single_qubit_enabled"]:
        for p_pauli in p_values:
            points = []
            for pair in pair_summaries:
                if abs(pair["p_pauli"] - float(p_pauli)) > 1e-15:
                    continue
                for threshold in pair["thresholds"]:
                    if threshold["case"] != case:
                        continue
                    if threshold["threshold_estimate"] is None:
                        continue
                    points.append((pair["q_check"], threshold["threshold_estimate"]))
            points.sort(key=lambda t: t[0])
            if not points:
                continue
            x = np.array([pt[0] for pt in points], dtype=float)
            y = np.array([pt[1] for pt in points], dtype=float)
            label = f"{case_to_label[case]}, p={p_pauli:.3g}"
            plt.plot(x, y, marker="o", linewidth=1.8, label=label)
            has_curve = True

    plt.xlabel("Check error probability q (FN=FP=q)")
    plt.ylabel("Estimated erasure threshold e_th")
    plt.title("Erasure Threshold vs Check Error (parameterized by Pauli p)")
    plt.grid(True, which="both", alpha=0.3)
    if has_curve:
        plt.legend(fontsize=8)
    plt.tight_layout()
    args.plot_out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.plot_out, dpi=220)
    plt.close()

    elapsed = time.perf_counter() - t0
    _log(f"\nSaved summary JSON: {args.json_out}")
    _log(f"Saved pair JSON directory: {args.pair_json_dir}")
    _log(f"Saved Plot: {args.plot_out}")
    _log(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
