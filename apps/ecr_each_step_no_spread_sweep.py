#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

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


def parse_modes(modes_text: str) -> list[str]:
    modes = [m.strip().upper() for m in modes_text.split(",") if m.strip()]
    allowed = {"DATA", "ANCILLA", "ALL"}
    for mode in modes:
        if mode not in allowed:
            raise ValueError(f"Invalid mode '{mode}'. Allowed: DATA, ANCILLA, ALL.")
    if not modes:
        raise ValueError("No valid modes parsed from --modes.")
    return modes


def make_p_values(
    p_values_text: str | None,
    p_min: float,
    p_max: float,
    points: int,
) -> list[float]:
    if p_values_text is not None and p_values_text.strip():
        values = [float(v.strip()) for v in p_values_text.split(",") if v.strip()]
        if not values:
            raise ValueError("--p-values provided, but no values were parsed.")
        if any(v <= 0 for v in values):
            raise ValueError("All p values must be > 0.")
        return values

    if p_min <= 0 or p_max <= 0 or p_min >= p_max:
        raise ValueError("Expected 0 < p-min < p-max for log sweep.")
    if points < 2:
        raise ValueError("--points must be >= 2.")
    return list(np.logspace(math.log10(p_min), math.log10(p_max), points))


def bernoulli_per_round(logical_error_rate: float, rounds: int) -> float:
    p = float(np.clip(logical_error_rate, 0.0, 1.0))
    return 1.0 - (1.0 - p) ** (1.0 / float(rounds))


def fit_leftmost_nonzero(points: list[tuple[float, float]], max_points: int = 7) -> tuple[float | None, int]:
    nonzero = sorted([(p, y) for p, y in points if y > 0.0], key=lambda t: t[0])[:max_points]
    if len(nonzero) < 2:
        return None, len(nonzero)
    xs = np.log10([p for p, _ in nonzero])
    ys = np.log10([y for _, y in nonzero])
    m, _ = np.polyfit(xs, ys, 1)
    return float(m), len(nonzero)


def rewrite_ecr_after_each_erase(circuit_text: str) -> str:
    lines = [ln.rstrip() for ln in circuit_text.splitlines() if ln.strip()]
    ecr_template = None
    for ln in lines:
        s = ln.strip()
        if s.startswith("ECR("):
            ecr_template = s
            break
    if ecr_template is None:
        raise RuntimeError("No ECR line found in circuit; cannot rewrite.")

    out: list[str] = []
    for ln in lines:
        s = ln.strip()
        if s.startswith("ECR("):
            continue
        out.append(s)
        if s.startswith("ERASE2(") or s.startswith("ERASE2_ANY("):
            out.append(ecr_template)
    return "\n".join(out) + "\n"


def run_single_point(
    distance: int,
    rounds: int,
    mode: str,
    p_tqe: float,
    shots: int,
    seed: int,
    num_threads: int,
    max_batch_bytes: int,
) -> dict:
    base_circuit = qe.SurfaceCodeRotated(distance).build_circuit(
        rounds=rounds,
        erasure_prob=p_tqe,
        erasable_qubits=mode,
        reset_failure_prob=0.0,
    )
    rewritten_text = rewrite_ecr_after_each_erase(str(base_circuit))
    circuit = qe.ErasureCircuit(rewritten_text)

    model = qe.ErasureModel(
        2,
        qe.PauliChannel(0.25, 0.25, 0.25),
        qe.PauliChannel(0.25, 0.25, 0.25),
        qe.TQGSpreadModel(
            qe.PauliChannel(0.0, 0.0, 0.0),
            qe.PauliChannel(0.0, 0.0, 0.0),
        ),
    )
    model.check_false_negative_prob = 0.0
    model.check_false_positive_prob = 0.0

    compiled = qe.CompiledErasureProgram(circuit, model)
    sampler = qe.StreamSampler(compiled)
    dem_builder = qe.SurfDemBuilder(compiled)
    decoder = qe.SurfaceCodeBatchDecoder(
        compiled,
        dem_builder=dem_builder,
        max_batch_bytes=max_batch_bytes,
    )

    t0 = time.perf_counter()
    dets, obs, checks = sampler.sample(num_shots=shots, seed=seed, num_threads=num_threads)
    preds = decoder.decode_batch(dets, checks)
    t1 = time.perf_counter()

    truths = obs if obs.ndim == 2 else obs[:, None]
    n_obs = min(truths.shape[1], preds.shape[1])
    mismatches = np.any(preds[:, :n_obs] != truths[:, :n_obs], axis=1)
    ler = float(np.mean(mismatches)) if len(mismatches) else 0.0

    return {
        "distance": distance,
        "qec_rounds": rounds,
        "erasable_qubits": mode,
        "shots": shots,
        "seed": seed,
        "p_two_qubit_erasure": p_tqe,
        "logical_error_rate": ler,
        "logical_error_rate_per_round": bernoulli_per_round(ler, rounds),
        "timing_seconds": {
            "total": t1 - t0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep p for circuits rewritten to place ECR immediately after each ERASE2/ERASE2_ANY. "
            "Uses zero spread and perfect checks."
        )
    )
    parser.add_argument("--configs", type=str, default="3,3;5,5;7,7")
    parser.add_argument("--modes", type=str, default="DATA,ANCILLA,ALL")
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=random.randint(0, 2**32 - 1))
    parser.add_argument("--points", type=int, default=10)
    parser.add_argument("--p-min", type=float, default=1e-2)
    parser.add_argument("--p-max", type=float, default=1e-1)
    parser.add_argument("--p-values", type=str, default=None)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--max-batch-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--gradient-points", type=int, default=7)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=REPO_ROOT / "apps" / "results" / "ecr_each_step_no_spread_no_check_sweep.json",
    )
    args = parser.parse_args()

    configs = parse_configs(args.configs)
    modes = parse_modes(args.modes)
    p_values = make_p_values(args.p_values, args.p_min, args.p_max, args.points)

    rows: list[dict] = []
    t0 = time.perf_counter()
    total = len(configs) * len(modes) * len(p_values)
    count = 0
    for cfg_idx, (distance, rounds) in enumerate(configs):
        for mode_idx, mode in enumerate(modes):
            for p_idx, p_tqe in enumerate(p_values):
                seed = args.seed + cfg_idx * 1_000_000 + mode_idx * 100_000 + p_idx
                row = run_single_point(
                    distance=distance,
                    rounds=rounds,
                    mode=mode,
                    p_tqe=float(p_tqe),
                    shots=args.shots,
                    seed=seed,
                    num_threads=args.num_threads,
                    max_batch_bytes=args.max_batch_bytes,
                )
                rows.append(row)
                count += 1
                print(
                    f"[{count}/{total}] (d={distance},r={rounds},mode={mode}) "
                    f"p={p_tqe:.6g} "
                    f"LER={row['logical_error_rate']:.6g} "
                    f"LER/round={row['logical_error_rate_per_round']:.6g}"
                )
    elapsed = time.perf_counter() - t0

    gradients: dict[str, dict[str, float | int | None]] = {}
    for distance, rounds in configs:
        for mode in modes:
            key = f"d{distance}_r{rounds}_{mode}"
            points = [
                (r["p_two_qubit_erasure"], r["logical_error_rate_per_round"])
                for r in rows
                if r["distance"] == distance
                and r["qec_rounds"] == rounds
                and r["erasable_qubits"] == mode
            ]
            slope, n_used = fit_leftmost_nonzero(points, max_points=args.gradient_points)
            gradients[key] = {
                "slope": slope,
                "points_used": n_used,
            }

    payload = {
        "configs": [{"distance": d, "qec_rounds": r} for d, r in configs],
        "modes": modes,
        "shots_per_point": args.shots,
        "p_values": [float(p) for p in p_values],
        "gradient_points": args.gradient_points,
        "elapsed_seconds": elapsed,
        "rows": rows,
        "gradients": gradients,
    }

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2))

    print("\nGradients (LER/round vs p, log-log):")
    for key in sorted(gradients.keys()):
        slope = gradients[key]["slope"]
        used = gradients[key]["points_used"]
        print(f"  {key}: slope={slope}, points_used={used}")
    print(f"\nSaved JSON: {args.json_out}")
    print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
