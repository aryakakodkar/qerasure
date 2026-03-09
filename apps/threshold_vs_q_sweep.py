#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


def parse_float_list(text: str, name: str) -> list[float]:
	values = [float(v.strip()) for v in text.split(",") if v.strip()]
	if not values:
		raise ValueError(f"No values parsed for {name}.")
	return values


def make_p_values(
	p_values_text: str | None,
	p_min: float,
	p_max: float,
	points: int,
) -> list[float]:
	if p_values_text is not None and p_values_text.strip():
		values = parse_float_list(p_values_text, "--p-values")
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


def run_single_point(
	distance: int,
	rounds: int,
	shots: int,
	p_tqe: float,
	q_check: float,
	seed: int,
	sample_threads: int,
	decode_threads: int,
	max_batch_bytes: int,
	single_qubit_errors: bool,
) -> dict:
	circuit = qe.SurfaceCodeRotated(distance).build_circuit(
		rounds=rounds,
		erasure_prob=p_tqe,
		erasable_qubits="ALL",
		reset_failure_prob=0.0,
		single_qubit_errors=single_qubit_errors,
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
	model.check_false_negative_prob = q_check
	model.check_false_positive_prob = q_check

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
	predictions = grouped_decoder.decode_batch(dets, checks, num_threads=decode_threads)
	t2 = time.perf_counter()

	truths = obs if obs.ndim == 2 else obs[:, None]
	n_obs = min(truths.shape[1], predictions.shape[1])
	mismatches = np.any(predictions[:, :n_obs] != truths[:, :n_obs], axis=1)
	ler = float(np.mean(mismatches)) if len(mismatches) else 0.0

	return {
		"distance": distance,
		"qec_rounds": rounds,
		"shots": shots,
		"seed": seed,
		"single_qubit_errors": single_qubit_errors,
		"q_check": q_check,
		"p_two_qubit_erasure": p_tqe,
		"logical_error_rate": ler,
		"logical_error_rate_per_round": bernoulli_per_round(ler, rounds),
		"timing_seconds": {
			"sample": t1 - t0,
			"decode": t2 - t1,
			"total": t2 - t0,
		},
	}


def crossings_linear(
	curve_small: list[tuple[float, float]],
	curve_large: list[tuple[float, float]],
) -> list[float]:
	if len(curve_small) != len(curve_large):
		raise ValueError("Distance curves have different point counts.")

	crossings: list[float] = []
	for i in range(len(curve_small) - 1):
		p0, y0s = curve_small[i]
		p1, y1s = curve_small[i + 1]
		q0, y0l = curve_large[i]
		q1, y1l = curve_large[i + 1]
		if abs(p0 - q0) > 1e-15 or abs(p1 - q1) > 1e-15:
			raise ValueError("Distance curves use different p grids.")

		d0 = y0s - y0l
		d1 = y1s - y1l

		if d0 == 0.0:
			crossings.append(p0)
			continue
		if d0 * d1 < 0.0:
			frac = -d0 / (d1 - d0)
			crossings.append(p0 + frac * (p1 - p0))
		elif d1 == 0.0:
			crossings.append(p1)

	unique: list[float] = []
	for x in crossings:
		if not unique or abs(x - unique[-1]) > 1e-12:
			unique.append(x)
	return unique


def estimate_threshold_for_slice(
	rows: list[dict],
) -> dict:
	curves: dict[int, list[tuple[float, float]]] = {}
	for row in rows:
		d = int(row["distance"])
		p = float(row["p_two_qubit_erasure"])
		y = float(row["logical_error_rate_per_round"])
		curves.setdefault(d, []).append((p, y))

	distances = sorted(curves.keys())
	for d in distances:
		curves[d].sort(key=lambda t: t[0])

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
	}


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Run p-sweeps for multiple check-error q values and estimate threshold vs q "
			"for both two-qubit-only and single-qubit-erasure-enabled surface-code builders."
		)
	)
	parser.add_argument("--configs", type=str, default="3,3;5,5;7,7")
	parser.add_argument("--shots", type=int, default=10_000)
	parser.add_argument("--seed", type=int, default=random.randint(0, 2**32 - 1))
	parser.add_argument("--points", type=int, default=10)
	parser.add_argument("--p-min", type=float, default=1e-2)
	parser.add_argument("--p-max", type=float, default=1e-1)
	parser.add_argument("--p-values", type=str, default=None)
	parser.add_argument(
		"--q-values",
		type=str,
		required=True,
		help="Comma-separated check-error rates q (used for both FN and FP).",
	)
	parser.add_argument("--num-threads", type=int, default=1)
	parser.add_argument("--decode-threads", type=int, default=None)
	parser.add_argument("--max-batch-bytes", type=int, default=256 * 1024 * 1024)
	parser.add_argument(
		"--json-out",
		type=Path,
		default=REPO_ROOT / "apps" / "results" / "threshold_vs_q_sweep.json",
	)
	parser.add_argument(
		"--plot-out",
		type=Path,
		default=REPO_ROOT / "apps" / "results" / "threshold_vs_q_sweep.png",
	)
	args = parser.parse_args()

	configs = parse_configs(args.configs)
	p_values = make_p_values(args.p_values, args.p_min, args.p_max, args.points)
	q_values = parse_float_list(args.q_values, "--q-values")
	if any(q < 0.0 or q > 1.0 for q in q_values):
		raise ValueError("All q-values must be in [0, 1].")
	decode_threads = args.decode_threads if args.decode_threads is not None else args.num_threads

	rows: list[dict] = []
	thresholds: list[dict] = []

	# two_qubit_only=False means single_qubit_errors=True in builder.
	cases = [
		{"label": "two_qubit_only", "single_qubit_errors": False},
		{"label": "single_qubit_enabled", "single_qubit_errors": True},
	]

	total_jobs = len(cases) * len(q_values) * len(configs) * len(p_values)
	job_idx = 0
	t0 = time.perf_counter()
	for case_i, case in enumerate(cases):
		for q_i, q in enumerate(q_values):
			for cfg_i, (distance, rounds) in enumerate(configs):
				for p_i, p in enumerate(p_values):
					seed = (
						args.seed
						+ case_i * 100_000_000
						+ q_i * 1_000_000
						+ cfg_i * 10_000
						+ p_i
					)
					row = run_single_point(
						distance=distance,
						rounds=rounds,
						shots=args.shots,
						p_tqe=float(p),
						q_check=float(q),
						seed=seed,
						sample_threads=args.num_threads,
						decode_threads=decode_threads,
						max_batch_bytes=args.max_batch_bytes,
						single_qubit_errors=bool(case["single_qubit_errors"]),
					)
					row["case"] = case["label"]
					rows.append(row)
					job_idx += 1
					print(
						f"[{job_idx}/{total_jobs}] "
						f"case={case['label']} q={q:.6g} (d={distance},r={rounds}) p={p:.6g} "
						f"LER/round={row['logical_error_rate_per_round']:.6g}"
					)

			slice_rows = [
				r
				for r in rows
				if r["case"] == case["label"] and abs(r["q_check"] - float(q)) < 1e-15
			]
			est = estimate_threshold_for_slice(slice_rows)
			thresholds.append(
				{
					"case": case["label"],
					"q_check": float(q),
					**est,
				}
			)
			print(
				f"threshold case={case['label']} q={q:.6g}: "
				f"{est['threshold_estimate']} (pairs={est['num_pairs_with_crossing']})"
			)

	elapsed = time.perf_counter() - t0

	payload = {
		"configs": [{"distance": d, "qec_rounds": r} for d, r in configs],
		"shots_per_point": args.shots,
		"p_values": [float(p) for p in p_values],
		"q_values": [float(q) for q in q_values],
		"cases": cases,
		"elapsed_seconds": elapsed,
		"rows": rows,
		"thresholds": thresholds,
	}
	args.json_out.parent.mkdir(parents=True, exist_ok=True)
	args.json_out.write_text(json.dumps(payload, indent=2))

	case_to_label = {
		"two_qubit_only": "Two-qubit erasure only",
		"single_qubit_enabled": "Single-qubit erasure enabled",
	}
	plt.figure(figsize=(8.6, 5.4))
	for case in ["two_qubit_only", "single_qubit_enabled"]:
		pts = [
			(t["q_check"], t["threshold_estimate"])
			for t in thresholds
			if t["case"] == case and t["threshold_estimate"] is not None
		]
		pts.sort(key=lambda t: t[0])
		if not pts:
			continue
		x = np.array([p[0] for p in pts], dtype=float)
		y = np.array([p[1] for p in pts], dtype=float)
		plt.plot(x, y, marker="o", linewidth=1.8, label=case_to_label[case])

	plt.xlabel("Check error probability q (FN=FP=q)")
	plt.ylabel("Estimated threshold p_th")
	plt.title("Threshold vs Check Error Probability")
	plt.grid(True, which="both", alpha=0.3)
	plt.legend()
	plt.tight_layout()
	args.plot_out.parent.mkdir(parents=True, exist_ok=True)
	plt.savefig(args.plot_out, dpi=220)
	plt.close()

	print(f"\nSaved JSON: {args.json_out}")
	print(f"Saved Plot: {args.plot_out}")
	print(f"Elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
	main()
