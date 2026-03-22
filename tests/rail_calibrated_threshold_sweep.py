#!/usr/bin/env python3
"""Compare threshold curves for calibrated rail-resolved decoding vs normal decoding."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_SRC = REPO_ROOT / "python"
if str(PYTHON_SRC) not in sys.path:
	sys.path.insert(0, str(PYTHON_SRC))

import qerasure as qe
import pymatching as pm


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


def parse_float_list(values_text: str) -> list[float]:
	values = [float(v.strip()) for v in values_text.split(",") if v.strip()]
	if not values:
		raise ValueError("No values parsed.")
	return values


def bernoulli_per_round(logical_error_rate: float, rounds: int) -> float:
	p = float(np.clip(logical_error_rate, 0.0, 1.0))
	return 1.0 - (1.0 - p) ** (1.0 / float(rounds))


def schedule_bucket(schedule_type: int) -> int:
	if schedule_type == 1:
		return 0
	if schedule_type == 2:
		return 1
	return -1


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
	d0 = rail_program.round_detector_index(int(round_index), int(slot0))
	d1 = rail_program.round_detector_index(int(round_index), int(slot1))
	if d0 < 0 or d1 < 0:
		return False
	return int(det_row[d0]) != int(det_row[d1])


def condition_bucket(
	rail_program: qe.RailSurfaceCompiledProgram,
	det_row: np.ndarray,
	data_qubit: int,
	check_round: int,
) -> int:
	inc_round1 = pair_inconsistency_in_round(rail_program, det_row, data_qubit, int(check_round) - 1)
	inc_round2 = pair_inconsistency_in_round(rail_program, det_row, data_qubit, int(check_round))
	if inc_round1 and inc_round2:
		return 3
	if inc_round1:
		return 1
	if inc_round2:
		return 2
	return 0


def map_onset_op_to_bin(
	rail_program: qe.RailSurfaceCompiledProgram,
	check_round: int,
	event_rows: list[dict],
	true_onset_op: int,
) -> int:
	prev_round = int(check_round) - 1
	curr_round = int(check_round)
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


def build_circuit(
	distance: int,
	rounds: int,
	erasure_prob: float,
	rounds_per_check: int,
	single_qubit_errors: bool,
	pauli_prob: float,
):
	return qe.SurfaceCodeRotated(distance).build_circuit(
		rounds=rounds,
		erasure_prob=float(erasure_prob),
		erasable_qubits="ALL",
		reset_failure_prob=0.0,
		single_qubit_errors=bool(single_qubit_errors),
		post_clifford_pauli_prob=float(pauli_prob),
		rounds_per_check=int(rounds_per_check),
	)


def build_model(check_prob: float):
	model = qe.ErasureModel(
		2,
		qe.PauliChannel(0.25, 0.25, 0.25),
		qe.PauliChannel(0.25, 0.25, 0.25),
		qe.TQGSpreadModel(
			qe.PauliChannel(0.0, 0.0, 0.0),
			qe.PauliChannel(0.0, 0.0, 0.5),
		),
	)
	model.check_false_negative_prob = float(check_prob)
	model.check_false_positive_prob = float(check_prob)
	return model


def calibrate_onset_posteriors(
	distance: int,
	rounds: int,
	shots: int,
	seed: int,
	erasure_prob: float,
	check_prob: float,
	rounds_per_check: int,
	single_qubit_errors: bool,
	pauli_prob: float,
) -> tuple[list[list[list[float]]], dict]:
	circuit = build_circuit(
		distance=distance,
		rounds=rounds,
		erasure_prob=erasure_prob,
		rounds_per_check=rounds_per_check,
		single_qubit_errors=single_qubit_errors,
		pauli_prob=pauli_prob,
	)
	model = build_model(check_prob=check_prob)
	rail_program = qe.RailSurfaceCompiledProgram(
		circuit=circuit,
		model=model,
		distance=distance,
		rounds=rounds,
	)
	sampler = qe.RailCalibrationSampler(rail_program)
	dem_builder = qe.RailSurfaceDemBuilder(rail_program)
	dets, _obs, checks, onset_ops = sampler.sample(
		num_shots=int(shots),
		seed=int(seed),
		num_threads=1,
	)

	counts = np.zeros((2, 4, 8), dtype=np.int64)
	events_per_bucket = np.zeros((2, 4), dtype=np.int64)
	for shot in range(int(dets.shape[0])):
		check_row = checks[shot].tolist()
		det_row = dets[shot]
		onset_row = onset_ops[shot]
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
			if not rail_program.data_qubit_is_full_interior(int(data_qubit)):
				continue
			if int(check_round) <= 0:
				continue
			s_bucket = schedule_bucket(int(schedule_type))
			if s_bucket < 0:
				continue
			c_bucket = condition_bucket(
				rail_program=rail_program,
				det_row=det_row,
				data_qubit=int(data_qubit),
				check_round=int(check_round),
			)
			true_onset_op = int(onset_row[int(check_event_index)])
			bin_index = map_onset_op_to_bin(
				rail_program=rail_program,
				check_round=int(check_round),
				event_rows=event_rows,
				true_onset_op=true_onset_op,
			)
			if bin_index < 0:
				continue
			counts[s_bucket, c_bucket, bin_index] += 1
			events_per_bucket[s_bucket, c_bucket] += 1

	posteriors = np.zeros((2, 4, 8), dtype=float)
	for s_bucket in range(2):
		for c_bucket in range(4):
			total = int(events_per_bucket[s_bucket, c_bucket])
			if total <= 0:
				# Fallback to uniform row if calibration bucket is empty.
				posteriors[s_bucket, c_bucket, :] = 1.0 / 8.0
			else:
				posteriors[s_bucket, c_bucket, :] = counts[s_bucket, c_bucket, :] / float(total)

	summary = {
		"counts": counts.tolist(),
		"events_per_bucket": events_per_bucket.tolist(),
		"posteriors": posteriors.tolist(),
	}
	return posteriors.tolist(), summary


def decode_with_rail(
	distance: int,
	rounds: int,
	shots: int,
	seed: int,
	erasure_prob: float,
	check_prob: float,
	pauli_prob: float,
	rounds_per_check: int,
	single_qubit_errors: bool,
	calibration_posteriors: list[list[list[float]]],
	calibration_erasure_prob: float,
) -> dict:
	circuit = build_circuit(
		distance=distance,
		rounds=rounds,
		erasure_prob=erasure_prob,
		rounds_per_check=rounds_per_check,
		single_qubit_errors=single_qubit_errors,
		pauli_prob=pauli_prob,
	)
	model = build_model(check_prob=check_prob)
	rail_program = qe.RailSurfaceCompiledProgram(
		circuit=circuit,
		model=model,
		distance=distance,
		rounds=rounds,
	)
	sampler = qe.RailStreamSampler(rail_program)
	dem_builder = qe.RailSurfaceDemBuilder(rail_program)
	dem_builder.set_calibrated_onset_posteriors(
		erasure_probability=float(calibration_erasure_prob),
		posteriors=calibration_posteriors,
		boost_nonzero_with_pe2=True,
	)

	t0 = time.perf_counter()
	dets, obs, checks = sampler.sample(num_shots=int(shots), seed=int(seed), num_threads=1)
	t1 = time.perf_counter()

	pred_rows = []
	decode_failures = 0
	for shot in range(int(dets.shape[0])):
		try:
			decoded_circuit = dem_builder.build_decoded_circuit(
				checks[shot].tolist(),
				dets[shot].tolist(),
				verbose=False,
			)
			decoded_dem = decoded_circuit.detector_error_model(
				decompose_errors=True,
				approximate_disjoint_errors=True,
			)
			matching = pm.Matching.from_detector_error_model(decoded_dem)
			pred = np.asarray(matching.decode(dets[shot]), dtype=np.uint8)
			if pred.ndim == 0:
				pred = pred.reshape(1)
			pred_rows.append(pred)
		except Exception:
			decode_failures += 1
			pred_rows.append(np.zeros((1,), dtype=np.uint8))
	t2 = time.perf_counter()

	width = max((int(row.shape[0]) for row in pred_rows), default=1)
	predictions = np.zeros((len(pred_rows), width), dtype=np.uint8)
	for i, row in enumerate(pred_rows):
		n = min(width, int(row.shape[0]))
		predictions[i, :n] = row[:n]

	truths = obs if obs.ndim == 2 else obs[:, None]
	n_obs = min(int(truths.shape[1]), int(predictions.shape[1]))
	mismatches = np.any(predictions[:, :n_obs] != truths[:, :n_obs], axis=1)
	ler = float(np.mean(mismatches)) if len(mismatches) else 0.0
	return {
		"logical_error_rate": ler,
		"logical_error_rate_per_round": bernoulli_per_round(ler, rounds),
		"decode_failures": int(decode_failures),
		"timing_seconds": {
			"sample": float(t1 - t0),
			"decode": float(t2 - t1),
			"total": float(t2 - t0),
		},
	}


def decode_with_normal(
	distance: int,
	rounds: int,
	shots: int,
	seed: int,
	erasure_prob: float,
	check_prob: float,
	pauli_prob: float,
	rounds_per_check: int,
	single_qubit_errors: bool,
) -> dict:
	circuit = build_circuit(
		distance=distance,
		rounds=rounds,
		erasure_prob=erasure_prob,
		rounds_per_check=rounds_per_check,
		single_qubit_errors=single_qubit_errors,
		pauli_prob=pauli_prob,
	)
	model = build_model(check_prob=check_prob)
	compiled = qe.CompiledErasureProgram(circuit, model)
	sampler = qe.StreamSampler(compiled)
	dem_builder = qe.SurfDemBuilder(compiled)
	decoder = qe.SurfaceCodeBatchDecoder(compiled, dem_builder=dem_builder)

	t0 = time.perf_counter()
	dets, obs, checks = sampler.sample(num_shots=int(shots), seed=int(seed), num_threads=1)
	t1 = time.perf_counter()
	predictions = np.asarray(decoder.decode_batch(dets, checks, num_threads=1), dtype=np.uint8)
	if predictions.ndim == 1:
		predictions = predictions[:, None]
	t2 = time.perf_counter()

	truths = obs if obs.ndim == 2 else obs[:, None]
	n_obs = min(int(truths.shape[1]), int(predictions.shape[1]))
	mismatches = np.any(predictions[:, :n_obs] != truths[:, :n_obs], axis=1)
	ler = float(np.mean(mismatches)) if len(mismatches) else 0.0
	return {
		"logical_error_rate": ler,
		"logical_error_rate_per_round": bernoulli_per_round(ler, rounds),
		"decode_failures": 0,
		"timing_seconds": {
			"sample": float(t1 - t0),
			"decode": float(t2 - t1),
			"total": float(t2 - t0),
		},
	}


def estimate_threshold(rows: list[dict], scheme: str) -> dict:
	scheme_rows = [row for row in rows if row["scheme"] == scheme]
	if not scheme_rows:
		return {
			"scheme": scheme,
			"threshold_estimate": None,
			"crossings": [],
		}
	curves: dict[int, list[tuple[float, float]]] = {}
	for row in scheme_rows:
		distance = int(row["distance"])
		curves.setdefault(distance, []).append(
			(float(row["erasure_prob"]), float(row["logical_error_rate_per_round"]))
		)
	for distance in curves:
		curves[distance].sort(key=lambda x: x[0])
	distances = sorted(curves.keys())
	crossings: list[dict] = []
	crossing_values: list[float] = []
	for i in range(len(distances) - 1):
		d_small = distances[i]
		d_large = distances[i + 1]
		small = curves[d_small]
		large = curves[d_large]
		if len(small) != len(large):
			continue
		selected = None
		for j in range(len(small) - 1):
			p0, y0s = small[j]
			p1, y1s = small[j + 1]
			_, y0l = large[j]
			_, y1l = large[j + 1]
			d0 = y0s - y0l
			d1 = y1s - y1l
			if d0 == 0.0:
				selected = p0
				break
			if d0 * d1 < 0.0:
				frac = -d0 / (d1 - d0)
				selected = p0 + frac * (p1 - p0)
				break
		crossings.append({"pair": [d_small, d_large], "selected": selected})
		if selected is not None and math.isfinite(selected):
			crossing_values.append(float(selected))
	threshold = float(np.mean(crossing_values)) if crossing_values else None
	return {
		"scheme": scheme,
		"threshold_estimate": threshold,
		"crossings": crossings,
	}


def main() -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Calibrate rail onset posteriors from latent data, apply them in "
			"RailSurfaceDemBuilder, and compare threshold sweeps to normal decoding."
		)
	)
	parser.add_argument("--configs", type=str, default="3,3;5,5;7,7")
	parser.add_argument("--erasure-values", type=str, default="0.0006,0.0008,0.001,0.0012,0.0014")
	parser.add_argument("--shots", type=int, default=1000)
	parser.add_argument("--seed", type=int, default=random.randint(0, 2**32 - 1))
	parser.add_argument("--check-prob", type=float, default=0.005)
	parser.add_argument("--pauli-prob", type=float, default=0.0)
	parser.add_argument("--rounds-per-check", type=int, default=2)
	parser.add_argument(
		"--single-qubit-errors",
		action=argparse.BooleanOptionalAction,
		default=True,
	)
	parser.add_argument("--calibration-shots", type=int, default=20000)
	parser.add_argument("--json-out", type=Path, default=REPO_ROOT / "tests" / "artifacts" / "rail_calibrated_threshold_sweep.json")
	args = parser.parse_args()

	configs = parse_configs(args.configs)
	erasure_values = parse_float_list(args.erasure_values)
	if args.shots <= 0:
		raise ValueError("--shots must be positive")
	if args.calibration_shots <= 0:
		raise ValueError("--calibration-shots must be positive")

	calibration_cache: dict[tuple[int, int, float, float, float, int, bool], tuple[list[list[list[float]]], dict]] = {}

	rows: list[dict] = []
	calibration_summaries: dict[str, dict] = {}
	for cfg_index, (distance, rounds) in enumerate(configs):
		for e_index, erasure_prob in enumerate(erasure_values):
			base_seed = int(args.seed) + cfg_index * 10_000 + e_index
			calib_key = (
				int(distance),
				int(rounds),
				float(erasure_prob),
				float(args.check_prob),
				float(args.pauli_prob),
				int(args.rounds_per_check),
				bool(args.single_qubit_errors),
			)
			if calib_key not in calibration_cache:
				calib_seed = (int(args.seed) ^ 0xA5A5A5A5) + cfg_index * 10_000 + e_index
				print(
					f"calibrating d={distance} r={rounds} e={erasure_prob:.6g} "
					f"q={args.check_prob:.6g} p={args.pauli_prob:.6g} shots={args.calibration_shots}",
					flush=True,
				)
				calibration_cache[calib_key] = calibrate_onset_posteriors(
					distance=int(distance),
					rounds=int(rounds),
					shots=int(args.calibration_shots),
					seed=int(calib_seed),
					erasure_prob=float(erasure_prob),
					check_prob=float(args.check_prob),
					rounds_per_check=int(args.rounds_per_check),
					single_qubit_errors=bool(args.single_qubit_errors),
					pauli_prob=float(args.pauli_prob),
				)
				calibration_summaries[
					f"d={distance},r={rounds},e={float(erasure_prob):.12g},p={float(args.pauli_prob):.12g},q={float(args.check_prob):.12g}"
				] = calibration_cache[calib_key][1]
			calibration_table, _calibration_summary = calibration_cache[calib_key]
			rail_row = decode_with_rail(
				distance=distance,
				rounds=rounds,
				shots=int(args.shots),
				seed=base_seed,
				erasure_prob=float(erasure_prob),
				check_prob=float(args.check_prob),
				pauli_prob=float(args.pauli_prob),
				rounds_per_check=int(args.rounds_per_check),
				single_qubit_errors=bool(args.single_qubit_errors),
				calibration_posteriors=calibration_table,
				calibration_erasure_prob=float(erasure_prob),
			)
			rail_row.update(
				{
					"scheme": "rail_calibrated",
					"distance": int(distance),
					"rounds": int(rounds),
					"erasure_prob": float(erasure_prob),
					"check_prob": float(args.check_prob),
					"pauli_prob": float(args.pauli_prob),
					"shots": int(args.shots),
					"seed": int(base_seed),
				}
			)
			rows.append(rail_row)
			print(
				f"rail_calibrated d={distance} r={rounds} e={erasure_prob:.6g} "
				f"LER/round={rail_row['logical_error_rate_per_round']:.6g} "
				f"decode_failures={rail_row['decode_failures']}",
				flush=True,
			)

			normal_row = decode_with_normal(
				distance=distance,
				rounds=rounds,
				shots=int(args.shots),
				seed=base_seed,
				erasure_prob=float(erasure_prob),
				check_prob=float(args.check_prob),
				pauli_prob=float(args.pauli_prob),
				rounds_per_check=int(args.rounds_per_check),
				single_qubit_errors=bool(args.single_qubit_errors),
			)
			normal_row.update(
				{
					"scheme": "normal",
					"distance": int(distance),
					"rounds": int(rounds),
					"erasure_prob": float(erasure_prob),
					"check_prob": float(args.check_prob),
					"pauli_prob": float(args.pauli_prob),
					"shots": int(args.shots),
					"seed": int(base_seed),
				}
			)
			rows.append(normal_row)
			print(
				f"normal d={distance} r={rounds} e={erasure_prob:.6g} "
				f"LER/round={normal_row['logical_error_rate_per_round']:.6g}",
				flush=True,
			)

	threshold_rail = estimate_threshold(rows, "rail_calibrated")
	threshold_normal = estimate_threshold(rows, "normal")
	print(
		f"threshold_estimate rail_calibrated={threshold_rail['threshold_estimate']} "
		f"normal={threshold_normal['threshold_estimate']}",
		flush=True,
	)

	payload = {
		"configs": [{"distance": d, "rounds": r} for d, r in configs],
		"erasure_values": [float(v) for v in erasure_values],
		"shots_per_point": int(args.shots),
		"check_prob": float(args.check_prob),
		"pauli_prob": float(args.pauli_prob),
		"rounds_per_check": int(args.rounds_per_check),
		"single_qubit_errors": bool(args.single_qubit_errors),
		"calibration": {
			"shots": int(args.calibration_shots),
			"strategy": "independent calibration per (distance, rounds, erasure_prob, pauli_prob, check_prob)",
			"summaries": calibration_summaries,
		},
		"thresholds": {
			"rail_calibrated": threshold_rail,
			"normal": threshold_normal,
		},
		"rows": rows,
	}
	args.json_out.parent.mkdir(parents=True, exist_ok=True)
	args.json_out.write_text(json.dumps(payload, indent=2))
	print(f"saved: {args.json_out}", flush=True)


if __name__ == "__main__":
	main()
