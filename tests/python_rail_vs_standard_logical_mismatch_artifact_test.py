#!/usr/bin/env python3
"""Find a shot where rail decode fails logically but standard decode succeeds.

Artifacts saved on first mismatch:
- erasure_circuit.qer
- logical_circuit.stim
- rail_virtual_circuit.stim
- rail_virtual_dem.dem
- standard_virtual_circuit.stim
- standard_virtual_dem.dem
- metadata.json
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pymatching as pm


def _import_qerasure():
	cwd = Path.cwd().resolve()
	repo_root = None
	for candidate in [cwd, *cwd.parents]:
		if (candidate / "python" / "qerasure").exists():
			repo_root = candidate
			break
	if repo_root is None:
		raise RuntimeError("Could not find repository root containing python/qerasure")
	sys.path.insert(0, str(repo_root / "python"))
	import qerasure as qe  # pylint: disable=import-error

	return repo_root, qe


def _schedule_bucket(schedule_type: int) -> int:
	if schedule_type == 1:
		return 0
	if schedule_type == 2:
		return 1
	return -1


def _pair_inconsistency_in_round(rail_program, det_row: np.ndarray, data_qubit: int, round_index: int) -> bool:
	if round_index < 0:
		return False
	slot0, slot1 = rail_program.data_z_ancilla_slots(data_qubit)
	if slot0 < 0 or slot1 < 0:
		return False
	d0 = rail_program.round_detector_index(round_index, slot0)
	d1 = rail_program.round_detector_index(round_index, slot1)
	if d0 < 0 or d1 < 0:
		return False
	return int(det_row[d0]) != int(det_row[d1])


def _condition_bucket(rail_program, det_row: np.ndarray, data_qubit: int, check_round: int) -> int:
	inc_round1 = _pair_inconsistency_in_round(rail_program, det_row, data_qubit, check_round - 1)
	inc_round2 = _pair_inconsistency_in_round(rail_program, det_row, data_qubit, check_round)
	if inc_round1 and inc_round2:
		return 3
	if inc_round1:
		return 1
	if inc_round2:
		return 2
	return 0


def _map_onset_op_to_bin(rail_program, check_round: int, event_rows: list[dict], true_onset_op: int) -> int:
	prev_round = check_round - 1
	curr_round = check_round
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


def _build_circuit(
	qe,
	*,
	distance: int,
	rounds: int,
	erasure_prob: float,
	pauli_prob: float,
	rounds_per_check: int,
):
	return qe.SurfaceCodeRotated(distance).build_circuit(
		rounds=rounds,
		erasure_prob=float(erasure_prob),
		erasable_qubits="ALL",
		reset_failure_prob=0.0,
		single_qubit_errors=True,
		post_clifford_pauli_prob=float(pauli_prob),
		rounds_per_check=int(rounds_per_check),
	)


def _build_model(qe, *, max_persistence: int, check_prob: float):
	model = qe.ErasureModel(
		int(max_persistence),
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


def _calibrate_posteriors(
	qe,
	*,
	rail_program,
	shots: int,
	seed: int,
	final_round_only: bool,
) -> tuple[list[list[list[float]]], dict]:
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
			is_final_round_check = int(check_round) == int(rail_program.rounds) - 1
			if final_round_only and not is_final_round_check:
				continue
			if not final_round_only and is_final_round_check:
				continue
			s_bucket = _schedule_bucket(int(schedule_type))
			if s_bucket < 0:
				continue
			c_bucket = _condition_bucket(
				rail_program=rail_program,
				det_row=det_row,
				data_qubit=int(data_qubit),
				check_round=int(check_round),
			)
			true_onset_op = int(onset_row[int(check_event_index)])
			bin_index = _map_onset_op_to_bin(
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
				posteriors[s_bucket, c_bucket, :] = 1.0 / 8.0
			else:
				posteriors[s_bucket, c_bucket, :] = counts[s_bucket, c_bucket, :] / float(total)

	summary = {
		"scope": "final_round_only" if final_round_only else "non_final_rounds",
		"counts": counts.tolist(),
		"events_per_bucket": events_per_bucket.tolist(),
	}
	return posteriors.tolist(), summary


def _decode_standard_one_shot(dem_builder, det_row: np.ndarray, check_row: np.ndarray) -> dict:
	try:
		virtual_circuit = dem_builder.build_decoded_circuit(
			np.asarray(check_row, dtype=np.uint8).tolist(),
			verbose=False,
		)
		virtual_circuit_text = str(virtual_circuit)
		virtual_dem = virtual_circuit.detector_error_model(
			decompose_errors=True,
			approximate_disjoint_errors=True,
		)
		virtual_dem_text = str(virtual_dem)
		matching = pm.Matching.from_detector_error_model(virtual_dem)
		pred = np.asarray(matching.decode(np.asarray(det_row, dtype=np.uint8)), dtype=np.uint8)
		if pred.ndim == 0:
			pred = pred.reshape(1)
		return {
			"ok": True,
			"pred": pred,
			"virtual_circuit_text": virtual_circuit_text,
			"virtual_dem_text": virtual_dem_text,
			"error": None,
		}
	except Exception as exc:  # pylint: disable=broad-except
		return {
			"ok": False,
			"pred": None,
			"virtual_circuit_text": None,
			"virtual_dem_text": None,
			"error": f"{type(exc).__name__}: {exc}",
		}


def _decode_rail_one_shot(dem_builder, det_row: np.ndarray, check_row: np.ndarray) -> dict:
	try:
		virtual_circuit = dem_builder.build_decoded_circuit(
			np.asarray(check_row, dtype=np.uint8).tolist(),
			np.asarray(det_row, dtype=np.uint8).tolist(),
			verbose=False,
		)
		virtual_circuit_text = str(virtual_circuit)
		virtual_dem = virtual_circuit.detector_error_model(
			decompose_errors=True,
			approximate_disjoint_errors=True,
		)
		virtual_dem_text = str(virtual_dem)
		matching = pm.Matching.from_detector_error_model(virtual_dem)
		pred = np.asarray(matching.decode(np.asarray(det_row, dtype=np.uint8)), dtype=np.uint8)
		if pred.ndim == 0:
			pred = pred.reshape(1)
		return {
			"ok": True,
			"pred": pred,
			"virtual_circuit_text": virtual_circuit_text,
			"virtual_dem_text": virtual_dem_text,
			"error": None,
		}
	except Exception as exc:  # pylint: disable=broad-except
		return {
			"ok": False,
			"pred": None,
			"virtual_circuit_text": None,
			"virtual_dem_text": None,
			"error": f"{type(exc).__name__}: {exc}",
		}


def _row_mismatch(pred: np.ndarray, truth: np.ndarray) -> bool:
	truth_1d = np.asarray(truth, dtype=np.uint8).reshape(-1)
	pred_1d = np.asarray(pred, dtype=np.uint8).reshape(-1)
	n = min(int(truth_1d.shape[0]), int(pred_1d.shape[0]))
	if n == 0:
		return False
	return bool(np.any(pred_1d[:n] != truth_1d[:n]))


def _save_artifacts(
	output_dir: Path,
	*,
	erasure_circuit_text: str,
	logical_circuit_text: str,
	rail_virtual_circuit_text: str,
	rail_virtual_dem_text: str,
	standard_virtual_circuit_text: str,
	standard_virtual_dem_text: str,
	calibration_text: str,
	metadata: dict,
) -> None:
	output_dir.mkdir(parents=True, exist_ok=True)
	(output_dir / "erasure_circuit.qer").write_text(erasure_circuit_text)
	(output_dir / "logical_circuit.stim").write_text(logical_circuit_text)
	(output_dir / "rail_virtual_circuit.stim").write_text(rail_virtual_circuit_text)
	(output_dir / "rail_virtual_dem.dem").write_text(rail_virtual_dem_text)
	(output_dir / "standard_virtual_circuit.stim").write_text(standard_virtual_circuit_text)
	(output_dir / "standard_virtual_dem.dem").write_text(standard_virtual_dem_text)
	(output_dir / "calibration_posteriors.txt").write_text(calibration_text)
	(output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def _format_calibration_text(
	*,
	cal_nonfinal: list[list[list[float]]],
	summary_nonfinal: dict,
	cal_final: list[list[list[float]]],
	summary_final: dict,
) -> str:
	schedule_labels = ["XZZX", "ZXXZ"]
	condition_labels = ["00", "10", "01", "11"]
	lines: list[str] = []

	def write_block(title: str, posteriors: list[list[list[float]]], summary: dict) -> None:
		lines.append(title)
		events_per_bucket = summary.get("events_per_bucket", [[0, 0, 0, 0], [0, 0, 0, 0]])
		for s_idx, schedule_label in enumerate(schedule_labels):
			lines.append(f"schedule={schedule_label}")
			for c_idx, condition_label in enumerate(condition_labels):
				n_events = int(events_per_bucket[s_idx][c_idx])
				prob_values = [float(v) for v in posteriors[s_idx][c_idx]]
				prob_str = " ".join(f"{v:.6f}" for v in prob_values)
				lines.append(
					f"  condition={condition_label} n={n_events} "
					f"onset_bins=[{prob_str}]"
				)
			lines.append("")

	write_block("non_final_rounds", cal_nonfinal, summary_nonfinal)
	write_block("final_round_only", cal_final, summary_final)
	return "\n".join(lines).rstrip() + "\n"


def main() -> int:
	parser = argparse.ArgumentParser(
		description=(
			"Sample repeatedly and save artifacts for a shot where rail has a logical "
			"failure but standard decode succeeds."
		)
	)
	parser.add_argument("--distance", type=int, default=7)
	parser.add_argument("--rounds", type=int, default=7)
	parser.add_argument("--erasure-prob", type=float, default=0.001)
	parser.add_argument("--check-prob", type=float, default=0.0)
	parser.add_argument("--pauli-prob", type=float, default=0.0)
	parser.add_argument("--rounds-per-check", type=int, default=2)
	parser.add_argument("--max-persistence", type=int, default=2)
	parser.add_argument("--calibration-shots", type=int, default=30000)
	parser.add_argument("--batch-shots", type=int, default=10000)
	parser.add_argument("--max-shots", type=int, default=200000)
	parser.add_argument("--seed", type=int, default=None)
	parser.add_argument(
		"--artifact-subdir",
		type=str,
		default="rail_vs_standard_logical_mismatch_d3",
	)
	parser.add_argument(
		"--allow-not-found",
		action=argparse.BooleanOptionalAction,
		default=False,
		help="Return success even if no mismatch shot is found.",
	)
	args = parser.parse_args()

	if args.distance <= 0 or args.rounds <= 0:
		raise ValueError("distance and rounds must be positive")
	if args.max_persistence <= 0:
		raise ValueError("--max-persistence must be positive")
	if args.rounds_per_check <= 0:
		raise ValueError("--rounds-per-check must be positive")
	if args.calibration_shots <= 0:
		raise ValueError("--calibration-shots must be positive")
	if args.batch_shots <= 0:
		raise ValueError("--batch-shots must be positive")
	if args.max_shots <= 0:
		raise ValueError("--max-shots must be positive")

	repo_root, qe = _import_qerasure()
	base_seed = int(args.seed) if args.seed is not None else int(secrets.randbits(32))
	print(f"base_seed={base_seed}", flush=True)

	erasure_circuit = _build_circuit(
		qe,
		distance=int(args.distance),
		rounds=int(args.rounds),
		erasure_prob=float(args.erasure_prob),
		pauli_prob=float(args.pauli_prob),
		rounds_per_check=int(args.rounds_per_check),
	)
	model = _build_model(
		qe,
		max_persistence=int(args.max_persistence),
		check_prob=float(args.check_prob),
	)
	compiled = qe.CompiledErasureProgram(erasure_circuit, model)
	rail_program = qe.RailSurfaceCompiledProgram(
		erasure_circuit,
		model,
		int(args.distance),
		int(args.rounds),
	)
	sampler = qe.RailStreamSampler(rail_program)

	print(
		f"calibrating onset posteriors: shots={int(args.calibration_shots)} "
		f"(non-final + final-round)",
		flush=True,
	)
	cal_nonfinal, summary_nonfinal = _calibrate_posteriors(
		qe,
		rail_program=rail_program,
		shots=int(args.calibration_shots),
		seed=int((base_seed ^ 0xA5A5A5A5) & 0xFFFFFFFF),
		final_round_only=False,
	)
	cal_final, summary_final = _calibrate_posteriors(
		qe,
		rail_program=rail_program,
		shots=int(args.calibration_shots),
		seed=int((base_seed ^ 0x6C6C6C6C) & 0xFFFFFFFF),
		final_round_only=True,
	)

	rail_dem_builder = qe.RailSurfaceDemBuilder(rail_program)
	rail_dem_builder.set_calibrated_onset_posteriors(
		float(args.erasure_prob),
		cal_nonfinal,
		True,
	)
	rail_dem_builder.set_final_round_calibrated_onset_posteriors(
		float(args.erasure_prob),
		cal_final,
		True,
	)
	standard_dem_builder = qe.SurfDemBuilder(compiled)

	max_batches = (int(args.max_shots) + int(args.batch_shots) - 1) // int(args.batch_shots)
	processed = 0
	rail_decode_errors = 0
	standard_decode_errors = 0

	for batch_index in range(max_batches):
		remaining = int(args.max_shots) - processed
		if remaining <= 0:
			break
		shots_in_batch = min(int(args.batch_shots), remaining)
		batch_seed = int((base_seed + batch_index * 1000003) & 0xFFFFFFFF)
		dets, obs, checks = sampler.sample(
			num_shots=int(shots_in_batch),
			seed=int(batch_seed),
			num_threads=1,
		)
		for local_shot in range(int(shots_in_batch)):
			global_shot = processed + local_shot
			det_row = np.asarray(dets[local_shot], dtype=np.uint8)
			check_row = np.asarray(checks[local_shot], dtype=np.uint8)
			obs_row = np.asarray(obs[local_shot], dtype=np.uint8).reshape(-1)

			rail_decoded = _decode_rail_one_shot(rail_dem_builder, det_row, check_row)
			standard_decoded = _decode_standard_one_shot(standard_dem_builder, det_row, check_row)
			if not rail_decoded["ok"]:
				rail_decode_errors += 1
				continue
			if not standard_decoded["ok"]:
				standard_decode_errors += 1
				continue

			rail_fail = _row_mismatch(rail_decoded["pred"], obs_row)
			standard_fail = _row_mismatch(standard_decoded["pred"], obs_row)
			if rail_fail and not standard_fail:
				logical_circuit_text, replayed_checks = sampler.sample_exact_shot(
					int(batch_seed),
					int(local_shot),
				)
				if not np.array_equal(
					np.asarray(replayed_checks, dtype=np.uint8),
					check_row,
				):
					raise RuntimeError(
						"Replayed check row mismatch for located mismatch shot; aborting artifact save."
					)

				artifacts_dir = repo_root / "tests" / "artifacts" / str(args.artifact_subdir)
				metadata = {
					"distance": int(args.distance),
					"rounds": int(args.rounds),
					"erasure_prob": float(args.erasure_prob),
					"check_prob": float(args.check_prob),
					"pauli_prob": float(args.pauli_prob),
					"rounds_per_check": int(args.rounds_per_check),
					"max_persistence": int(args.max_persistence),
					"seed": int(base_seed),
					"batch_seed": int(batch_seed),
					"batch_index": int(batch_index),
					"local_shot": int(local_shot),
					"global_shot_index": int(global_shot),
					"rail_decode_errors_before_match": int(rail_decode_errors),
					"standard_decode_errors_before_match": int(standard_decode_errors),
					"obs_row": [int(v) for v in obs_row.tolist()],
					"check_row": [int(v) for v in check_row.tolist()],
					"detector_row": [int(v) for v in det_row.tolist()],
					"rail_prediction": [
						int(v) for v in np.asarray(rail_decoded["pred"], dtype=np.uint8).reshape(-1).tolist()
					],
					"standard_prediction": [
						int(v) for v in np.asarray(standard_decoded["pred"], dtype=np.uint8).reshape(-1).tolist()
					],
					"calibration_nonfinal_events_per_bucket": summary_nonfinal["events_per_bucket"],
					"calibration_final_events_per_bucket": summary_final["events_per_bucket"],
				}
				calibration_text = _format_calibration_text(
					cal_nonfinal=cal_nonfinal,
					summary_nonfinal=summary_nonfinal,
					cal_final=cal_final,
					summary_final=summary_final,
				)
				_save_artifacts(
					artifacts_dir,
					erasure_circuit_text=erasure_circuit.to_string(),
					logical_circuit_text=logical_circuit_text,
					rail_virtual_circuit_text=str(rail_decoded["virtual_circuit_text"]),
					rail_virtual_dem_text=str(rail_decoded["virtual_dem_text"]),
					standard_virtual_circuit_text=str(standard_decoded["virtual_circuit_text"]),
					standard_virtual_dem_text=str(standard_decoded["virtual_dem_text"]),
					calibration_text=calibration_text,
					metadata=metadata,
				)
				print("python_rail_vs_standard_logical_mismatch_artifact_test", flush=True)
				print(f"status: mismatch found and artifacts saved to {artifacts_dir}", flush=True)
				print(f"global_shot_index: {global_shot}", flush=True)
				return 0

		processed += int(shots_in_batch)
		print(
			f"processed={processed}/{int(args.max_shots)} "
			f"(rail_decode_errors={rail_decode_errors}, standard_decode_errors={standard_decode_errors})",
			flush=True,
		)

	message = (
		"No rail-only logical mismatch shot found within search budget "
		f"({processed} shots)."
	)
	if args.allow_not_found:
		print(message, flush=True)
		return 0
	raise RuntimeError(message)


if __name__ == "__main__":
	raise SystemExit(main())
