#!/usr/bin/env python3
"""End-to-end Python decode test for d=9, rounds=9 surface-code erasures."""

from __future__ import annotations

import json
import sys
import time
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

    return qe


def _decode_obs_with_matching(matching: pm.Matching, det_row: np.ndarray) -> np.ndarray:
    pred = np.asarray(matching.decode(det_row), dtype=np.uint8)
    if pred.ndim == 0:
        pred = pred.reshape(1)
    return pred


def main() -> int:
    qe = _import_qerasure()

    distance = 9
    rounds = 9
    shots = 10_000
    seed = 12345
    erasure_prob = 0.01
    max_persistence = 2
    artifacts_dir = Path("tests") / "artifacts" / "invalid_decoded_circuits"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for stale in artifacts_dir.glob("shot_*"):
        if stale.is_file():
            stale.unlink()

    model = qe.ErasureModel(
        max_persistence,
        qe.PauliChannel(0.25, 0.25, 0.25),
        qe.PauliChannel(0.25, 0.25, 0.25),
        qe.TQGSpreadModel(
            qe.PauliChannel(0.5, 0.0, 0.0),
            qe.PauliChannel(0.0, 0.0, 0.5),
        ),
    )
    model.check_false_negative_prob = 0.01
    model.check_false_positive_prob = 0.0

    circuit = qe.SurfaceCodeRotated(distance).build_circuit(
        rounds=rounds,
        erasure_prob=erasure_prob,
        erasable_qubits="DATA",
        reset_failure_prob=0.0,
    )
    (artifacts_dir / "input_erasure_circuit.stim").write_text(str(circuit))
    program = qe.CompiledErasureProgram(circuit, model)

    sampler = qe.StreamSampler(program)
    decoder = qe.SurfHMMDecoder(program)

    t0 = time.perf_counter()
    dets, obs, checks = sampler.sample(num_shots=shots, seed=seed, num_threads=1)
    t1 = time.perf_counter()

    if dets.shape[0] != shots or obs.shape[0] != shots or checks.shape[0] != shots:
        raise RuntimeError("Unexpected sampled array row count.")
    if dets.dtype != np.uint8 or obs.dtype != np.uint8 or checks.dtype != np.uint8:
        raise RuntimeError("Expected uint8 arrays from stream sampling.")

    num_obs = obs.shape[1] if obs.ndim == 2 else 1
    predictions = np.zeros((shots, max(1, num_obs)), dtype=np.uint8)

    decode_start = time.perf_counter()
    for shot in range(shots):
        check_row = checks[shot]
        try:
            decoded_circuit = decoder.decode(check_row, verbose=False)
        except Exception as ex:
            violations = decoder.find_probability_violations(check_row)
            flagged_checks = []
            links = program.check_lookback_links
            for idx, bit in enumerate(check_row.tolist()):
                if int(bit) != 1:
                    continue
                link = links[idx]
                flagged_checks.append(
                    {
                        "check_event_index": int(idx),
                        "check_op_index": int(link.check_op_index),
                        "qubit_index": int(link.qubit_index),
                    }
                )
            debug_text = decoder.debug_decoded_circuit_text(check_row, verbose=False)
            prefix = f"shot_{shot:05d}"
            (artifacts_dir / f"{prefix}_decode_error.txt").write_text(str(ex))
            (artifacts_dir / f"{prefix}_check_flags.txt").write_text(
                " ".join(str(int(v)) for v in check_row.tolist())
            )
            (artifacts_dir / f"{prefix}_flagged_checks.json").write_text(
                json.dumps(flagged_checks, indent=2)
            )
            (artifacts_dir / f"{prefix}_probability_violations.json").write_text(
                json.dumps(violations, indent=2)
            )
            (artifacts_dir / f"{prefix}_decoded_circuit_debug.stim").write_text(debug_text)
            raise RuntimeError(
                f"Failed to parse decoded circuit at shot {shot}. "
                f"Artifacts saved under {artifacts_dir}."
            ) from ex

        decoded_dem = decoded_circuit.detector_error_model(
            decompose_errors=True,
            approximate_disjoint_errors=True,
        )
        matching = pm.Matching.from_detector_error_model(decoded_dem)
        pred = _decode_obs_with_matching(matching, dets[shot])
        n = min(pred.shape[0], predictions.shape[1])
        predictions[shot, :n] = pred[:n]
    decode_end = time.perf_counter()

    truths = obs
    if truths.ndim == 1:
        truths = truths[:, None]
    n_obs = min(truths.shape[1], predictions.shape[1])
    mismatches = np.any(predictions[:, :n_obs] != truths[:, :n_obs], axis=1)
    logical_error_rate = float(np.mean(mismatches))

    print("python_stream_decode_pymatching_d9r9_test")
    print(f"distance: {distance}")
    print(f"rounds: {rounds}")
    print(f"shots: {shots}")
    print(f"erasure_prob: {erasure_prob}")
    print(f"max_persistence: {max_persistence}")
    print(f"erasable_qubits: DATA")
    print(f"sample_time_s: {t1 - t0:.6f}")
    print(f"decode_time_s: {decode_end - decode_start:.6f}")
    print(f"logical_error_rate: {logical_error_rate:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
