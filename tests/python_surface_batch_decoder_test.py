#!/usr/bin/env python3
"""Regression test for grouped SurfaceCodeBatchDecoder output parity."""

from __future__ import annotations

import sys
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

    distance = 5
    rounds = 5
    shots = 800
    seed = 12345

    model = qe.ErasureModel(
        2,
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
        erasure_prob=0.01,
        erasable_qubits="DATA",
        reset_failure_prob=0.0,
    )
    program = qe.CompiledErasureProgram(circuit, model)
    sampler = qe.StreamSampler(program)

    dets, obs, checks = sampler.sample(num_shots=shots, seed=seed, num_threads=1)

    dem_builder = qe.SurfDemBuilder(program)
    grouped_decoder = qe.SurfaceCodeBatchDecoder(program, dem_builder=dem_builder)
    grouped_preds = grouped_decoder.decode_batch(dets, checks)

    baseline_preds = np.zeros_like(grouped_preds)
    for i in range(shots):
        decoded_circuit = dem_builder.build_decoded_circuit(checks[i], verbose=False)
        decoded_dem = decoded_circuit.detector_error_model(
            decompose_errors=True,
            approximate_disjoint_errors=True,
        )
        matching = pm.Matching.from_detector_error_model(decoded_dem)
        pred = _decode_obs_with_matching(matching, dets[i])
        n = min(baseline_preds.shape[1], pred.shape[0])
        baseline_preds[i, :n] = pred[:n]

    if grouped_preds.shape != baseline_preds.shape:
        raise RuntimeError(
            f"Prediction shape mismatch: grouped={grouped_preds.shape}, baseline={baseline_preds.shape}"
        )
    if not np.array_equal(grouped_preds, baseline_preds):
        mismatch_count = int(np.sum(np.any(grouped_preds != baseline_preds, axis=1)))
        raise RuntimeError(f"Grouped decoder predictions differ from baseline on {mismatch_count} shots")

    truths = obs if obs.ndim == 2 else obs[:, None]
    n_obs = min(truths.shape[1], grouped_preds.shape[1])
    ler = float(np.mean(np.any(grouped_preds[:, :n_obs] != truths[:, :n_obs], axis=1)))

    print("python_surface_batch_decoder_test")
    print(f"distance: {distance}")
    print(f"rounds: {rounds}")
    print(f"shots: {shots}")
    print(f"logical_error_rate: {ler:.8f}")
    print("status: grouped decode matches baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
