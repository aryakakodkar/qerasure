#!/usr/bin/env python3
"""Find a grouped decode failure and save the failing circuits as artifacts."""

from __future__ import annotations

import json
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
    import stim  # noqa: F401  pylint: disable=import-error,unused-import
    import qerasure as qe  # pylint: disable=import-error

    return repo_root, qe


def _build_components(
    qe,
    *,
    distance: int,
    rounds: int,
    erasure_prob: float,
    single_qubit_errors: bool,
    max_persistence: int,
    check_error_prob: float,
):
    circuit = qe.SurfaceCodeRotated(distance).build_circuit(
        rounds=rounds,
        erasure_prob=erasure_prob,
        erasable_qubits="ALL",
        reset_failure_prob=0.0,
        single_qubit_errors=single_qubit_errors,
        post_clifford_pauli_prob=0.0,
    )

    model = qe.ErasureModel(
        int(max_persistence),
        qe.PauliChannel(0.25, 0.25, 0.25),
        qe.PauliChannel(0.25, 0.25, 0.25),
        qe.TQGSpreadModel(
            qe.PauliChannel(0.25, 0.25, 0.25),
            qe.PauliChannel(0.25, 0.25, 0.25),
        ),
    )
    model.check_false_negative_prob = float(check_error_prob)
    model.check_false_positive_prob = float(check_error_prob)

    program = qe.CompiledErasureProgram(circuit, model)
    sampler = qe.StreamSampler(program)
    dem_builder = qe.SurfDemBuilder(program)
    grouped_decoder = qe.SurfaceCodeBatchDecoder(program, dem_builder=dem_builder)
    return circuit, program, sampler, dem_builder, grouped_decoder


def _pack_check_row(check_row: np.ndarray) -> bytes:
    return np.packbits(np.asarray(check_row, dtype=np.uint8), bitorder="little").tobytes()


def _find_failure(dem_builder, grouped_decoder, dets: np.ndarray, checks: np.ndarray):
    try:
        grouped_decoder.decode_batch(dets, checks, num_threads=1)
        return None
    except Exception as exc:  # pylint: disable=broad-except
        batch_error = f"{type(exc).__name__}: {exc}"

    packed_checks = np.packbits(checks, axis=1, bitorder="little")
    groups: dict[bytes, list[int]] = {}
    for shot_idx in range(int(checks.shape[0])):
        key = packed_checks[shot_idx].tobytes()
        groups.setdefault(key, []).append(shot_idx)

    for key, shot_indices in groups.items():
        packed_row = np.frombuffer(key, dtype=np.uint8)
        unpacked = np.unpackbits(packed_row, bitorder="little")
        check_row = unpacked[: checks.shape[1]].astype(np.uint8, copy=False)
        det_group = dets[np.asarray(shot_indices, dtype=np.int64)]

        try:
            decoded_circuit = dem_builder.build_decoded_circuit(check_row, verbose=False)
        except Exception as exc:  # pylint: disable=broad-except
            return {
                "stage": "build_decoded_circuit",
                "batch_error": batch_error,
                "error": f"{type(exc).__name__}: {exc}",
                "check_row": check_row,
                "shot_indices": shot_indices,
                "failing_shot": int(shot_indices[0]),
                "decoded_circuit_text": None,
            }

        try:
            decoded_dem = decoded_circuit.detector_error_model(
                decompose_errors=True,
                approximate_disjoint_errors=True,
            )
            matching = pm.Matching.from_detector_error_model(decoded_dem)
        except Exception as exc:  # pylint: disable=broad-except
            return {
                "stage": "matching_from_dem",
                "batch_error": batch_error,
                "error": f"{type(exc).__name__}: {exc}",
                "check_row": check_row,
                "shot_indices": shot_indices,
                "failing_shot": int(shot_indices[0]),
                "decoded_circuit_text": str(decoded_circuit),
            }

        try:
            matching.decode_batch(det_group)
        except Exception as exc:  # pylint: disable=broad-except
            failing_shot = int(shot_indices[0])
            for group_pos, shot_idx in enumerate(shot_indices):
                try:
                    matching.decode(det_group[group_pos])
                except Exception:  # pylint: disable=broad-except
                    failing_shot = int(shot_idx)
                    break
            return {
                "stage": "matching_decode",
                "batch_error": batch_error,
                "error": f"{type(exc).__name__}: {exc}",
                "check_row": check_row,
                "shot_indices": shot_indices,
                "failing_shot": failing_shot,
                "decoded_circuit_text": str(decoded_circuit),
            }

    raise RuntimeError(
        "Grouped decode failed, but the failing check pattern could not be isolated."
    )


def _replay_logical_circuit(
    sampler,
    *,
    seed: int,
    failing_shot: int,
    expected_check_row: np.ndarray,
):
    logical_circuit_text, replayed_check_row = sampler.sample_exact_shot(seed, failing_shot)
    if not np.array_equal(replayed_check_row, np.asarray(expected_check_row, dtype=np.uint8)):
        raise RuntimeError(
            "Replayed check bits do not match the sampled failing shot; "
            "refusing to save a mismatched logical circuit."
        )
    return logical_circuit_text, replayed_check_row


def _save_artifacts(
    output_dir: Path,
    *,
    erasure_circuit_text: str,
    logical_circuit_text: str,
    decoded_circuit_text: str | None,
    metadata: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "erasure_circuit.qer").write_text(erasure_circuit_text)
    (output_dir / "logical_circuit.stim").write_text(logical_circuit_text)
    if decoded_circuit_text is not None:
        (output_dir / "virtual_circuit.stim").write_text(decoded_circuit_text)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def main() -> int:
    repo_root, qe = _import_qerasure()

    # Focus the search on the regime requested by the user.
    search_points = [
        {"distance": 3, "rounds": 3, "erasure_prob": 0.00172034, "single_qubit_errors": True},
        {"distance": 3, "rounds": 3, "erasure_prob": 0.00245508, "single_qubit_errors": True},
        {"distance": 3, "rounds": 3, "erasure_prob": 0.00350363, "single_qubit_errors": True},
        {"distance": 3, "rounds": 3, "erasure_prob": 0.005, "single_qubit_errors": True},
    ]
    shots = 20000
    max_seed_tries = 48
    seed_base = 920000
    max_persistence = 3
    check_error_prob = 0.02

    artifact_dir = repo_root / "tests" / "artifacts" / "surface_batch_decode_failure"

    for point_idx, point in enumerate(search_points):
        erasure_circuit, _program, sampler, dem_builder, grouped_decoder = _build_components(
            qe,
            distance=int(point["distance"]),
            rounds=int(point["rounds"]),
            erasure_prob=float(point["erasure_prob"]),
            single_qubit_errors=bool(point["single_qubit_errors"]),
            max_persistence=max_persistence,
            check_error_prob=check_error_prob,
        )

        for seed_offset in range(max_seed_tries):
            seed = seed_base + point_idx * 1000 + seed_offset
            dets, _obs, checks = sampler.sample(num_shots=shots, seed=seed, num_threads=1)
            failure = _find_failure(dem_builder, grouped_decoder, dets, checks)
            if failure is None:
                continue

            logical_circuit_text = _replay_logical_circuit(
                sampler,
                seed=seed,
                failing_shot=int(failure["failing_shot"]),
                expected_check_row=checks[int(failure["failing_shot"])],
            )
            logical_circuit_text, replayed_check_row = logical_circuit_text

            metadata = {
                "distance": int(point["distance"]),
                "rounds": int(point["rounds"]),
                "erasure_prob": float(point["erasure_prob"]),
                "single_qubit_errors": bool(point["single_qubit_errors"]),
                "max_persistence": max_persistence,
                "check_error_prob": check_error_prob,
                "shots": shots,
                "seed": seed,
                "stage": failure["stage"],
                "batch_error": failure["batch_error"],
                "error": failure["error"],
                "failing_shot": int(failure["failing_shot"]),
                "shot_indices": [int(v) for v in failure["shot_indices"]],
                "check_row": [int(v) for v in np.asarray(failure["check_row"], dtype=np.uint8)],
                "replayed_check_row": [int(v) for v in np.asarray(replayed_check_row, dtype=np.uint8)],
            }
            _save_artifacts(
                artifact_dir,
                erasure_circuit_text=erasure_circuit.to_string(),
                logical_circuit_text=logical_circuit_text,
                decoded_circuit_text=failure["decoded_circuit_text"],
                metadata=metadata,
            )

            print("python_surface_batch_decode_failure_artifact_test")
            print(f"saved: {artifact_dir / 'erasure_circuit.qer'}")
            print(f"saved: {artifact_dir / 'logical_circuit.stim'}")
            if failure["decoded_circuit_text"] is not None:
                print(f"saved: {artifact_dir / 'virtual_circuit.stim'}")
            print(f"saved: {artifact_dir / 'metadata.json'}")
            print(f"seed: {seed}")
            print(f"stage: {failure['stage']}")
            print(f"error: {failure['error']}")
            return 0

    raise RuntimeError(
        "Failed to trigger a grouped decode error in the surface-batch search window. "
        "Increase shots, seed range, or search points."
    )


if __name__ == "__main__":
    raise SystemExit(main())
