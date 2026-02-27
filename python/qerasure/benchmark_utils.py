"""High-throughput benchmark helpers for logical/virtual translation + decoding."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .code_utils import RotatedSurfaceCode
from .lowering_utils import LoweredErrorParams, Lowerer, LoweringParams, LoweringResult, PauliError, SpreadProgram
from .noise_utils import NoiseChannel, NoiseParams
from .sim_utils import ErasureQubitSelection, ErasureSimParams, ErasureSimulator
from .translation_utils import build_logical_stabilizer_circuit_object
from .virtual_translation_utils import build_virtual_decoder_stim_circuit_object


@dataclass
class GroupedBenchmarkResult:
    """Structured result returned by grouped virtual-decode benchmark runs."""

    distance: int
    qec_rounds: int
    shots_requested: int
    shots_attempted: int
    shots_failed: int
    two_qubit_erasure_probability: float
    seed: int
    condition_on_erasure_in_round: bool
    signature_groups: int
    signature_cache_hits: int
    signature_cache_misses: int
    logical_error_rate: float | None
    timing_seconds: dict[str, float]
    throughput_shots_per_sec: float | None
    failure_artifacts_dir: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "distance": self.distance,
            "qec_rounds": self.qec_rounds,
            "shots_requested": self.shots_requested,
            "shots_attempted": self.shots_attempted,
            "shots_failed": self.shots_failed,
            "two_qubit_erasure_probability": self.two_qubit_erasure_probability,
            "seed": self.seed,
            "condition_on_erasure_in_round": self.condition_on_erasure_in_round,
            "signature_groups": self.signature_groups,
            "signature_cache_hits": self.signature_cache_hits,
            "signature_cache_misses": self.signature_cache_misses,
            "logical_error_rate": self.logical_error_rate,
            "timing_seconds": self.timing_seconds,
            "throughput_shots_per_sec": self.throughput_shots_per_sec,
            "failure_artifacts_dir": self.failure_artifacts_dir,
        }


def default_notebook_lowering_params() -> LoweringParams:
    """LoweringParams mirroring `apps/qerasure_sims.ipynb`."""

    program = SpreadProgram()
    program.append("Z_ERROR(0.5) X_1; Z_ERROR(0.5) X_2")
    program.append("COND_X_ERROR(0.5) Z_1; ELSE_X_ERROR(1.0) Z_2")
    reset = LoweredErrorParams(PauliError.DEPOLARIZE, 1.0)
    return LoweringParams(program, reset)


def _shot_signature(
    lowering_result: LoweringResult,
    shot_index: int,
    *,
    distance: int,
    qec_rounds: int,
    p_tqe: float,
    condition_on_erasure_in_round: bool,
) -> bytes:
    """Build a stable signature for grouping shots with identical translated circuits."""

    h = hashlib.blake2b(digest_size=16)
    h.update(distance.to_bytes(2, "little", signed=False))
    h.update(qec_rounds.to_bytes(2, "little", signed=False))
    h.update(np.float64(p_tqe).tobytes())
    h.update(b"\x01" if condition_on_erasure_in_round else b"\x00")

    offsets = lowering_result.clifford_timestep_offsets[shot_index]
    for off in offsets:
        h.update(int(off).to_bytes(4, "little", signed=False))

    for event in lowering_result.sparse_cliffords[shot_index]:
        h.update(int(event.qubit_idx).to_bytes(2, "little", signed=False))
        h.update(int(event.error_type).to_bytes(1, "little", signed=False))
        h.update(int(event.origin).to_bytes(1, "little", signed=False))

    if shot_index < len(lowering_result.erasure_round_flags):
        h.update(bytes(int(v) for v in lowering_result.erasure_round_flags[shot_index]))
    if shot_index < len(lowering_result.reset_round_qubits):
        for round_idx, qubit_idx in lowering_result.reset_round_qubits[shot_index]:
            h.update(int(round_idx).to_bytes(2, "little", signed=False))
            h.update(int(qubit_idx).to_bytes(2, "little", signed=False))

    return h.digest()


def _write_failure_artifacts(
    out_dir: Path,
    shot_index: int,
    err: Exception,
    erasure_results,
    lowering_result: LoweringResult,
    logical_circuit,
    virtual_circuit,
) -> None:
    shot_dir = out_dir / f"shot_{shot_index:05d}"
    shot_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "shot_index": shot_index,
        "error_type": type(err).__name__,
        "error_message": str(err),
    }
    (shot_dir / "error.json").write_text(json.dumps(meta, indent=2))

    debug = {
        "erasure_timestep_offsets": erasure_results.erasure_timestep_offsets[shot_index],
        "sparse_erasures": [
            {"qubit_idx": int(e.qubit_idx), "event_type": int(e.event_type)}
            for e in erasure_results.sparse_erasures[shot_index]
        ],
        "lowering_timestep_offsets": lowering_result.clifford_timestep_offsets[shot_index],
        "sparse_cliffords": [
            {
                "qubit_idx": int(e.qubit_idx),
                "error_type": int(e.error_type),
                "origin": int(e.origin),
            }
            for e in lowering_result.sparse_cliffords[shot_index]
        ],
        "check_error_round_flags": list(lowering_result.check_error_round_flags[shot_index])
        if shot_index < len(lowering_result.check_error_round_flags)
        else [],
        "erasure_round_flags": list(lowering_result.erasure_round_flags[shot_index])
        if shot_index < len(lowering_result.erasure_round_flags)
        else [],
        "reset_round_qubits": [
            [int(r), int(q)] for (r, q) in lowering_result.reset_round_qubits[shot_index]
        ]
        if shot_index < len(lowering_result.reset_round_qubits)
        else [],
    }
    (shot_dir / "shot_debug.json").write_text(json.dumps(debug, indent=2))

    if logical_circuit is not None:
        (shot_dir / "logical_circuit.stim").write_text(str(logical_circuit))
    if virtual_circuit is not None:
        (shot_dir / "virtual_circuit.stim").write_text(str(virtual_circuit))


def _decode_with_partitioned_batches(matching, syndromes: np.ndarray, num_obs: int) -> tuple[np.ndarray, np.ndarray]:
    """Decode with batch splitting; isolates failing syndromes without full per-shot fallback."""

    num_shots = syndromes.shape[0]
    predictions = np.zeros((num_shots, max(1, num_obs)), dtype=np.uint8)
    failed_mask = np.zeros(num_shots, dtype=bool)

    if not hasattr(matching, "decode_batch"):
        for i in range(num_shots):
            try:
                pred = np.asarray(matching.decode(syndromes[i]), dtype=np.uint8)
                if pred.ndim == 0:
                    pred = pred.reshape(1)
                if pred.ndim == 1:
                    pred = pred.reshape(1, -1)
                n = min(pred.shape[1], predictions.shape[1])
                predictions[i, :n] = pred[0, :n]
            except Exception:
                failed_mask[i] = True
        return predictions, failed_mask

    # Use a stack to avoid recursion depth issues on large groups.
    stack: list[np.ndarray] = [np.arange(num_shots, dtype=np.int64)]
    while stack:
        indices = stack.pop()
        if indices.size == 0:
            continue
        try:
            pred = np.asarray(matching.decode_batch(syndromes[indices]), dtype=np.uint8)
            if pred.ndim == 1:
                pred = pred[:, None]
            n = min(pred.shape[1], predictions.shape[1])
            predictions[indices, :n] = pred[:, :n]
        except Exception:
            if indices.size == 1:
                failed_mask[indices[0]] = True
                continue
            split = indices.size // 2
            stack.append(indices[:split])
            stack.append(indices[split:])
    return predictions, failed_mask


def run_grouped_virtual_decode_benchmark(
    *,
    distance: int = 3,
    qec_rounds: int = 3,
    shots: int = 10_000,
    two_qubit_erasure_probability: float = 0.01,
    seed: int = 12345,
    lowering_params: LoweringParams | None = None,
    condition_on_erasure_in_round: bool = True,
    write_failure_artifacts: bool = True,
    failure_artifacts_dir: Path | None = None,
) -> GroupedBenchmarkResult:
    """Run high-throughput grouped benchmark for logical/virtual translation + decoding."""

    try:
        import pymatching as pm
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime dependency
        raise ModuleNotFoundError(
            "The Python `pymatching` package is required for grouped virtual decoding benchmarks."
        ) from exc

    if lowering_params is None:
        lowering_params = default_notebook_lowering_params()

    code = RotatedSurfaceCode(distance)
    noise = NoiseParams()
    noise.set(NoiseChannel.TWO_QUBIT_ERASURE, two_qubit_erasure_probability)

    sim_params = ErasureSimParams(
        code=code,
        noise=noise,
        qec_rounds=qec_rounds,
        shots=shots,
        seed=seed,
        erasure_selection=ErasureQubitSelection.DATA_QUBITS,
    )

    t0 = time.perf_counter()
    erasure_results = ErasureSimulator(sim_params).simulate()
    t1 = time.perf_counter()

    lowering_result = Lowerer(code, lowering_params).lower(erasure_results)
    t2 = time.perf_counter()

    groups: dict[bytes, list[int]] = defaultdict(list)
    for shot_index in range(shots):
        signature = _shot_signature(
            lowering_result,
            shot_index,
            distance=distance,
            qec_rounds=qec_rounds,
            p_tqe=two_qubit_erasure_probability,
            condition_on_erasure_in_round=condition_on_erasure_in_round,
        )
        groups[signature].append(shot_index)

    if failure_artifacts_dir is None:
        failure_artifacts_dir = Path("benchmarks") / "failures" / time.strftime("%Y%m%d_%H%M%S")

    attempted = 0
    failed = 0
    mismatches = 0

    t_build_logical = 0.0
    t_build_virtual = 0.0
    t_dem = 0.0
    t_sampler_compile = 0.0
    t_sampler_sample = 0.0
    t_match_build = 0.0
    t_decode = 0.0

    for group_shots in groups.values():
        rep = group_shots[0]
        logical_circuit = None
        virtual_circuit = None
        matcher = None
        detector_samples = None
        observable_flips = None

        try:
            st = time.perf_counter()
            logical_circuit = build_logical_stabilizer_circuit_object(
                code=code,
                lowering_result=lowering_result,
                shot_index=rep,
            )
            t_build_logical += time.perf_counter() - st

            st = time.perf_counter()
            virtual_circuit = build_virtual_decoder_stim_circuit_object(
                code=code,
                qec_rounds=qec_rounds,
                lowering_params=lowering_params,
                lowering_result=lowering_result,
                shot_index=rep,
                two_qubit_erasure_probability=two_qubit_erasure_probability,
                condition_on_erasure_in_round=condition_on_erasure_in_round,
            )
            t_build_virtual += time.perf_counter() - st

            st = time.perf_counter()
            virtual_dem = virtual_circuit.detector_error_model(
                decompose_errors=True,
                approximate_disjoint_errors=True,
            )
            t_dem += time.perf_counter() - st

            st = time.perf_counter()
            logical_sampler = logical_circuit.compile_detector_sampler()
            t_sampler_compile += time.perf_counter() - st

            st = time.perf_counter()
            matcher = pm.Matching.from_detector_error_model(virtual_dem)
            t_match_build += time.perf_counter() - st

            st = time.perf_counter()
            detector_samples, observable_flips = logical_sampler.sample(
                shots=len(group_shots), separate_observables=True
            )
            t_sampler_sample += time.perf_counter() - st

            syndromes = np.asarray(detector_samples, dtype=np.uint8)
            truths = np.asarray(observable_flips, dtype=np.uint8)
            if syndromes.ndim == 1:
                syndromes = syndromes[None, :]
            if truths.ndim == 1:
                truths = truths[:, None]

            st = time.perf_counter()
            predictions, failed_mask = _decode_with_partitioned_batches(matcher, syndromes, truths.shape[1])
            t_decode += time.perf_counter() - st

            if predictions.ndim == 1:
                predictions = predictions[:, None]
            if failed_mask.any():
                failed_rows = np.flatnonzero(failed_mask)
                failed += int(failed_rows.size)
                if write_failure_artifacts:
                    row_i = int(failed_rows[0])
                    err = RuntimeError(
                        f"No perfect matching for {failed_rows.size} syndrome(s) in this signature group."
                    )
                    _write_failure_artifacts(
                        out_dir=failure_artifacts_dir,
                        shot_index=group_shots[row_i],
                        err=err,
                        erasure_results=erasure_results,
                        lowering_result=lowering_result,
                        logical_circuit=logical_circuit,
                        virtual_circuit=virtual_circuit,
                    )
            n_obs = min(predictions.shape[1], truths.shape[1])
            if n_obs == 0:
                # No observable comparisons available; count as attempted without mismatch.
                attempted += len(group_shots) - int(np.count_nonzero(failed_mask))
            else:
                group_mismatch = np.any((predictions[:, :n_obs] ^ truths[:, :n_obs]) != 0, axis=1)
                group_mismatch = np.logical_and(group_mismatch, np.logical_not(failed_mask))
                mismatches += int(np.count_nonzero(group_mismatch))
                attempted += len(group_shots) - int(np.count_nonzero(failed_mask))

        except Exception as err:
            failed += len(group_shots)
            if write_failure_artifacts:
                _write_failure_artifacts(
                    out_dir=failure_artifacts_dir,
                    shot_index=group_shots[0],
                    err=err,
                    erasure_results=erasure_results,
                    lowering_result=lowering_result,
                    logical_circuit=logical_circuit,
                    virtual_circuit=virtual_circuit,
                )

        # Drop per-group heavy objects.
        del logical_circuit
        del virtual_circuit
        del matcher
        del detector_samples
        del observable_flips

    t3 = time.perf_counter()

    timings = {
        "simulate": t1 - t0,
        "lower": t2 - t1,
        "build_logical_grouped": t_build_logical,
        "build_virtual_grouped": t_build_virtual,
        "virtual_dem_grouped": t_dem,
        "sampler_compile_grouped": t_sampler_compile,
        "sampler_sample_grouped": t_sampler_sample,
        "matching_build_grouped": t_match_build,
        "decode_grouped": t_decode,
        "grouped_loop_total": t3 - t2,
        "total": t3 - t0,
    }

    result = GroupedBenchmarkResult(
        distance=distance,
        qec_rounds=qec_rounds,
        shots_requested=shots,
        shots_attempted=attempted,
        shots_failed=failed,
        two_qubit_erasure_probability=two_qubit_erasure_probability,
        seed=seed,
        condition_on_erasure_in_round=condition_on_erasure_in_round,
        signature_groups=len(groups),
        signature_cache_hits=shots - len(groups),
        signature_cache_misses=len(groups),
        logical_error_rate=(mismatches / attempted) if attempted else None,
        timing_seconds=timings,
        throughput_shots_per_sec=(attempted / (t3 - t2)) if (t3 - t2) > 0 else None,
        failure_artifacts_dir=str(failure_artifacts_dir) if failed and write_failure_artifacts else None,
    )

    return result
