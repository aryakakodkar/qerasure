"""Python wrappers for circuit-model compile + stream sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from ._bindings import cpp

OpCode = cpp.OpCode
PauliChannel = cpp.PauliChannel
TQGSpreadModel = cpp.TQGSpreadModel
ErasureModel = cpp.ErasureModel
SurfaceCodeRotated = cpp.SurfaceCodeRotated


class ErasureCircuit:
    """Python facade for `cpp.ErasureCircuit` with Stim-like append ergonomics."""

    def __init__(self, circuit_text: str | None = None):
        self._cpp_circuit = cpp.ErasureCircuit()
        if circuit_text is not None:
            self.from_string(circuit_text)

    @classmethod
    def from_file(cls, filepath: str) -> "ErasureCircuit":
        circuit = cls()
        circuit._cpp_circuit.from_file(filepath)
        return circuit

    def append(self, op: str | object, targets: Sequence[int], arg: float = 0.0) -> "ErasureCircuit":
        int_targets = [int(t) for t in targets]
        if isinstance(op, str):
            self._cpp_circuit.safe_append(op, int_targets, float(arg))
        else:
            self._cpp_circuit.append(op, int_targets, float(arg))
        return self

    def append_detector(self, rec_lookbacks: Sequence[int]) -> "ErasureCircuit":
        self._cpp_circuit.append_detector([int(v) for v in rec_lookbacks])
        return self

    def append_observable_include(self, rec_lookbacks: Sequence[int]) -> "ErasureCircuit":
        self._cpp_circuit.append_observable_include([int(v) for v in rec_lookbacks])
        return self

    def from_string(self, circuit_str: str) -> "ErasureCircuit":
        self._cpp_circuit.from_string(circuit_str)
        return self

    def to_string(self) -> str:
        return str(self._cpp_circuit.to_string())

    def _to_cpp_circuit(self):
        return self._cpp_circuit

    def __str__(self) -> str:
        return self.to_string()


@dataclass
class CompiledErasureProgram:
    """Python holder around `cpp.CompiledErasureProgram`."""

    circuit: ErasureCircuit | object
    model: ErasureModel

    def __post_init__(self):
        cpp_circuit = (
            self.circuit._to_cpp_circuit() if isinstance(self.circuit, ErasureCircuit) else self.circuit
        )
        self._cpp_program = cpp.CompiledErasureProgram(cpp_circuit, self.model)

    @property
    def num_checks(self) -> int:
        return int(self._cpp_program.num_checks)

    @property
    def max_qubit_index(self) -> int:
        return int(self._cpp_program.max_qubit_index)

    @property
    def max_persistence(self) -> int:
        return int(self._cpp_program.max_persistence)

    def _to_cpp_program(self):
        return self._cpp_program

    @property
    def check_lookback_links(self):
        return self._cpp_program.check_lookback_links

class StreamSampler:
    """Python stream sampler that returns detector/observable/check arrays."""

    def __init__(self, program: CompiledErasureProgram | object):
        if isinstance(program, CompiledErasureProgram):
            self._program = program
            cpp_program = program._to_cpp_program()
            self._num_checks = program.num_checks
        else:
            self._program = program
            cpp_program = program
            self._num_checks = int(program.num_checks)
        self._cpp_sampler = cpp.StreamSampler(cpp_program)

    @property
    def num_checks(self) -> int:
        return self._num_checks

    def sample(
        self,
        num_shots: int,
        seed: int,
        num_threads: int = 1,
    ):
        shots = int(num_shots)
        threads = int(num_threads)
        if shots < 0:
            raise ValueError("num_shots must be non-negative")
        if threads < 0:
            raise ValueError("num_threads must be non-negative")
        return self._cpp_sampler.sample_syndromes(shots, int(seed), threads)

    def sample_with_callback(
        self,
        num_shots: int,
        seed: int,
        callback: Optional[Callable[[str, object], None]] = None,
        num_threads: int = 1,
    ):
        import numpy as np

        shots = int(num_shots)
        threads = int(num_threads)
        if shots < 0:
            raise ValueError("num_shots must be non-negative")
        if callback is None:
            return self._cpp_sampler.sample_with_callback(shots, int(seed), None, threads)

        def wrapped(circuit_obj, check_flags):
            callback(circuit_obj, np.asarray(check_flags, dtype=np.uint8))

        return self._cpp_sampler.sample_with_callback(shots, int(seed), wrapped, threads)


class SurfDemBuilder:
    """Python wrapper for the surface-code decoded-circuit/DEM builder."""

    def __init__(self, program: CompiledErasureProgram | object):
        if isinstance(program, CompiledErasureProgram):
            cpp_program = program._to_cpp_program()
        else:
            cpp_program = program
        self._cpp_builder = cpp.SurfDemBuilder(cpp_program)

    def build_decoded_circuit(self, check_results: Sequence[int], verbose: bool = False):
        """Build the decoded Stim circuit from erasure-check flags."""
        checks = [int(v) for v in check_results]
        try:
            import stim
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The Python `stim` package is required for SurfDemBuilder.build_decoded_circuit."
            ) from exc
        return stim.Circuit(self._cpp_builder.build_decoded_circuit(checks, bool(verbose)))

    def find_probability_violations(self, check_results: Sequence[int]):
        """Return PAULI_CHANNEL_1 events whose disjoint probabilities sum above 1."""
        checks = [int(v) for v in check_results]
        return self._cpp_builder.find_probability_violations(checks)

    def build_decoded_circuit_text(self, check_results: Sequence[int], verbose: bool = False) -> str:
        """Return a textual decoded-circuit dump for debugging invalid probability tuples."""
        checks = [int(v) for v in check_results]
        return str(self._cpp_builder.build_decoded_circuit_text(checks, bool(verbose)))


class SurfaceCodeBatchDecoder:
    """Grouped batch decoder using one decoded DEM per unique check-flag pattern."""

    def __init__(
        self,
        program: CompiledErasureProgram | object,
        dem_builder: Optional[SurfDemBuilder] = None,
        max_batch_bytes: int = 256 * 1024 * 1024,
    ):
        if isinstance(program, CompiledErasureProgram):
            cpp_program = program._to_cpp_program()
        else:
            cpp_program = program

        self._num_checks = int(cpp_program.num_checks)
        self._dem_builder = dem_builder if dem_builder is not None else SurfDemBuilder(cpp_program)
        self._max_batch_bytes = int(max_batch_bytes)
        if self._max_batch_bytes <= 0:
            raise ValueError("max_batch_bytes must be positive.")

    def _compute_shots_per_chunk(self, num_detectors: int, num_checks: int) -> int:
        packed_check_bytes = (int(num_checks) + 7) // 8
        # Detector row + packed-check key + prediction row + grouping/index overhead.
        bytes_per_shot = int(num_detectors) + packed_check_bytes + 1 + 32
        return max(1, self._max_batch_bytes // max(1, bytes_per_shot))

    def decode_batch(self, detector_samples, check_flags):
        """Decode a full batch of shots, grouping by identical check flags."""
        import numpy as np
        import pymatching as pm

        dets = np.asarray(detector_samples, dtype=np.uint8)
        checks = np.asarray(check_flags, dtype=np.uint8)
        if dets.ndim != 2:
            raise ValueError("detector_samples must be a 2D uint8 array.")
        if checks.ndim != 2:
            raise ValueError("check_flags must be a 2D uint8 array.")
        if dets.shape[0] != checks.shape[0]:
            raise ValueError("detector_samples and check_flags must have the same number of shots.")
        if checks.shape[1] != self._num_checks:
            raise ValueError(
                f"check_flags width mismatch: expected {self._num_checks}, got {checks.shape[1]}."
            )

        shots = int(dets.shape[0])
        if shots == 0:
            return np.zeros((0, 0), dtype=np.uint8)

        shots_per_chunk = self._compute_shots_per_chunk(dets.shape[1], checks.shape[1])
        predictions = None

        for start in range(0, shots, shots_per_chunk):
            end = min(shots, start + shots_per_chunk)
            det_chunk = dets[start:end]
            check_chunk = checks[start:end]

            packed = np.packbits(check_chunk, axis=1, bitorder="little")
            groups = {}
            for local_idx in range(packed.shape[0]):
                key = packed[local_idx].tobytes()
                groups.setdefault(key, []).append(local_idx)

            for key, local_indices in groups.items():
                if check_chunk.shape[1] == 0:
                    check_row = np.zeros((0,), dtype=np.uint8)
                else:
                    packed_row = np.frombuffer(key, dtype=np.uint8)
                    unpacked = np.unpackbits(packed_row, bitorder="little")
                    check_row = unpacked[: check_chunk.shape[1]].astype(np.uint8, copy=False)

                decoded_circuit = self._dem_builder.build_decoded_circuit(check_row, verbose=False)
                decoded_dem = decoded_circuit.detector_error_model(
                    decompose_errors=True,
                    approximate_disjoint_errors=True,
                )
                matching = pm.Matching.from_detector_error_model(decoded_dem)
                group_detectors = det_chunk[np.asarray(local_indices, dtype=np.int64)]

                if hasattr(matching, "decode_batch"):
                    group_preds = np.asarray(matching.decode_batch(group_detectors), dtype=np.uint8)
                    if group_preds.ndim == 1:
                        group_preds = group_preds[:, None]
                else:
                    pred_rows = []
                    for row in group_detectors:
                        pred = np.asarray(matching.decode(row), dtype=np.uint8)
                        if pred.ndim == 0:
                            pred = pred.reshape(1)
                        pred_rows.append(pred)
                    width = max(1, max((p.shape[0] for p in pred_rows), default=0))
                    group_preds = np.zeros((len(pred_rows), width), dtype=np.uint8)
                    for i, pred in enumerate(pred_rows):
                        n = min(width, pred.shape[0])
                        group_preds[i, :n] = pred[:n]

                if predictions is None:
                    predictions = np.zeros((shots, max(1, int(group_preds.shape[1]))), dtype=np.uint8)
                elif group_preds.shape[1] > predictions.shape[1]:
                    grown = np.zeros((shots, int(group_preds.shape[1])), dtype=np.uint8)
                    grown[:, : predictions.shape[1]] = predictions
                    predictions = grown

                for group_pos, local_idx in enumerate(local_indices):
                    n = min(predictions.shape[1], group_preds.shape[1])
                    predictions[start + local_idx, :n] = group_preds[group_pos, :n]

        if predictions is None:
            return np.zeros((shots, 0), dtype=np.uint8)
        return predictions


def compile_erasure_sampler(
    circuit: ErasureCircuit | object,
    model: ErasureModel,
) -> StreamSampler:
    """Convenience helper: circuit + model -> compiled program -> stream sampler."""
    program = CompiledErasureProgram(circuit=circuit, model=model)
    return StreamSampler(program)


def build_surface_code_erasure_circuit(
    distance: int,
    rounds: int,
    erasure_prob: float,
    erasable_qubits: str = "ALL",
    reset_failure_prob: float = 0.0,
) -> ErasureCircuit:
    """Build a rotated-surface-code erasure circuit using the C++ generator."""
    generator = SurfaceCodeRotated(int(distance))
    cpp_circuit = generator.build_circuit(
        int(rounds),
        float(erasure_prob),
        str(erasable_qubits),
        float(reset_failure_prob),
    )
    wrapped = ErasureCircuit()
    wrapped._cpp_circuit = cpp_circuit
    return wrapped
