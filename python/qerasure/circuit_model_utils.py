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


class SurfHMMDecoder:
    """Python wrapper for the surface-code HMM decoder circuit builder."""

    def __init__(self, program: CompiledErasureProgram | object):
        if isinstance(program, CompiledErasureProgram):
            cpp_program = program._to_cpp_program()
        else:
            cpp_program = program
        self._cpp_decoder = cpp.SurfHMMDecoder(cpp_program)

    def decode(self, check_results: Sequence[int], verbose: bool = False):
        """Build the decoded Stim circuit from erasure-check flags."""
        checks = [int(v) for v in check_results]
        try:
            import stim
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The Python `stim` package is required for SurfHMMDecoder.decode."
            ) from exc
        return stim.Circuit(self._cpp_decoder.decode(checks, bool(verbose)))

    def find_probability_violations(self, check_results: Sequence[int]):
        """Return PAULI_CHANNEL_1 events whose disjoint probabilities sum above 1."""
        checks = [int(v) for v in check_results]
        return self._cpp_decoder.find_probability_violations(checks)

    def debug_decoded_circuit_text(self, check_results: Sequence[int], verbose: bool = False) -> str:
        """Return a textual decoded-circuit dump for debugging invalid probability tuples."""
        checks = [int(v) for v in check_results]
        return str(self._cpp_decoder.debug_decoded_circuit_text(checks, bool(verbose)))


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
