"""Python wrappers for circuit-model compile + stream sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, Sequence

from ._bindings import cpp

if TYPE_CHECKING:
    import numpy as np

OpCode = cpp.OpCode
PauliChannel = cpp.PauliChannel
TQGSpreadModel = cpp.TQGSpreadModel
ErasureModel = cpp.ErasureModel


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


def _estimate_output_bytes(num_shots: int, num_detectors: int, num_observables: int, num_checks: int) -> int:
    return int(num_shots) * int(num_detectors + num_observables + num_checks)


class StreamSampler:
    """Python stream sampler with optional default Stim detector-sampler callback."""

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
        callback: Optional[Callable[[object, object], None]] = None,
        max_output_bytes: int = 2 * 1024 * 1024 * 1024,
    ):
        import numpy as np

        shots = int(num_shots)
        if shots < 0:
            raise ValueError("num_shots must be non-negative")
        if int(num_threads) != 1:
            raise ValueError(
                "Python StreamSampler.sample currently requires num_threads=1 because callbacks run in Python."
            )

        if callback is not None:
            def wrapped(circuit_obj, check_flags):
                callback(circuit_obj, np.asarray(check_flags, dtype=np.uint8))

            self._cpp_sampler.sample(shots, int(seed), wrapped, 1)
            return None

        try:
            import stim  # noqa: F401
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The Python `stim` package is required for StreamSampler default callback mode."
            ) from exc

        checks_out = np.zeros((shots, self._num_checks), dtype=np.uint8)
        if shots == 0:
            return (
                np.zeros((0, 0), dtype=np.uint8),
                np.zeros((0, 0), dtype=np.uint8),
                checks_out,
            )

        dets_out = None
        obs_out = None
        shot_index = 0

        def default_callback(circuit_obj, check_flags):
            nonlocal shot_index, dets_out, obs_out
            det_sampler = circuit_obj.compile_detector_sampler()
            dets, obs = det_sampler.sample(shots=1, separate_observables=True)
            det_row = np.asarray(dets[0], dtype=np.uint8)
            obs_row = np.asarray(obs[0], dtype=np.uint8)

            if dets_out is None:
                estimated = _estimate_output_bytes(shots, det_row.size, obs_row.size, self._num_checks)
                if estimated > int(max_output_bytes):
                    raise MemoryError(
                        f"Requested output arrays need {estimated} bytes, exceeding max_output_bytes={max_output_bytes}."
                    )
                dets_out = np.empty((shots, det_row.size), dtype=np.uint8)
                obs_out = np.empty((shots, obs_row.size), dtype=np.uint8)

            dets_out[shot_index, :] = det_row
            obs_out[shot_index, :] = obs_row
            checks_out[shot_index, :] = np.asarray(check_flags, dtype=np.uint8)
            shot_index += 1

        self._cpp_sampler.sample(shots, int(seed), default_callback, 1)
        if dets_out is None or obs_out is None:
            raise RuntimeError("StreamSampler returned no shots unexpectedly.")
        return dets_out, obs_out, checks_out


def compile_erasure_sampler(
    circuit: ErasureCircuit | object,
    model: ErasureModel,
) -> StreamSampler:
    """Convenience helper: circuit + model -> compiled program -> stream sampler."""
    program = CompiledErasureProgram(circuit=circuit, model=model)
    return StreamSampler(program)
