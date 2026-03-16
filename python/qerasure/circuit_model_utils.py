"""Python wrappers for circuit-model compile + stream sampling."""

from __future__ import annotations

from dataclasses import dataclass
import re
import sys
from typing import Callable, Mapping, Optional, Sequence

import numpy as np

from ._bindings import cpp

OpCode = cpp.OpCode
PauliChannel = cpp.PauliChannel
TQGSpreadModel = cpp.TQGSpreadModel
SurfaceCodeRotated = cpp.SurfaceCodeRotated
_CppErasureModel = cpp.ErasureModel
_UINT32_MAX = (1 << 32) - 1
_STIM_TEXT_FALLBACK_WARNED = False


def _normalize_u32_seed(seed: int) -> int:
    """Coerce Python ints into the uint32 range expected by C++ samplers."""
    return int(seed) & _UINT32_MAX


def _validate_probability(value: float, *, name: str) -> float:
    p = float(value)
    if p < 0.0 or p > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {p}.")
    return p


def _pauli_channel_from_components(px: float, py: float, pz: float, *, slot_name: str) -> PauliChannel:
    p_x = _validate_probability(px, name=f"{slot_name}.p_x")
    p_y = _validate_probability(py, name=f"{slot_name}.p_y")
    p_z = _validate_probability(pz, name=f"{slot_name}.p_z")
    if p_x + p_y + p_z > 1.0 + 1e-12:
        raise ValueError(
            f"{slot_name} channel is invalid: p_x + p_y + p_z must be <= 1. "
            f"Got {p_x + p_y + p_z}."
        )
    return PauliChannel(p_x, p_y, p_z)


def _parse_single_float(text: str, *, slot_name: str, op_name: str) -> float:
    try:
        return _validate_probability(float(text), name=f"{slot_name}:{op_name}")
    except ValueError:
        raise ValueError(f"Invalid probability '{text}' in {slot_name} spec '{op_name}(... )'.")


def _parse_channel_spec(spec: object, *, slot_name: str) -> PauliChannel:
    if spec is None:
        return PauliChannel()
    if isinstance(spec, PauliChannel):
        return _pauli_channel_from_components(spec.p_x, spec.p_y, spec.p_z, slot_name=slot_name)
    if isinstance(spec, (tuple, list)) and len(spec) == 3:
        return _pauli_channel_from_components(spec[0], spec[1], spec[2], slot_name=slot_name)
    if not isinstance(spec, str):
        raise TypeError(
            f"{slot_name} must be one of: string spec, PauliChannel, or length-3 tuple/list. "
            f"Got type {type(spec).__name__}."
        )

    text = spec.strip()
    match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)\((.*)\)", text)
    if match is None:
        raise ValueError(
            f"Invalid channel spec for {slot_name}: '{spec}'. "
            "Expected forms like X_ERROR(p), Z_ERROR(p), DEPOLARIZE1(p), PAULI_CHANNEL(px,py,pz)."
        )
    op_name = match.group(1).upper()
    args_text = match.group(2).strip()
    parts = [part.strip() for part in args_text.split(",")] if args_text else []

    if op_name == "PAULI_CHANNEL":
        if len(parts) != 3:
            raise ValueError(
                f"{slot_name} PAULI_CHANNEL must have 3 args, got {len(parts)} in '{spec}'."
            )
        try:
            px, py, pz = (float(parts[0]), float(parts[1]), float(parts[2]))
        except ValueError:
            raise ValueError(f"Invalid PAULI_CHANNEL args in '{spec}'.")
        return _pauli_channel_from_components(px, py, pz, slot_name=slot_name)

    if len(parts) != 1:
        raise ValueError(f"{slot_name} {op_name} must have 1 arg, got {len(parts)} in '{spec}'.")
    p = _parse_single_float(parts[0], slot_name=slot_name, op_name=op_name)

    if op_name == "X_ERROR":
        return _pauli_channel_from_components(p, 0.0, 0.0, slot_name=slot_name)
    if op_name == "Y_ERROR":
        return _pauli_channel_from_components(0.0, p, 0.0, slot_name=slot_name)
    if op_name == "Z_ERROR":
        return _pauli_channel_from_components(0.0, 0.0, p, slot_name=slot_name)
    if op_name == "DEPOLARIZE1":
        return _pauli_channel_from_components(p / 3.0, p / 3.0, p / 3.0, slot_name=slot_name)

    raise ValueError(
        f"Unsupported channel op '{op_name}' in {slot_name}. "
        "Supported: PAULI_CHANNEL, X_ERROR, Y_ERROR, Z_ERROR, DEPOLARIZE1."
    )


def _apply_check_probabilities(
    model: "ErasureModel",
    *,
    check_error_prob: float | None,
    check_false_negative_prob: float | None,
    check_false_positive_prob: float | None,
) -> None:
    if check_error_prob is not None:
        q = _validate_probability(check_error_prob, name="check_error_prob")
        model.check_error_prob = q
    if check_false_negative_prob is not None:
        model.check_false_negative_prob = _validate_probability(
            check_false_negative_prob, name="check_false_negative_prob"
        )
    if check_false_positive_prob is not None:
        model.check_false_positive_prob = _validate_probability(
            check_false_positive_prob, name="check_false_positive_prob"
        )


class ErasureModel:
    """User-facing wrapper for `cpp.ErasureModel`.

    Supports two styles:
    1. Legacy constructor forwarding to the C++ constructor.
    2. `from_specs(...)` for named channel slots using string specs.
    """

    def __init__(self, *args, **kwargs):
        self._cpp_model = _CppErasureModel(*args, **kwargs)

    @classmethod
    def from_specs(
        cls,
        *,
        max_persistence: int = _UINT32_MAX,
        channels: Optional[Mapping[str, object]] = None,
        onset: object = None,
        reset: object = None,
        spread_control: object = None,
        spread_target: object = None,
        check_error_prob: float | None = None,
        check_false_negative_prob: float | None = None,
        check_false_positive_prob: float | None = None,
    ) -> "ErasureModel":
        allowed = {"onset", "reset", "spread_control", "spread_target"}
        channels = dict(channels or {})
        unknown_keys = set(channels.keys()) - allowed
        if unknown_keys:
            keys = ", ".join(sorted(unknown_keys))
            raise ValueError(
                f"Unknown channel slot(s): {keys}. Expected one of: onset, reset, "
                "spread_control, spread_target."
            )

        onset_spec = onset if onset is not None else channels.get("onset")
        reset_spec = reset if reset is not None else channels.get("reset")
        spread_control_spec = (
            spread_control if spread_control is not None else channels.get("spread_control")
        )
        spread_target_spec = spread_target if spread_target is not None else channels.get("spread_target")

        onset_channel = _parse_channel_spec(onset_spec, slot_name="onset")
        reset_channel = _parse_channel_spec(reset_spec, slot_name="reset")
        spread_control_channel = _parse_channel_spec(spread_control_spec, slot_name="spread_control")
        spread_target_channel = _parse_channel_spec(spread_target_spec, slot_name="spread_target")

        model = cls(
            int(max_persistence),
            onset_channel,
            reset_channel,
            spread_control_channel,
            spread_target_channel,
        )
        _apply_check_probabilities(
            model,
            check_error_prob=check_error_prob,
            check_false_negative_prob=check_false_negative_prob,
            check_false_positive_prob=check_false_positive_prob,
        )
        return model

    def _to_cpp_model(self):
        return self._cpp_model

    @property
    def max_persistence(self) -> int:
        return int(self._cpp_model.max_persistence)

    @max_persistence.setter
    def max_persistence(self, value: int) -> None:
        self._cpp_model.max_persistence = int(value)

    @property
    def onset(self):
        return self._cpp_model.onset

    @onset.setter
    def onset(self, value) -> None:
        self._cpp_model.onset = value

    @property
    def reset(self):
        return self._cpp_model.reset

    @reset.setter
    def reset(self, value) -> None:
        self._cpp_model.reset = value

    @property
    def spread(self):
        return self._cpp_model.spread

    @spread.setter
    def spread(self, value) -> None:
        self._cpp_model.spread = value

    @property
    def check_false_negative_prob(self) -> float:
        return float(self._cpp_model.check_false_negative_prob)

    @check_false_negative_prob.setter
    def check_false_negative_prob(self, value: float) -> None:
        self._cpp_model.check_false_negative_prob = _validate_probability(
            value, name="check_false_negative_prob"
        )

    @property
    def check_false_positive_prob(self) -> float:
        return float(self._cpp_model.check_false_positive_prob)

    @check_false_positive_prob.setter
    def check_false_positive_prob(self, value: float) -> None:
        self._cpp_model.check_false_positive_prob = _validate_probability(
            value, name="check_false_positive_prob"
        )

    @property
    def check_error_prob(self) -> float | None:
        fn = self.check_false_negative_prob
        fp = self.check_false_positive_prob
        if abs(fn - fp) < 1e-15:
            return fn
        return None

    @check_error_prob.setter
    def check_error_prob(self, value: float) -> None:
        q = _validate_probability(value, name="check_error_prob")
        self._cpp_model.check_false_negative_prob = q
        self._cpp_model.check_false_positive_prob = q

    def explain(self) -> str:
        spread = self._cpp_model.spread
        return (
            "ErasureModel(\n"
            f"  max_persistence={int(self._cpp_model.max_persistence)},\n"
            f"  onset=PAULI_CHANNEL({self._cpp_model.onset.p_x}, {self._cpp_model.onset.p_y}, {self._cpp_model.onset.p_z}),\n"
            f"  reset=PAULI_CHANNEL({self._cpp_model.reset.p_x}, {self._cpp_model.reset.p_y}, {self._cpp_model.reset.p_z}),\n"
            f"  spread_control=PAULI_CHANNEL({spread.control_spread.p_x}, {spread.control_spread.p_y}, {spread.control_spread.p_z}),\n"
            f"  spread_target=PAULI_CHANNEL({spread.target_spread.p_x}, {spread.target_spread.p_y}, {spread.target_spread.p_z}),\n"
            f"  check_false_negative_prob={self._cpp_model.check_false_negative_prob},\n"
            f"  check_false_positive_prob={self._cpp_model.check_false_positive_prob}\n"
            ")"
        )

    def __repr__(self) -> str:
        return self.explain()


def make_erasure_model(
    *,
    max_persistence: int = _UINT32_MAX,
    channels: Optional[Mapping[str, object]] = None,
    onset: object = None,
    reset: object = None,
    spread_control: object = None,
    spread_target: object = None,
    check_error_prob: float | None = None,
    check_false_negative_prob: float | None = None,
    check_false_positive_prob: float | None = None,
) -> ErasureModel:
    """Build an ErasureModel using named channel slots and string specs."""
    return ErasureModel.from_specs(
        max_persistence=max_persistence,
        channels=channels,
        onset=onset,
        reset=reset,
        spread_control=spread_control,
        spread_target=spread_target,
        check_error_prob=check_error_prob,
        check_false_negative_prob=check_false_negative_prob,
        check_false_positive_prob=check_false_positive_prob,
    )


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
    model: ErasureModel | object

    def __post_init__(self):
        cpp_circuit = (
            self.circuit._to_cpp_circuit() if isinstance(self.circuit, ErasureCircuit) else self.circuit
        )
        cpp_model = self.model._to_cpp_model() if isinstance(self.model, ErasureModel) else self.model
        self._cpp_program = cpp.CompiledErasureProgram(cpp_circuit, cpp_model)

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
        return self._cpp_sampler.sample_syndromes(
            shots,
            _normalize_u32_seed(seed),
            threads,
        )

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
            return self._cpp_sampler.sample_with_callback(
                shots,
                _normalize_u32_seed(seed),
                None,
                threads,
            )

        def wrapped(circuit_obj, check_flags):
            callback(circuit_obj, np.asarray(check_flags, dtype=np.uint8))

        return self._cpp_sampler.sample_with_callback(
            shots,
            _normalize_u32_seed(seed),
            wrapped,
            threads,
        )

    def sample_exact_shot(self, seed: int, shot: int):
        """Reconstruct one exact shot from the `sample_syndromes` stream."""
        circuit_text, check_flags = self._cpp_sampler.sample_exact_shot(
            _normalize_u32_seed(seed),
            int(shot),
        )
        return circuit_text, np.asarray(check_flags, dtype=np.uint8)


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
        # Keep stim imported so pybind can resolve and return native stim.Circuit.
        _ = stim
        try:
            return self._cpp_builder.build_decoded_circuit(checks, bool(verbose))
        except TypeError as exc:
            global _STIM_TEXT_FALLBACK_WARNED
            message = str(exc)
            if "stim::Circuit" not in message and "Unable to convert function return value" not in message:
                raise
            if not _STIM_TEXT_FALLBACK_WARNED:
                print(
                    "WARNING: qerasure fell back to stringified decoded-circuit conversion "
                    "because native stim::Circuit cast failed; this will lose performance.",
                    file=sys.stderr,
                )
                _STIM_TEXT_FALLBACK_WARNED = True
            return stim.Circuit(str(self._cpp_builder.build_decoded_circuit_text(checks, bool(verbose))))

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
        self._cpp_program = cpp_program

    def _compute_shots_per_chunk(self, num_detectors: int, num_checks: int) -> int:
        packed_check_bytes = (int(num_checks) + 7) // 8
        # Detector row + packed-check key + prediction row + grouping/index overhead.
        bytes_per_shot = int(num_detectors) + packed_check_bytes + 1 + 32
        return max(1, self._max_batch_bytes // max(1, bytes_per_shot))

    def _decode_group_with_matching(self, group_detectors, matching):
        """Decode one detector-group using a prebuilt matching graph."""
        import numpy as np

        if hasattr(matching, "decode_batch"):
            group_preds = np.asarray(matching.decode_batch(group_detectors), dtype=np.uint8)
            if group_preds.ndim == 1:
                group_preds = group_preds[:, None]
            return group_preds

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
        return group_preds

    def _group_check_patterns(self, check_chunk):
        """Group identical check rows and return (group_indices, unique_check_rows)."""
        import numpy as np

        shots = int(check_chunk.shape[0])
        if shots == 0:
            return [], np.zeros((0, int(check_chunk.shape[1])), dtype=np.uint8)

        packed = np.packbits(check_chunk, axis=1, bitorder="little")
        unique_packed, inverse = np.unique(packed, axis=0, return_inverse=True)
        num_groups = int(unique_packed.shape[0])

        if check_chunk.shape[1] == 0:
            unique_checks = np.zeros((num_groups, 0), dtype=np.uint8)
        else:
            unique_checks = np.unpackbits(unique_packed, axis=1, bitorder="little")[
                :, : check_chunk.shape[1]
            ].astype(np.uint8, copy=False)

        order = np.argsort(inverse, kind="stable")
        counts = np.bincount(inverse, minlength=num_groups)
        boundaries = np.cumsum(counts[:-1], dtype=np.int64)
        group_indices = np.split(order.astype(np.int64, copy=False), boundaries)
        return group_indices, unique_checks

    def _decode_chunk_grouped(self, det_chunk, check_chunk, dem_builder):
        """Decode one shot-chunk by grouping identical check patterns.

        Decoding steps inside this function:
        1. Pack check rows and group shots with identical check flags.
        2. Build one decoded circuit/DEM/matching graph per unique check pattern.
        3. Decode all detector rows in that group with the shared matching graph.
        4. Scatter group predictions back into original shot order for the chunk.
        """
        import numpy as np
        import pymatching as pm

        shots = int(det_chunk.shape[0])
        if shots == 0:
            return np.zeros((0, 0), dtype=np.uint8)

        group_indices, group_check_rows = self._group_check_patterns(check_chunk)

        predictions = None
        for group_id, local_indices in enumerate(group_indices):
            check_row = group_check_rows[group_id]
            decoded_circuit = dem_builder.build_decoded_circuit(check_row, verbose=False)
            decoded_dem = decoded_circuit.detector_error_model(
                decompose_errors=True,
                approximate_disjoint_errors=True,
            )
            matching = pm.Matching.from_detector_error_model(decoded_dem)
            group_detectors = det_chunk[local_indices]
            group_preds = self._decode_group_with_matching(group_detectors, matching)

            if predictions is None:
                predictions = np.zeros((shots, max(1, int(group_preds.shape[1]))), dtype=np.uint8)
            elif group_preds.shape[1] > predictions.shape[1]:
                grown = np.zeros((shots, int(group_preds.shape[1])), dtype=np.uint8)
                grown[:, : predictions.shape[1]] = predictions
                predictions = grown

            n = min(predictions.shape[1], group_preds.shape[1])
            predictions[local_indices, :n] = group_preds[:, :n]

        if predictions is None:
            return np.zeros((shots, 0), dtype=np.uint8)
        return predictions

    def _decode_chunk_grouped_parallel(self, det_chunk, check_chunk, num_threads: int):
        """Decode one shot-chunk by grouping once, then decoding groups in parallel."""
        import numpy as np
        import pymatching as pm
        import threading
        from concurrent.futures import ThreadPoolExecutor

        shots = int(det_chunk.shape[0])
        if shots == 0:
            return np.zeros((0, 0), dtype=np.uint8)

        group_indices, group_check_rows = self._group_check_patterns(check_chunk)
        num_groups = len(group_indices)
        workers = min(max(1, int(num_threads)), num_groups)
        if workers <= 1:
            return self._decode_chunk_grouped(det_chunk, check_chunk, self._dem_builder)

        thread_state = threading.local()

        def worker_decode(group_id: int):
            dem_builder = getattr(thread_state, "dem_builder", None)
            if dem_builder is None:
                dem_builder = SurfDemBuilder(self._cpp_program)
                thread_state.dem_builder = dem_builder

            check_row = group_check_rows[group_id]
            decoded_circuit = dem_builder.build_decoded_circuit(check_row, verbose=False)
            decoded_dem = decoded_circuit.detector_error_model(
                decompose_errors=True,
                approximate_disjoint_errors=True,
            )
            matching = pm.Matching.from_detector_error_model(decoded_dem)
            local_indices = group_indices[group_id]
            group_detectors = det_chunk[local_indices]
            group_preds = self._decode_group_with_matching(group_detectors, matching)
            return group_id, group_preds

        predictions = None
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for group_id, group_preds in pool.map(worker_decode, range(num_groups)):
                if predictions is None:
                    predictions = np.zeros((shots, max(1, int(group_preds.shape[1]))), dtype=np.uint8)
                elif group_preds.shape[1] > predictions.shape[1]:
                    grown = np.zeros((shots, int(group_preds.shape[1])), dtype=np.uint8)
                    grown[:, : predictions.shape[1]] = predictions
                    predictions = grown

                local_indices = group_indices[group_id]
                n = min(predictions.shape[1], group_preds.shape[1])
                predictions[local_indices, :n] = group_preds[:, :n]

        if predictions is None:
            return np.zeros((shots, 0), dtype=np.uint8)
        return predictions

    def _decode_range(
        self,
        dets,
        checks,
        start: int,
        end: int,
        shots_per_chunk: int,
        dem_builder=None,
    ):
        """Decode a contiguous shot-range, chunking internally for memory limits."""
        import numpy as np

        if dem_builder is None:
            dem_builder = SurfDemBuilder(self._cpp_program)
        range_shots = end - start
        if range_shots <= 0:
            return np.zeros((0, 0), dtype=np.uint8)

        range_preds = None
        local_write = 0
        for chunk_start in range(start, end, shots_per_chunk):
            chunk_end = min(end, chunk_start + shots_per_chunk)
            det_chunk = dets[chunk_start:chunk_end]
            check_chunk = checks[chunk_start:chunk_end]
            chunk_preds = self._decode_chunk_grouped(det_chunk, check_chunk, dem_builder)

            if range_preds is None:
                range_preds = np.zeros(
                    (range_shots, max(1, int(chunk_preds.shape[1]))), dtype=np.uint8
                )
            elif chunk_preds.shape[1] > range_preds.shape[1]:
                grown = np.zeros((range_shots, int(chunk_preds.shape[1])), dtype=np.uint8)
                grown[:, : range_preds.shape[1]] = range_preds
                range_preds = grown

            span = chunk_end - chunk_start
            n = min(range_preds.shape[1], chunk_preds.shape[1])
            range_preds[local_write : local_write + span, :n] = chunk_preds[:, :n]
            local_write += span

        if range_preds is None:
            return np.zeros((range_shots, 0), dtype=np.uint8)
        return range_preds

    def decode_batch(self, detector_samples, check_flags, num_threads: int = 1):
        """Decode a full batch of shots."""
        import numpy as np

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
        if int(num_threads) <= 0:
            raise ValueError("num_threads must be positive.")

        shots = int(dets.shape[0])
        if shots == 0:
            return np.zeros((0, 0), dtype=np.uint8)

        shots_per_chunk = self._compute_shots_per_chunk(dets.shape[1], checks.shape[1])
        predictions = None
        write_offset = 0
        for chunk_start in range(0, shots, shots_per_chunk):
            chunk_end = min(shots, chunk_start + shots_per_chunk)
            det_chunk = dets[chunk_start:chunk_end]
            check_chunk = checks[chunk_start:chunk_end]

            if int(num_threads) == 1:
                chunk_preds = self._decode_chunk_grouped(
                    det_chunk, check_chunk, self._dem_builder
                )
            else:
                chunk_preds = self._decode_chunk_grouped_parallel(
                    det_chunk, check_chunk, int(num_threads)
                )

            if predictions is None:
                predictions = np.zeros((shots, max(1, int(chunk_preds.shape[1]))), dtype=np.uint8)
            elif chunk_preds.shape[1] > predictions.shape[1]:
                grown = np.zeros((shots, int(chunk_preds.shape[1])), dtype=np.uint8)
                grown[:, : predictions.shape[1]] = predictions
                predictions = grown

            span = chunk_end - chunk_start
            n = min(predictions.shape[1], chunk_preds.shape[1])
            predictions[write_offset : write_offset + span, :n] = chunk_preds[:, :n]
            write_offset += span

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
    ecr_after_each_step: bool = False,
    single_qubit_errors: bool = False,
    post_clifford_pauli_prob: float = 0.0,
) -> ErasureCircuit:
    """Build a rotated-surface-code erasure circuit using the C++ generator."""
    generator = SurfaceCodeRotated(int(distance))
    cpp_circuit = generator.build_circuit(
        int(rounds),
        float(erasure_prob),
        str(erasable_qubits),
        float(reset_failure_prob),
        bool(ecr_after_each_step),
        bool(single_qubit_errors),
        float(post_clifford_pauli_prob),
    )
    wrapped = ErasureCircuit()
    wrapped._cpp_circuit = cpp_circuit
    return wrapped
