"""Python wrappers and visualization helpers for lowering events."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Sequence

from ._bindings import cpp
from .sim_utils import ErasureSimParams, ErasureSimResult, _normalize_qubit_subset

PauliError = cpp.PauliError
LoweredEventOrigin = cpp.LoweredEventOrigin
PartnerSlot = cpp.PartnerSlot
try:
    SpreadInstructionType = cpp.SpreadInstructionType
except AttributeError:
    # Backward compatibility when Python code is newer than compiled extension.
    class SpreadInstructionType(IntEnum):
        X_ERROR = 0
        Y_ERROR = 1
        Z_ERROR = 2
        DEPOLARIZE1 = 3
        COND_X_ERROR = 4
        COND_Y_ERROR = 5
        COND_Z_ERROR = 6
        ELSE_X_ERROR = 7
        ELSE_Y_ERROR = 8
        ELSE_Z_ERROR = 9


@dataclass
class LoweredErrorParams:
    """Python facade around C++ LoweredErrorParams."""

    error_type: object
    probability: float

    def __init__(self, error_type: object = PauliError.NO_ERROR, probability: float = 0.0):
        # Accept legacy-swapped positional usage: LoweredErrorParams(probability, error_type).
        if isinstance(error_type, (float, int)) and probability in (
            PauliError.NO_ERROR,
            PauliError.X_ERROR,
            PauliError.Z_ERROR,
            PauliError.Y_ERROR,
            PauliError.DEPOLARIZE,
        ):
            error_type, probability = probability, error_type
        self.error_type = error_type
        self.probability = float(probability)

    def _to_cpp(self):
        params = cpp.LoweredErrorParams()
        params.error_type = self.error_type
        params.probability = float(self.probability)
        return params


@dataclass
class SpreadTargetOp:
    error_type: object
    slot: object

    def __init__(self, error_type: object, slot: object):
        self.error_type = error_type
        self.slot = slot

    def _to_cpp(self):
        return cpp.SpreadTargetOp(self.error_type, self.slot)


@dataclass
class SpreadInstruction:
    """Python representation of one spread-program instruction."""

    type: object
    probability: float
    target: SpreadTargetOp | None

    @classmethod
    def from_cpp(cls, cpp_instruction) -> "SpreadInstruction":
        target = SpreadTargetOp(
            error_type=PauliError.NO_ERROR,
            slot=cpp_instruction.target_slot,
        )
        return cls(
            type=cpp_instruction.type,
            probability=float(cpp_instruction.probability),
            target=target,
        )


@dataclass
class SpreadProgram:
    """Stim-like lowering instruction program."""

    _cpp_program: object

    def __init__(self):
        self._cpp_program = cpp.SpreadProgram()

    def append(self, stim_like_program: str) -> None:
        """Append one or many semicolon-separated Stim-like instruction strings."""
        self._cpp_program.append(str(stim_like_program))

    @staticmethod
    def _normalize_target(target: object | None) -> object | None:
        if target is None:
            raise ValueError("Each spread instruction must include exactly one target.")
        if hasattr(target, "slot"):
            return target.slot
        if isinstance(target, (list, tuple)):
            if len(target) > 1:
                raise ValueError("Each spread instruction must have at most one target.")
            if len(target) == 0:
                raise ValueError("Each spread instruction must include exactly one target.")
            first = target[0]
            return first.slot if hasattr(first, "slot") else first
        return target

    def add_instruction(self, instruction_type: object, probability: float, target: object) -> None:
        normalized_target = self._normalize_target(target)
        self._cpp_program.add_instruction(instruction_type, float(probability), normalized_target)

    def add_x_error(self, probability: float, target: object) -> None:
        self._cpp_program.add_x_error(float(probability), self._normalize_target(target))

    def add_y_error(self, probability: float, target: object) -> None:
        self._cpp_program.add_y_error(float(probability), self._normalize_target(target))

    def add_z_error(self, probability: float, target: object) -> None:
        self._cpp_program.add_z_error(float(probability), self._normalize_target(target))

    def add_depolarize1(self, probability: float, target: object) -> None:
        self._cpp_program.add_depolarize1(float(probability), self._normalize_target(target))

    def add_cond_x_error(self, probability: float, target: object) -> None:
        self._cpp_program.add_cond_x_error(float(probability), self._normalize_target(target))

    def add_cond_y_error(self, probability: float, target: object) -> None:
        self._cpp_program.add_cond_y_error(float(probability), self._normalize_target(target))

    def add_cond_z_error(self, probability: float, target: object) -> None:
        self._cpp_program.add_cond_z_error(float(probability), self._normalize_target(target))

    def add_else_x_error(self, probability: float, target: object) -> None:
        self._cpp_program.add_else_x_error(float(probability), self._normalize_target(target))

    def add_else_y_error(self, probability: float, target: object) -> None:
        self._cpp_program.add_else_y_error(float(probability), self._normalize_target(target))

    def add_else_z_error(self, probability: float, target: object) -> None:
        self._cpp_program.add_else_z_error(float(probability), self._normalize_target(target))

    # Backward-compatible helpers.
    def add_error_channel(self, probability: float, targets: list[SpreadTargetOp]) -> None:
        self._cpp_program.add_error_channel(float(probability), [t._to_cpp() for t in targets])

    def add_correlated_error(self, probability: float, targets: list[SpreadTargetOp]) -> None:
        self._cpp_program.add_correlated_error(float(probability), [t._to_cpp() for t in targets])

    def add_else_correlated_error(self, probability: float, targets: list[SpreadTargetOp]) -> None:
        self._cpp_program.add_else_correlated_error(float(probability), [t._to_cpp() for t in targets])

    @property
    def instructions(self) -> list[SpreadInstruction]:
        """Return the program instructions with type, probability, and target operations."""
        return [SpreadInstruction.from_cpp(inst) for inst in self._cpp_program.instructions]

    def _to_cpp(self):
        return self._cpp_program


@dataclass
class LoweringParams:
    """Python facade around C++ LoweringParams."""

    reset_params: LoweredErrorParams
    x_ancillas: object | None = None
    z_ancillas: object | None = None
    default_program: SpreadProgram | None = None
    _cpp_params: object | None = None

    def __init__(
        self,
        reset_params: LoweredErrorParams | SpreadProgram | None = None,
        x_ancillas: object | LoweredErrorParams | None = None,
        z_ancillas: object | None = None,
        default_program: SpreadProgram | None = None,
    ):
        # Accept intuitive positional form: LoweringParams(program, reset_params).
        if isinstance(reset_params, SpreadProgram):
            default_program = reset_params
            reset_params = (
                x_ancillas
                if isinstance(x_ancillas, LoweredErrorParams)
                else LoweredErrorParams(PauliError.NO_ERROR, 0.0)
            )
            x_ancillas = None
            z_ancillas = None

        self.reset_params = (
            reset_params if reset_params is not None else LoweredErrorParams(PauliError.NO_ERROR, 0.0)
        )
        self.x_ancillas = x_ancillas
        self.z_ancillas = z_ancillas
        self.default_program = default_program

        reset_cpp = self.reset_params._to_cpp()
        if default_program is not None:
            self._cpp_params = cpp.LoweringParams(default_program._to_cpp(), reset_cpp)
            return

        if isinstance(self.x_ancillas, tuple):
            x_cpp = (self.x_ancillas[0]._to_cpp(), self.x_ancillas[1]._to_cpp())
            z_cpp = (self.z_ancillas[0]._to_cpp(), self.z_ancillas[1]._to_cpp())
            self._cpp_params = cpp.LoweringParams(reset_cpp, x_cpp, z_cpp)
            return
        if self.z_ancillas is None and self.x_ancillas is not None:
            self._cpp_params = cpp.LoweringParams(reset_cpp, self.x_ancillas._to_cpp())
            return
        if self.x_ancillas is not None and self.z_ancillas is not None:
            self._cpp_params = cpp.LoweringParams(
                reset_cpp, self.x_ancillas._to_cpp(), self.z_ancillas._to_cpp()
            )
            return

        # Default to an empty program configuration for concise construction.
        self._cpp_params = cpp.LoweringParams(cpp.SpreadProgram(), reset_cpp)

    @classmethod
    def with_program(
        cls,
        program: SpreadProgram,
        reset_params: LoweredErrorParams | None = None,
    ) -> "LoweringParams":
        """Preferred construction path for Stim-like lowering programs."""
        return cls(reset_params=reset_params, default_program=program)

    def _to_cpp(self):
        return self._cpp_params

    def set_default_data_program(self, program: SpreadProgram) -> None:
        self._cpp_params.set_default_data_program(program._to_cpp())

    def set_data_qubit_program(self, data_qubit_idx: int, program: SpreadProgram) -> None:
        self._cpp_params.set_data_qubit_program(int(data_qubit_idx), program._to_cpp())


@dataclass
class LoweringResult:
    qec_rounds: int
    sparse_cliffords: list
    clifford_timestep_offsets: list
    check_error_round_flags: list
    erasure_round_flags: list
    reset_round_qubits: list
    _cpp_result: object | None = None

    @classmethod
    def from_cpp(cls, cpp_result) -> "LoweringResult":
        return cls(
            qec_rounds=int(getattr(cpp_result, "qec_rounds", 0)),
            sparse_cliffords=cpp_result.sparse_cliffords,
            clifford_timestep_offsets=cpp_result.clifford_timestep_offsets,
            check_error_round_flags=getattr(cpp_result, "check_error_round_flags", []),
            erasure_round_flags=getattr(cpp_result, "erasure_round_flags", []),
            reset_round_qubits=getattr(cpp_result, "reset_round_qubits", []),
            _cpp_result=cpp_result,
        )


class Lowerer:
    """High-level Python lowering wrapper using C++ backend."""

    def __init__(self, code, params: LoweringParams):
        self._cpp_lowerer = cpp.Lowerer(code._to_cpp_code(), params._to_cpp())

    def lower(self, sim_result: ErasureSimResult) -> LoweringResult:
        cpp_result = getattr(sim_result, "_cpp_result", sim_result)
        return LoweringResult.from_cpp(self._cpp_lowerer.lower(cpp_result))


def visualize_lowering(
    lowering_result: LoweringResult,
    sim_params: ErasureSimParams,
    shot_idx: int = 0,
    qubits: Optional[Sequence[int]] = None,
    show_round_guides: bool = True,
):
    """Render lowering timeline heatmap plus per-timestep event counts for one shot."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import BoundaryNorm, ListedColormap

    if shot_idx < 0 or shot_idx >= len(lowering_result.sparse_cliffords):
        raise ValueError(f"shot_idx must be in [0, {len(lowering_result.sparse_cliffords) - 1}]")

    selected_qubits = _normalize_qubit_subset(sim_params.code.num_qubits, qubits)
    qubit_to_col = {q: i for i, q in enumerate(selected_qubits)}

    num_timesteps = sim_params.qec_rounds * 4 + 1
    matrix = np.zeros((num_timesteps, len(selected_qubits)), dtype=np.uint8)

    events = lowering_result.sparse_cliffords[shot_idx]
    offsets = lowering_result.clifford_timestep_offsets[shot_idx]
    if len(offsets) != num_timesteps + 1:
        raise ValueError(
            f"Lowering offsets length {len(offsets)} does not match expected {num_timesteps + 1}"
        )

    per_timestep_counts = np.zeros(num_timesteps, dtype=np.int32)

    for t in range(num_timesteps):
        start = offsets[t]
        end = offsets[t + 1]
        per_timestep_counts[t] = end - start

        for event in events[start:end]:
            q = int(event.qubit_idx)
            if q not in qubit_to_col:
                continue
            col = qubit_to_col[q]
            if event.error_type == PauliError.X_ERROR:
                matrix[t, col] = 1
            elif event.error_type == PauliError.Z_ERROR:
                matrix[t, col] = 2
            elif event.error_type == PauliError.Y_ERROR:
                matrix[t, col] = 3
            elif event.error_type == PauliError.DEPOLARIZE:
                matrix[t, col] = 4

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        gridspec_kw={"height_ratios": [4, 1]},
        sharex=True,
    )

    cmap = ListedColormap(["#f8fafc", "#2563eb", "#dc2626", "#16a34a", "#0f172a"])
    norm = BoundaryNorm([0, 1, 2, 3, 4, 5], cmap.N)
    im = ax0.imshow(matrix.T, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    if show_round_guides:
        for t in range(0, num_timesteps, 4):
            ax0.axvline(t - 0.5, color="#111827", linestyle="--", linewidth=0.6, alpha=0.7)
            ax1.axvline(t - 0.5, color="#111827", linestyle="--", linewidth=0.6, alpha=0.7)

    cbar = fig.colorbar(im, ax=ax0, ticks=[0.5, 1.5, 2.5, 3.5, 4.5], pad=0.01)
    cbar.ax.set_yticklabels(["No event", "X", "Z", "Y", "Depolarize"])

    ax0.set_title(f"Lowering Timeline (shot={shot_idx}, qubits={len(selected_qubits)})")
    ax0.set_ylabel("Qubit index (subset order)")

    ax1.plot(np.arange(num_timesteps), per_timestep_counts, color="#0f172a", linewidth=1.5)
    ax1.fill_between(np.arange(num_timesteps), per_timestep_counts, color="#cbd5e1", alpha=0.6)
    ax1.set_ylabel("Events")
    ax1.set_xlabel("Timestep")

    fig.tight_layout()
    return fig, (ax0, ax1)
