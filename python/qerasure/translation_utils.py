"""Python wrappers for Stim-circuit translation utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._bindings import cpp
from .code_utils import RotatedSurfaceCode
from .lowering_utils import LoweringResult

if TYPE_CHECKING:
    import stim


def _build_stim_circuit_from_text(circuit_text: str) -> "stim.Circuit":
    """Parse a Stim-format circuit string into a `stim.Circuit` object."""
    try:
        import stim
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The Python `stim` package is required for circuit-object builders. "
            "Install/import `stim` to use this function."
        ) from exc
    return stim.Circuit(circuit_text)


def build_surf_stabilizer_circuit(code: RotatedSurfaceCode, qec_rounds: int) -> str:
    """Generate a Stim-format rotated-surface stabilizer circuit string."""
    return str(cpp.build_surf_stabilizer_circuit(code._to_cpp_code(), int(qec_rounds)))


def build_surface_code_stim_circuit(code: RotatedSurfaceCode, qec_rounds: int) -> str:
    """Backward-compatible alias for build_surf_stabilizer_circuit."""
    return build_surf_stabilizer_circuit(code, qec_rounds)


def build_logical_stabilizer_circuit(
    code: RotatedSurfaceCode, lowering_result: LoweringResult, shot_index: int = 0
) -> str:
    """Generate a Stim circuit with deterministic lowered-erasure errors injected by timestep."""
    cpp_result = getattr(lowering_result, "_cpp_result", lowering_result)
    return str(
        cpp.build_logical_stabilizer_circuit(
            code._to_cpp_code(), cpp_result, int(shot_index)
        )
    )


def build_logical_stabilizer_circuit_object(
    code: RotatedSurfaceCode, lowering_result: LoweringResult, shot_index: int = 0
) -> "stim.Circuit":
    """Generate a `stim.Circuit` with deterministic lowered-erasure errors injected by timestep."""
    return _build_stim_circuit_from_text(
        build_logical_stabilizer_circuit(code, lowering_result, shot_index)
    )


def build_logically_equivalent_erasure_stim_circuit(
    code: RotatedSurfaceCode, lowering_result: LoweringResult, shot_index: int = 0
) -> str:
    """Backward-compatible alias for build_logical_stabilizer_circuit."""
    return build_logical_stabilizer_circuit(code, lowering_result, shot_index)


def build_logically_equivalent_erasure_stim_circuit_object(
    code: RotatedSurfaceCode, lowering_result: LoweringResult, shot_index: int = 0
) -> "stim.Circuit":
    """Backward-compatible alias for build_logical_stabilizer_circuit_object."""
    return build_logical_stabilizer_circuit_object(code, lowering_result, shot_index)
