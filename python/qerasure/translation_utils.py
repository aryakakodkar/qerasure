"""Python wrappers for Stim-circuit translation utilities."""

from __future__ import annotations

from ._bindings import cpp
from .code_utils import RotatedSurfaceCode
from .lowering_utils import LoweringResult


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


def build_logically_equivalent_erasure_stim_circuit(
    code: RotatedSurfaceCode, lowering_result: LoweringResult, shot_index: int = 0
) -> str:
    """Backward-compatible alias for build_logical_stabilizer_circuit."""
    return build_logical_stabilizer_circuit(code, lowering_result, shot_index)
