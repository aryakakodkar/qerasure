"""Python wrappers for virtual decoder circuit builders."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._bindings import cpp
from .code_utils import RotatedSurfaceCode
from .lowering_utils import LoweringParams, LoweringResult, SpreadProgram

if TYPE_CHECKING:
    import stim


def build_virtual_decoder_stim_circuit(
    code: RotatedSurfaceCode,
    qec_rounds: int,
    lowering_params: LoweringParams | SpreadProgram,
    lowering_result: LoweringResult,
    two_qubit_erasure_probability: float,
    shot_index: int = 0,
    condition_on_erasure_in_round: bool = True,
) -> str:
    """Generate a virtual decoder circuit string with probabilistic spread injection."""
    if isinstance(lowering_params, SpreadProgram):
        lowering_params = LoweringParams(lowering_params)
    return str(
        cpp.build_virtual_decoder_stim_circuit(
            code._to_cpp_code(),
            int(qec_rounds),
            lowering_params._to_cpp(),
            getattr(lowering_result, "_cpp_result", lowering_result),
            int(shot_index),
            float(two_qubit_erasure_probability),
            bool(condition_on_erasure_in_round),
        )
    )


def build_virtual_decoder_stim_circuit_object(
    code: RotatedSurfaceCode,
    qec_rounds: int,
    lowering_params: LoweringParams | SpreadProgram,
    lowering_result: LoweringResult,
    two_qubit_erasure_probability: float,
    shot_index: int = 0,
    condition_on_erasure_in_round: bool = True,
) -> "stim.Circuit":
    """Generate a `stim.Circuit` virtual decoder circuit."""
    try:
        import stim  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "The Python `stim` package is required for circuit-object builders. "
            "Install/import `stim` to use this function."
        ) from exc
    if isinstance(lowering_params, SpreadProgram):
        lowering_params = LoweringParams(lowering_params)
    return cpp.build_virtual_decoder_stim_circuit_object(
        code._to_cpp_code(),
        int(qec_rounds),
        lowering_params._to_cpp(),
        getattr(lowering_result, "_cpp_result", lowering_result),
        int(shot_index),
        float(two_qubit_erasure_probability),
        bool(condition_on_erasure_in_round),
    )


# Backward-compatible aliases for previous naming.
def build_virtual_logical_stabilizer_circuit(
    code: RotatedSurfaceCode,
    qec_rounds: int,
    lowering_params: LoweringParams | SpreadProgram,
    lowering_result: LoweringResult,
    p_two_qubit_erasure: float,
    shot_index: int = 0,
    *,
    condition_on_erasure_in_round: bool = True,
) -> str:
    """Backward-compatible alias for build_virtual_decoder_stim_circuit."""
    return build_virtual_decoder_stim_circuit(
        code=code,
        qec_rounds=qec_rounds,
        lowering_params=lowering_params,
        lowering_result=lowering_result,
        shot_index=shot_index,
        two_qubit_erasure_probability=p_two_qubit_erasure,
        condition_on_erasure_in_round=condition_on_erasure_in_round,
    )


def build_virtual_logical_stabilizer_circuit_object(
    code: RotatedSurfaceCode,
    qec_rounds: int,
    lowering_params: LoweringParams | SpreadProgram,
    lowering_result: LoweringResult,
    p_two_qubit_erasure: float,
    shot_index: int = 0,
    *,
    condition_on_erasure_in_round: bool = True,
) -> "stim.Circuit":
    """Backward-compatible alias for build_virtual_decoder_stim_circuit_object."""
    return build_virtual_decoder_stim_circuit_object(
        code=code,
        qec_rounds=qec_rounds,
        lowering_params=lowering_params,
        lowering_result=lowering_result,
        shot_index=shot_index,
        two_qubit_erasure_probability=p_two_qubit_erasure,
        condition_on_erasure_in_round=condition_on_erasure_in_round,
    )
