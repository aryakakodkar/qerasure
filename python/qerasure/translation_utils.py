"""Python wrappers for Stim-circuit translation utilities."""

from __future__ import annotations

from ._bindings import cpp
from .code_utils import RotatedSurfaceCode


def build_surface_code_stim_circuit(code: RotatedSurfaceCode, qec_rounds: int) -> str:
    """Generate a Stim-format rotated-surface-code circuit string."""
    return str(cpp.build_surface_code_stim_circuit(code._to_cpp_code(), int(qec_rounds)))
