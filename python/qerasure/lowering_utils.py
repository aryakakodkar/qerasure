"""Python wrappers and visualization helpers for lowering events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from ._bindings import cpp
from .sim_utils import ErasureSimParams, ErasureSimResult, _normalize_qubit_subset

PauliError = cpp.PauliError


@dataclass
class LoweredErrorParams:
    """Python facade around C++ LoweredErrorParams."""

    error_type: object
    probability: float

    def __init__(self, error_type: object = PauliError.NO_ERROR, probability: float = 0.0):
        self.error_type = error_type
        self.probability = float(probability)

    def _to_cpp(self):
        params = cpp.LoweredErrorParams()
        params.error_type = self.error_type
        params.probability = float(self.probability)
        return params


@dataclass
class LoweringParams:
    """Python facade around C++ LoweringParams."""

    reset_params: LoweredErrorParams
    x_ancillas: object
    z_ancillas: object | None = None

    def _to_cpp(self):
        reset_cpp = self.reset_params._to_cpp()
        if isinstance(self.x_ancillas, tuple):
            x_cpp = (self.x_ancillas[0]._to_cpp(), self.x_ancillas[1]._to_cpp())
            z_cpp = (self.z_ancillas[0]._to_cpp(), self.z_ancillas[1]._to_cpp())
            return cpp.LoweringParams(reset_cpp, x_cpp, z_cpp)
        if self.z_ancillas is None:
            return cpp.LoweringParams(reset_cpp, self.x_ancillas._to_cpp())
        return cpp.LoweringParams(reset_cpp, self.x_ancillas._to_cpp(), self.z_ancillas._to_cpp())


@dataclass
class LoweringResult:
    sparse_cliffords: list
    clifford_timestep_offsets: list

    @classmethod
    def from_cpp(cls, cpp_result) -> "LoweringResult":
        return cls(
            sparse_cliffords=cpp_result.sparse_cliffords,
            clifford_timestep_offsets=cpp_result.clifford_timestep_offsets,
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
