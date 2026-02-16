"""Python wrappers and visualization helpers for erasure simulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from ._bindings import cpp
from .code_utils import RotatedSurfaceCode
from .noise_utils import NoiseParams

EventType = cpp.EventType
ErasureQubitSelection = cpp.ErasureQubitSelection


@dataclass
class ErasureSimParams:
    """Python facade around C++ ErasureSimParams."""

    code: RotatedSurfaceCode
    noise: NoiseParams
    qec_rounds: int
    shots: int
    seed: Optional[int] = None
    erasure_selection: object = ErasureQubitSelection.ALL_QUBITS
    erasable_qubits: Optional[Sequence[int]] = None

    def _to_cpp_params(self):
        selection = self.erasure_selection
        if self.erasable_qubits is not None:
            selection = ErasureQubitSelection.EXPLICIT
        return cpp.ErasureSimParams(
            self.code._to_cpp_code(),
            self.noise._to_cpp_noise_params(),
            int(self.qec_rounds),
            int(self.shots),
            self.seed,
            selection,
            [] if self.erasable_qubits is None else [int(q) for q in self.erasable_qubits],
        )


@dataclass
class ErasureSimResult:
    sparse_erasures: list
    erasure_timestep_offsets: list
    _cpp_result: object | None = None

    @classmethod
    def from_cpp(cls, cpp_result) -> "ErasureSimResult":
        return cls(
            sparse_erasures=cpp_result.sparse_erasures,
            erasure_timestep_offsets=cpp_result.erasure_timestep_offsets,
            _cpp_result=cpp_result,
        )


class ErasureSimulator:
    """High-level Python simulator wrapper using C++ backend."""

    def __init__(self, params: ErasureSimParams):
        self.params = params
        self._cpp_simulator = cpp.ErasureSimulator(params._to_cpp_params())

    def simulate(self) -> ErasureSimResult:
        return ErasureSimResult.from_cpp(self._cpp_simulator.simulate())


def _normalize_qubit_subset(total_qubits: int, qubits: Optional[Sequence[int]]) -> list[int]:
    if qubits is None:
        if total_qubits > 128:
            raise ValueError(
                "Code has more than 128 qubits; pass an explicit qubit subset for visualization."
            )
        return list(range(total_qubits))

    if len(qubits) == 0:
        raise ValueError("Qubit subset cannot be empty")

    unique_qubits = sorted(set(int(q) for q in qubits))
    if unique_qubits[0] < 0 or unique_qubits[-1] >= total_qubits:
        raise ValueError(f"Qubit indices must be in [0, {total_qubits - 1}]")
    if len(unique_qubits) > 200:
        raise ValueError("Visualizing >200 qubits is not recommended for test inspection")
    return unique_qubits


def visualize_erasures(
    sim_result: ErasureSimResult,
    params: ErasureSimParams,
    shot_idx: int = 0,
    qubits: Optional[Sequence[int]] = None,
    show_round_guides: bool = True,
):
    """Render erasure timeline heatmap plus per-timestep event counts for one shot."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import BoundaryNorm, ListedColormap

    if shot_idx < 0 or shot_idx >= len(sim_result.sparse_erasures):
        raise ValueError(f"shot_idx must be in [0, {len(sim_result.sparse_erasures) - 1}]")

    selected_qubits = _normalize_qubit_subset(params.code.num_qubits, qubits)
    qubit_to_col = {q: i for i, q in enumerate(selected_qubits)}

    num_timesteps = params.qec_rounds * 4 + 1
    matrix = np.zeros((num_timesteps, len(selected_qubits)), dtype=np.uint8)

    events = sim_result.sparse_erasures[shot_idx]
    offsets = sim_result.erasure_timestep_offsets[shot_idx]

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
            if event.event_type == EventType.ERASURE:
                matrix[t, col] = 1
            elif event.event_type == EventType.RESET:
                matrix[t, col] = 2
            elif event.event_type == EventType.CHECK_ERROR:
                matrix[t, col] = 3

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        gridspec_kw={"height_ratios": [4, 1]},
        sharex=True,
    )

    cmap = ListedColormap(["#f8fafc", "#2563eb", "#f97316", "#dc2626"])
    norm = BoundaryNorm([0, 1, 2, 3, 4], cmap.N)

    im = ax0.imshow(matrix.T, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    if show_round_guides:
        for t in range(0, num_timesteps, 4):
            ax0.axvline(t - 0.5, color="#111827", linestyle="--", linewidth=0.6, alpha=0.7)
            ax1.axvline(t - 0.5, color="#111827", linestyle="--", linewidth=0.6, alpha=0.7)

    cbar = fig.colorbar(im, ax=ax0, ticks=[0.5, 1.5, 2.5, 3.5], pad=0.01)
    cbar.ax.set_yticklabels(["No event", "Erasure", "Reset", "Check error"])

    ax0.set_title(f"Erasure Timeline (shot={shot_idx}, qubits={len(selected_qubits)})")
    ax0.set_ylabel("Qubit index (subset order)")

    ax1.plot(np.arange(num_timesteps), per_timestep_counts, color="#0f172a", linewidth=1.5)
    ax1.fill_between(np.arange(num_timesteps), per_timestep_counts, color="#cbd5e1", alpha=0.6)
    ax1.set_ylabel("Events")
    ax1.set_xlabel("Timestep")

    fig.tight_layout()
    return fig, (ax0, ax1)
