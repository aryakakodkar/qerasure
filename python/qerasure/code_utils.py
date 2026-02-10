"""Python wrappers and visualization helpers for rotated surface codes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ._bindings import cpp


@dataclass
class RotatedSurfaceCode:
    """Python wrapper around the C++ RotatedSurfaceCode object."""

    _cpp_code: object

    def __init__(self, distance: int):
        self._cpp_code = cpp.RotatedSurfaceCode(distance)

    def _to_cpp_code(self):
        return self._cpp_code

    @property
    def distance(self) -> int:
        return int(self._cpp_code.distance)

    @property
    def num_qubits(self) -> int:
        return int(self._cpp_code.num_qubits)

    @property
    def x_anc_offset(self) -> int:
        return int(self._cpp_code.x_anc_offset)

    @property
    def z_anc_offset(self) -> int:
        return int(self._cpp_code.z_anc_offset)

    @property
    def index_to_coord(self) -> dict[int, tuple[int, int]]:
        return dict(self._cpp_code.index_to_coord)

    @property
    def gates(self):
        return self._cpp_code.gates

    @property
    def partner_map(self):
        return self._cpp_code.partner_map

    def draw(self, annotate: bool = True, figsize: tuple[int, int] = (7, 7)):
        import matplotlib.pyplot as plt

        positions = self.index_to_coord
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor("#fafbfc")

        data_nodes = list(range(0, self.x_anc_offset))
        x_anc_nodes = list(range(self.x_anc_offset, self.z_anc_offset))
        z_anc_nodes = list(range(self.z_anc_offset, self.num_qubits))

        for step_gates in self.gates:
            for control, target in step_gates:
                x1, y1 = positions[control]
                x2, y2 = positions[target]
                ax.plot([x1, x2], [y1, y2], color="#d7dde8", linewidth=1.2, zorder=1)

        def scatter_nodes(indices: Iterable[int], color: str, label: str):
            xs = [positions[i][0] for i in indices]
            ys = [positions[i][1] for i in indices]
            ax.scatter(xs, ys, s=220, c=color, edgecolors="white", linewidths=1.2, label=label, zorder=3)

        scatter_nodes(data_nodes, "#1f2937", "Data")
        scatter_nodes(x_anc_nodes, "#e11d48", "X ancilla")
        scatter_nodes(z_anc_nodes, "#0ea5e9", "Z ancilla")

        if annotate:
            for idx, (x, y) in positions.items():
                ax.text(x, y, str(idx), ha="center", va="center", fontsize=8, color="white", zorder=4)

        ax.set_aspect("equal")
        ax.set_title(f"Rotated Surface Code (d={self.distance}, n={self.num_qubits})")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(color="#e5e7eb", linewidth=0.6)
        ax.legend(loc="upper right")
        fig.tight_layout()
        return fig, ax

    def draw_gates(self, annotate: bool = False, figsize: tuple[int, int] = (12, 10)):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        positions = self.index_to_coord
        fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)

        for step in range(4):
            ax = axes.flat[step]
            ax.set_facecolor("#fafbfc")
            step_gates = self.gates[step]

            involved = set()
            for control, target in step_gates:
                involved.add(control)
                involved.add(target)
                x1, y1 = positions[control]
                x2, y2 = positions[target]
                ax.plot([x1, x2], [y1, y2], color="#334155", linewidth=1.4, zorder=2)
                ax.scatter([x1], [y1], c="#16a34a", s=55, zorder=3)
                ax.scatter([x2], [y2], facecolors="none", edgecolors="#ef4444", s=110, linewidths=1.6, zorder=4)

            xs = [positions[i][0] for i in range(self.num_qubits)]
            ys = [positions[i][1] for i in range(self.num_qubits)]
            ax.scatter(xs, ys, c="#cbd5e1", s=40, zorder=1)

            if annotate:
                for idx in involved:
                    x, y = positions[idx]
                    ax.text(x + 0.05, y + 0.05, str(idx), fontsize=7, color="#0f172a")

            ax.set_title(f"Step {step + 1}: {len(step_gates)} CNOTs")
            ax.grid(color="#e5e7eb", linewidth=0.5)
            ax.set_aspect("equal")

        legend_handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#16a34a", markersize=7, label="Control"),
            Line2D([0], [0], marker="o", color="#ef4444", markerfacecolor="none", markersize=9, label="Target"),
        ]
        fig.legend(handles=legend_handles, loc="upper center", ncol=2, frameon=False)
        fig.suptitle("Syndrome Extraction Schedule", y=0.98)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        return fig, axes
