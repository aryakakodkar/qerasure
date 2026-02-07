from qerasure_python import RotatedSurfaceCode as _CPP_RotatedSurfaceCode
import matplotlib.pyplot as plt
import matplotlib as mpl

class RotatedSurfaceCode:
    """
    A Python wrapper around the C++ RotatedSurfaceCode class.

    This class provides a Python interface to the C++ implementation of the rotated surface code.
    It initializes the code with a given distance and exposes properties such as the number of qubits,
    the offsets for X and Z ancillas, the mapping from qubit indices to coordinates, and the gates in the syndrome extraction circuit.
    """
    def __init__(self, distance):
        self.cpp_code = _CPP_RotatedSurfaceCode(distance)
        self.num_qubits = self.cpp_code.num_qubits
        self.x_anc_offset = self.cpp_code.x_anc_offset
        self.z_anc_offset = self.cpp_code.z_anc_offset
        self.index_to_coord = self.cpp_code.index_to_coord
        self.gates = self.cpp_code.gates

    def draw(self):
        # Color scheme
        data_color = "#222222"
        x_anc_color = "#E57373"
        z_anc_color = "#64B5F6"
        edge_color = "#DDDDDD"

        # Prepare color and type lists
        colors = []
        types = []
        for i in range(self.num_qubits):
            if i < self.x_anc_offset:
                colors.append(data_color)
                types.append("data")
            elif i < self.z_anc_offset:
                colors.append(x_anc_color)
                types.append("x_anc")
            else:
                colors.append(z_anc_color)
                types.append("z_anc")

        # Extract positions
        positions = {idx: (x, y) for idx, (x, y) in self.index_to_coord.items()}

        # Find grid points (data qubits)
        data_indices = [i for i, t in enumerate(types) if t == "data"]
        data_points = [positions[i] for i in data_indices]

        # Draw grid lines between data qubits
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_aspect('equal')
        for (x, y) in data_points:
            # Draw horizontal lines
            if (x+2, y) in data_points:
                ax.plot([x, x+2], [y, y], color=edge_color, linewidth=4, zorder=1)
            # Draw vertical lines
            if (x, y+2) in data_points:
                ax.plot([x, x], [y, y+2], color=edge_color, linewidth=4, zorder=1)

        # Draw all qubits
        for idx, (x, y) in positions.items():
            ax.scatter(x, y, color=colors[idx], s=600, zorder=2, edgecolors='white', linewidths=2)
            ax.text(x, y, str(idx), fontsize=12, ha='center', va='center', zorder=3, color='white', weight='bold')

        # Remove axes, ticks, and gridlines
        ax.axis('off')
        plt.tight_layout()
        plt.show()
            
        mpl.rcParams['font.family'] = 'DejaVu Sans'  # Clean sans-serif font
        mpl.rcParams['font.size'] = 14

    def draw_gates(self):
        # This function will visualize the gates in the code object
        pass
