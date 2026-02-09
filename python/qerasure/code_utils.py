from qerasure_python import RotatedSurfaceCode as _CPP_RotatedSurfaceCode
import matplotlib.pyplot as plt
import matplotlib as mpl

# TODO: these diagrams are ok, but not particularly beautiful. They're more for testing than anything else, but it would
# be nice to make them look better.

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
        self.partner_map = self.cpp_code.partner_map

    def draw(self):
        """
        Visualize the layout of the qubits in the rotated surface code.
        Data qubits are shown in black, X ancillas in red, and Z ancillas
        in blue. The qubits are plotted according to their (x, y) coordinates.
        
        Args:
            None
            
        Returns:
            None (displays a plot of the qubit layout)
        """
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
        """
        Visualize the gates in the syndrome extraction circuit across 4 timesteps.
        Shows CNOT gates with control dots and target circles, colored by ancilla type
        (red for X-ancillas, blue for Z-ancillas).
        
        Args:
            None
            
        Returns:
            None (displays a 2x2 plot of the gates for each timestep)
        """
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
        
        positions = {idx: (x, y) for idx, (x, y) in self.index_to_coord.items()}
        data_indices = [i for i, t in enumerate(types) if t == "data"]
        data_points = [positions[i] for i in data_indices]
        
        # Create 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.flatten()
        
        for step in range(4):
            ax = axes[step]
            ax.set_aspect('equal')
            
            # Draw grid lines between data qubits
            for (x, y) in data_points:
                if (x+2, y) in data_points:
                    ax.plot([x, x+2], [y, y], color=edge_color, linewidth=3, zorder=1)
                if (x, y+2) in data_points:
                    ax.plot([x, x], [y, y+2], color=edge_color, linewidth=3, zorder=1)
            
            # Draw all qubits
            for idx, (x, y) in positions.items():
                ax.scatter(x, y, color=colors[idx], s=400, zorder=2, edgecolors='white', linewidths=2)
            
            # Track which qubits are involved in gates
            qubits_in_gates = set()
            
            # Draw gates for this timestep
            for control, target in self.gates[step]:
                qubits_in_gates.add(control)
                qubits_in_gates.add(target)
                
                # Determine gate color based on which qubit is the ancilla
                if control >= self.x_anc_offset and control < self.z_anc_offset:
                    # Control is X-ancilla
                    gate_color = x_anc_color
                elif control >= self.z_anc_offset:
                    # Control is Z-ancilla
                    gate_color = z_anc_color
                elif target >= self.x_anc_offset and target < self.z_anc_offset:
                    # Target is X-ancilla
                    gate_color = x_anc_color
                elif target >= self.z_anc_offset:
                    # Target is Z-ancilla
                    gate_color = z_anc_color
                else:
                    gate_color = data_color
                
                x_ctrl, y_ctrl = positions[control]
                x_targ, y_targ = positions[target]
                
                # Draw line connecting control and target
                ax.plot([x_ctrl, x_targ], [y_ctrl, y_targ], color=gate_color, linewidth=2.5, zorder=4)
                
                # Draw control dot (filled circle) - more prominent
                ax.scatter(x_ctrl, y_ctrl, color=gate_color, s=180, zorder=7, edgecolors='white', linewidths=2)
                
                # Draw target symbol (hollow circle with white background + crosshair)
                # First draw a white background circle to ensure visibility
                ax.scatter(x_targ, y_targ, color='white', s=350, zorder=6)
                # Then draw the hollow colored circle
                ax.scatter(x_targ, y_targ, facecolors='none', edgecolors=gate_color, s=350, linewidths=3, zorder=7)
                # Draw crosshair on target - make it very visible
                cross_size = 0.35
                ax.plot([x_targ - cross_size, x_targ + cross_size], [y_targ, y_targ], color=gate_color, linewidth=3, zorder=8)
                ax.plot([x_targ, x_targ], [y_targ - cross_size, y_targ + cross_size], color=gate_color, linewidth=3, zorder=8)
            
            # Draw qubit labels - offset slightly for qubits involved in gates
            for idx, (x, y) in positions.items():
                if idx in qubits_in_gates:
                    # Offset label slightly up and to the right for qubits in gates
                    ax.text(x + 0.15, y - 0.15, str(idx), fontsize=9, ha='center', va='center', zorder=10, 
                           color='white', weight='bold', 
                           bbox=dict(boxstyle='circle,pad=0.15', facecolor=colors[idx], edgecolor='white', linewidth=1))
                else:
                    ax.text(x, y, str(idx), fontsize=10, ha='center', va='center', zorder=10, color='white', weight='bold')
                ax.plot([x_targ - cross_size, x_targ + cross_size], [y_targ, y_targ], color=gate_color, linewidth=2, zorder=6)
                ax.plot([x_targ, x_targ], [y_targ - cross_size, y_targ + cross_size], color=gate_color, linewidth=2, zorder=6)
            
            ax.set_title(f'CHECK {step + 1}', fontsize=14, weight='bold', pad=10)
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()

