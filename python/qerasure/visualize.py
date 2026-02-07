from qerasure_python import RotatedSurfaceCode
import matplotlib.pyplot as plt

def visualize_gates(code):
    """
    Visualize the gates for each of the four timesteps.
    For each timestep, plot lines between qubits involved in the same gate.
    Show a 2x2 grid of subplots (one for each timestep).
    """
    qubit_positions = code.index_to_coord
    gates = code.gates
    x_anc_offset = code.x_anc_offset
    z_anc_offset = code.z_anc_offset

    # Color map for qubits
    colors = []
    for i in range(code.num_qubits):
        if i < x_anc_offset:
            colors.append('black')  # data qubits
        elif i < z_anc_offset:
            colors.append('red')    # X ancillas
        else:
            colors.append('blue')   # Z ancillas

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    for t in range(4):
        ax = axes[t]
        # Plot all qubits
        for idx, (x, y) in qubit_positions.items():
            ax.scatter(x, y, color=colors[idx], s=100, zorder=2)
            ax.text(x, y, str(idx), fontsize=8, ha='right', zorder=3, color='white', bbox=dict(facecolor=colors[idx], edgecolor='none', boxstyle='round,pad=0.2'))
        # Plot gates as lines
        for q1, q2 in gates[t]:
            x1, y1 = qubit_positions[q1]
            x2, y2 = qubit_positions[q2]
            ax.plot([x1, x2], [y1, y2], color='green', linewidth=2, zorder=1)
        ax.set_title(f'Timestep {t+1}')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid()
        ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

def visualize_code(code):
    # This function will visualize the code object

    qubit_positions = code.index_to_coord
    x_anc_offset = code.x_anc_offset
    z_anc_offset = code.z_anc_offset

    print(x_anc_offset, z_anc_offset)

    # Add a nice discrete color map: black for data qubits, red for X ancillas, blue for Z ancillas
    colors = []
    for i in range(code.num_qubits):
        if i < x_anc_offset:
            colors.append('black')  # data qubits
        elif i < z_anc_offset:
            colors.append('red')    # X ancillas
        else:
            colors.append('blue')   # Z ancillas

    # Use matplotlib to visualize the qubits
    plt.figure(figsize=(6, 6))
    for idx, (x, y) in qubit_positions.items():
        plt.scatter(x, y, color=colors[idx], s=100, zorder=2)
        plt.text(x, y, str(idx), fontsize=8, ha='right', zorder=3, color='white', bbox=dict(facecolor=colors[idx], edgecolor='none', boxstyle='round,pad=0.2'))
    plt.title('Qubit Layout of the Rotated Surface Code')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid()
    plt.show()


