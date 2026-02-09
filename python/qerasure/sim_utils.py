from qerasure_python import ErasureSimParams as _CPP_ErasureSimParams
from qerasure_python import EventType as _CPP_EventType
from qerasure_python import ErasureSimEvent as _CPP_ErasureSimEvent
from qerasure_python import ErasureSimResult as _CPP_ErasureSimResult
from qerasure_python import ErasureSimulator as _CPP_ErasureSimulator

from .code_utils import RotatedSurfaceCode
from .noise_utils import NoiseParams

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np

class ErasureSimParams:
    """
    A Python wrapper around the C++ ErasureSimParams struct.

    This class provides a Python interface to the C++ implementation of the parameters for an erasure simulator.
    It allows users to easily create and manipulate simulation parameters in Python before passing them to the simulator.
    """
    def __init__(self, code: RotatedSurfaceCode, noise: NoiseParams, qec_rounds: int, shots: int):
        self.code = code
        self.noise = noise
        self.qec_rounds = qec_rounds
        self.shots = shots

    def _to_cpp_params(self):
        cpp_params = _CPP_ErasureSimParams(self.code._to_cpp_code(), self.noise._to_cpp_noise_params(), self.qec_rounds, self.shots)
        return cpp_params

class ErasureSimResult:
    """
    A Python wrapper around the C++ ErasureSimResult struct.

    This class provides a Python interface to the C++ implementation of the results from an erasure simulation.
    It allows users to easily access and manipulate the simulation results in Python after running the simulator.
    """
    def __init__(self, cpp_result: _CPP_ErasureSimResult):
        self.sparse_erasures = cpp_result.sparse_erasures
        self.erasure_timestep_offsets = cpp_result.erasure_timestep_offsets

class ErasureSimEvent:
    """
    A Python wrapper around the C++ ErasureSimEvent struct.

    This class provides a Python interface to the C++ implementation of an erasure simulation event.
    It allows users to easily access and manipulate the details of an erasure event in Python after running the simulator.
    """
    def __init__(self, cpp_event: _CPP_ErasureSimEvent):
        self.qubit_idx = cpp_event.qubit_idx
        self.event_type = cpp_event.event_type

class ErasureSimulator:
    """
    A Python wrapper around the C++ ErasureSimulator class.

    This class provides a Python interface to the C++ implementation of an erasure simulator for quantum error correction.
    It initializes the simulator with given parameters and exposes methods to run simulations and retrieve results.
    """
    def __init__(self, params: ErasureSimParams):
        self.params = params._to_cpp_params()
        self.simulator = _CPP_ErasureSimulator(params._to_cpp_params())

    def simulate(self):
        return ErasureSimResult(self.simulator.simulate())

def visualize_erasures(sim_result: ErasureSimResult, params: ErasureSimParams, shot_idx: int = 0, qubits: list[int] = None):
    """
    A helper function to visualize the erasure events from a simulation result.

    This function takes an ErasureSimResult object and creates a visual representation of the erasure events over time.
    It can be used to better understand the distribution and frequency of erasures during the simulation.

    Args:
        sim_result (ErasureSimResult): The result object containing the erasure events to visualize.
        params (ErasureSimParams): The parameters used for the simulation, needed to determine the number of qubits and rounds.
        shot_idx (int): The index of the shot to visualize (default is 0).
        qubits (list[int], optional): A list of qubit indices to visualize. If None, all qubits will be visualized. 
            Must be between 0 and the total number of qubits in the code, and no more than 50 qubits can be visualized at once.

    Raises:
        ValueError: If the specified shot index is out of range, or if the qubit indices are invalid or too many qubits are requested for visualization.

    Returns:
        None: This function displays a plot of the erasure events and does not return any value
    """
    if qubits is None:
        if params.code.num_qubits > 50:
            raise ValueError("Code has more than 50 qubits, please specify a subset of qubits to visualize.")
        qubits = list(range(params.code.num_qubits))
    elif qubits < 0 or max(qubits) >= params.code.num_qubits:
        raise ValueError("Qubit indices must be between 0 and {}".format(params.code.num_qubits - 1))
    elif len(qubits) > 50:
        raise ValueError("Cannot visualize more than 50 qubits at once, please specify a smaller subset of qubits.")

    # Create a figure and axis for plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract the erasure events for the specified shot
    erasure_events = sim_result.sparse_erasures[shot_idx]

    # Create a 2D array to represent the erasure events for each qubit and shot
    erasure_matrix = np.zeros((params.qec_rounds * 4 + 1, len(qubits)), dtype=int)

    for t, offset in enumerate(sim_result.erasure_timestep_offsets[shot_idx][:-1]):
        for event in erasure_events[offset:sim_result.erasure_timestep_offsets[shot_idx][t+1]]:
            if event.qubit_idx in qubits:
                if event.event_type == _CPP_EventType.ERASURE:
                    erasure_matrix[t, event.qubit_idx] = 1
                elif event.event_type == _CPP_EventType.RESET:
                    erasure_matrix[t, event.qubit_idx] = 2
                elif event.event_type == _CPP_EventType.CHECK_ERROR:
                    erasure_matrix[t, event.qubit_idx] = 3

    cmap = ListedColormap(['#f0f0f0', '#1f77b4', '#ff7f0e', '#2ca02c'])  # light gray, blue, orange, green
    bounds = [0, 1, 2, 3, 4]
    norm = BoundaryNorm(bounds, cmap.N)
    # Transpose so timesteps are on x-axis, qubits on y-axis
    cax = ax.imshow(erasure_matrix.T, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')
    cbar = fig.colorbar(cax, ax=ax, ticks=[0.5, 1.5, 2.5, 3.5])
    cbar.ax.set_yticklabels(['No Event', 'Erasure', 'Reset', 'Check Error'])
    cbar.set_label('Event Type')

    # Add vertical lines at every timestep 4n + 1 to indicate the start of each QEC round
    for t in range(0, params.qec_rounds * 4 + 1, 4):
        ax.axvline(x=t, color='black', linestyle='--', linewidth=0.5)

    ax.set_xlabel('Timestep')
    ax.set_ylabel('Qubit Index')
    ax.set_title('Erasure Events for Shot {}'.format(shot_idx))
    plt.show()
