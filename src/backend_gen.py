from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap

def generate_regular_coupling_map(num_qubits_x, num_qubits_y):
    """ Generate a regular coupling map for a 2D grid of qubits. """
    coupling_map = []
    for i in range(num_qubits_x):
        for j in range(num_qubits_y):
            # Current qubit index
            current_qubit = i * num_qubits_y + j
            
            # Connect to the right neighbor
            if i < num_qubits_x - 1:
                right_neighbor = (i + 1) * num_qubits_y + j
                coupling_map.append([current_qubit, right_neighbor])
                coupling_map.append([right_neighbor, current_qubit])  # Symmetric connection
            
            # Connect to the bottom neighbor
            if j < num_qubits_y - 1:
                bottom_neighbor = i * num_qubits_y + (j + 1)
                coupling_map.append([current_qubit, bottom_neighbor])
                coupling_map.append([bottom_neighbor, current_qubit])  # Symmetric connection

    return CouplingMap(coupling_map)


def generate_regular_backend(num_qubits_x, num_qubits_y):
    """ Generate a regular backend with a 2D grid of qubits. """
    num_qubits = num_qubits_x * num_qubits_y
    coupling_map = generate_regular_coupling_map(num_qubits_x, num_qubits_y)
    gates = ['h', 'x', 'y', 'z', 'cx']

    backend = GenericBackendV2(
        num_qubits=num_qubits,
        basis_gates=gates,
        coupling_map=coupling_map.get_edges(),
        control_flow=False,    # Assume no control flow support
        dtm=1e-9,              # 1 nanosecond time resolution
        seed=42,               # Set a random seed for deterministic behavior
        pulse_channels=False,  # No pulse channels
        noise_info=True        # Include noise information
    )
    return backend
    
