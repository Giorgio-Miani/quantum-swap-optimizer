import sys
import os
import pickle

# Manually specify the path to the 'src' directory
src_path = os.path.abspath(os.path.join(os.getcwd(), './src'))

# Add the specified path to the system path
sys.path.append(src_path)

# Local application/library imports
import backend_gen as backendGen
import circuit_gen as circuitGen
import qubit_mapping as qMap

num_modules        = 4
module_max_qubits  = 4
module_max_gates   = 6
buffer_distance    = 0
reduced_distance   = 3
max_allowed_weight = 3
num_qubits_x       = 5
num_qubits_y       = 5
heuristic          = False
save_backend       = False

# Generate random circuit
circuit = circuitGen.RandomCircuit(num_modules, module_max_qubits, module_max_gates)
circuit.gen_random_circuit()

# Ploy dependency graph
circuit.draw_dependency_graph()

# Generate the Quantum Circuit Mapping
q_map = qMap.QubitMapping(
    circuit, 
    backend=backendGen.generate_regular_backend(num_qubits_x, num_qubits_y),
    buffer_distance=buffer_distance, 
    reduced_distance=reduced_distance, 
    max_allowed_weight=max_allowed_weight,
    heuristic=heuristic
)
q_map.generate_qubit_mapping()

# Save the backend
if save_backend:
    with open('backends/backend.pkl', 'wb') as file:
        pickle.dump(q_map.backend, file)