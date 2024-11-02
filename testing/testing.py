import sys
import os
import pickle
import random

# Manually specify the path to the 'src' directory
src_path = os.path.abspath(os.path.join(os.getcwd(), './src'))

# Add the specified path to the system path
sys.path.append(src_path)

# Local application/library imports
import backend_gen as backendGen
import circuit_gen as circuitGen
import qubit_mapping as qMap

num_modules        = 7
module_max_qubits  = 4
module_max_gates   = 10
reduced_distance   = None
max_allowed_weight = 5
num_qubits_x       = 100
num_qubits_y       = 100
heuristic          = False
save_backend       = False
seed               = random.randint(1, int(1e4))

# Generate random circuit
circuit = circuitGen.RandomCircuit(num_modules, module_max_qubits, module_max_gates, seed)
circuit.gen_random_circuit()

# Ploy dependency graph
circuit.draw_dependency_graph()

# Generate the Quantum Circuit Mapping
q_map = qMap.QubitMapping(
    circuit, 
    backend=backendGen.generate_regular_backend(num_qubits_x, num_qubits_y),
    coupling_map_dims=(num_qubits_x, num_qubits_y),
    reduced_distance=reduced_distance, 
    max_allowed_weight=max_allowed_weight,
    heuristic=heuristic
)
q_map.generate_ASAP_qubit_mapping()
print(f"Qubit modules:  {q_map.modules_qubits}")
print(f"Qubit mapping: {q_map.qubit_mapping}")

# Save the backend
if save_backend:
    with open('backends/backend.pkl', 'wb') as file:
        pickle.dump(q_map.backend, file)