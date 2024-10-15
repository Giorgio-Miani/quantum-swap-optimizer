import sys
import os

# Manually specify the path to the 'src' directory
src_path = os.path.abspath(os.path.join(os.getcwd(), './src'))

# Add the specified path to the system path
sys.path.append(src_path)

# Third-party libraries
import networkx as nx
from qiskit import transpile
from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2
import mapomatic as mm

# Local application/library imports
import circuit_gen as circuitGen
import max_clique as maxClique
import qubit_mapping as qMap


num_modules       = 4
module_max_qubits = 4
module_max_gates  = 6
buffer_distance   = 1

# Generate random circuit
circuit = circuitGen.RandomCircuit(num_modules, module_max_qubits, module_max_gates)
circuit.gen_random_circuit()

# Ploy dependency graph
circuit.draw_dependency_graph()

# Generate the Quantum Circuit Mapping
q_map = qMap.QubitMapping(circuit, backend=FakeGuadalupeV2(), buffer_distance=buffer_distance)
q_map.generate_qubit_mapping()