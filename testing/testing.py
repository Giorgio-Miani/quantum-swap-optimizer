import sys
import os

# Manually specify the path to the 'src' directory
src_path = os.path.abspath(os.path.join(os.getcwd(), './src'))

# Add the specified path to the system path
sys.path.append(src_path)

# Third-party libraries
import networkx as nx

# Local application/library imports
import compatibility_graph as compGraph
import circuit_gen_2 as circuitGen
import max_clique as maxClique

num_modules       = 4
module_max_qubits = 4
module_max_gates  = 6
buffer_distance   = 1

# Generate random circuit
circuit = circuitGen.RandomCircuit(num_modules, module_max_qubits, module_max_gates)
circuit.gen_random_circuit()

# Generate compatibility graph
comp_graph = compGraph.CompatibilityGraph(modules=circuit.modules,dependency_graph=circuit.dependency_graph, buffer_distance=buffer_distance, modules_qubits=circuit.modules_qubits)
comp_graph.generate_compatibility_graph()
graph = comp_graph.graph