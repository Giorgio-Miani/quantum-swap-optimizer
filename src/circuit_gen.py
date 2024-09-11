import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
from networkx.drawing.nx_pydot import graphviz_layout
from qiskit import QuantumCircuit

class RandomCircuit:
    def __init__(self, num_modules, module_max_qubits, module_max_gates):
        self.num_modules       = num_modules
        self.module_max_gates  = module_max_gates
        self.module_max_qubits = module_max_qubits
        self.dependency_graph  = nx.random_tree(self.num_modules)
        self.dependency_graph  = nx.DiGraph([(u,v) for (u,v) in self.dependency_graph.edges() if u<v])
        self.modules           = []  

    def gen_random_circuit(self):
        """Generate a random circuit."""
        # Generate adjacency matrix from the graph
        adj_matrix = nx.to_numpy_array(self.dependency_graph, nodelist=sorted(self.dependency_graph.nodes()))
        
        # Print the adjacency matrix
        print(adj_matrix)             
    
    def draw_dependency_graph(self):
        """Draw the dependency graph of the circuit."""
        # Generate layout for the graph
        try:
            pos = graphviz_layout(self.dependency_graph, prog="dot")
        except:
            pos = nx.spring_layout(self.dependency_graph)  # Fallback if Graphviz is not installed
        
        # Draw the graph
        nx.draw(self.dependency_graph, pos, with_labels=True, node_size=500, 
                node_color="lightblue", font_size=10, font_weight="bold", 
                arrows=True)
        plt.show()

    @staticmethod
    def gen_random_module(num_qubits, num_gates):
        """Generate a random module."""
        module = QuantumCircuit(num_qubits)
        for _ in range(num_gates):
            gate = random.choice(['h', 'x', 'y', 'z', 'cx'])
            if gate == 'cx':
                if num_qubits > 1:
                    qubits = random.sample(range(num_qubits), 2)
                    module.cx(qubits[0], qubits[1])
            else:
                qubit = random.choice(range(num_qubits))
                getattr(module, gate)(qubit)
        return module