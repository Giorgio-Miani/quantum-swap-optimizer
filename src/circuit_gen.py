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
        self.modules           = {}  
        self.modules_qubits    = {}

    def gen_random_circuit(self):
        """Generate a random circuit."""
        # Generate adjacency matrix from the graph
        adj_matrix = nx.to_numpy_array(self.dependency_graph, nodelist=sorted(self.dependency_graph.nodes()))

        # Initialize dictionaries
        in_arrow       = {i: set() for i in range(self.num_modules)}
        in_dictionary  = {i: set() for i in range(self.num_modules)}
        out_dictionary = {i: set() for i in range(self.num_modules)}
        
        # Fill dictionaries based on the adjacency matrix
        for i in range(self.num_modules):
            for j in range(self.num_modules):
                if adj_matrix[i, j] == 1:
                    in_arrow[j].add(i)
                    in_dictionary[j].add(i)
                    out_dictionary[i].add(j)
        
        in_arrow       = {k: v for k, v in in_arrow.items() if len(v) > 0}
        in_dictionary  = {k: v for k, v in in_dictionary.items() if len(v) > 0}
        out_dictionary = {k: v for k, v in out_dictionary.items() if len(v) > 0}

        # Define current_nodes as the set of nodes with no incoming edges
        current_nodes = list({i for i in range(self.num_modules) if i not in in_dictionary})

        # Init the qubit counter
        qubit_counter = 0

        # Init the module qubit assignment dictionary
        self.modules_qubits = {i: [] for i in range(self.num_modules)}
        qubit_availability  = {i: [] for i in range(self.num_modules)}
        
        while current_nodes:
            # print(f"Current Nodes: {current_nodes}")  
            current_node = current_nodes.pop(0)

            # Generate module associated with current_node
            if current_node in in_dictionary or current_node in out_dictionary:
                in_dict_value = len(in_dictionary[current_node]) if current_node in in_dictionary else 1
                out_dict_value = len(out_dictionary[current_node]) if current_node in out_dictionary else 1
                min_qubits = max(in_dict_value, out_dict_value)
            else:
                min_qubits = 1

            if min_qubits == self.module_max_qubits:
                num_qubits = min_qubits
            else:
                num_qubits = random.randint(min_qubits, self.module_max_qubits)
            self.modules[current_node] = self.gen_random_module(num_qubits,
                                                                random.randint(1, self.module_max_gates))
            
            # Qubit assignment
            if current_node in in_dictionary:
                for node in in_dictionary[current_node]:
                    assigned_qubits = random.choice(qubit_availability[node])
                    self.modules_qubits[current_node].append(assigned_qubits)
                    qubit_availability[current_node].append(assigned_qubits)
                    qubit_availability[node].remove(assigned_qubits)
                    num_qubits -= 1

            for i in range(num_qubits):
                qubit_counter += 1
                self.modules_qubits[current_node].append(qubit_counter)
                qubit_availability[current_node].append(qubit_counter)

            # Update in_arrow
            for key, values in in_arrow.items():
                if current_node in values:
                    values.remove(current_node)
            in_arrow = {k: v for k, v in in_arrow.items() if len(v) > 0}
            if current_node in out_dictionary:
                nodes_to_add = [node for node in out_dictionary[current_node] if node not in in_arrow]
                current_nodes.extend(nodes_to_add)         
    
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