import random
import time

import matplotlib.pyplot as plt
import networkx as nx
from qiskit import QuantumCircuit, transpile
from networkx.drawing.nx_pydot import graphviz_layout


def gen_random_module(num_qubits, num_gates, seed):
    """Generate a random quantum module with a given number of qubits and gates."""
    # Check that num_qubits is less than or equal to num_gates
    random.seed(seed)
    if num_gates < num_qubits:
        raise ValueError("Number of gates must be greater than or equal to the number of qubits.")

    # Check that num_qubits is greater than 1
    if num_qubits < 2:
        raise ValueError("Number of qubits must be greater than 1")

    # Initialize a list of gates
    gates = ['h', 'cx', 's', 'sdg', 'x', 't', 'tdg']

    # Create a QuantumCircuit with the specified number of qubits
    module = QuantumCircuit(num_qubits)

    # Creation of the random connected graph
    graph = nx.gnm_random_graph(num_qubits, num_qubits - 1, seed=seed)
    seed += 1

    # Check if the graph is connected, if not, regenerate it until a connected graph is obtained
    while not nx.is_connected(graph):
        graph = nx.gnm_random_graph(num_qubits, num_qubits - 1, seed=seed)
        seed += 1

    # Get the edges of the graph
    edges = list(graph.edges())

    # Build the module
    num_1qubit_gates = 0
    num_2qubit_gates = 0
    max_1qubit_gates = num_gates - num_qubits + 1

    while num_1qubit_gates < max_1qubit_gates or num_2qubit_gates < num_qubits - 1:
        gate = random.choice(gates)
        if gate == 'cx' and edges:
            qubits = edges.pop()
            module.cx(qubits[0], qubits[1])
            num_2qubit_gates += 1
        elif gate != 'cx' and num_1qubit_gates < max_1qubit_gates:
            qubit = random.choice(range(num_qubits))
            getattr(module, gate)(qubit)
            num_1qubit_gates += 1
    return module


class RandomCircuit:
    def __init__(self, num_modules, module_max_qubits, module_max_gates, seed):
        self.seed = seed
        random.seed(self.seed)
        self.num_modules = num_modules
        self.module_max_qubits = module_max_qubits
        self.module_max_gates = module_max_gates
        self.qubit_counter = 0
        self.dependency_graph = nx.random_tree(self.num_modules, seed=self.seed)
        self.dependency_graph = nx.DiGraph([(u, v) for (u, v) in self.dependency_graph.edges() if u < v])
        self.modules = {}
        self.modules_qubits = {}

    def gen_random_circuit(self):
        """Generate a random circuit."""
        # Generate adjacency matrix from the graph
        adj_matrix = nx.to_numpy_array(self.dependency_graph, nodelist=sorted(self.dependency_graph.nodes()))

        # Initialize lists to store incoming and outgoing edges
        incoming_edges = [[] for i in range(self.num_modules)]
        outgoing_edges = [[] for i in range(self.num_modules)]
        active_incoming_edges = [[] for i in range(self.num_modules)]

        # Fill lists based on the adjacency matrix
        for i in range(self.num_modules):
            for j in range(self.num_modules):
                if adj_matrix[i, j] == 1:
                    incoming_edges[j].append(i)
                    outgoing_edges[i].append(j)
                    active_incoming_edges[j].append(i)

        # Define current_nodes as the list of nodes with no incoming edges
        current_nodes = [i for i in range(self.num_modules) if len(incoming_edges[i]) == 0]

        # Initialize qubit_availability
        qubit_availability = {i: [] for i in range(self.num_modules)}

        # Initialize modules_qubits
        self.modules_qubits = {i: [] for i in range(self.num_modules)}

        # Init the qubit counter
        qubit_counter = -1

        while current_nodes:
            current_node = current_nodes.pop(0)

            # Compute the minimum number of qubits required for the current module
            min_qubits = max(len(incoming_edges[current_node]),
                             len(outgoing_edges[current_node]),
                             2)

            # Check that min_qubits is less than or equal to the maximum number of qubits
            if min_qubits > self.module_max_qubits:
                raise ValueError("Increase the maximum number of qubits per module.")

            # Select the number of qubits for the module
            if min_qubits < self.module_max_qubits:
                num_qubits = random.randint(min_qubits, self.module_max_qubits)
            else:
                num_qubits = self.module_max_qubits

            # Check that num_qubits is less than or equal to the maximum number of gates
            if num_qubits > self.module_max_gates:
                raise ValueError("Increase the maximum number of gates per module.")

            # Select the number of gates for the module
            if num_qubits < self.module_max_gates:
                num_gates = random.randint(num_qubits, self.module_max_gates)
            else:
                num_gates = self.module_max_gates

            # Generate a random module
            self.modules[current_node] = gen_random_module(num_qubits, num_gates, self.seed)

            # Qubit assignment
            for node in incoming_edges[current_node]:
                assigned_qubit = random.choice(qubit_availability[node])
                self.modules_qubits[current_node].append(assigned_qubit)
                qubit_availability[current_node].append(assigned_qubit)
                qubit_availability[node].remove(assigned_qubit)
                num_qubits -= 1

            for i in range(num_qubits):
                qubit_counter += 1
                self.modules_qubits[current_node].append(qubit_counter)
                qubit_availability[current_node].append(qubit_counter)

            # Update active_incoming_edges and current_nodes
            for i in range(len(active_incoming_edges)):
                if current_node in active_incoming_edges[i]:
                    active_incoming_edges[i].remove(current_node)
            if outgoing_edges[current_node]:
                nodes_to_process = [node for node in outgoing_edges[current_node] if not active_incoming_edges[node]]
                current_nodes.extend(nodes_to_process)

        # Set the qubit counter
        self.qubit_counter = qubit_counter + 1

    def get_circuit(self):
        """Return the generated circuit."""
        # Create a QuantumCircuit with the specified number of qubits
        circuit = QuantumCircuit(self.qubit_counter)

        # Generate adjacency matrix from the graph
        adj_matrix = nx.to_numpy_array(self.dependency_graph, nodelist=sorted(self.dependency_graph.nodes()))

        # Initialize lists to store incoming and outgoing edges
        outgoing_edges = [[] for i in range(self.num_modules)]
        active_incoming_edges = [[] for i in range(self.num_modules)]

        # Fill lists based on the adjacency matrix
        for i in range(self.num_modules):
            for j in range(self.num_modules):
                if adj_matrix[i, j] == 1:
                    outgoing_edges[i].append(j)
                    active_incoming_edges[j].append(i)

        # Define current_nodes as the list of nodes with no incoming edges
        current_nodes = [i for i in range(self.num_modules) if len(active_incoming_edges[i]) == 0]

        while current_nodes:
            current_node = current_nodes.pop(0)

            # Add the module to the circuit
            circuit.compose(self.modules[current_node], self.modules_qubits[current_node], inplace=True)

            # Update active_incoming_edges and current_nodes
            for i in range(len(active_incoming_edges)):
                if current_node in active_incoming_edges[i]:
                    active_incoming_edges[i].remove(current_node)
            if outgoing_edges[current_node]:
                nodes_to_process = [node for node in outgoing_edges[current_node] if not active_incoming_edges[node]]
                current_nodes.extend(nodes_to_process)

        return circuit

    def get_benchmark_metrics(self, backend, coupling_map, optimization_level=3):
        """Return the benchmark metric of the circuit."""
        circuit = self.get_circuit()
        basis_gates = ['h', 'cx', 's', 'sdg', 'x', 't', 'tdg']

        # Transpile the circuit
        optimized_circuit = transpile(
            circuit,
            backend=backend,
            routing_method='sabre',
            basis_gates=basis_gates,
            coupling_map=coupling_map,
            optimization_level=optimization_level
        )

        # Extract basic metrics
        depth = optimized_circuit.depth()
        qubits_occupied = set()
        for instruction in optimized_circuit.data:
            qubits_occupied.update(instruction.qubits)

        total_qubits = len(qubits_occupied)
        gate_count = optimized_circuit.size()

        # Extract swap count
        swap_count = 0

        for i, (instr, qargs, _) in enumerate(optimized_circuit.data):
            if instr.name == 'cx':
                control_qubit = qargs[0]._index
                target_qubit = qargs[1]._index

                for j in range(i + 1, len(optimized_circuit.data)):
                    next_instr, next_qargs, _ = optimized_circuit.data[j]

                    # Check if the next gate is applied to either control or target qubit and is not a CNOT
                    if ((next_qargs[0]._index == control_qubit or next_qargs[0]._index == target_qubit) and next_instr.name != 'cx'):
                        break

                    if next_instr.name == 'cx':
                        next_control = next_qargs[0]._index
                        next_target = next_qargs[1]._index

                        # Check if it has reversed control and target
                        if next_control == target_qubit and next_target == control_qubit:
                            
                            for k in range(j + 1, len(optimized_circuit.data)):
                                third_instr, third_qargs, _ = optimized_circuit.data[k]

                                # Check if the gate is applied to control or target and is not a CNOT
                                if ((third_qargs[0]._index == control_qubit or third_qargs[0]._index == target_qubit) and third_instr.name != 'cx'):
                                    break

                                if third_instr.name == 'cx':
                                    third_control = third_qargs[0]._index
                                    third_target = third_qargs[1]._index

                                    if third_control == control_qubit and third_target == target_qubit:
                                        swap_count += 1
                                        break
                                    else:
                                        break
                            break
                        else:
                            break

        # Calculate T-count
        t_count = 0
        for gate in optimized_circuit.data:
            if gate[0].name == 't' or gate[0].name == 'tdg':
                t_count += 1


        metrics = {
            'optimization_level': optimization_level,
            'depth': depth,
            'total_qubits': total_qubits,
            'gate_count': gate_count,
            'swap_count': swap_count,  # Updated swap count based on enhanced detection
            't_count': t_count
        }

        return metrics

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
        plt.savefig('./dep_graph.png')
        plt.show()

    def generate_dependency_graph(self, path) -> float:
        """Generates the dependency graph of the circuit and saves it as a png file
        at the position specified by path, without showing it. It's used for collecting results."""
        start_time = time.time()
        # Generate layout for the graph
        try:
            pos = graphviz_layout(self.dependency_graph, prog="dot")
        except:
            pos = nx.spring_layout(self.dependency_graph)  # Fallback if Graphviz is not installed

        # Draw the graph
        nx.draw(self.dependency_graph, pos, with_labels=True, node_size=500,
                node_color="lightblue", font_size=10, font_weight="bold",
                arrows=True)
        end_time = time.time()
        total_time = end_time - start_time
        # Save the figure
        path = path + '/dep_graph.png'
        plt.savefig(path)
        plt.close()  # To avoid overlapping graphs on multiple successive calls
        print("Dependency graph successfully saved at " + path)

        return total_time

