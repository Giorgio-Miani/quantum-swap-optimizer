import networkx as nx
from qiskit import transpile
from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2
import mapomatic as mm
import b_overlap as overlap
import max_clique as maxClique
from qiskit.transpiler import CouplingMap
from collections import deque

def generate_layouts(module, backend, coupling_map = None):
    """ Searches for, optimizes and evaluates quantum circuit layouts for a specified backend. """
    if coupling_map is not None:
        trans_qc = transpile(module, backend, optimization_level=3, coupling_map=coupling_map)
        small_qc = mm.deflate_circuit(trans_qc)
        layouts = mm.matching_layouts(small_qc, coupling_map)
    else:
        trans_qc = transpile(module, backend, optimization_level=3)
        small_qc = mm.deflate_circuit(trans_qc)
        layouts = mm.matching_layouts(small_qc, backend)
    return layouts


class QubitMapping:
    def __init__(self, circuit, backend=FakeGuadalupeV2(), buffer_distance=1, reduced_distance=2, max_allowed_weight=3):
        self.backend = backend
        self.coupling_map = backend.coupling_map
        self.buffer_distance = buffer_distance
        self.reduced_distance = reduced_distance
        self.max_allowed_weight = max_allowed_weight
        self.dependency_graph = circuit.dependency_graph
        self.modules = circuit.modules
        self.modules_qubits = circuit.modules_qubits
        self.qubit_mapping = []
        self.reduced_coupling_maps = []

    def find_qubits_within_distance(self, start_qubit, target_distance):
        """ Obtain all qubits up to a target distance using BFS (Breadth-First Search). """
        visited = {start_qubit}              # Set of visited qubits, initially contains the starting qubit
        queue   = deque([(start_qubit, 0)])  # Queue for BFS, holds tuples of (qubit, current_distance)
        result  = set([start_qubit])         # Result set that includes the starting qubit
        
        # Perform BFS to explore qubits
        while queue:
            current_qubit, current_distance = queue.popleft() # Get the next qubit and its distance
            # If the current distance is less than the target, explore its neighbors
            if current_distance < target_distance:
                for neighbor in self.coupling_map.neighbors(current_qubit):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        result.add(neighbor)
                        queue.append((neighbor, current_distance + 1))
        return result
    
    def get_coupling_map_up_to_reduced_distance(self, qubits):
        """ Obtain a reduced coupling map based on the qubits within a specified reduced distance. """
        # Retrieve the coupling list from the backend configuration
        coupling_list = self.backend.configuration().coupling_map

        # Create a set to store all relevant qubits
        relevant_qubits = set()

        # For each starting qubit, find all qubits that are within the specified reduced_distance
        for qubit in qubits:
            qubits_up_to_distance = self.find_qubits_within_distance(qubit, self.reduced_distance)
            relevant_qubits.update(qubits_up_to_distance)

        # Identify all connections (edges) between the relevant qubits to create a new CouplingMap
        reduced_coupling_list = [edge for edge in coupling_list 
                                if edge[0] in relevant_qubits and edge[1] in relevant_qubits]
        
        return CouplingMap(reduced_coupling_list)
    
    def compute_edge_weight(self, dependentModules, module_idx1, module_idx2, layout1, layout2):
        """ Computes the weight of the edge between two nodes in the compatibility graph. """
        common_dependences = [element for element in dependentModules[module_idx2] if 
                              element in dependentModules[module_idx1]]

        edge_weight = 0
        for dep in common_dependences:
            qubits_dependences = [
                (qubit1, qubit2)
                for qubit1 in self.modules_qubits[dep] if qubit1 in self.modules_qubits[module_idx1]
                for qubit2 in self.modules_qubits[dep] if qubit2 in self.modules_qubits[module_idx2]
            ]

            for qubits_couple in qubits_dependences:
                idx1 = self.modules_qubits[module_idx1].index(qubits_couple[0])
                idx2 = self.modules_qubits[module_idx2].index(qubits_couple[1])
                distance = self.coupling_map.distance(layout1[idx1], layout2[idx2])
                edge_weight += distance

        return edge_weight
    
    def find_common_qubits(self, outModule, inModule, layout):
        """ Finds common qubits between two modules and returns their corresponding positions in the topology. """
        # Initialize an empty list to store common qubits
        common_qubits = []

        # Create a list of qubits that are present in both inModule and outModule
        qubits_dependences = [
            qubit1
            for qubit1 in inModule if qubit1 in outModule
        ]

        # Find the positions of the common qubits in the topology
        for qubit in qubits_dependences:
            idx = outModule.index(qubit)
            common_qubits.append(layout[idx])

        return common_qubits
    
    def get_layouts(self, idx_module, incomingModules):
                # Add nodes to the graph
        module = self.modules[idx_module]
        inModule = self.modules_qubits[idx_module]
        output_qubits = []
        qubit_mapping_dict = {module: layout for dict in self.qubit_mapping for module, layout in dict.items()}
        for outModule_idx in incomingModules[idx_module]:
            if len(self.qubit_mapping) > 0:
                if qubit_mapping_dict.get(outModule_idx) is not None:
                    output_qubits.extend(self.find_common_qubits(inModule=inModule, outModule=self.modules_qubits[outModule_idx], layout=qubit_mapping_dict[outModule_idx]))
        if len(output_qubits) > 0:
            reduced_coupling_map = self.get_coupling_map_up_to_reduced_distance(qubits=output_qubits)
            self.reduced_coupling_maps.append(reduced_coupling_map)
            layouts = generate_layouts(module, self.backend, coupling_map=reduced_coupling_map)
        else:
            layouts = generate_layouts(module, self.backend)

        return layouts

    def generate_qubit_mapping(self):
        """ Generates a compatibility graph. """
        # Check if buffer distance and coupling map are set
        if self.buffer_distance is None or self.coupling_map is None:
            raise ValueError("The buffer distance or the coupling map has not been set yet.")

        # Generate adjacency matrix from the graph
        adj_matrix = nx.to_numpy_array(self.dependency_graph, nodelist=sorted(self.dependency_graph.nodes()))

        # Initialize lists to store active incoming and outgoing edges
        outgoing_edges = [[] for i in range(len(adj_matrix))]
        active_incoming_edges = [[] for i in range(len(adj_matrix))]
        static_incoming_edges = [[] for i in range(len(adj_matrix))]

        # Fill lists based on the adjacency matrix
        for i in range(len(adj_matrix)):
            for j in range(len(adj_matrix)):
                if adj_matrix[i, j] == 1:
                    outgoing_edges[i].append(j)
                    active_incoming_edges[j].append(i)
                    static_incoming_edges[j].append(i)
        
        # Define current_nodes as the list of nodes with no active incoming edges
        current_nodes = [i for i in range(len(adj_matrix)) if len(active_incoming_edges[i]) == 0]
        # Initialize mapped_nodes
        mapped_nodes = []

        while current_nodes:
            if len(current_nodes) == 1:
                # Choose the first layout for the only module in the current set of modules
                layouts = self.get_layouts(idx_module=current_nodes[0], incomingModules=static_incoming_edges)

                #Choose random layout
                layout = layouts[0]
                
                #generate_layouts(self.modules[current_nodes[0]], self.backend)[0]
                max_clique_layouts = {current_nodes[0]:layout}
            else:
                # Build the compatibility graph for the current set of modules (nodes)
                comp_graph = self.build_compatibility_graph(current_nodes, outgoing_edges, self.max_allowed_weight, static_incoming_edges)
                # Find the maximum clique in the compatibility graph
                max_clique, max_clique_weight = maxClique.find_max_clique(comp_graph.to_undirected())

                max_clique_layouts = {}

                for vertex in max_clique:
                    max_clique_layouts[vertex[0]] = comp_graph.nodes[vertex]['layout']

            # Add the maximum clique to the qubit mapping
            self.qubit_mapping.append(max_clique_layouts)

            # Update active_incoming_edges
            for current_node in current_nodes:
                for i in range(len(active_incoming_edges)):
                    if current_node in active_incoming_edges[i]:
                        active_incoming_edges[i].remove(current_node)
            # Get nodes with no active incoming edges that have not been mapped yet
            mapped_nodes.extend(current_nodes)
            current_nodes = [
                node for node in range(len(adj_matrix))
                if len(active_incoming_edges[node]) == 0 and node not in mapped_nodes
            ]

    def build_compatibility_graph(self, current_modules_idx, dependentModules, max_allowed_weight, incomingModules):
        """ Builds a compatibility graph for a given set of modules. """
        # Check if buffer distance and coupling map are set
        if self.buffer_distance is None or self.coupling_map is None:
            raise ValueError("The buffer distance or the coupling map has not been set yet.")

        # Initialize compatibility graph
        comp_graph = nx.DiGraph()

        # Max weight
        max_weight = 0

        # Add nodes to the graph
        for idx_module in current_modules_idx:

            layouts = self.get_layouts(idx_module=idx_module, incomingModules=incomingModules)

            for idx_layout, layout in enumerate(layouts):
                comp_graph.add_node((idx_module, idx_layout), layout=layout)

        # Add edges to the graph
        for v1, attributes1 in comp_graph.nodes(data=True):
            for v2, attributes2 in comp_graph.nodes(data=True):
                if v1[0] != v2[0]:
                    layout1 = attributes1['layout']
                    layout2 = attributes2['layout']

                    # Check if layouts b-overlap
                    overlapping = overlap.check_b_overlap(layout1, layout2, self.coupling_map, self.buffer_distance)

                    # finds common dependencies between 2 modules of the circuit

                    if not overlapping:
                        # print(f"layout1: {layout1}, layout2: {layout2}")
                        edge_weight = self.compute_edge_weight(dependentModules=dependentModules, module_idx1=v1[0], module_idx2=v2[0], layout1=layout1, layout2=layout2)

                        if edge_weight <= max_allowed_weight:

                            # Update maximum weight
                            if edge_weight > max_weight:
                                max_weight = edge_weight

                            # Add edge to the graph with the computed weight
                            comp_graph.add_edge(v1, v2, weight=edge_weight)

        for u, v, data in comp_graph.edges(data=True):
            comp_graph[u][v]['weight'] = max_weight - data['weight']

        return comp_graph
