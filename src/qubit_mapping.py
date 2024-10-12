import networkx as nx
from qiskit import transpile
from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2
import mapomatic as mm
import b_overlap as overlap
import max_clique as maxClique

def generate_layouts(module, backend):
    """ Searches for, optimizes and evaluates quantum circuit layouts for a specified backend. """    
    trans_qc = transpile(module, backend, optimization_level=3)
    small_qc = mm.deflate_circuit(trans_qc)
    layouts = mm.matching_layouts(small_qc, backend)
    return layouts

class QubitMapping:
    def __init__(self, circuit, backend=FakeGuadalupeV2(), buffer_distance=1):
        self.backend          = backend
        self.buffer_distance  = buffer_distance
        self.coupling_map     = backend.coupling_map
        self.dependency_graph = circuit.dependency_graph
        self.modules          = circuit.modules
        self.modules_qubits   = circuit.modules_qubits
        self.qubit_mapping    = []

    def generate_qubit_mapping(self):
        """ Generates a compatibility graph. """
        # Check if buffer distance and coupling map are set
        if self.buffer_distance is None or self.coupling_map is None:
            raise ValueError("The buffer distance or the coupling map has not been set yet.")
        
        # Generate adjacency matrix from the graph
        adj_matrix = nx.to_numpy_array(self.dependency_graph, nodelist=sorted(self.dependency_graph.nodes()))
        # print(f"Adjacency matrix: {adj_matrix}")

        # Initialize lists to store active incoming and outgoing edges
        outgoing_edges = [[] for i in range(len(adj_matrix))]
        active_incoming_edges = [[] for i in range(len(adj_matrix))]

        # Fill lists based on the adjacency matrix
        for i in range(len(adj_matrix)):
            for j in range(len(adj_matrix)):
                if adj_matrix[i, j] == 1:
                    outgoing_edges[i].append(j)
                    active_incoming_edges[j].append(i)

        # Define current_nodes as the list of nodes with no active incoming edges
        current_nodes = [i for i in range(len(adj_matrix)) if len(active_incoming_edges[i]) == 0]

        mapped_nodes = []

        # print(f"Outgoing edges: {outgoing_edges}")
        # print(f"Active incoming edges: {active_incoming_edges}")
        # print(f"Current nodes: {current_nodes}")
        # print(f"Mapped nodes: {mapped_nodes}")

        while current_nodes:
            # print(f"Outgoing edges: {outgoing_edges}")
            # print(f"Active incoming edges: {active_incoming_edges}")
            # print(f"Current nodes: {current_nodes}")
            # print(f"Mapped nodes: {mapped_nodes}")

            # Build the compatibility graph for the current set of modules (nodes)
            comp_graph = self.build_compatibility_graph(current_nodes, outgoing_edges)

            # Find the maximum clique in the compatibility graph
            max_clique, max_clique_weight = maxClique.find_max_clique(comp_graph. to_undirected())

            # Add the maximum clique to the qubit mapping
            self.qubit_mapping.append(max_clique)

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

    def build_compatibility_graph(self, current_modules_idx, dependentModules):
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
            module = self.modules[idx_module]
            layouts = generate_layouts(module, self.backend)
            for idx_layout, layout in enumerate(layouts):
                comp_graph.add_node((idx_module, idx_layout))
        
        # Add edges to the graph
        for v1 in comp_graph.nodes:
            for v2 in comp_graph.nodes:
                if v1[0] != v2[0]:
                    has_common_dependence = bool(set(dependentModules[v1[0]]) & set(dependentModules[v2[0]]))
                    if has_common_dependence:
                        # print(f"v1: {v1}, v2: {v2}")
                        layout1 = generate_layouts(self.modules[v1[0]], self.backend)[v1[1]]
                        layout2 = generate_layouts(self.modules[v2[0]], self.backend)[v2[1]]
                        # print(f"layout1: {layout1}, layout2: {layout2}")

                        # Check if layouts b-overlap
                        overlapping = overlap.check_b_overlap(layout1, layout2, self.coupling_map, self.buffer_distance)

                        # TODO: Controllare se il codice che segue fino alla riga 142 è corretto.
                        common_dependences = [element for element in dependentModules[v2[0]] if element in dependentModules[v1[0]]]
                        
                        if not overlapping:
                            # print(f"layout1: {layout1}, layout2: {layout2}")
                            edge_weight = 0
                            for dep in common_dependences:
                                qubits_dependences = [
                                    (qubit1, qubit2)
                                    for qubit1 in self.modules_qubits[dep] if qubit1 in self.modules_qubits[v1[0]]
                                    for qubit2 in self.modules_qubits[dep] if qubit2 in self.modules_qubits[v2[0]]
                                ]

                                # print(f"qubits_dependences: {qubits_dependences}")

                                for qubits_couple in qubits_dependences:
                                    idx1 = self.modules_qubits[v1[0]].index(qubits_couple[0])
                                    idx2 = self.modules_qubits[v2[0]].index(qubits_couple[1])
                                    distance = self.coupling_map.distance(layout1[idx1], layout2[idx2])
                                    edge_weight += distance
                                
                                # Update maximum weight
                                if edge_weight > max_weight:
                                    max_weight = edge_weight
                        
                            # Add edge to the graph with the computed weight
                            comp_graph.add_edge(v1, v2, weight=edge_weight)

        for u, v, data in comp_graph.edges(data=True):
            comp_graph[u][v]['weight'] = max_weight - data['weight']

        return comp_graph