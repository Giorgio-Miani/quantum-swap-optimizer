# Standard library import
from collections import deque

# Third-party imports
import networkx as nx
from qiskit import transpile
from qiskit.transpiler import CouplingMap
from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2

# Local imports
import mapomatic as mm
import b_overlap as overlap
import max_clique as maxClique

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
    def __init__(self, 
                 circuit, 
                 backend=FakeGuadalupeV2(), 
                 buffer_distance=1, 
                 reduced_distance=2, 
                 max_allowed_weight=3,
                 heuristic=False):
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
        self.heuristic = heuristic

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
        coupling_list = self.coupling_map.get_edges()

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
    
    def compute_edge_weight(self, 
                            dependentModules, 
                            incomingModules,
                            module_idx1, 
                            module_idx2, 
                            layout1, 
                            layout2):
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

        in_qubits1 = self.locate_qubit_dependencies(module_idx1, incomingModules)
        in_qubits2 = self.locate_qubit_dependencies(module_idx2, incomingModules)

        for qubit1, pos1 in in_qubits1.items():
            edge_weight += self.coupling_map.distance(pos1, layout1[self.modules_qubits[module_idx1].index(qubit1)])

        for qubit2, pos2 in in_qubits2.items():
            edge_weight += self.coupling_map.distance(pos2, layout2[self.modules_qubits[module_idx2].index(qubit2)])

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
        module   = self.modules[idx_module]
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
    
    def locate_qubit_dependencies(self, idx_module, incomingModules):
        """ Given a specific module, return the positions of the qubits in the topology that it depends on. """
        in_module_qubits = self.modules_qubits[idx_module]
        qubit_mapping_dict = {
            module: layout 
            for dict in self.qubit_mapping 
            for module, layout in dict.items()
        }
        qubit_positions = {}
        if len(self.qubit_mapping) > 0:
            for idx_out_module in incomingModules[idx_module]:
                if qubit_mapping_dict.get(idx_out_module) is not None:
                    qubits_dependences = [
                        qubit
                        for qubit in in_module_qubits if qubit in self.modules_qubits[idx_out_module]
                    ]

                    for qubit in qubits_dependences:
                        idx = self.modules_qubits[idx_out_module].index(qubit)
                        qubit_positions[qubit] = qubit_mapping_dict[idx_out_module][idx]
        
        return qubit_positions        

    def generate_qubit_mapping(self):
        """ Generates a qubit mapping for the circuit. """
        # Check if buffer distance and coupling map are set
        if self.buffer_distance is None or self.coupling_map is None:
            raise ValueError("The buffer distance or the coupling map has not been set yet.")

        # Generate adjacency matrix from the graph
        adj_matrix = nx.to_numpy_array(
            self.dependency_graph, 
            nodelist=sorted(self.dependency_graph.nodes())
        )

        # Initialize lists to store incoming and outgoing edges
        incoming_edges = [[] for i in range(len(adj_matrix))]
        outgoing_edges = [[] for i in range(len(adj_matrix))]
        active_outgoing_edges = [[] for i in range(len(adj_matrix))]

        # Fill lists based on the adjacency matrix
        for i in range(len(adj_matrix)):
            for j in range(len(adj_matrix)):
                if adj_matrix[i, j] == 1:
                    incoming_edges[j].append(i)
                    outgoing_edges[i].append(j)
                    active_outgoing_edges[i].append(j)
        
        # Initialize nodes_order lists
        nodes_order = []
        mapped_nodes = set()
        while len(mapped_nodes) < len(incoming_edges):
            group_nodes = [
                i for i, outgoing_edges_i in enumerate(active_outgoing_edges)
                if len(outgoing_edges_i) == 0 and i not in mapped_nodes
            ]
            mapped_nodes.update(group_nodes)
            for node in group_nodes:
                for j in range(len(active_outgoing_edges)):
                    if node in active_outgoing_edges[j]:
                        active_outgoing_edges[j].remove(node)
            nodes_order = [group_nodes] + nodes_order

        print(f'Nodes order: {nodes_order} - Mapped nodes: {mapped_nodes}')

        mapped_qubits_to_preserve = []

        for current_nodes in nodes_order:
            if len(current_nodes) == 1:
                # Update the mapped qubits to preserve list
                in_qubits = self.locate_qubit_dependencies(
                    current_nodes[0], 
                    incoming_edges
                )
                mapped_qubits_to_preserve = [qubit for qubit in mapped_qubits_to_preserve if qubit not in list(in_qubits.values())]
                # If there is only one node in the current set of modules, select the layout that 
                # requires the fewest swap gates if the module depends on others. If not, choose 
                # the first layout from the list.
                layouts = self.get_layouts(
                    idx_module=current_nodes[0], 
                    incomingModules=incoming_edges
                )
                layouts = [
                    layout for layout in layouts
                    if not any(preserved_qubit in layout for preserved_qubit in mapped_qubits_to_preserve)
                ]
                if len(incoming_edges[current_nodes[0]]) > 0:
                    layout_swap_gate_count = []
                    swap_gate_count = 0
                    for layout in layouts:
                        for qubit, pos in in_qubits.items():
                            swap_gate_count += self.coupling_map.distance(
                                pos, 
                                layout[self.modules_qubits[current_nodes[0]].index(qubit)]
                            )
                        layout_swap_gate_count.append(swap_gate_count)
                    layout_idx = layout_swap_gate_count.index(min(layout_swap_gate_count))
                else:
                    layout_idx = 0
                
                chosen_layout = layouts[layout_idx]
                max_clique_layouts = {current_nodes[0]:chosen_layout}

                # Update the mapped qubits to preserve list
                for outgoing_edge in outgoing_edges[current_nodes[0]]:
                    for i in range(len(self.modules_qubits[current_nodes[0]])):
                        if self.modules_qubits[current_nodes[0]][i] in self.modules_qubits[outgoing_edge]:
                            mapped_qubits_to_preserve.append(max_clique_layouts[current_nodes[0]][i])
                        
            else:
                # Build the compatibility graph for the current set of modules (nodes)
                comp_graph = self.build_compatibility_graph(
                    current_nodes, 
                    outgoing_edges, 
                    self.max_allowed_weight, 
                    incoming_edges,
                    mapped_qubits_to_preserve
                )
                # Find the maximum clique in the compatibility graph
                max_clique, max_clique_weight = maxClique.find_max_clique(
                    G=comp_graph.to_undirected(), 
                    heuristic=self.heuristic
                )
                max_clique_layouts = {}
                for vertex in max_clique:
                    max_clique_layouts[vertex[0]] = comp_graph.nodes[vertex]['layout']

                for node in current_nodes:
                    for outgoing_edge in outgoing_edges[node]:
                        for i in range(len(self.modules_qubits[node])):
                            if self.modules_qubits[node][i] in self.modules_qubits[outgoing_edge]:
                                mapped_qubits_to_preserve.append(max_clique_layouts[node][i])

            # Add the maximum clique to the qubit mapping
            self.qubit_mapping.append(max_clique_layouts)
        
            print(f"Qubit to preserve: {mapped_qubits_to_preserve}")


    def build_compatibility_graph(self, 
                                  current_modules_idx, 
                                  dependentModules, 
                                  max_allowed_weight, 
                                  incomingModules,
                                  mapped_qubits_to_preserve):
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
            in_qubits = self.locate_qubit_dependencies(idx_module, incomingModules)
            mapped_qubits_to_preserve[:] = [qubit for qubit in mapped_qubits_to_preserve if qubit not in list(in_qubits.values())]
            layouts = self.get_layouts(idx_module=idx_module, incomingModules=incomingModules)
            layouts = [
                layout for layout in layouts
                if not any(preserved_qubit in layout for preserved_qubit in mapped_qubits_to_preserve)
            ]
            for idx_layout, layout in enumerate(layouts):
                comp_graph.add_node((idx_module, idx_layout), layout=layout)

        # Add edges to the graph
        for v1, attributes1 in comp_graph.nodes(data=True):
            for v2, attributes2 in comp_graph.nodes(data=True):
                if v1[0] != v2[0]:
                    layout1 = attributes1['layout']
                    layout2 = attributes2['layout']

                    # Check if layouts b-overlap
                    overlapping = overlap.check_b_overlap(
                        layout1, 
                        layout2, 
                        self.coupling_map, 
                        self.buffer_distance
                    )

                    if not overlapping:
                        edge_weight = self.compute_edge_weight(
                            dependentModules=dependentModules, 
                            incomingModules=incomingModules,
                            module_idx1=v1[0], 
                            module_idx2=v2[0], 
                            layout1=layout1, 
                            layout2=layout2
                        )

                        if edge_weight <= max_allowed_weight:
                            # Update maximum weight
                            if edge_weight > max_weight:
                                max_weight = edge_weight

                            # Add edge to the graph with the computed weight
                            comp_graph.add_edge(v1, v2, weight=edge_weight)

        for u, v, data in comp_graph.edges(data=True):
            comp_graph[u][v]['weight'] = max_weight - data['weight']

        return comp_graph
