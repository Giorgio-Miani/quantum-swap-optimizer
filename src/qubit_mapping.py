# Standard library import
from collections import deque

# Third-party imports
import networkx as nx
from qiskit import transpile
from qiskit.transpiler import CouplingMap

# Local imports
import mapomatic as mm
import backend_gen as bg
import max_clique as maxClique

def generate_layouts(module, backend, coupling_map = None):
    """ Searches for, optimizes and evaluates quantum circuit layouts for a specified backend. """
    basis_gates = ['h', 'cx', 's', 'sdg', 'x', 't', 'tdg']
    
    if coupling_map is not None:
        trans_qc = transpile(module, 
                             backend, 
                             routing_method='sabre',
                             basis_gates=basis_gates,
                             optimization_level=3, 
                             coupling_map=coupling_map)
        small_qc = mm.deflate_circuit(trans_qc)
        layouts = mm.matching_layouts(small_qc, coupling_map)
    else:
        trans_qc = transpile(module, 
                             backend, 
                             routing_method='sabre',
                             basis_gates=basis_gates, 
                             optimization_level=3)
        small_qc = mm.deflate_circuit(trans_qc)
        layouts = mm.matching_layouts(small_qc, backend)
    return layouts

def get_benchmark_metrics(module, backend, coupling_map = None):
    """Return the benchmark metric of the circuit. """
    basis_gates = ['h', 'cx', 's', 'sdg', 'x', 't', 'tdg']

    optimized_circuit = transpile(module, 
                                  backend, 
                                  routing_method='sabre',
                                  basis_gates=basis_gates,
                                  optimization_level=3, 
                                  coupling_map=coupling_map)
    
    # Extract basic metrics
    depth = optimized_circuit.depth()
    total_qubits = optimized_circuit.num_qubits
    gate_count = optimized_circuit.size()

    # Calculate T-count and T-depth
    t_count = 0
    t_depth = 0
    current_depth = 0
    for gate in optimized_circuit.data:
        if gate[0].name == 't':
            t_count += 1
            current_depth += 1  # Increment current depth for T-gate
        elif gate[0].name == 'tdg':
            t_count += 1
            current_depth += 1  # Increment current depth for T-dg gate
        else:
            # If it's a different gate, update the T-depth if needed
            if current_depth > t_depth:
                t_depth = current_depth
            current_depth = 0  # Reset for non-T gates

    # Final check for the last sequence of T-gates
    if current_depth > t_depth:
        t_depth = current_depth

    metrics = {'depth': depth, 
               'total_qubits': total_qubits, 
               'gate_count': gate_count, 
               't_count': t_count, 
               't_depth': t_depth}
    
    return metrics

class QubitMapping:
    def __init__(self, 
                 circuit, 
                 backend, 
                 coupling_map_dims,
                 reduced_distance=None, 
                 max_allowed_weight=10,
                 heuristic=False):
        # Initialize essential attributes
        self.backend = backend
        self.coupling_map = backend.coupling_map
        self.coupling_map_dims = coupling_map_dims
        self.reduced_distance = reduced_distance
        self.max_allowed_weight = max_allowed_weight
        self.heuristic = heuristic

        # Initialize the circuit-related attributes
        self.dependency_graph = circuit.dependency_graph
        self.descendants = {}
        for node in self.dependency_graph:
            self.descendants[node] = nx.descendants(self.dependency_graph, node)
        self.modules = circuit.modules
        self.modules_qubits = circuit.modules_qubits
        self.modules_dependencies = {}

        # Initialize the qubit mapping-related attributes
        self.qubit_mapping = []
        self.reduced_coupling_maps = {}

        # Initialize the benchmark metrics
        self.benchmark_metrics = {
            'depth': 0,
            'total_qubits': 0,
            'gate_count': 0,
            'swap_count':0,
            't_count': 0,
            't_depth': 0
        }
        self.modules_metrics = {}
   
    def generate_ASAP_qubit_mapping(self):
        """ Generates a qubit mapping for the circuit using the As Soon As Possible (ASAP) scheduling. """
        if self.coupling_map is None:
            raise ValueError("The coupling map has not been set yet.")
        
        num_nodes = self.dependency_graph.number_of_nodes()
        incoming_edges = [[] for _ in range(num_nodes)]
        outgoing_edges = [[] for _ in range(num_nodes)]

        # Fill lists based on edges in the graph
        for u, v in self.dependency_graph.edges():
            outgoing_edges[u].append(v)  # u -> v
            incoming_edges[v].append(u)  # u -> v

        mapped_nodes = set()
        mapped_qubits_to_preserve = []

        # ASAP scheduling
        nodes_order = self.generate_ASAP_module_order(mapped_nodes)
        print(f"Nodes order: {nodes_order}")

        self.map_fist_executable_modules_ASAP(nodes_order, 
                                              mapped_nodes, 
                                              mapped_qubits_to_preserve, 
                                              incoming_edges, 
                                              outgoing_edges)
        
        while len(nodes_order) > 0:
            self.map_executable_modules_ASAP(nodes_order, 
                                             mapped_nodes, 
                                             mapped_qubits_to_preserve, 
                                             incoming_edges, 
                                             outgoing_edges)

        self.update_benchmark_metrics()
   
    def generate_ASAP_module_order(self, mapped_nodes):
        """ Generates the As Soon As Possible (ASAP) module order. """
        nodes_order = []
        processed_nodes = set()
        topological_order = list(nx.topological_sort(self.dependency_graph))
        topological_order = [node for node in topological_order if node not in mapped_nodes]

        while topological_order:
            batch = []
            for node in topological_order:
                # Check if the node has dependencies that are already processed
                if all(pred in processed_nodes for pred in self.dependency_graph.predecessors(node)):
                    batch.append(node)

            if batch:
                nodes_order.append(batch)
                processed_nodes.update(batch)
                # Remove the processed nodes from the topological order
                topological_order = [node for node in topological_order if node not in batch]
        
        return nodes_order

    def map_fist_executable_modules_ASAP(self,
                                         nodes_order,
                                         mapped_nodes,
                                         mapped_qubits_to_preserve,
                                         incoming_edges,
                                         outgoing_edges):
        """ Maps the first executable modules in the circuit. """
        current_nodes = nodes_order.pop(0)

        self.modules_dependencies = {
            node_index: self.locate_qubit_dependencies(node_index, incoming_edges) for node_index in self.dependency_graph.nodes
        }

        if len(current_nodes) == 1:
            layout_idx = 0
            idx_module = current_nodes[0]
            reduced_distance = self.reduced_distance if self.reduced_distance is not None else len(self.modules_qubits[idx_module])
            reduced_coupling_map = bg.generate_regular_coupling_map(
                reduced_distance, 
                reduced_distance
            )
            self.reduced_coupling_maps[idx_module] = reduced_coupling_map

            # Generate layouts and metrics for the module
            layouts = generate_layouts(
                self.modules[idx_module], 
                self.backend, 
                coupling_map=reduced_coupling_map
            )
            self.modules_metrics[idx_module] = get_benchmark_metrics(
                self.modules[idx_module], 
                self.backend, 
                coupling_map=reduced_coupling_map
            )
            chosen_reduced_layout = layouts[layout_idx]

            # Center the layout in the true topology
            chosen_layout = [] 
            central_row_idx = (self.coupling_map_dims[0] - reduced_distance) // 2
            central_col_idx = (self.coupling_map_dims[1] - reduced_distance) // 2
            for qubit in chosen_reduced_layout:
                qubit_coords_on_reduced_topology = (qubit // reduced_distance, qubit % reduced_distance)
                chosen_layout.append((central_row_idx + qubit_coords_on_reduced_topology[0]) * self.coupling_map_dims[1] + 
                                      central_col_idx + qubit_coords_on_reduced_topology[1])                            

            max_clique_layouts = {idx_module:chosen_layout}

            # Update the mapped nodes set
            mapped_nodes.update(current_nodes)

            # Update the mapped qubits to preserve list
            for outgoing_edge in outgoing_edges[idx_module]:
                for i in range(len(self.modules_qubits[idx_module])):
                    if self.modules_qubits[idx_module][i] in self.modules_qubits[outgoing_edge]:
                        mapped_qubits_to_preserve.append(max_clique_layouts[idx_module][i])
                    
        else:
            # Build the compatibility graph for the current set of modules (nodes)
            comp_graph = self.build_first_compatibility_graph(
                current_nodes, 
                outgoing_edges
            )
            # Find the maximum clique in the compatibility graph
            max_clique, max_clique_weight = maxClique.find_max_clique(
                G=comp_graph.to_undirected(), 
                heuristic=self.heuristic,
                modules=self.modules_qubits
            )
            max_clique_layouts = {}
            for vertex in max_clique:
                max_clique_layouts[vertex[0]] = comp_graph.nodes[vertex]['layout']

            # Update the mapped nodes set
            mapped_nodes.update({node for node in current_nodes if max_clique_layouts.get(node) is not None})

            # Update the mapped qubits to preserve list
            for node in current_nodes:
                for outgoing_edge in outgoing_edges[node]:
                    for i in range(len(self.modules_qubits[node])):
                        if self.modules_qubits[node][i] in self.modules_qubits[outgoing_edge] and max_clique_layouts.get(node) is not None:
                            mapped_qubits_to_preserve.append(max_clique_layouts[node][i])
            
            if len(current_nodes) > len(max_clique_layouts):
                nodes_order = self.generate_ASAP_module_order(mapped_nodes)
                print(f"Nodes order: {nodes_order}")

        # Add the maximum clique to the qubit mapping
        self.qubit_mapping.append(max_clique_layouts)
    
        print(f"Qubit to preserve: {mapped_qubits_to_preserve}")
    
    def build_first_compatibility_graph(self, 
                                        current_modules_idx, 
                                        dependentModules):
        """ Builds the compatibility graph for the first set of modules. """
        if self.coupling_map is None:
            raise ValueError("The coupling map has not been set yet.")

        # Initialize compatibility graph
        comp_graph = nx.DiGraph()

        # Max weight
        max_weight = 0

        reduced_distance = self.reduced_distance if self.reduced_distance is not None else max(len(self.modules_qubits[idx_module]) for idx_module in current_modules_idx)
        central_row_idx = (self.coupling_map_dims[0] - reduced_distance) // 2
        central_col_idx = (self.coupling_map_dims[1] - reduced_distance) // 2
        reduced_coupling_map = bg.generate_regular_coupling_map(
            reduced_distance, 
            reduced_distance
        )

        # Add nodes to the graph
        for idx_module in current_modules_idx:
            self.reduced_coupling_maps[idx_module] = reduced_coupling_map
            layouts = generate_layouts(
                self.modules[idx_module], 
                self.backend, 
                coupling_map=reduced_coupling_map
            )
            self.modules_metrics[idx_module] = get_benchmark_metrics(
                self.modules[idx_module], 
                self.backend, 
                coupling_map=reduced_coupling_map
            )
            for idx_layout, reduced_layout in enumerate(layouts):
                layout = []
                for qubit in reduced_layout:
                    qubit_coords_on_reduced_topology = (qubit // reduced_distance, 
                                                        qubit % reduced_distance)
                    layout.append((central_row_idx + qubit_coords_on_reduced_topology[0]) * self.coupling_map_dims[1] + 
                                   central_col_idx + qubit_coords_on_reduced_topology[1])  
                comp_graph.add_node((idx_module, idx_layout), 
                                    layout=layout)

        # Add edges to the graph
        for v1, attributes1 in comp_graph.nodes(data=True):
            for v2, attributes2 in comp_graph.nodes(data=True):
                if v1[0] != v2[0]:
                    layout1 = attributes1['layout']
                    layout2 = attributes2['layout']

                    # Check if layouts overlap
                    overlapping = bool(set(layout1) & set(layout2))

                    if not overlapping:
                        edge_weight = self.compute_edge_weight(
                            dependentModules=dependentModules, 
                            module_idx1=v1[0], 
                            module_idx2=v2[0], 
                            layout1=layout1, 
                            layout2=layout2
                        )

                        if edge_weight <= self.max_allowed_weight:
                            # Update maximum weight
                            if edge_weight > max_weight:
                                max_weight = edge_weight

                            # Add edge to the graph with the computed weight
                            comp_graph.add_edge(v1, v2, weight=edge_weight)

        for u, v, data in comp_graph.edges(data=True):
            comp_graph[u][v]['weight'] = max_weight - data['weight']

        return comp_graph
    
    def map_executable_modules_ASAP(
            self,
            nodes_order,
            mapped_nodes,
            mapped_qubits_to_preserve,
            incoming_edges,
            outgoing_edges
        ):
        current_nodes = nodes_order.pop(0)

        self.modules_dependencies = {
            node_index: self.locate_qubit_dependencies(node_index, incoming_edges) for node_index in self.dependency_graph.nodes
        }

        if len(current_nodes) == 1:
            # Update the mapped qubits to preserve list
            in_qubits = self.modules_dependencies[current_nodes[0]]
            mapped_qubits_to_preserve = [qubit for qubit in mapped_qubits_to_preserve if qubit not in list(in_qubits.values())]

            # Select the layout that requires the fewest swap gates.
            layouts, module_metrics = self.get_layouts_and_metrics(
                idx_module=current_nodes[0], 
                incomingModules=incoming_edges
            )
            layouts = [
                layout for layout in layouts
                if not any(preserved_qubit in layout for preserved_qubit in mapped_qubits_to_preserve)
            ]
            layout_swap_gate_count = []
            swap_gate_count = 0
            for layout in layouts:
                for qubit, pos in in_qubits.items():
                    swap_gate_count += self.coupling_map.distance(
                        pos, 
                        layout[self.modules_qubits[current_nodes[0]].index(qubit)]
                    )
                layout_swap_gate_count.append(swap_gate_count)
                swap_gate_count = 0
            layout_idx = layout_swap_gate_count.index(min(layout_swap_gate_count))
            
            chosen_layout = layouts[layout_idx]
            max_clique_layouts = {current_nodes[0]:chosen_layout}

            # Update the mapped nodes set
            mapped_nodes.update(current_nodes)

            # Update the benchmark metrics
            self.modules_metrics[current_nodes[0]] = module_metrics

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
                incoming_edges,
                mapped_qubits_to_preserve
            )
            # Find the maximum clique in the compatibility graph
            max_clique, max_clique_weight = maxClique.find_max_clique(
                G=comp_graph.to_undirected(), 
                heuristic=self.heuristic,
                modules=self.modules_qubits
            )
            max_clique_layouts = {}
            for vertex in max_clique:
                max_clique_layouts[vertex[0]] = comp_graph.nodes[vertex]['layout']

            # Update the mapped nodes set
            mapped_nodes.update({node for node in current_nodes if max_clique_layouts.get(node) is not None})

            # Update the mapped qubits to preserve list
            for node in current_nodes:
                for outgoing_edge in outgoing_edges[node]:
                    for i in range(len(self.modules_qubits[node])):
                        if self.modules_qubits[node][i] in self.modules_qubits[outgoing_edge] and max_clique_layouts.get(node) is not None:
                            mapped_qubits_to_preserve.append(max_clique_layouts[node][i])
            
            if len(current_nodes) > len(max_clique_layouts):
                nodes_order = self.generate_ASAP_module_order(mapped_nodes)
                print(f"Nodes order: {nodes_order}")

        # Add the maximum clique to the qubit mapping
        self.qubit_mapping.append(max_clique_layouts)
    
        print(f"Qubit to preserve: {mapped_qubits_to_preserve}")

    def build_compatibility_graph(self, 
                                  current_modules_idx, 
                                  dependentModules, 
                                  incomingModules,
                                  mapped_qubits_to_preserve):
        """ Builds a compatibility graph for a given set of modules. """
        if self.coupling_map is None:
            raise ValueError("The coupling map has not been set yet.")

        # Initialize compatibility graph
        comp_graph = nx.DiGraph()

        # Max weight
        max_weight = 0

        # Add nodes to the graph
        for idx_module in current_modules_idx:
            in_qubits = self.modules_dependencies[idx_module]
            mapped_qubits_to_preserve[:] = [qubit for qubit in mapped_qubits_to_preserve if qubit not in list(in_qubits.values())]
            layouts, module_metrics = self.get_layouts_and_metrics(
                idx_module=idx_module, 
                incomingModules=incomingModules
            )
            layouts = [
                layout for layout in layouts
                if not any(preserved_qubit in layout for preserved_qubit in mapped_qubits_to_preserve)
            ]
            self.modules_metrics[idx_module] = module_metrics
            for idx_layout, layout in enumerate(layouts):
                comp_graph.add_node((idx_module, idx_layout), 
                                    layout=layout, 
                                    num_qubits=len(layout))

        # Add edges to the graph
        for v1, attributes1 in comp_graph.nodes(data=True):
            for v2, attributes2 in comp_graph.nodes(data=True):
                if v1[0] != v2[0]:
                    layout1 = attributes1['layout']
                    layout2 = attributes2['layout']

                    # Check if layouts overlap
                    overlapping = bool(set(layout1) & set(layout2))

                    if not overlapping:
                        edge_weight = self.compute_edge_weight(
                            dependentModules=dependentModules, 
                            module_idx1=v1[0], 
                            module_idx2=v2[0], 
                            layout1=layout1, 
                            layout2=layout2
                        )

                        if edge_weight <= self.max_allowed_weight:
                            # Update maximum weight
                            if edge_weight > max_weight:
                                max_weight = edge_weight

                            # Add edge to the graph with the computed weight
                            comp_graph.add_edge(v1, v2, weight=edge_weight)

        for u, v, data in comp_graph.edges(data=True):
            comp_graph[u][v]['weight'] = max_weight - data['weight']

        return comp_graph

    def compute_edge_weight(self, 
                            dependentModules, 
                            module_idx1, 
                            module_idx2, 
                            layout1, 
                            layout2):
        """ Computes the weight of the edge between two nodes in the compatibility graph. """
        common_dependences = [element for element in self.descendants[module_idx1] if 
                              element in self.descendants[module_idx2]]

        edge_weight = 0
        for dep in common_dependences:
            next_node1 = nx.shortest_path(self.dependency_graph, source=module_idx1, target=dep)[1]
            next_node2 = nx.shortest_path(self.dependency_graph, source=module_idx2, target=dep)[1]
            qubits_dependences = [
                (qubit1, qubit2)
                for qubit1 in self.modules_qubits[next_node1] if qubit1 in self.modules_qubits[module_idx1]
                for qubit2 in self.modules_qubits[next_node2] if qubit2 in self.modules_qubits[module_idx2]
            ]

            for qubits_couple in qubits_dependences:
                idx1 = self.modules_qubits[module_idx1].index(qubits_couple[0])
                idx2 = self.modules_qubits[module_idx2].index(qubits_couple[1])
                x1 = layout1[idx1] // self.coupling_map_dims[1]
                y1 = layout1[idx1] % self.coupling_map_dims[1]
                x2 = layout2[idx2] // self.coupling_map_dims[1]
                y2 = layout2[idx2] % self.coupling_map_dims[1]
                distance = abs(x1 - x2) + abs(y1 - y2)
                edge_weight += distance

        in_qubits1 = self.modules_dependencies[module_idx1]
        in_qubits2 = self.modules_dependencies[module_idx2]

        for qubit1, pos1 in in_qubits1.items():
            edge_weight += self.coupling_map.distance(
                pos1, 
                layout1[self.modules_qubits[module_idx1].index(qubit1)]
            )

        for qubit2, pos2 in in_qubits2.items():
            edge_weight += self.coupling_map.distance(
                pos2, 
                layout2[self.modules_qubits[module_idx2].index(qubit2)]
            )

        return edge_weight

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

    def get_layouts_and_metrics(self, idx_module, incomingModules):
        module   = self.modules[idx_module]
        inModule = self.modules_qubits[idx_module]
        output_qubits = []
        qubit_mapping_dict = {module: layout for dict in self.qubit_mapping for module, layout in dict.items()}
        for outModule_idx in incomingModules[idx_module]:
            if len(self.qubit_mapping) > 0:
                if qubit_mapping_dict.get(outModule_idx) is not None:
                    output_qubits.extend(
                        self.find_common_qubits(
                            inModule=inModule, 
                            outModule=self.modules_qubits[outModule_idx], 
                            layout=qubit_mapping_dict[outModule_idx]
                        )
                    )
        reduced_distance = self.reduced_distance if self.reduced_distance is not None else len(inModule)
        reduced_coupling_map = self.get_coupling_map_up_to_reduced_distance(
            qubits=output_qubits, 
            reduced_distance=reduced_distance
        )
        self.reduced_coupling_maps[idx_module] = reduced_coupling_map
        layouts = generate_layouts(
            module, 
            self.backend, 
            coupling_map=reduced_coupling_map
        )
        module_metrics = get_benchmark_metrics(
            module, 
            self.backend, 
            coupling_map=reduced_coupling_map
        )

        return layouts, module_metrics

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

    def get_coupling_map_up_to_reduced_distance(self, qubits, reduced_distance):
        """ Obtain a reduced coupling map based on the qubits within a specified reduced distance. """
        # Retrieve the coupling list from the backend configuration
        coupling_list = self.coupling_map.get_edges()

        # Create a set to store all relevant qubits
        relevant_qubits = set()

        # For each starting qubit, find all qubits that are within the specified reduced_distance
        for qubit in qubits:
            qubits_up_to_distance = self.find_qubits_within_distance(qubit, reduced_distance)
            relevant_qubits.update(qubits_up_to_distance)

        # Identify all connections (edges) between the relevant qubits to create a new CouplingMap
        reduced_coupling_list = [edge for edge in coupling_list 
                                if edge[0] in relevant_qubits and edge[1] in relevant_qubits]
        
        return CouplingMap(reduced_coupling_list)

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

    def update_benchmark_metrics(self):
        """ Update the benchmark metrics based on each module's metrics and the qubit scheduling. """
        # Compute the depth along the critical path
        def dfs(node, current_weight):
            current_weight += self.modules_metrics.get(node, {}).get('depth', 0)
            if not self.dependency_graph[node]:
                return current_weight
            return max(dfs(child, current_weight) for child in self.dependency_graph[node])
        
        self.benchmark_metrics['depth'] = max(dfs(node, 0) for node in self.dependency_graph)

        # Compute the total qubits, gate count, T-count    
        used_qubits = set()
        for qubit_mapping_element in self.qubit_mapping:
            for module, assigned_qubits in qubit_mapping_element.items():
                self.benchmark_metrics['gate_count'] += self.modules_metrics[module]['gate_count']
                self.benchmark_metrics['t_count'] += self.modules_metrics[module]['t_count']
                used_qubits.update(assigned_qubits)
        self.benchmark_metrics['total_qubits'] = len(used_qubits)

        # Compute the T-depth


        # Compute the swap count
        adj_matrix = nx.to_numpy_array(
            self.dependency_graph, 
            nodelist=sorted(self.dependency_graph.nodes())
        )
        swap_distance = 0
        for module_idx in range(len(self.modules)):
            outgoing_modules = []
            for j in range(len(adj_matrix)):
                if adj_matrix[module_idx,j] != 0:
                    outgoing_modules.append(j)
            for out_module in outgoing_modules:
                for qubit_idx, qubits in enumerate(self.modules_qubits[module_idx]):
                    if qubits in self.modules_qubits[out_module]:
                        for timestep in self.qubit_mapping:
                            if timestep.get(module_idx) is not None:
                                output = timestep[module_idx][qubit_idx]
                            if timestep.get(out_module) is not None:
                                pos = self.modules_qubits[out_module].index(qubits)
                                input = timestep[out_module][pos]
                        swap_distance += self.coupling_map.distance(output, input)
        self.benchmark_metrics['swap_count'] = swap_distance