# Note: The `layout` variable is structured as follows:
#   - `layout[0]`: Contains the actual layout.
#   - `layout[1]`: Represents the noise associated with this layout.
#
# The `module` variable is expected to be structured as:
#   - `module[i][0]`: Contains the layouts for the `i`-th module.
#   - `module[i][1]`: Represents the normalized circuit area (A_i) for the `i`-th module.

import networkx as nx
from qiskit import transpile
from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2
import mapomatic as mm
import b_overlap as overlap

def convert_module_for_comp_graph(module, backend):
    """ Convert a module into a format suitable for the CompatibilityGraph class. """
    # Calculate the area of the module
    area = module.num_qubits * module.depth()

    # Initialize the compatibility graph module format with an empty list for layouts and the computed area
    comp_graph_module = ([], area)

    try:
        # Generate possible layouts for the module on the given backend
        layouts = generate_layouts(module, backend)

        # Append all generated layouts to the first element of comp_graph_module
        comp_graph_module[0].extend(layouts)
    except Exception as e:
        # Handle possible errors during layout generation
        print(f"Error generating layouts: {e}")

    return comp_graph_module

def convert_modules_for_comp_graph(modules, backend):
    """ Convert a dictionary of modules to their compatibility graph representations. """
    try:
        # Convert each module in modules
        comp_modules = [convert_module_for_comp_graph(module, backend) for module in modules.values()]
    except Exception as e:
        print(f"Error converting modules: {e}")
        return []

    return comp_modules

def generate_layouts(module, backend):
    """ Searches for, optimizes and evaluates quantum circuit layouts for a specified backend. """    
    trans_qc = transpile(module, backend, optimization_level=3)
    small_qc = mm.deflate_circuit(trans_qc)
    layouts = mm.matching_layouts(small_qc, backend)
    scores = mm.evaluate_layouts(small_qc, layouts, backend)
    return scores

class CompatibilityGraph:
    def __init__(self, modules, dependency_graph, modules_qubits, backend=FakeGuadalupeV2(), buffer_distance=1):
        self.graph            = nx.DiGraph()
        self.buffer_distance  = buffer_distance
        self.backend          = backend
        self.coupling_map     = backend.coupling_map
        self.modules          = convert_modules_for_comp_graph(modules, backend)
        self.maxWeight        = 0
        self.modules_qubits = modules_qubits
        self.dependency_graph = dependency_graph

    def generate_compatibility_graph(self, coeff_1 = 0.01, coeff_2 = 0.01):
        """ Generates a compatibility graph based on the modules, their layouts, and the coupling map. """
        # Check if buffer distance and coupling map are set
        if self.buffer_distance is None or self.coupling_map is None:
            raise ValueError("The buffer distance or the coupling map has not been set yet.")
        
        # Add nodes to the graph
        for module_index, module in enumerate(self.modules):
            for layout_index, layout in enumerate(module[0]):
                vertex = (module_index, layout_index)
                self.graph.add_node(vertex)
        
        # Compute edge weights based on compatibility
        for v1 in self.graph.nodes:
            for v2 in self.graph.nodes:
                if v1[0] != v2[0]:
                    layout1 = self.modules[v1[0]][0][v1[1]]
                    layout2 = self.modules[v2[0]][0][v2[1]]

                    # Check if layouts b-overlap
                    overlapping, layout_distance = overlap.check_b_overlap(layout1[0], layout2[0], self.coupling_map, self.buffer_distance, self.modules_qubits[v1[0]], self.modules_qubits[v2[0]])

                    if not overlapping:
                        # Compute edge weight
                        weight_layout1   = layout1[1]
                        weight_layout2   = layout2[1]
                        normalized_area1 = self.modules[v1[0]][1]
                        normalized_area2 = self.modules[v2[0]][1]
                        # if nx.has_path(self.dependency_graph, source=v1[0], target=v2[0]):
                        if self.dependency_graph.has_edge(v1[0], v2[0]):
                            print(self.modules_qubits[v1[0]])
                            print(self.modules_qubits[v2[0]])
                            edge_weight = weight_layout1 * normalized_area1 + weight_layout2 * normalized_area2 - ((coeff_1 / nx.shortest_path_length(self.dependency_graph, source=v1[0], target=v2[0])) + (coeff_2 / layout_distance))
                        else:
                            edge_weight = weight_layout1 * normalized_area1 + weight_layout2 * normalized_area2
                        
                        # Update maximum weight
                        if edge_weight > self.maxWeight:
                            self.maxWeight = edge_weight
                        
                        # Add edge to the graph with the computed weight
                        self.graph.add_edge(v1, v2, weight=edge_weight)
        
        for u, v, data in self.graph.edges(data=True):
            self.graph[u][v]['weight'] = self.maxWeight - data['weight']
    
    def set_buffer_distance(self, distance):
        self.buffer_distance = distance

    def set_coupling_map(self, coupling_map):
        self.coupling_map = coupling_map
