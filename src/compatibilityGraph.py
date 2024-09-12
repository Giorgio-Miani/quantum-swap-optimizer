import networkx as nx
from src import b_overlap as overlap
from src import layoutGenerator as layoutGen
from qiskit_ibm_runtime.fake_provider import FakeGuadalupeV2


# note: I imagine that layout is given me like this: layout[0] = actual layout, layout[1] = noise indicator
# I also imagine that module[0][0] = actual module with its layout, module[0][1] = module's normalized circuit area Ai

def convertSingleModuleForCompGraph(module, backend):
    # obtaining module's normalized circuit area
    area = module.num_qubits * module.depth()
    compModule = ([], area)
    layouts = layoutGen.generateLayouts(module, backend)
    for layout in layouts:
        compModule[0].append(layout)
    return compModule


def convertModulesForCompGraph(modules, backend):
    compModules = []
    for module in modules.values():
        compModules.append(convertSingleModuleForCompGraph(module, backend))
    return compModules


class CompatibilityGraph:
    def __init__(self, buffer_distance=None, modules=None, backend=FakeGuadalupeV2()):
        self.graph = nx.DiGraph()
        self.buffer_distance = buffer_distance
        self.backend = backend
        self.coupling_map = self.backend.coupling_map
        self.modules = convertModulesForCompGraph(modules, self.backend)
        self.maxWeight = 0

    def addModule(self, module):
        self.modules.append(convertSingleModuleForCompGraph(module, self.backend))

    def removeModule(self, module):
        self.modules.remove(convertSingleModuleForCompGraph(module, self.backend))

    # Note: here layout and module need to be in the right format accepted by this class
    def addLayout(self, layout, module):
        for mod in self.modules:
            if mod == module:
                if layout not in mod:
                    mod[0].append(layout)

    # Note: here layout and module need to be in the right format accepted by this class
    def removeLayout(self, layout, module):
        for mod in self.modules:
            if mod == module:
                if layout in mod:
                    mod[0].remove(layout)

    def generateCompatibilityGraph(self):
        if self.buffer_distance is None or self.coupling_map is None:
            print("error, the buffer distance or coupling map has not been set yet\n")
        for mod_index, mod in enumerate(self.modules):
            for lay_index, lay in enumerate(mod[0]):
                vertex = (mod_index, lay_index)
                self.graph.add_node(vertex)
        #print(self.graph.number_of_nodes())
        for v1 in self.graph.nodes:
            for v2 in self.graph.nodes:
                if v1[0] != v2[0]:
                    layout1 = self.modules[v1[0]][0][v1[1]]
                    layout2 = self.modules[v2[0]][0][v2[1]]
                    if not overlap.check_b_overlap(layout1[0], layout2[0], self.coupling_map, self.buffer_distance):
                        weightLay1 = layout1[1]
                        weightLay2 = layout2[1]
                        normalizedArea1 = self.modules[v1[0]][1]
                        normalizedArea2 = self.modules[v2[0]][1]
                        myWeight = weightLay1 * normalizedArea1 + weightLay2 * normalizedArea2
                        if myWeight > self.maxWeight:
                            self.maxWeight = myWeight
                        self.graph.add_edge(v1, v2, weight=myWeight)
        for u, v, data in self.graph.edges(data=True):
            self.graph[u][v]['weight'] = self.maxWeight - data['weight']
        return self.graph

    def setBufferDistance(self, distance):
        self.buffer_distance = distance

    def setCouplingMap(self, myMap):
        self.coupling_map = myMap
