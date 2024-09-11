import networkx as nx
import b_overlap as overlap


# note: I imagine that layout is given me like this: layout[0] = actual layout, layout[1] = noise indicator
# I also imagine that module[0][0] = actual module with its layout, module[0][1] = module's normalized circuit area Ai
class CompatibilityGraph:
    def __init__(self, buffer_distance=None, coupling_map=None, modules=None):
        self.graph = nx.DiGraph()
        self.buffer_distance = buffer_distance
        self.coupling_map = coupling_map
        self.modules = modules
        self.maxWeight = 0

    def addModule(self, module):
        self.modules.append(module)

    def removeModule(self, module):
        self.modules.remove(module)

    def addLayout(self, layout, module):
        for mod in self.modules:
            if mod == module:
                if layout not in mod:
                    mod.append(layout)

    def removeLayout(self, layout, module):
        for mod in self.modules:
            if mod == module:
                if layout in mod:
                    mod.remove(layout)

    def generateCompatibilityGraph(self):
        if self.buffer_distance is None or self.coupling_map is None:
            print("error, the buffer distance or coupling map has not been set yet\n")
            for mod_index, mod in enumerate(self.modules):
                for lay_index, lay in enumerate(mod):
                    vertex = (mod_index, lay_index)
                    self.graph.add_node(vertex)
        for v1 in self.graph.nodes:
            for v2 in self.graph.nodes:
                if v1 != v2:
                    layout1 = self.modules[v1[0]][0][v1[1]]
                    layout2 = self.modules[v2[0]][0][v2[1]]
                    if not overlap.check_b_overlap(layout1[0], layout2[0], self.buffer_distance, self.coupling_map):
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
