from src import compatibilityGraph as compGraph
from src import circuit_gen as circGen
import networkx as nx


numModules = 2
moduleMaxQubits = 5
moduleMaxGates = 3
circuit = circGen.RandomCircuit(numModules, moduleMaxQubits, moduleMaxGates)
circuit.gen_random_circuit()

bufferDistance = 2
#it's possible to define a specific backend. if nothing is specified, the FakeGuadalupeV2 will be used
compatibilityGraph = compGraph.CompatibilityGraph(bufferDistance, circuit.modules)
resultingGraph = compatibilityGraph.generateCompatibilityGraph()
nx.draw(resultingGraph)
