import networkx as nx
from networkx.algorithms.approximation import max_clique

def clique_weight_sum(G, clique):
    """ Calculate the sum of weights in the given clique. """
    return sum(G[u][v]['weight'] for i, u in enumerate(clique) for v in clique[i+1:])

def clique_num_qubits(clique, modules):
    """ Calculate the number of qubits in the given clique. """
    num_qubits = 0
    for vertex in clique:
        num_qubits += len(modules[vertex[0]])
    return num_qubits

def find_max_clique(G,modules, heuristic=False):
    """ Find the clique with the maximum sum of weights in the graph. """
    if not heuristic:
        # Initialize variables to keep track of the maximum weight and corresponding clique
        max_weight = -float('inf')
        max_clique_num_qubits = 0
        maximal_clique = None

        # Iterate over all cliques in the graph
        for clique in nx.find_cliques(G):
            # Calculate the sum of weights in the current clique
            weight_sum = clique_weight_sum(G, clique)

            # Compute the number of qubits in the current clique
            num_qubits = clique_num_qubits(clique, modules)

            if weight_sum == max_weight and num_qubits > max_clique_num_qubits:
                maximal_clique = clique
                max_clique_num_qubits = num_qubits
            
            # Update the maximum weight and corresponding clique if the current clique has a higher weight
            if weight_sum > max_weight:
                max_weight = weight_sum
                maximal_clique = clique
                max_clique_num_qubits = num_qubits
        
        return maximal_clique, max_weight
    else:
        maximal_clique = list(max_clique(G))
        max_weight = clique_weight_sum(G, maximal_clique)

        return maximal_clique, max_weight

