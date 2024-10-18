import networkx as nx
from networkx.algorithms.approximation import max_clique

def clique_weight_sum(G, clique):
    """ Calculate the sum of weights in the given clique. """
    return sum(G[u][v]['weight'] for i, u in enumerate(clique) for v in clique[i+1:])

def find_max_clique(G, heuristic=False):
    """ Find the clique with the maximum sum of weights in the graph. """
    if not heuristic:
        # Initialize variables to keep track of the maximum weight and corresponding clique
        max_weight = -float('inf')
        maximal_clique = None

        # Iterate over all cliques in the graph
        for clique in nx.find_cliques(G):
            # Calculate the sum of weights in the current clique
            weight_sum = clique_weight_sum(G, clique)

            # Update the maximum weight and corresponding clique if the current clique has a higher weight
            if weight_sum > max_weight:
                max_weight = weight_sum
                maximal_clique = clique
        
        # print(f"Maximal clique: {maximal_clique}, Maximal weight: {max_weight}")
        return maximal_clique, max_weight
    else:
        maximal_clique = list(max_clique(G))
        max_weight = clique_weight_sum(G, maximal_clique)

        # print(f"Maximal clique: {maximal_clique}, Maximal weight: {max_weight}")
        return maximal_clique, max_weight

