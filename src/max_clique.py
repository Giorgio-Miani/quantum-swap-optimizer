import networkx as nx

def clique_weight_sum(G, clique):
    """ Calculate the sum of weights in the given clique. """
    return sum(G[u][v]['weight'] for i, u in enumerate(clique) for v in clique[i+1:])

def find_max_clique(g):
    """ Find the clique with the maximum sum of weights in the graph. """
    # Initialize variables to keep track of the maximum weight and corresponding clique
    max_weight = -float('inf')
    max_clique = None

    # Iterate over all cliques in the graph
    for clique in nx.find_cliques(g):
        # Calculate the sum of weights in the current clique
        weight_sum = clique_weight_sum(g, clique)

        # Update the maximum weight and corresponding clique if the current clique has a higher weight
        if weight_sum > max_weight:
            max_weight = weight_sum
            max_clique = clique

    return max_clique, max_weight

