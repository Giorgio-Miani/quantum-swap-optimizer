import networkx as nx

# Funzione per calcolare la somma dei pesi di un clique
def clique_weight_sum(G, clique):
    return sum(G[u][v]['weight'] for i, u in enumerate(clique) for v in clique[i+1:])

def find_max_clique(g):
    # Trova il clique con la somma dei pesi massima
    max_weight = -float('inf')
    max_clique = None
    for clique in nx.find_cliques(g):
        weight_sum = clique_weight_sum(g, clique)
        if weight_sum > max_weight:
            max_weight = weight_sum
            max_clique = clique

    return max_clique, max_weight

