""" Given a size of a graph, each method generates the graph in
the corresponding topology, that is closest to the given size.
"""

import math
import networkx as nx

def complete(size):
    G = nx.complete_graph(size)
    G.name = 'complete'
    return G

def complete_bipartite(size):
    m = n = int(size/2)
    G = nx.complete_bipartite_graph(m,n)
    G.name = 'bipartite'
    return G

def grid_2d(size):
    m = n = int(math.sqrt(size))
    G = nx.grid_2d_graph(m,n)
    G.name = 'grid2d'
    return G

def hypercube(size):
    n = int(math.log(size,2))
    G = nx.hypercube_graph(n)
    G.name = 'hypercube'
    return G

def rooks(size):
    m = n = int(math.sqrt(size))
    G = nx.complete_graph(n)
    H = nx.complete_graph(m)
    F = nx.cartesian_product(G,H)
    F.name = 'rooks'
    return F

def grid_3d(size):
    m = n = t = int(size**(1./3.))
    G = nx.grid_graph(dim=[m,n,t])
    G.name = 'grid3d'
    return G
