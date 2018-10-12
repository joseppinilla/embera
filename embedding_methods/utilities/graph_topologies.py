""" Given a size of a graph, each method generates the graph in
the corresponding topology that is closest to the given size.

Some graphs include a layout, or 2D-position for each node.

Graph Attributes:
    name:
    pos: dictionary with (x,y) values for

"""

import math
import random as rand
import networkx as nx


def complete_graph(size):
    G = nx.complete_graph(size)
    G.name = 'complete'
    return G

def complete_bipartite_graph(size):
    m = n = round(size/2)
    G = nx.complete_bipartite_graph(m,n)
    G.name = 'bipartite'
    return G

def grid_2d_graph(size):
    m = n = round(math.sqrt(size))
    G = nx.grid_2d_graph(m,n)
    G.name = 'grid2d'
    G.graph['pos'] = {v:list(v) for v in G}
    return G

def hypercube_graph(size):
    n = round(math.log(size,2))
    G = nx.hypercube_graph(n)
    G.name = 'hypercube'
    return G

def rooks_graph(size):
    n = round(math.sqrt(size))
    G = nx.complete_graph(n)
    H = nx.complete_graph(n)
    F = nx.cartesian_product(G,H)
    F.name = 'rooks'
    F.graph['pos'] = {v:list(v) for v in F}
    return F

def grid_3d_graph(size):
    m = n = t = round(size**(1./3.))
    G = nx.grid_graph(dim=[m,n,t])
    G.name = 'grid3d'
    G.graph['pos'] = {(x,y,z):[x+z,y+z] for (x,y,z) in G}
    return G

def random_graph(size, max_degree=None, seed=None):
    if not max_degree: max_degree=round(size/4)
    G = nx.empty_graph(size)
    rand.seed(seed)
    for v in G:
         N = rand.randrange(1, max_degree)
         node_set = set(G.nodes)
         node_set.remove(v)
         randedges = [ (v,n) for n in rand.sample(node_set, N)]
         G.add_edges_from(randedges)
    G.name = 'random%s' % max_degree
    return G

""" When using graph generators, pruning edges of the source graph can be
done using the following method. (Default to 5% of edges removed).
Example:
>> # Generate a K16 graph with 5% of the edges removed
>> prune_graph(complete_graph)(16)
>> # Generate a K8,8 (16 vertices) with 10% of the edges removed
>> prune_graph(complete_bipartite_graph, edge_yield=0.90)(16)
"""
def _prune(graph, edge_yield):
    num_edges = round( (1.0 - edge_yield) * len(graph))
    for val in range(num_edges):
        while (True):
            u, v = rand.choice(list(graph.edges))
            if len(graph[u]) > 1 and len(graph[v]) > 1:
                break
        graph.remove_edge(u,v)
    return graph
def prune_graph(graph_method, edge_yield=0.95):
    graph_gen = lambda x: _prune(graph_method(x), edge_yield)
    graph_gen.__name__ = graph_method.__name__ + str(edge_yield)
    return graph_gen
