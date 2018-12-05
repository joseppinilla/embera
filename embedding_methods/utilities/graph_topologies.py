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


def complete_graph(n):
    G = nx.complete_graph(n)
    G.name = 'complete'
    return G

def complete_bipartite_graph(n, m=None):
    if m is None:
        m = n = round(n/2)
    G = nx.complete_bipartite_graph(m,n)
    G.name = 'bipartite'
    return G

def grid_2d_graph(n, m=None):
    if m is None:
        m = n = round(math.sqrt(n))
    G = nx.grid_2d_graph(m,n)
    G.name = 'grid2d'
    G.graph['pos'] = {v:list(v) for v in G}
    return G

def hypercube_graph(n=None, dim=None):
    if n is None and dim is None:
        raise ValueError('Expected either n or dim')
    if dim is None:
        dim = round(math.log(n,2))
    G = nx.hypercube_graph(dim)
    G.name = 'hypercube'
    return G

def rooks_graph(n, m=None):
    if m is None:
        n = m = round(math.sqrt(n))
    G = nx.complete_graph(n)
    H = nx.complete_graph(m)
    F = nx.cartesian_product(G,H)
    F.name = 'rooks'
    F.graph['pos'] = {v:list(v) for v in F}
    return F

def grid_3d_graph(n, m=None, t=2):
    if m is None:
        m = n = t = round(n**(1./3.))
    G = nx.grid_graph(dim=[m,n,t])
    G.name = 'grid3d'
    G.graph['pos'] = {(x,y,z):[x+z,y+z] for (x,y,z) in G}
    return G

def random_graph(n, max_degree=None, seed=None):
    if not max_degree: max_degree=round(n/4)
    G = nx.empty_graph(n)
    rand.seed(seed)
    for v in G:
         n = rand.randrange(1, max_degree)
         node_set = set(G.nodes)
         node_set.remove(v)
         randedges = [ (v,n) for n in rand.sample(node_set, n)]
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
