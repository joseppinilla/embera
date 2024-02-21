import math
import networkx as nx

from itertools import combinations


def complete_graph(n):
    G = nx.complete_graph(n)
    G.name = 'complete'
    return G

def complete_bipartite_graph(n, m=None):
    if m is None:
        m = n
    G = nx.complete_bipartite_graph(m,n)
    G.name = 'bipartite'
    G.graph['pos'] = nx.bipartite_layout(G, nx.bipartite.sets(G)[0])
    return G

def complete_multipartite_graph(n,m=None,o=10): #TODO: More than 3 layers?
    if m is None:
        m = n
    G = nx.complete_multipartite_graph(n,m,o)
    G.name = 'multipartite'
    #TODO: pos based on multipartite_layout gist
    return G

def grid_2d_graph(n, m=None):
    if m is None:
        m = n
    G = nx.grid_2d_graph(m,n)
    G.name = 'grid2d'
    G.graph['pos'] = {v:list(v) for v in G}
    return G

def hypercube_graph(n=None, dim=None):
    if n is None and dim is None:
        return nx.empty_graph()
    if dim is None:
        dim = round(math.log(n,2))
    G = nx.hypercube_graph(dim)
    G.name = 'hypercube'
    return G

def rooks_graph(n, m=None):
    if m is None:
        m = n
    G = nx.complete_graph(n)
    H = nx.complete_graph(m)
    F = nx.cartesian_product(G,H)
    F.name = 'rooks'
    F.graph['pos'] = {v:list(v) for v in F}
    return F

def triangular_lattice_graph(m,n):
    G = nx.triangular_lattice_graph(m,n,with_positions=True)
    F = nx.Graph(name='triangular')
    F.add_edges_from(G.edges)
    F.graph['pos'] = nx.get_node_attributes(G,'pos')
    return F

def grid_3d_graph(n, m=None, t=2):
    if m is None:
        m = n
    G = nx.grid_graph(dim=[m,n,t])
    G.name = 'grid3d'
    G.graph['pos'] = {(z,y,x):[x+z,y+z] for (z,y,x) in G}
    return G

def prism_graph(k,m):
    G = nx.grid_2d_graph(k,m,periodic=True)
    G.name = 'prism'
    nlist = [[] for _ in range(m)]
    for j, i in G:
        nlist[i].append((j,i))
    G.graph['pos'] = nx.shell_layout(G,nlist=nlist)
    return G

def barbell_graph(k,m):
    G = nx.barbell_graph(k,m)
    G.name = 'barbell'
    return G

@nx.utils.np_random_state(4)
def dbg_graph(layers, nodes, max_conn, probability, seed=42):
    # Names used in [5]
    number_of_layers=layers
    nodes_per_layer=nodes
    max_connectivity_range_layer=max_conn
    connectivity_probability=probability
    # Start from empty graph, add nodes, add edges with probability p
    G = nx.Graph()
    G.name = 'dbg'
    for l in range(number_of_layers):
        G.add_nodes_from([(l,v) for v in range(nodes_per_layer)])
    for (u,v) in combinations(G.nodes,2):
        l1,v1 = u
        l2,v2 = v
        p = [1.0-connectivity_probability,connectivity_probability]
        if seed.choice([0,1],p=p) and l2-l1<=max_connectivity_range_layer:
            G.add_edge(u,v)
    G.graph['pos'] = {(l,v):(l/number_of_layers,v/nodes_per_layer) for l,v in G.nodes}
    return G
