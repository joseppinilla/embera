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

def embera_bench():
    """ Set of benchmarks used to evaluate embera:
            | name          | node      | edges     |
            | ------------- |:---------:| ---------:|
            | D-Wave [1]    | See below | See below |
            | QCA [2]       | See below | See below |
            | grid_16x16    | 256       | 480       |
            | rooks_8x8     | 64        | 448       |
            | grid3d_10x10  | 200       | 460       |
            | hypercube     | 128       | 448       |

            TODO:

            | LANL1*        | 269           | 490    |

    """
    benchmark_set = dwave_bench()
    benchmark_set.extend(qca_bench())

    benchmark_set.append(rooks_graph(8,8))
    benchmark_set.append(grid_2d_graph(16,16))
    benchmark_set.append(grid_3d_graph(10,10))
    benchmark_set.append(hypercube_graph(128))
    # TODO: LANL1
    return benchmark_set


def dwave_bench():
    """ Set of benchmarks to replicate results in [1]:

        clique: complete graphs, Kn for n = 20 to n = 29

        biclique: complete bipartite graphs, Kn,n for n = 22 to n = 31

        circular: circular complete graphs, K4n/n for n = 10 to n = 19 — these
        are graphs on 4n nodes, with edges between i = [0, ..., 2n − 1] and
        (i + n + j) mod 4n for j = [0, ..., 2n − 1],

        nae3sat: not-all-equal-3SAT graphs near the critical threshold; 10
        instances with size 35,

        Erdös-Rényi random graphs, G(n, p), with 10 instances each of – gnp25:
        G(70,.25), – gnp50: G(60, .50), and – gnp75: G(50, .75).

    In [1] the authors clain that "Pegasus consistently achieves around a 50-60%
    reduction in chainlength over Chimera." on this set of benchmarks.

    [1] Boothby, K., Bunyk, P., Raymond, J., & Roy, A. (2019). Next-Generation
    Topology of D-Wave Quantum Processors. Technical Report. Retrieved from:
    https://www.dwavesys.com/sites/default/files/14-1026A-C_Next-Generation-Topology-of-DW-Quantum-Processors.pdf
    """
    benchmark_set = []
    for n in range(20,29):
        G = nx.complete_graph(n)
        G.name = 'clique'
        benchmark_set.append(G)

    for n in range(22,31):
        G = nx.complete_bipartite_graph(n,n)
        G.name = 'biclique'
        benchmark_set.append(G)

    for n in range(10,19):
        G = nx.empty_graph(4*n)
        for i in range(2*n):
            for j in range(2*n):
                u = (i+n+j)%(4*n)
                G.add_edge(i,u)
        G.name = 'circular'

    for _ in range(10):
        G = nx.generators.k_random_intersection_graph(2*35, 35, 3)
        G.name = 'nae3sat'
        benchmark_set.append(G)

    for n,p in [(70,25),(60,50),(50,75)]:
        for _ in range(10):
            G = nx.erdos_renyi_graph(n,p/100)
            G.name = f'gnp{p}'
            print(G.name)
            benchmark_set.append(G)

    return benchmark_set

def qca_bench():
    """ Set of benchmarks to replicate Quantum-Dot Cellular Automata results
    in [2]:

        |name           | nodes         | edges  |
        | ------------- |:-------------:| ------:|
        |QCA_XOR        | 77            | 256    |
        |QCA_FULLADD    | 101           | 305    |
        |QCA_SERADD     | 128           | 391    |
        |QCA_LOOPMEM    | 129           | 412    |
        |QCA_4BMUX      | 210           | 719    |
        |QCA_4BACCUM    | 290           | 883    |

    [2] Pinilla, J. P., & Wilton, S. J. E. (2019). Layout-aware embedding for
    quantum annealing processors. In Lecture Notes in Computer Science
    (Vol. 11501 LNCS, pp. 121–139). https://doi.org/10.1007/978-3-030-20656-7_7
    """

    # Copy MNIST download
    # Download
    # Unzip
    # Unpickle
    pass


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
