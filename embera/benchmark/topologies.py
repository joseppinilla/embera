""" Given a size of a graph, each method generates the graph in
the corresponding topology that is closest to the given size.

Some graphs include a layout, or 2D-position for each node.

Graph Attributes:
    name:
    pos: dictionary with (x,y) values for

"""
import os
import math
import tarfile
import requests

import networkx as nx

from itertools import combinations

def embera_bench():
    """ Set of benchmarks used to evaluate embera:
            | name          | node      | edges     |
            | ------------- |:---------:| ---------:|
            | D-Wave        | See below | See below |
            | QCA           | See below | See below |
            | Geometry      | See below | See below |
            | Misc          | See below | See below |
    """
    benchmark_set = []
    benchmark_set.extend(dwave_bench(N=1))
    benchmark_set.extend(qca_bench()[0:4])
    benchmark_set.extend(misc_bench())
    benchmark_set.extend(geometry_bench())

    return benchmark_set

def geometry_bench():
    """ Set of benchmarks for geometric graphs:
            | name          | node      | edges     |
            | ------------- |:---------:| ---------:|
            | grid          | 256       | 480       |
            | rooks         | 64        | 448       |
            | triangular    | 216       | 590       |
            | prism         | 288       | 576       |
            | grid3d        | 200       | 460       |
            | hypercube     | 128       | 448       |
            | barbell       | 410       | 1221      |

    """
    benchmark_set = []
    benchmark_set.append(grid_2d_graph(16,16))
    benchmark_set.append(rooks_graph(8,8))
    benchmark_set.append(triangular_lattice_graph(15,25))
    benchmark_set.append(prism_graph(24,12))
    benchmark_set.append(grid_3d_graph(10,10))
    benchmark_set.append(hypercube_graph(128))
    benchmark_set.append(barbell_graph(30,350))
    return benchmark_set


def dwave_bench(N=10,seed=42):
    """ Set of benchmarks to replicate results in [1].

    Parameters:
        N: (int, default=10)
            Number of samples from each type of benchmark, with increasing size.
            For N==1 to N==10:
                | name          | node      | edges     |
                | ------------- |:---------:| ---------:|
                | clique        | 20-29     | 190-406   |
                | biclique      | 44-62     | 484-961   |
                | circular      | 40-76     | 355-1273  |
                | nae3sat       | 70        | ~581      |
                | gnp25         | 70        | ~604      |
                | gnp50         | 60        | ~872      |
                | gnp75         | 50        | ~919      |

    Benchmarks:
        clique: complete graphs, Kn for n = 20 to n = 29

        biclique: complete bipartite graphs, Kn,n for n = 22 to n = 31

        circular: circular complete graphs, K4n/n for n = 10 to n = 19 — these
        are graphs on 4n nodes, with edges between i = [0, ..., 2n − 1] and
        (i + n + j) mod 4n for j = [0, ..., 2n − 1],

        nae3sat: not-all-equal-3SAT graphs near the critical threshold; 10
        instances with size 35,

        Erdös-Rényi random graphs, G(n, p), with 10 instances each of
            – gnp25: G(70,.25),
            – gnp50: G(60, .50), and
            – gnp75: G(50, .75).

    In [1] the authors clain that "Pegasus consistently achieves around a 50-60%
    reduction in chainlength over Chimera." on this set of benchmarks.

    [1] Boothby, K., Bunyk, P., Raymond, J., & Roy, A. (2019). Next-Generation
    Topology of D-Wave Quantum Processors. Technical Report. Retrieved from:
    https://www.dwavesys.com/sites/default/files/14-1026A-C_Next-Generation-Topology-of-DW-Quantum-Processors.pdf
    """
    benchmark_set = []
    for n in range(20, 20+N):
        G = nx.complete_graph(n)
        G.name = 'clique'
        benchmark_set.append(G)

    for n in range(22, 22+N):
        G = nx.complete_bipartite_graph(n,n)
        G.name = 'biclique'
        G.graph['pos'] = nx.bipartite_layout(G, nx.bipartite.sets(G)[0])
        benchmark_set.append(G)

    for n in range(10,10+N):
        G = nx.empty_graph(4*n)
        for i in range(2*n):
            for j in range(2*n):
                u = (i+n+j)%(4*n)
                G.add_edge(i,u)
        G.name = 'circular'
        G.graph['pos'] = nx.circular_layout(G)
        benchmark_set.append(G)

    for _ in range(N):
        G = nx.generators.k_random_intersection_graph(2*35,35,3,seed=seed)
        G.name = 'nae3sat'
        benchmark_set.append(G)

    for n,p in [(70,25),(60,50),(50,75)]:
        for _ in range(N):
            G = nx.erdos_renyi_graph(n,p/100,seed=seed)
            G.name = f'gnp{p}'
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
    benchmark_set = []
    path = "./qca.tar.gz"
    url = "http://www.ece.ubc.ca/~jpinilla/resources/embera/qca/qca.tar.gz"

    # Download
    if not os.path.isfile(path):
        print(f"-> Downloading QCA benchmarks to {path}")
        with open(path, 'wb') as f:
            response = requests.get(url)
            f.write(response.content)
    # Unzip, untar, unpickle
    with tarfile.open(path) as contents:
        for member in contents.getmembers():
            f = contents.extractfile(member)
            G = nx.read_gpickle(f)
            benchmark_set.append(G)

    return benchmark_set

def misc_bench():
    """ Set of miscellaneous benchmark graphs:
                | name          | node      | edges     |
                | ------------- |:---------:| ---------:|
                | LANL1 [2]     | 269       | 490       |
                | Maze(6x6) [3] | 326       | 564       |
                | MNIST [4]     | 74        | 1664      |
                | DBG [5]       | 120       | 481       |

    [3] Scott Pakin. "A Quantum Macro Assembler". In Proceedings of the 20th
    Annual IEEE High Performance Extreme Computing Conference (HPEC 2016),
    Waltham, Massachusetts, USA, 13–15 September 2016. DOI:
    10.1109/HPEC.2016.7761637.
    [4] Adachi, S. H., & Henderson, M. P. (2015). Application of Quantum
    Annealing to Training of Deep Neural Networks. ArXiv Preprint
    ArXiv:1510.00635, 18. https://doi.org/10.1038/nature10012
    [5] Bass, G. et al.: Optimizing the Optimizer: Decomposition Techniques
    for Quantum Annealing. (2020).https://arxiv.org/abs/2001.06079

    """
    benchmark_set = []
    path = "./misc.tar.gz"
    url = "http://www.ece.ubc.ca/~jpinilla/resources/embera/misc/misc.tar.gz"

    # Download
    if not os.path.isfile(path):
        print(f"-> Downloading miscellaneous benchmarks to {path}")
        with open(path, 'wb') as f:
            response = requests.get(url)
            f.write(response.content)
    # Unzip, untar, unpickle
    with tarfile.open(path) as contents:
        for member in contents.getmembers():
            f = contents.extractfile(member)
            G = nx.read_gpickle(f)
            benchmark_set.append(G)

    # Deep Boltzmann Graph from function
    benchmark_set.append(dbg_graph(6,20,2,0.1))

    return benchmark_set

def download_pickle(path, url):
    # Download
    if not os.path.isfile(path):
        print(f"-> Downloading {url} to {path}")
        with open(path, 'wb') as f:
            response = requests.get(url)
            f.write(response.content)
    # Unpickle
    G = nx.read_gpickle(path)
    return G

def complete_graph(n):
    G = nx.complete_graph(n)
    G.name = 'complete'
    return G

def complete_bipartite_graph(n, m=None):
    if m is None:
        m = n
    G = nx.complete_bipartite_graph(m,n)
    G.name = 'bipartite'
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
    """
        l,n,c,p,seed=42number_of_layers,nodes_per_layer,max_connectivity_range_layer,connectivity_probability,seed

    """
    # Names used in [5]
    number_of_layers=layers
    nodes_per_layer=nodes
    max_connectivity_range_layer=max_conn
    connectivity_probability=probability
    # Start from empty graph, add nodes, add nodes with probability p
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
