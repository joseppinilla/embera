
import traceback
import networkx as nx
import dwave_networkx as dnx


__max_chimera__ = 16
__max_pegasus__ = 16


__all__ = ["find_embedding"]


class Options:
    def __init__(self, **params):
        self.topology = None
        self.enable_migration = True
        self.random_seed = None
        self.tries = 1
        self.family = 'chimera'
        self.verbose = 0

def _loc2tile(loc, family):

    tile = None

    return tile

def _read_source_graph(S, opts):

    Sg = nx.Graph(S, coordinates=opts.topology)

    return Sg

def _read_target_graph(T, opts):
    if opts.family=='chimera':
        Tg = _read_chimera_graph(T, opts)
    elif opts.family=='pegasus':
        Tg = _read_pegasus_graph(T, opts)
    else:
        raise RuntimeError("Target architecture graph %s not recognized" % opts.family)
    return Tg

def _read_chimera_graph(T, opts):

    Tg = dnx.chimera_graph(16, edge_list = T)

    if opts.verbose >= 1:
        print("Drawing Chimera Graph")
        plt.clf()
        nx.draw(Tg, with_labels=True)
        plt.show()

    return Tg

def _read_pegasus_graph(T):

    Tg = dnx.pegasus_graph(16,edge_list = T)

    return Tg

def _simulated_annealing_placement():

    return init_loc

def _scale():
    """ Transform node locations to in-scale values of the dimension
    of the chimera.
    """
    return scale_loc

def _migrate():

    return node_loc

def _route():

    return chains

def _parse_params(**params):
    """ Parse the optional parameters, assign default values or perform
    necessary actions.

    """
    # Parse optional parameters
    names = {"topology", "enable_migration", "random_seed", "family", "tries", "verbose"}

    for name in params:
        if name not in names:
            raise ValueError("%s is not a valid parameter for topological find_embedding"%name)

    opts = Options()

    # If a topology of the graph is not provided
    try: opts.topology =  params['topology']
    except KeyError: opts.topology = _simulated_annealing_placement(S)

    try: opts.enable_migration = params['enable_migration']
    except KeyError: opts.enable_migration = True

    try: opts.random_seed = params['random_seed']
    except KeyError: opts.random_seed = None

    try: opts.tries = params['tries']
    except KeyError: opts.tries = 1

    try: opts.family = params['family']
    except KeyError: opts.family = 'chimera'

    try: opts.verbose = params['verbose']
    except KeyError: opts.verbose = False

    return opts

def find_embedding(S, T, **params):
    """
    Heuristically attempt to find a minor-embedding of a graph representing an
    Ising/QUBO into a target graph.

    Args:

        S: an iterable of label pairs representing the edges in the source graph

        T: an iterable of label pairs representing the edges in the target graph

        **params (optional): see below
    Returns:

        embedding: a dict that maps labels in S to lists of labels in T

    Optional parameters:

        topology ({<node>:(<x>,<y>),...}):
            Dict of 2D positions assigned to the source graph nodes.

        random_seed (int):

        tries (int):

        family {'chimera','pegasus'}:
            Target graph architecture family

        verbose (int):
            Verbosity level
            0: Quiet mode
            1: Print statements
            2: NetworkX graph drawings

    """

    opts = _parse_params(**params)

    Sg = _read_source_graph(S, opts)

    Tg = _read_target_graph(T, opts)

    scale_loc = _scale(Sg, Tg, opts)

    node_loc = _migrate(scale_loc, opts)

    embedding = _route(S, T, node_loc, opts)

    return embedding


#Temporary standalone test
if __name__== "__main__":
    import matplotlib.pyplot as plt

    m = 2
    S = nx.grid_2d_graph(4,4)
    topology = {v:v for v in S}

    plt.clf()
    nx.draw(S, with_labels=True)
    plt.show()

    family = 'pegasus'
    family = 'chimera'

    if family == 'chimera':
        T = dnx.chimera_graph(m, coordinates=True)
    elif family == 'pegasus':
        T = dnx.pegasus_graph(m, coordinates=True)

    plt.clf()
    nx.draw(T, with_labels=True)
    plt.show()

    S_edgelist = list(S.edges())
    T_edgelist = list(T.edges())

    try:
        find_embedding(S_edgelist, T_edgelist, topology=topology, family=family, verbose=2)
    except:
        traceback.print_exc()
