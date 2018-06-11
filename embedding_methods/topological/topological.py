
import traceback
import networkx as nx
import dwave_networkx as dnx
from embedding_methods.utilities import *

__max_chimera__ = 16
__max_pegasus__ = 16

__default_construction__ =  {"family": "chimera", "rows": 16, "columns": 16,
                            "tile": 4, "data": True, "labels": "coordinate"}


__all__ = ["find_embedding"]

class Tile:
    """Tile for migration stage
    """
    def __init__(self, Tg, family, i, j, m, n, t):
        self.name = (i,j)
        self.index = j*m + i
        self.nodes = set()
        self.concentration = 0.0
        self.velocity_x = 0.0
        self.velocity_y = 0.0

        if family=='chimera':
            self._get_chimera_qubits(Tg, t, i, j)
        elif family=='pegasus':
            self._get_pegasus_qubits(Tg, t, i, j)

        print (self.qubits)

    def _get_chimera_qubits(self, Tg, family, i, j, m, n, t):

        self.qubits = set()
        v = 0

        for u in range(2):
            for k in range(t):
                qubit_index = (i, j, u, k)
                if qubit_index in Tg.nodes:
                    self.qubits.add(qubit_index)
                    v += 1


    def _get_pegasus_qubits(self, Tg, t, w, z):

        self.qubits = set()
        v=0

        for u in range(2):
            for k in range(t):
                qubit_index = (u, w, k, z)
                if qubit_index in Tg.nodes:
                    self.qubits.add(qubit_index)
                    v += 1



class Tiling:
    """Tiling for migration stage
    """
    def __init__(self, Tg, opts):
        m = opts.construction['rows']
        n = opts.construction['columns']
        t = opts.construction['tile']
        family = opts.construction['family']
        self.size = m*n
        self.tiles = {}
        self.family = family
        for i in range(m):
            for j in range(n):
                tile = (i,j)
                self.tiles[tile] = Tile(Tg, family, i, j, m, n, t)

class TopologicalOptions(EmbedderOptions):
    def __init__(self, **params):
        EmbedderOptions.__init__(self, **params)
        # Parse optional parameters
        self.names.update({"topology", "enable_migration"})

        for name in params:
            if name not in self.names:
                raise ValueError("%s is not a valid parameter for topological find_embedding"%name)

        # If a topology of the graph is not provided, generate one
        try: self.topology =  params['topology']
        except KeyError: self.topology = self._simulated_annealing_placement(S)

        try: self.enable_migration = params['enable_migration']
        except: self.enable_migration = True


    def _simulated_annealing_placement(self, S):

        rng = self.rng
        m = self.construction['rows']
        n = self.construction['columns']
        family = self.construction['family']

        init_loc = {}
        for node in S:
            init_loc[node] = (rng.uniform(0,n),rng.uniform(0,m))

        return init_loc



def _scale(Sg, Tg, opts):
    """ Transform node locations to in-scale values of the dimension
    of the target graph.
    """

    m = opts.construction['rows']
    n = opts.construction['columns']

    ###### Find dimensions of source graph S
    Sx_min = Sy_min = float("inf")
    Sx_max = Sy_max = 0.0
    # Loop through all source graph nodes to find dimensions
    for s in Sg:
        sx,sy = Sg.nodes[s]['coordinate']
        Sx_min = min(sx,Sx_min)
        Sx_max = max(sx,Sx_max)
        Sy_min = min(sy,Sy_min)
        Sy_max = max(sy,Sy_max)
    # Source graph width
    Swidth =  (Sx_max - Sx_min)
    Sheight = (Sx_max - Sx_min)

    ###### Normalize and scale
    scale_loc = {}
    for s in Sg:
        sx,sy = Sg.nodes[s]['coordinate']
        norm_sx = sx / Swidth
        norm_sy = sy / Sheight
        scaled_sx = norm_sx * n
        scaled_sy = norm_sy * m
        scale_loc[s] = scaled_sx, scaled_sy

    return scale_loc

def _get_velocity(tile):
    pass


def _step(tiling, scale_loc, opts):

    N = len(scale_loc)
    m = opts.construction['rows']
    n = opts.construction['columns']

    center_x, center_y = m/2.0, n/2.0

    for name, tile in tiling.tiles.items():

        velocity_step = _get_velocity(tile)

        for node in nodes:
            node_x, node_y = scale_loc[node]
            v_x, v_y = velocity_step
            new_x, new_y = node_x + v_x, node_y + v_y
            dist_accum += (new_x-center_x)**2 + (new_y-center_x)**2
            new_tiling[tile].add(node)

    dispersion = dist_accum/N
    return dispersion

def _get_demand(Sg, tiling, opts):


    demand = len(tile.nodes)
    supply = len(tile.qubits)
    concentration = demand/supply

    pass

def _migrate(Sg, Tg, scale_loc, opts):

    N = len(scale_loc)
    m = opts.construction['rows']
    n = opts.construction['columns']
    familiy = opts.construction['family']

    tiling = Tiling(Tg, opts)

    migrating = opts.enable_migration
    while migrating:
        _get_demand(Sg, tiling, opts)
        _step(tiling, scale_loc, opts)


    return tiling, node_loc

def _route(Sg, Tg, tiling, node_loc, opts):

    return chains

def find_embedding(S, T, **params):
    """
    Heuristically attempt to find a minor-embedding of a graph representing an
    Ising/QUBO into a target graph.

    Args:

        S: an iterable of label pairs representing the edges in the source graph

        T: an iterable of label pairs representing the edges in the target graph
            The node labels for the different target archictures should be either
            node indices or coordinates as given from dwave_networkx_.


        **params (optional): see below
    Returns:

        embedding: a dict that maps labels in S to lists of labels in T

    Optional parameters:

        topology ({<node>:(<x>,<y>),...}):
            Dict of 2D positions assigned to the source graph nodes.

        random_seed (int):

        tries (int):

        construction (dict of construction parameters of the graph):
            family {'chimera','pegasus'}: Target graph architecture family
            rows (int)
            columns (int)
            labels {'coordinate', 'int'}
            data (bool)
            **family_parameters:


        verbose (int):
            Verbosity level
            0: Quiet mode
            1: Print statements
            2: NetworkX graph drawings

    """

    opts = TopologicalOptions(**params)

    print(vars(opts))

    Sg = read_source_graph(S, opts)

    Tg = read_target_graph(T, opts)

    scale_loc = _scale(Sg, Tg, opts)

    tiling, node_loc = _migrate(Sg, Tg, scale_loc, opts)

    embedding = _route(Sg, Tg, tiling, node_loc, opts)

    return embedding


#Temporary standalone test
if __name__== "__main__":

    verbose = 2
    import matplotlib.pyplot as plt

    m = 2
    S = nx.grid_2d_graph(4,4)
    topology = {v:v for v in S}

    #T = dnx.chimera_graph(m, coordinates=True)
    T = dnx.pegasus_graph(m, coordinates=True)

    S_edgelist = list(S.edges())
    T_edgelist = list(T.edges())

    try:
        find_embedding(S_edgelist, T_edgelist, topology=topology, construction=T.graph, verbose=verbose)
        #find_embedding(S_edgelist, T_edgelist, construction=T.graph, verbose=verbose)
    except:
        traceback.print_exc()
