
import traceback
import networkx as nx
import dwave_networkx as dnx
from math import floor, sqrt
from embedding_methods.utilities import *

__chimera_qubits__ = 8
__pegasus_qubits__ = 20
__default_construction__ =  {"family": "chimera", "rows": 16, "columns": 16,
                            "tile": 4, "data": True, "labels": "coordinate"}


__all__ = ["find_embedding"]


def i2c(index, n):
    """ Convert tile index to coordinate
    """
    return divmod(index,n)

def _get_neighbours(i, j, m, n, index, vicinity):
    """ Calculate indices and names of negihbouring tiles to use recurrently
        during migration and routing.
        The vicinity parameter is later used to prune out the neighbours of
        interest.
        Uses cardinal notation north, south, ...
    """
    north = i2c(index - n, n)     if (j > 0)      else   None
    south = i2c(index + n, n)     if (j < m-1)    else   None
    west =  i2c(index - 1, n)     if (i > 0)      else   None
    east =  i2c(index + 1, n)     if (i < n-1)    else   None

    if vicinity>1 or vicinity<4:
        nw = i2c(index - n - 1, n)  if (j > 0    and i > 0)    else None
        ne = i2c(index - n + 1, n)  if (j > 0    and i < n-1)  else None
        se = i2c(index + n + 1, n)  if (j < m-1  and i < n-1)  else None
        sw = i2c(index + n - 1, n)  if (j < m-1  and i > 0)    else None
        return (north,south,west,east,nw,ne,se,sw)
    else:
        raise ValueError("%s is not a valid value for \
            the topological embedding vicinity parameter."%name)

    return (north,south,west,east)

class DummyTile:
    def __init__(self):
        # Treat as a fully occupied tile
        self.concentration = 1.0
        # Dummy empty set to skip calculations
        self.nodes = set()

class Tile:
    """ Tile for migration stage
    """
    def __init__(self, Tg, i, j, opts):

        m = opts.construction['rows']
        n = opts.construction['columns']
        t = opts.construction['tile']
        family = opts.construction['family']
        vicinity = opts.vicinity
        index = j*m + i

        self.name = (i,j)
        self.index = index
        self.nodes = set()
        self.concentration = 0.0
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.neighbours = _get_neighbours(i, j, m, n, index, vicinity)

        if family=='chimera':
            self.supply = self._get_chimera_qubits(Tg, t, i, j)
        elif family=='pegasus':
            self.supply = self._get_pegasus_qubits(Tg, t, i, j)

    def add_node(self, node):
        self.nodes.add(node)

    def _get_chimera_qubits(self, Tg, t, i, j):
        """ Finds the avilable qubits associated to tile (i,j) of the Chimera
            Graph and returns the supply or number of qubits found.

            The notation (i, j, u, k) is called the chimera index:
                i : indexes the row of the Chimera tile from 0 to m inclusive
                j : indexes the column of the Chimera tile from 0 to n inclusive
                u : qubit orientation (0 = left-hand nodes, 1 = right-hand nodes)
                k : indexes the qubit within either the left- or right-hand shore
                    from 0 to t inclusive
        """
        self.qubits = set()
        v = 0
        for u in range(2):
            for k in range(t):
                chimera_index = (i, j, u, k)
                if chimera_index in Tg.nodes:
                    self.qubits.add(chimera_index)
                    v += 1
        return v


    def _get_pegasus_qubits(self, Tg, t, i, j):
        """ Finds the avilable qubits associated to tile (i,j) of the Pegasus
            Graph and returns the supply or number of qubits found.

            The notation (u, w, k, z) is called the pegasus index:
                u : qubit orientation (0 = vertical, 1 = horizontal)
                w : orthogonal major offset
                k : orthogonal minor offset
                z : parallel offset
        """
        self.qubits = set()
        v=0
        for u in range(2):
            for k in range(t):
                pegasus_index = (u, j, k, i)
                if pegasus_index in Tg.nodes:
                    self.qubits.add(pegasus_index)
                    v += 1
        return v

class Tiling:
    """Tiling for migration stage
    """
    def __init__(self, Tg, opts):
        m = opts.construction['rows']
        n = opts.construction['columns']
        t = opts.construction['tile']
        family = opts.construction['family']
        self.size = m*n
        self.qubits = len(Tg)
        self.family = family
        # Add Tile objects
        self.tiles = {}
        for i in range(m):
            for j in range(n):
                tile = (i,j)
                self.tiles[tile] = Tile(Tg, i, j, opts)
        # Dummy tile to represent boundaries
        self.tiles[None] = DummyTile()

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
    for s in Sg:
        sx,sy = Sg.nodes[s]['coordinate']
        norm_sx = sx / Swidth
        norm_sy = sy / Sheight
        scaled_sx = norm_sx * n
        scaled_sy = norm_sy * m
        Sg.nodes[s]['coordinate'] = (scaled_sx, scaled_sy)

def _get_gradient(tile):

    return 0.0,0.0


def _step(Sg, tiling, opts):

    # Problem size
    P = len(Sg)
    # Number of Qubits
    Q = tiling.qubits

    m = opts.construction['rows']
    n = opts.construction['columns']
    delta_t = opts.delta_t
    viscosity = opts.viscosity

    center_x, center_y = n/2.0, m/2.0
    dist_accum = 0.0

    # Diffusivity
    D = (viscosity*P) / Q

    # Iterate over tiles
    for name, tile in tiling.tiles.items():

        del_x, del_y = _get_gradient(tile)
        # Iterate over nodes in tile and migrate
        for s in tile.nodes:
            x, y = Sg.nodes[s]['coordinate']
            l_x = 2.0*x/n
            l_y = 2.0*y/m
            v_x = l_x * del_x
            v_y = l_y * del_y
            x_1 = x + (1 - D) * v_x * delta_t
            y_1 = y + (1 - D) * v_y * delta_t
            Sg.nodes[s]['coordinate'] = (x_1, y_1)
            dist_accum += (x_1-center_x)**2 + (y_1-center_x)**2

    dispersion = dist_accum/P
    return dispersion

def _get_demand(Sg, tiling, opts):

    m = opts.construction['rows']
    n = opts.construction['columns']

    for name, node in Sg.nodes(data=True):
        x,y = node['coordinate']
        i = min(floor(x), n-1)
        j = min(floor(y), m-1)
        tile = (i,j)
        tiling.tiles[tile].add_node(name)

def _migrate(Sg, Tg, opts):
    """
    """
    m = opts.construction['rows']
    n = opts.construction['columns']
    familiy = opts.construction['family']

    tiling = Tiling(Tg, opts)

    migrating = opts.enable_migration
    while migrating:
        _get_demand(Sg, tiling, opts)
        dispersion = _step(Sg, tiling, opts)

        migrating=False #TODO: Test mode

    return tiling

def _route(Sg, Tg, tiling, opts):
    chains = {}
    return chains

class TopologicalOptions(EmbedderOptions):
    def __init__(self, **params):
        EmbedderOptions.__init__(self, **params)
        # Parse optional parameters
        self.names.update({"topology", "enable_migration", "vicinity"})

        for name in params:
            if name not in self.names:
                raise ValueError("%s is not a valid parameter for topological find_embedding"%name)

        # If a topology of the graph is not provided, generate one
        try: self.topology =  params['topology']
        except KeyError: self.topology = self._simulated_annealing_placement(S)

        try: self.enable_migration = params['enable_migration']
        except: self.enable_migration = True

        try: self.vicinity = params['vicinity']
        except: self.vicinity = 0

        try: self.delta_t = params['delta_t']
        except: self.delta_t = 0.2

        try: self.viscosity = params['viscosity']
        except: self.viscosity = 3.0


    def _simulated_annealing_placement(self, S):

        rng = self.rng
        m = self.construction['rows']
        n = self.construction['columns']
        family = self.construction['family']

        init_loc = {}
        for node in S:
            init_loc[node] = (rng.uniform(0,n),rng.uniform(0,m))

        return init_loc

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

        vicinity (int): Configuration of the set of tiles that will provide the
            candidate qubits.
            0: Single tile
            1: Immediate neighbours (north, south, east, west)
            2: Extended neighbours (Immediate) + diagonals
            3: Directed (Single) + 3 tiles closest to the node


        random_seed (int):

        tries (int):

        construction (dict of construction parameters of the graph):
            family {'chimera','pegasus'}: Target graph architecture family
            rows (int)
            columns (int)
            labels {'coordinate', 'int'}
            data (bool)
            **family_parameters:


        verbose (int): Verbosity level
            0: Quiet mode
            1: Print statements
            2: NetworkX graph drawings

    """

    opts = TopologicalOptions(**params)

    Sg = read_source_graph(S, opts)

    Tg = read_target_graph(T, opts)

    _scale(Sg, Tg, opts)

    tiling = _migrate(Sg, Tg, opts)

    embedding = _route(Sg, Tg, tiling, opts)

    return embedding


#Temporary standalone test
if __name__== "__main__":

    verbose = 2
    import matplotlib.pyplot as plt

    m = 2
    S = nx.grid_2d_graph(4,4)
    topology = {v:v for v in S}

    T = dnx.chimera_graph(m, coordinates=True)
    #T = dnx.pegasus_graph(m, coordinates=True)

    S_edgelist = list(S.edges())
    T_edgelist = list(T.edges())

    try:
        find_embedding(S_edgelist, T_edgelist, topology=topology, construction=T.graph, verbose=verbose)
        #find_embedding(S_edgelist, T_edgelist, construction=T.graph, verbose=verbose)
    except:
        traceback.print_exc()
