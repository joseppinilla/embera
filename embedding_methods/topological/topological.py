
import traceback
import networkx as nx
import dwave_networkx as dnx
from math import floor, sqrt
from embedding_methods.utilities import *

# Concentration limit
__d_lim__ = 0.75

__default_construction__ =  {"family": "chimera", "rows": 16, "columns": 16,
                            "tile": 4, "data": True, "labels": "coordinate"}

__all__ = ["find_embedding"]

def i2c(index, n):
    """ Convert tile index to coordinate
    """
    j,i = divmod(index,n)
    return i,j

def _get_neighbours(i, j, n, m, index):
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

    nw = i2c(index - n - 1, n)  if (j > 0    and i > 0)    else None
    ne = i2c(index - n + 1, n)  if (j > 0    and i < n-1)  else None
    se = i2c(index + n + 1, n)  if (j < m-1  and i < n-1)  else None
    sw = i2c(index + n - 1, n)  if (j < m-1  and i > 0)    else None

    return (north,south,west,east,nw,ne,se,sw)

class DummyTile:
    def __init__(self):
        # Keyed in tile dictionary as None
        self.name = None
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
        index = j*n + i

        self.name = (i,j)
        self.index = index
        self.nodes = set()
        self.concentration = 0.0
        self.neighbours = _get_neighbours(i, j, n, m, index)

        if family=='chimera':
            self.supply = self._get_chimera_qubits(Tg, t, i, j)
        elif family=='pegasus':
            self.supply = self._get_pegasus_qubits(Tg, t, i, j)

    def add_node(self, node):
        self.nodes.add(node)

    def remove_node(self, node):
        self.nodes.remove(node)

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
        v = 0.0
        for u in range(2):
            for k in range(t):
                chimera_index = (i, j, u, k)
                if chimera_index in Tg.nodes:
                    self.qubits.add(chimera_index)
                    v += 1.0
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
        v=0.0
        for u in range(2):
            for k in range(t):
                pegasus_index = (u, j, k, i)
                if pegasus_index in Tg.nodes:
                    self.qubits.add(pegasus_index)
                    v += 1.0
        return v

class Tiling:
    """Tiling for migration stage
    """
    def __init__(self, Tg, opts):
        m = opts.construction['rows']
        n = opts.construction['columns']
        t = opts.construction['tile']
        family = opts.construction['family']
        self.m = m
        self.n = n
        self.t = t
        self.qubits = 1.0*len(Tg)
        self.family = family
        # Add Tile objects
        self.tiles = {}
        for i in range(m):
            for j in range(n):
                tile = (i,j)
                self.tiles[tile] = Tile(Tg, i, j, opts)
        # Dummy tile to represent boundaries
        self.tiles[None] = DummyTile()
        # Dispersion cost accumulator for termination
        self.dispersion_accum = None

"""
"""

def _scale(Sg, tiling, opts):
    """ Assign node locations to in-scale values of the dimension
    of the target graph.
    """
    P = len(Sg)
    m = opts.construction['rows']
    n = opts.construction['columns']
    topology = opts.topology

    ###### Find dimensions of source graph S
    Sx_min = Sy_min = float("inf")
    Sx_max = Sy_max = 0.0
    # Loop through all source graph nodes to find dimensions
    for name, node in Sg.nodes(data=True):
        sx,sy = topology[name]
        Sx_min = min(sx,Sx_min)
        Sx_max = max(sx,Sx_max)
        Sy_min = min(sy,Sy_min)
        Sy_max = max(sy,Sy_max)
    # Source graph width
    Swidth =  (Sx_max - Sx_min)
    Sheight = (Sx_max - Sx_min)

    center_x, center_y = n/2.0, m/2.0
    dist_accum = 0.0
    ###### Normalize and scale
    for name, node in Sg.nodes(data=True):
        x,y = topology[name]
        norm_x = (x-Sx_min) / Swidth
        norm_y = (y-Sy_min) / Sheight
        scaled_x = norm_x * (n-1) + 0.5
        scaled_y = norm_y * (m-1) + 0.5
        node['coordinate'] = (scaled_x, scaled_y)
        tile = min(floor(scaled_x), n-1), min(floor(scaled_y), m-1)
        node['tile'] = tile
        tiling.tiles[tile].nodes.add(name)
        dist_accum += (scaled_x-center_x)**2 + (scaled_y-center_x)**2

    # Initial dispersion
    dispersion = dist_accum/P
    tiling.dispersion_accum = [dispersion] * 3

def _get_attractors(tiling, i, j):

    n,s,w,e,nw,ne,se,sw = tiling.tiles[(i,j)].neighbours
    lh = (i >= 0.5*tiling.n)
    lv = (j >= 0.5*tiling.m)

    if (lh):
        if (lv):    return w,n,nw
        else:       return w,s,sw
    else:
        if (lv):    return e,n,ne
        else:       return e,s,se

def _get_gradient(tile, tiling):

    d_ij = tile.concentration
    if d_ij == 0.0 or tile.name==None: return 0.0, 0.0
    h, v, hv = _get_attractors(tiling, *tile.name)
    d_h = tiling.tiles[h].concentration
    d_v = tiling.tiles[v].concentration
    d_hv = tiling.tiles[hv].concentration
    del_x = - (__d_lim__ - (d_h + 0.5*d_hv)) / (2.0 * d_ij)
    del_y = - (__d_lim__ - (d_v + 0.5*d_hv)) / (2.0 * d_ij)
    return del_x, del_y


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
    D = min( (viscosity*P) / Q, 1.0)

    # Iterate over tiles
    for name, tile in tiling.tiles.items():

        del_x, del_y = _get_gradient(tile, tiling)
        # Iterate over nodes in tile and migrate
        for node in tile.nodes:
            x, y = Sg.nodes[node]['coordinate']
            l_x = (2.0*x/n)-1.0
            l_y = (2.0*y/m)-1.0
            v_x = l_x * del_x
            v_y = l_y * del_y
            x_1 = x + (1.0 - D) * v_x * delta_t
            y_1 = y + (1.0 - D) * v_y * delta_t
            Sg.nodes[node]['coordinate'] = (x_1, y_1)
            dist_accum += (x_1-center_x)**2 + (y_1-center_y)**2

    dispersion = dist_accum/P
    return dispersion

def _get_demand(Sg, tiling, opts):

    m = opts.construction['rows']
    n = opts.construction['columns']

    for name, node in Sg.nodes(data=True):
        x,y = node['coordinate']
        tile = node['tile']
        i = min(floor(x), n-1)
        j = min(floor(y), m-1)
        new_tile = (i,j)
        tiling.tiles[tile].nodes.remove(name)
        tiling.tiles[new_tile].nodes.add(name)
        node['tile'] = new_tile

    for name, tile in tiling.tiles.items():
        if name!=None:
            tile.concentration = len(tile.nodes)/tile.supply

def _condition(tiling, dispersion):
    """ The algorithm iterates until the dispersion, or average distance of
    the cells from the centre of the tile array, increases or has a cumulative
    variance lower than 1%
    """
    tiling.dispersion_accum.pop(0)
    tiling.dispersion_accum.append(dispersion)
    mean = sum(tiling.dispersion_accum) / 3.0
    prev_val = 0.0
    diff_accum = 0.0
    increasing = True
    for value in tiling.dispersion_accum:
        sq_diff = (value-mean)**2
        diff_accum = diff_accum + sq_diff
        if (value<=prev_val):
            increasing = False
        prev_val = value
    variance = (diff_accum/3.0)
    spread = variance > 0.01
    return spread and not increasing



def _migrate(Sg, tiling, opts):
    """
    """
    m = opts.construction['rows']
    n = opts.construction['columns']
    familiy = opts.construction['family']

    migrating = opts.enable_migration
    while migrating:
        _get_demand(Sg, tiling, opts)
        if verbose==3:
            concentrations = {name : "d=%s"%tile.concentration for name, tile in tiling.tiles.items() if name!=None}
            draw_tiled_graph(Sg,n,m,concentrations)
            plt.show()
        dispersion = _step(Sg, tiling, opts)
        migrating = _condition(tiling, dispersion)

    return tiling

"""

"""
def _route(Sg, Tg, tiling, opts):
    chains = {}
    return chains


"""

"""
class TopologicalOptions(EmbedderOptions):
    def __init__(self, **params):
        EmbedderOptions.__init__(self, **params)
        # Parse optional parameters
        self.names.update({"topology", "enable_migration", "vicinity", "delta_t", "viscosity"})

        for name in params:
            if name not in self.names:
                raise ValueError("%s is not a valid parameter for topological find_embedding"%name)

        # If a topology of the graph is not provided, generate one
        try: self.topology =  params['topology']
        except KeyError: self.topology = None

        try: self.enable_migration = params['enable_migration']
        except: self.enable_migration = True

        try: self.vicinity = params['vicinity']
        except: self.vicinity = 0

        try: self.delta_t = params['delta_t']
        except: self.delta_t = 0.2

        try: self.viscosity = params['viscosity']
        except: self.viscosity = 0.0


def _place(Sg, tiling, opts):
    """

    """
    if opts.topology:
        _scale(Sg, tiling, opts)
        _migrate(Sg, tiling, opts)
    else:
        opts.topology = nx.spring_layout(Sg, center=(1.0,1.0))
        _scale(Sg, tiling, opts)
        _migrate(Sg, tiling, opts)

    #TODO: Plugin different placement methods
    #elif:
    #_simulated_annealing(Sg, tiling, opts)


def _simulated_annealing(Sg, tiling, opts):
    rng = opts.rng
    m = opts.construction['rows']
    n = opts.construction['columns']
    family = opts.construction['family']

    init_loc = {}
    for node in S:
        init_loc[node] = (rng.randint(0,n),rng.randint(0,m))

    opts.enable_migration = False
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
            3: Migration process

    """

    opts = TopologicalOptions(**params)

    Sg = read_source_graph(S, opts)

    Tg = read_target_graph(T, opts)

    tiling = Tiling(Tg, opts)

    _place(Sg, tiling, opts)

    embedding = _route(Sg, Tg, tiling, opts)

    return embedding


#Temporary standalone test
if __name__== "__main__":

    verbose = 3
    import matplotlib.pyplot as plt

    p = 12
    S = nx.grid_2d_graph(p,p)
    topology = {v:v for v in S}

    #S = nx.cycle_graph(p)
    #topology = nx.circular_layout(S)

    m = 4
    T = dnx.chimera_graph(m, coordinates=True)
    #T = dnx.pegasus_graph(m, coordinates=True)

    S_edgelist = list(S.edges())
    T_edgelist = list(T.edges())

    try:
        find_embedding(S_edgelist, T_edgelist, topology=topology, construction=T.graph, verbose=verbose)
        #find_embedding(S_edgelist, T_edgelist, construction=T.graph, verbose=verbose)
    except:
        traceback.print_exc()
