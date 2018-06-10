
import traceback
import networkx as nx
import dwave_networkx as dnx


__max_chimera__ = 16
__max_pegasus__ = 16

__default_construction__ =  {"family": "chimera", "rows": 16, "columns": 16,
                            "tile": 4, "data": True, "labels": "coordinate"}


__all__ = ["find_embedding"]

class Tile:
    def __init__(self, i, j, m, n):
        self.name = (i,j)
        self.index = j*m + i
        self.nodes = set()
        self.supply = 0
        self.concentration = 0.0
        self.velocity_x = 0.0
        self.velocity_y = 0.0

    def add_node(self, node):
        self.nodes.add(node)


class Tiling:
    def __init__(self, m, n):
        self.size = m*n
        self.tiles = {}
        for i in m:
            for j in n:
                self.tiles[(i,j)] = Tile(i,j,m,m)



class Options:
    def __init__(self, **params):
        self.topology = None
        self.enable_migration = True
        self.random_seed = None
        self.tries = 1
        self.construction = __default_construction__
        self.coordinates = None
        self.verbose = 0

def _read_source_graph(S, opts):

    Sg = nx.Graph(S)
    nx.set_node_attributes(Sg, topology, 'coordinate')

    if opts.verbose >= 1:
        print("Drawing Source Graph")
        plt.clf()
        nx.draw(Sg, with_labels=True)
        plt.show()

    return Sg

def _read_target_graph(T, opts):

    if opts.construction['family'] =='chimera':
        Tg = _read_chimera_graph(T, opts)
    elif opts.construction['family'] =='pegasus':
        Tg = _read_pegasus_graph(T, opts)
    else:
        raise RuntimeError("Target architecture graph %s not recognized" % opts.construction['family'])
    return Tg

def _read_chimera_graph(T, opts):

    m = opts.construction['rows']
    n = opts.construction['columns']
    t = opts.construction['tile']
    data = opts.construction['data']
    coordinates = opts.construction['labels'] == 'coordinate'

    Tg = dnx.chimera_graph(m, n, t, edge_list=T, data=data, coordinates=coordinates)

    if opts.verbose >= 1:
        print("Drawing Chimera Graph")
        plt.clf()
        dnx.draw_chimera(Tg, with_labels=True)
        plt.show()

    return Tg

def _read_pegasus_graph(T, opts):

    m = opts.construction['rows']
    n = opts.construction['columns']
    t = opts.construction['tile']
    data = opts.construction['data']
    vertical_offsets = opts.construction['vertical_offsets']
    horizontal_offsets = opts.construction['horizontal_offsets']
    coordinates = opts.construction['labels'] == 'coordinate'

    Tg = dnx.pegasus_graph(m, edge_list=T, data=True,
        offset_lists=(vertical_offsets,horizontal_offsets),
        coordinates=coordinates)

    if opts.verbose >= 1:
        print("Drawing Pegasus Graph")
        plt.clf()
        dnx.draw_pegasus(Tg, with_labels=True)
        plt.show()

    return Tg


def _loc2tile(loc, family):

    tile = None

    return tile

def _simulated_annealing_placement():

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



def _step(Tsupply, scale_loc, tiling, opts):

    N = len(scale_loc)
    m = opts.construction['rows']
    n = opts.construction['columns']

    center_x, center_y = m/2.0, n/2.0

    # Store new node sets per tile
    new_tiling = n*m*[set()]

    for tile in tiling:
        nodes = tiling[tile]['nodes'] #TODO: Use this dict structure
        demand_tile = len(nodes)
        concentration = demand_tile/Tsupply[]

        velocity_step = _get_velocity()

        for node in nodes:
            node_x, node_y = scale_loc[node]
            v_x, v_y = velocity_step
            new_x, new_y = node_x + v_x, node_y + v_y
            dist_accum += (new_x-center_x)**2 + (new_y-center_x)**2
            new_tiling[tile].add(node)

    dispersion = dist_accum/N
    return dispersion

def _get_supply(Tg):
    #TODO: Implement _get_supply from Target
    pass


def _migrate(Sg, Tg, scale_loc, opts):

    N = len(scale_loc)
    m = opts.construction['rows']
    n = opts.construction['columns']

    Tsupply = _get_supply(Tg)

    tiling = m*n*[(0, set(), 0.0)] # supply, node_list, velocity_step
    tile_velocity = N*[0.0]
    concentration = m*n*[0.0]

    concentration, tile, migrating = _step(Tsupply, scale_loc, tiling, opts)

    while migrating:
        velocity_step =  _gradient(concentration, tile)
        concentration, tile, migrating = _step(velocity_step)


    return node_loc

def _route():

    return chains

def _parse_params(**params):
    """ Parse the optional parameters, assign default values or perform
    necessary actions.

    """
    # Parse optional parameters
    names = {"topology", "enable_migration", "random_seed", "construction", "tries", "verbose"}

    for name in params:
        if name not in names:
            raise ValueError("%s is not a valid parameter for topological find_embedding"%name)

    opts = Options()

    # If a topology of the graph is not provided
    try: opts.topology =  params['topology']
    except KeyError: opts.topology = _simulated_annealing_placement(S)

    # For the other parameters, keep defeaults
    try: opts.enable_migration = params['enable_migration']
    except: pass

    try: opts.random_seed = params['random_seed']
    except: pass

    try: opts.tries = params['tries']
    except: pass

    try: opts.construction = params['construction']
    except: pass

    try: opts.verbose = params['verbose']
    except: pass

    return opts

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

    opts = _parse_params(**params)

    Sg = _read_source_graph(S, opts)

    Tg = _read_target_graph(T, opts)

    scale_loc = _scale(Sg, Tg, opts)

    node_loc = _migrate(Sg, Tg, scale_loc, opts)

    embedding = _route(S, T, node_loc, opts)

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
    except:
        traceback.print_exc()
