import random
import networkx as nx
import matplotlib.pyplot as plt
import dwave_networkx as dnx

__all__ = ["EmbedderOptions","read_source_graph","read_target_graph","draw_source_graph", "draw_tiled_graph"]

class EmbedderOptions:
    def __init__(self, **params):
        # Parse optional parameters
        self.names = {"random_seed", "construction", "tries", "verbose"}

        # Random Number Generator
        try: self.random_seed = params['random_seed']
        except: self.random_seed = None
        self.rng = random.Random(self.random_seed)

        try: self.tries = params['tries']
        except: self.tries = 1

        try: self.construction = params['construction']
        except: self.construction = __default_construction__

        try: self.verbose = params['verbose']
        except: self.verbose = 0

def read_source_graph(S, opts):


    Sg = nx.Graph(S)

    if opts.verbose > 1:
        print("Drawing Source Graph")
        plt.clf()
        try:    nx.draw(Sg, pos=opts.topology, with_labels=True)
        except: nx.draw(Sg, with_labels=True)
        plt.show()

    return Sg

def read_target_graph(T, opts):

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

    if opts.verbose > 1:
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

    if opts.verbose > 1:
        print("Drawing Pegasus Graph")
        plt.clf()
        dnx.draw_pegasus(Tg, with_labels=True)
        plt.show()

    return Tg

## Drawings

def draw_tiled_graph(G, n, m, tile_labels={}, **kwargs):
    layout = {name:node['coordinate'] for name,node in G.nodes(data=True)}
    dnx.drawing.qubit_layout.draw_qubit_graph(G, layout,**kwargs)
    plt.grid('on')
    plt.axis('on')
    plt.axis([0,n,0,m])
    x_ticks = range(0, n) # steps are width/width = 1 without scaling
    y_ticks = range(0, m)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    # Label tiles
    for (i,j), label in tile_labels.items():
        plt.text(i,j,label)

def draw_source_graph(G, **kwargs):

    layout = {name:node['coordinate'] for name,node in G.nodes(data=True)}

    dnx.drawing.qubit_layout.draw_qubit_graph(G, layout,**kwargs)
