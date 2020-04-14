""" Generators for architecture graphs.

    All architecture graphs use the same parameters. Additional parameters
    for the underlying generators are allowed but discouraged.

    Parameters
    ----------
    data : bool (optional, default True)
        If True, each node has a `<family>_index attribute`
    coordinates : bool (optional, default False)
        If True, node labels are tuples, equivalent to the <family>_index
        attribute as above.  In this case, the `data` parameter controls the
        existence of a `linear_index attribute`, which is an int

    Returns
    -------
    G : NetworkX Graph of the chosen architecture.
        Nodes are labeled by integers.

"""
import networkx as nx
import dwave_networkx as dnx

__all__ = ['graph_from_solver','dwave_online',
           'rainier_graph', 'vesuvius_graph', 'dw2x_graph', 'dw2000q_graph',
           'p6_graph', 'p16_graph',
           'h20k_graph',
           'coordinates']

""" ========================== D-Wave Solver Solutions ===================== """

def graph_from_solver(solver, **kwargs):
    """ D-Wave architecture graph from Dimod Structured Solver
    """
    topology = solver.properties.get('topology','N/A')
    type = topology['type']
    shape = topology['shape']

    edgelist = solver.properties['couplers']
    kwargs['edge_list'] = edgelist

    if type=='chimera':
        target_graph = dnx.generators.chimera_graph(*shape, **kwargs)
        indices = nx.get_node_attributes(target_graph,'chimera_index')
        target_graph.graph['pos'] = {v:(i,j) for v, (i,j,u,k) in indices.items()}
    elif type=='pegasus':
        target_graph = dnx.generators.pegasus_graph(*shape, **kwargs)
        indices = nx.get_node_attributes(target_graph,'pegasus_index')
        target_graph.graph['pos'] = {v:(x,y) for v, (t,y,x,u,k) in indices.items()}
    else:
        raise TypeError("Solver provided is not any of the supported types.")

    target_graph.name = solver.properties['chip_id']
    return target_graph

def dwave_online(squeeze=True, **kwargs):
    """ Architecture graphs from D-Wave devices `online`"""
    import dwave.cloud
    with dwave.cloud.Client.from_config(**kwargs) as client:
        solvers = client.get_solvers()
    graphs = [graph_from_solver(s) for s in solvers if s.properties.get('topology')]
    if squeeze:
        return graphs[0] if len(graphs)==1 else graphs
    else:
        return graphs

""" =========================== D-Wave Architectures ======================= """

def rainier_graph(**kwargs):
    """ D-Wave One 'Rainier' Quantum Annealer graph
        https://en.wikipedia.org/wiki/D-Wave_Systems
    """
    target_graph = dnx.generators.chimera_graph(4, 4, 4, **kwargs)
    target_graph.name = 'Rainier'
    return target_graph

def vesuvius_graph(**kwargs):
    """ D-Wave Two 'Vesuvius' Quantum Annealer graph
        https://en.wikipedia.org/wiki/D-Wave_Systems
    """
    target_graph = dnx.generators.chimera_graph(8, 8, 4, **kwargs)
    target_graph.name = 'Vesuvius'
    return target_graph

def dw2x_graph(**kwargs):
    """ D-Wave 2X Quantum Annealer graph
        https://en.wikipedia.org/wiki/D-Wave_Systems
    """
    target_graph = dnx.generators.chimera_graph(12, 12, 4, **kwargs)
    target_graph.name = 'DW2X'
    return target_graph

def dw2000q_graph(**kwargs):
    """ D-Wave 2000Q Quantum Annealer graph
        https://en.wikipedia.org/wiki/D-Wave_Systems
    """
    target_graph = dnx.generators.chimera_graph(16, 16, 4, **kwargs)
    target_graph.name = 'DW2000Q'
    return target_graph

def p6_graph(**kwargs):
    """ Pegasus 6 graph
        https://www.dwavesys.com/sites/default/files/mwj_dwave_qubits2018.pdf
    """
    target_graph = dnx.generators.pegasus_graph(6, **kwargs)
    target_graph.name = 'P6'
    return target_graph

def p16_graph(**kwargs):
    """ Pegasus 16 graph
        https://www.dwavesys.com/sites/default/files/mwj_dwave_qubits2018.pdf
    """
    target_graph = dnx.generators.pegasus_graph(16, **kwargs)
    target_graph.name = 'P16'
    return target_graph

""" ============================== Miscellaneous =========================== """

def h20k_graph(data=True, coordinates=False):
    """ HITACHI 20k-Spin CMOS digital annealer graph.
        https://ieeexplore.ieee.org/document/7350099/
    """
    n, m, t = 128, 80, 2

    target_graph = nx.grid_graph(dim=[t, m, n])

    target_graph.name = 'HITACHI 20k'
    construction = (("family", "hitachi"),
                    ("rows", 5), ("columns", 4),
                    ("data", data),
                    ("labels", "coordinate" if coordinates else "int"))

    target_graph.graph.update(construction)

    if coordinates:
        if data:
            for t_node in target_graph:
                (z_coord, y_coord, x_coord) = t_node
                linear = x_coord + n*(y_coord + m*z_coord)
                target_graph.nodes[t_node]['linear_index'] = linear
    else:
        coordinate_labels = {(x, y, z):x+n*(y+m*z) for (x, y, z) in target_graph}
        if data:
            for t_node in target_graph:
                target_graph.nodes[t_node]['grid_index'] = t_node
        target_graph = nx.relabel_nodes(target_graph, coordinate_labels)

    return target_graph


""" ============== Architecture agnostic coordinate converter ============== """

class coordinates:
    def __init__(self, **graph):
        """
        Provides a coordinate converter that is architecture-agnostic.

        Parameter
        ---------
        graph : dict
            family : | chimera | pegasus |
            columns : int
            rows : int
            tile : int
            labels : | int | coordinate | nice |
        """

        try:
            family = graph["family"]
            m = graph["columns"]
            n = graph["rows"]
            t = graph["tile"]
            labels = graph["labels"]
        except:
            raise ValueError("Target graph needs to have family, columns, \
            rows, tile, and labels attributes.")

        if family=="chimera":
            self._converter = dnx.chimera.chimera_coordinates(m,n=n,t=t)
            if labels=="int":
                self._int_getter = lambda v : v
                self._tuple_getter = self._converter.linear_to_chimera
                self._shore_getter = lambda v: self._tuple_getter(v)[2]
                self._tile_getter = lambda v: self._tuple_getter(v)[:2]
            elif labels=="coordinate":
                self._int_getter = lambda *v : self._converter.int(v)
                self._tuple_getter = lambda *v : v
                self._shore_getter = lambda i, j, u, k: u
                self._tile_getter = lambda i, j, u, k: (i, j)
            else:
                raise ValueError("Unsupported label type.")
            self._nice_getter = self._tuple_getter

        elif family=="pegasus":
            self._converter = dnx.pegasus.pegasus_coordinates(m)
            if labels=="int":
                self._int_getter = lambda v : v
                self._tuple_getter = self._converter.linear_to_pegasus
                self._nice_getter = self._converter.linear_to_nice
                self._shore_getter = lambda v: self._tuple_getter(v)[0]
                self._tile_getter = lambda v: self.nice_getter(v)[:3]
            elif labels=="coordinate":
                self._int_getter = lambda *v : self._converter.int(v)
                self._tuple_getter = lambda *v : v
                self._nice_getter = self._converter.pegasus_to_nice
                self._shore_getter = lambda u, w, k, z : u
                self._tile_getter = lambda *v : self._nice_getter(v)[:3]
            elif labels=="nice":
                n2p = pegasus.get_nice_to_pegasus_fn()
                self._int_getter = self._converter.nice_to_linear
                self._tuple_getter = self._converter.nice_to_pegasus
                self._nice_getter = lambda *v : v
                self._shore_getter = lambda t, i, j, u, k: u
                self._tile_getter = lambda t, i, j, u, k : (t, i, j)
            else:
                raise ValueError("Unsupported label type.")
        else:
            raise ValueError("Unsupported graph family.")

    def get_int(self, *v):
        return self._int_getter(*v)

    def get_tuple(self, *v):
        return self._tuple_getter(*v)

    def get_shore(self, *v):
        return self._shore_getter(*v)

    def get_tile(self, *v):
        return self._tile_getter(*v)

    def get_nice(self, *v):
        return self._nice_getter(*v)
