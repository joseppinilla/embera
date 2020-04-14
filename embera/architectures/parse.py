
import dwave_networkx  as dnx

from decorator import decorator

""" =============================== Decorators ============================= """

def nx_graph(*graph_index):
    """ Supports multiple graph arguments but return NetworkX
    """
    if isinstance(graph_index,int):
        graph_index = [graph_index]

    def _parse_graph(G):
        if hasattr(G, 'edges') and hasattr(G, 'nodes'):
            return G


    @decorator
    def _graph_argument(func, *args, **kwargs):
        for i in graph_index:
            arg[i] = _parse_graph(arg[i])
        return func(*args, **kwargs)
    return _graph_argument

def dnx_graph(*graph_index,labels='coordinate'):
    """ Supports multiple D-Wave graph arguments but return dwave_networkx
        with given label type.
    """
    if not isinstance(graph_index,list):
        graph_index = [graph_index]

    if not isinstance(labels,list):
        labels = [labels]*len(graph_index)

    def _parse_graph(G):
        if hasattr(G, 'graph'):
            if G.graph.get('labels') == labels:
                return G
        else:
            raise ValueError("D-Wave NetworkX graph must have `graph` attribute")

        family = G.graph.get('family')

        if family=='chimera':
            n = G.graph['columns']
            m = G.graph['rows']
            t = G.graph['tile']
            converter = dnx.chimera_coordinates(m, n, t)
            if labels == 'int':
                node_list = converter.iter_linear_to_chimera(G.nodes)
                edge_list = converter.iter_linear_to_chimera_pairs(G.nodes)
            elif labels == 'coordinate':
                node_list = converter.iter_chimera_to_linear(G.nodes)
                edge_list = converter.iter_chimera_to_linear_pairs(G.nodes)
            else:
                raise ValueError("Label type not supported.")
            H = dnx.chimera_graph(m, n, t, node_list=node_list, edge_list=edge_list)

        elif family=='pegasus':
            m = G.graph['rows']
            converter = dnx.pegasus_coordinates(m)
            if labels == 'linear':
                node_list = converter.iter_linear_to_pegasus(G.nodes)
                edge_list = converter.iter_linear_to_pegasus_pairs(G.nodes)
            elif labels == 'coordinate':
                node_list = converter.iter_pegasus_to_linear(G.nodes)
                edge_list = converter.iter_pegasus_to_linear_pairs(G.nodes)
            elif labels == 'nice':
                node_list = converter.iter_nice_to_linear(G.nodes)
                edge_list = converter.iter_nice_to_linear_pairs(G.nodes)
            else:
                raise ValueError("Label type not supported.")

    @decorator
    def _graph_argument(func, *args, **kwargs):
        for i in graph_index:
            arg[i] = _parse_graph(arg[i])
        return func(*args, **kwargs)
    return _graph_argument

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
