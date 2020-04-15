import networkx as nx
import dwave_networkx  as dnx

from decorator import decorator

""" =============================== NetworkX ============================= """

def nx_graph(*graph_index):
    """ Supports multiple graph arguments but return NetworkX
    """
    def _parse_graph(G):
        if isinstance(G, nx.Graph):
            H = G
        elif isinstance(G, list):
            H = nx.Graph()
            H.add_edges_from(G)
        else:
            raise TypeError("Unsupported type of graph.")
        return H

    @decorator
    def _graph_argument(func, *args, **kwargs):
        new_args = list(args)
        for i in graph_index:
            new_args[i] =  _parse_graph(new_args[i])
        return func(*new_args, **kwargs)
    return _graph_argument

""" =========================== D-Wave NetworkX ============================ """

def dnx_graph(*graph_index,nice_coordinates=False):
    """ Supports multiple D-Wave graph arguments but return dwave_networkx
        with given label type.
    """
    def _parse_graph(G):
        if hasattr(G, 'graph'):
            labels = G.graph.get('labels')
            if labels=='coordinate' and not nice_coordinates: return G
            elif labels=='nice' and nice_coordinates: return G
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
                edge_list = list(converter.iter_linear_to_chimera_pairs(G.edges))
            else:
                raise ValueError("Label type not supported.")
            H = dnx.chimera_graph(m, n, t,
                                 node_list=node_list,
                                 edge_list=edge_list,
                                 coordinates=True)
        elif family=='pegasus':
            m = G.graph['rows']
            converter = dnx.pegasus_coordinates(m)
            if nice_coordinates:
                if labels == 'int':
                    node_list = converter.iter_linear_to_nice(G.nodes)
                    edge_list = converter.iter_linear_to_nice_pairs(G.edges)
                elif labels == 'coordinate':
                    node_list = converter.iter_pegasus_to_nice(G.nodes)
                    edge_list = converter.iter_pegasus_to_nice_pairs(G.edges)
                else:
                    raise ValueError("Label type not supported.")
            else:
                if labels == 'int':
                    node_list = converter.iter_linear_to_pegasus(G.nodes)
                    edge_list = converter.iter_linear_to_pegasus_pairs(G.edges)
                elif labels == 'nice':
                    node_list = converter.iter_nice_to_pegasus(G.nodes)
                    edge_list = converter.iter_nice_to_pegasus_pairs(G.edges)
                else:
                    raise ValueError("Label type not supported.")
            H = dnx.pegasus_graph(m,
                                  node_list=node_list,
                                  edge_list=edge_list,
                                  nice_coordinates=True)
        return H

    @decorator
    def _graph_argument(func, *args, **kwargs):
        new_args = list(args)
        for i in graph_index:
            new_args[i] =  _parse_graph(new_args[i])
        return func(*new_args, **kwargs)
    return _graph_argument
