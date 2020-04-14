import dwave_networkx  as dnx

from decorator import decorator

""" =============================== NetworkX ============================= """

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

""" =========================== D-Wave NetworkX ============================ """

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

            H = dnx.pegasus_graph(m,  node_list=node_list, edge_list=edge_list)

        return H

    @decorator
    def _graph_argument(func, *args, **kwargs):
        for i in graph_index:
            arg[i] = _parse_graph(arg[i])
        return func(*args, **kwargs)
    return _graph_argument
