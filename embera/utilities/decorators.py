import networkx as nx
import dwave_networkx  as dnx

from decorator import decorator
from embera.architectures import dwave_coordinates


""" ToDo:
     - Graph inputs could also be adjacency matrices, or dictionaries
     - Chimera coordinates can also be turned into `nice_coordinates` by
       appending a 0 on the 3r tile dimension.
     - Embedding label conversions could identify the label type of the target
       graph and convert accordingly.
"""

""" =============================== NetworkX ============================= """

def nx_graph(*graph_index):
    """ Decorator to support input graphs as NetworkX Graphs or lists of edges.
        Chosen arguments are parsed into NetworkX Graphs.

        Args:
            graph_index (iter):
                One or more numbers representing where in the argument list of
                the wrapped function is the `networkx.Graph` object or list.

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

def edgelist_graph(*graph_index):
    """ Decorator to support input graphs as NetworkX Graphs or lists of edges.
        Chosen arguments are parsed into graph edgelists.

        Args:
            graph_index (iter):
                One or more numbers representing where in the argument list of
                the wrapped function is the `networkx.Graph` object or list.

    """
    def _parse_graph(G):
        if isinstance(G, nx.Graph):
            H = list(G.edges)
        elif isinstance(G, list):
            H = G
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

def dnx_graph(*graph_index, nice_coordinates=False):
    """ Decorator to support input graphs as D-Wave NetworkX Graphs with
        different types of labels.

        Args:
            graph_index (iter):
                One or more numbers representing where in the argument list of
                the wrapped function is a `dwave_networkx` graph.

            nice_coordinates (bool):
                Wether or not to return nice_coordinates. Only applies to
                Pegasus architectures. Chimera coordinates are `nice`.
                # TODO: Adding a 5th label to Chimera coordinates
    """

    def _parse_graph(G):
        labels = G.graph['labels']
        converter = dwave_coordinates.from_graph_dict(G.graph)
        if nice_coordinates:
            if labels is 'int':
                node_list = converter.iter_linear_to_nice(G.nodes)
                edge_list = converter.iter_linear_to_nice_pairs(G.edges)
            elif labels is 'coordinate':
                node_list = converter.iter_coordinate_to_nice(G.nodes)
                edge_list = converter.iter_coordinate_to_nice_pairs(G.edges)
            elif labels is 'nice':
                return G
            else:
                raise ValueError("Label type not supported.")
        else:
            if labels is 'int':
                node_list = converter.iter_linear_to_coordinate(G.nodes)
                edge_list = converter.iter_linear_to_coordinate_pairs(G.edges)
            elif labels is 'nice':
                node_list = converter.iter_nice_to_coordinate(G.nodes)
                edge_list = converter.iter_nice_to_coordinate_pairs(G.edges)
            elif labels is 'coordinate':
                return G
            else:
                raise ValueError("Label type not supported.")

        family = G.graph.get('family')
        if family is 'chimera':
            n = G.graph['columns']
            m = G.graph['rows']
            t = G.graph['tile']
            H = dnx.chimera_graph(m,n,t,node_list=node_list,edge_list=edge_list,
                                  coordinates=True)
        elif family=='pegasus':
            m = G.graph['rows']
            H = dnx.pegasus_graph(m,node_list=node_list,edge_list=edge_list,
                                  nice_coordinates=nice_coordinates)
        return H

    @decorator
    def _dnx_graph_argument(func, *args, **kwargs):
        new_args = list(args)
        for i in graph_index:
            new_args[i] =  _parse_graph(new_args[i])
        return func(*new_args, **kwargs)
    return _dnx_graph_argument

""" ===================== D-Wave NetworkX w/ Embedding ===================== """

def dnx_graph_embedding(dnx_graph_index, *embedding_index):
    """ Decorator to transform embeddings from `int` labels into `coordinate`
        labels.

        Args:
            dnx_graph_index (int):
                Integer representing where in the argument list of the wrapped
                function is the dwave_networkx Graph of the target coordinates.
            embedding_index (iter):
                One or more numbers representing where in the argument list of
                the wrapped function is an embedding object, i.e. dict.

    """
    def _translate_labels(embedding, dnx_coords):
        if all([isinstance(q,int) for chain in embedding.values() for q in chain]):
            return {v:dnx_coords.iter_linear_to_coordinate(chain)
                    for v,chain in embedding.items()}
        else:
            return embedding

    @decorator
    def _embedding_argument(func, *args, **kwargs):
        new_args = list(args)
        dnx_graph = new_args[dnx_graph_index]
        dnx_coords = dwave_coordinates.from_dwave_networkx(dnx_graph)
        for i in embedding_index:
            new_args[i] =  _translate_labels(new_args[i], dnx_coords)
        embedding = func(*new_args, **kwargs)
        return _translate_labels(embedding,dnx_coords)
    return _embedding_argument
