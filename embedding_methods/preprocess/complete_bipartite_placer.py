"""
Systematic method to find the embedding of a complete biaprtite graph into the
Chimera graph representing the D-Wave Ising Sampler. The method here finds an
embedding such as in [1]. This method uses a naive window sliding approach to
assign fewer unavailable qubits.

[1] https://arxiv.org/abs/1510.06356

NOTE: Because this systematic node mapping does not guarantee a valid
embedding due to faulty qubits, these assignments are deemed candidates.

NOTE 2: This method is only applicable to Chimera graphs.
"""

import networkx as nx

__all__ = ['find_candidates']

def _parse_target(Tg):
    """ Parse Target graph
     Use coordinates if available, otherwise get chimera indices
     Returns:
        qubit_cols, qubit_rows (int, int):
            Number of available rows and columns of interconnected qubits

     """
    if Tg.graph['family'] != 'chimera':
        raise ValueError("Invalid target graph family. Only valid for 'chimera' graphs")
    if Tg.graph['labels'] == 'coordinate':
        pass
    elif Tg.graph['labels'] == 'int':
        if not Tg.graph['data']:
            coordinate_obj = chimera_coordinates(Tg.graph['columns'])
            coordinates = {v: coordinate_obj.tuple(v) for v in Tg.nodes()}
        else:
            try:
                coordinates = {v : data['chimera_index'] for v, data in Tg.nodes(data=True)}
            except:
                raise ValueError("Chimera index not found in node data.")
        nx.relabel_nodes(Tg, coordinates, copy=False)
        Tg.graph['labels'] = 'coordinate'
    else:
        raise ValueError("Invalid label type {coordinate, int}.")

    m = Tg.graph['columns']
    n = Tg.graph['rows']
    t = Tg.graph['tile']

    qubit_cols = m * t
    qubit_rows = n * t

    return qubit_cols, qubit_rows


def _parse_source(S):
    """ Parse Source graph.
    Args:
        S:
            an iterable of label pairs representing the edges in the
            source graph. This needs to be a complete bipartite graph of
            dimensions (p,q) where max(p,q) <= max(m*t, n*t)
            OR
            a tuple with the size of the sets S = (p, q)
    """
    try:
        P, Q = nx.bipartite.sets(nx.Graph(S))
    except:
        try:
            Sg = nx.complete_bipartite_graph(*S)
            P = set()
            Q = set()
            for v, data in Sg.nodes(data=True):
                if data['bipartite']==1: Q.add(v)
                else: P.add(v)
        except:
            raise RuntimeError("Input must be iterable of edges of a complete "
                                "bipartite graph, or tuple of dimensions (p,q)")
    return P, Q

def _slide_window(p, q, qubit_cols, qubit_rows, Tg):
    """ The sliding window method is a naive approach in which for a given size
    of a bipartite graph, it only assigns columns and rows of qubits that are
    immediately adjacent.

    Note: The topology of the graph allows for embeddings that are not all
    immediately adjacent. But naive exploration of these mappings is too
    expensive.
    """

    best_origin = None
    lowest_count = p*q
    orientation = None

    # Try both orientations
    for orientation, width, height in [(0, p, q), (1, q, p)]:
        if width <= qubit_cols and height <= qubit_rows:
            end_col = qubit_cols-width+1
            origins_i = range(end_col)
            end_row = qubit_rows-height+1
            origins_j = range(end_row)
            for i in origins_i:
                for j in origins_j:
                    count_faults = _find_faults((i, j), width, height, Tg)
                    if count_faults < lowest_count:
                        best_origin = i, j
                        lowest_count = count_faults
                        best_orientation = orientation

    if best_origin is None:
        raise RuntimeError('Cannot fit problem in target graph.')

    width, height = (q, p) if best_orientation else (p, q)
    candidates, faults = _assign_window_nodes(best_origin, width, height, Tg)
    return candidates, faults

def _find_faults(origin, width, height, Tg):
    """ Traverse the target graph, starting at the given "origin" and count
    the number of faults found.
    """
    faults = 0
    t = Tg.graph['tile']
    origin_i, origin_j = origin
    for col in range(origin_i, origin_i+width):
        i, k = divmod(col, t)
        j_init = (origin_j) // t
        j_end =  (origin_j+height-1) // t + 1
        for j in range(j_init, j_end):
            chimera_index = (j, i, 0, k)
            if chimera_index not in Tg.nodes:
                faults += 1

    for row in range(origin_j, origin_j+height):
        j, k = divmod(row, t)
        i_init = (origin_i) // t
        i_end =  (origin_i+width-1) // t + 1
        for i in range(i_init, i_end):
            chimera_index = (j, i, 1, k)
            if chimera_index not in Tg.nodes:
                faults += 1

    return faults

def _assign_window_nodes(origin, width, height, Tg):
    """ Traverse the target graph, starting at the given "origin" and assign
    the valid qubits found.
    """
    candidates = {}
    faults = []
    t = Tg.graph['tile']
    origin_i, origin_j = origin
    for node, col in enumerate(range(origin_i, origin_i+width)):
        i, k = divmod(col, t)
        j_init = (origin_j) // t
        j_end =  (origin_j+height-1) // t + 1
        candidates.setdefault(node, [])
        for j in range(j_init, j_end):
            chimera_index = (j, i, 0, k)
            if chimera_index in Tg.nodes:
                candidates[node].append(chimera_index)
            else:
                faults.append(chimera_index)

    for node, row in enumerate(range(origin_j, origin_j+height), width):
        j, k = divmod(row, t)
        i_init = (origin_i) // t
        i_end =  (origin_i+width-1) // t + 1
        candidates.setdefault(node, [])
        for i in range(i_init, i_end):
            chimera_index = (j, i, 1, k)
            if chimera_index in Tg.nodes:
                candidates[node].append(chimera_index)
            else:
                faults.append(chimera_index)

    return candidates, faults

def find_candidates(S, Tg, **params):
    """ Given a complete complete bipartite source graph and a target chimera
    graph of dimensions (m,n,t). Systematically find a mapping with a low
    number of fault qubits in the qubit chains.

        Args:
            S:  an iterable of label pairs representing the edges in the
                source graph. This needs to be a complete bipartite graph of
                dimensions (p,q) where max(p,q) <= max(m*t, n*t)
                OR
                a tuple with the size of the sets S = (p, q)
            Tg: a NetworkX Graph with construction parameters such as those
                generated using dwave_networkx_:
                    family : {'chimera','pegasus', ...}
                    rows : (int)
                    columns : (int)
                    labels : {'coordinate', 'int'}
                    data : (bool)
                    **family_parameters

            **params (optional):
                origin: Tuple(int, int) (default None)
                    If not None assign nodes to rows and columns starting
                    from origin.
                coordinates : bool (default False)
                    If True, node labels are 4-tuples, equivalent to the chimera_index
                    attribute as above.  In this case, the `data` parameter controls the
                    existence of a `linear_index attribute`, which is an int

        Returns:
            candidates: a dict that maps labels in S to lists of labels in T.

    """

    # Ensure correcteness of input graphs and obtained sizes
    qubit_cols, qubit_rows = _parse_target(Tg)
    P, Q = _parse_source(S)

    origin = params.pop('origin', None)
    if origin is not None:
        # Assign window of nodes starting from origin
        candidates, faults = _assign_window_nodes(origin, len(P), len(Q), Tg)
    else:
        # Use a sliding window to test all immediately adjacent columns and rows
        # and find lowest number of faults
        candidates, faults = _slide_window(len(P), len(Q), qubit_cols, qubit_rows, Tg)

    # Recover original graph node names
    names_list = list(P|Q)
    return { names_list[v]:qubits for v, qubits in candidates.items()}
