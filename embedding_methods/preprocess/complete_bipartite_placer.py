"""
Systematic method to find the embedding of a complete biaprtite graph into the
Chimera graph representing the D-Wave Ising Sampler. The method here finds an
embedding such as in [1], but guarantees the smallest number of qubit faults
possible.

[1] https://arxiv.org/abs/1510.06356

NOTE: Because this systematic node mapping does not guarantee a valid
embedding, these assignments are deemed candidates.

NOTE 2: This method is only applicable to Chimera graphs.
"""

import warnings
import networkx as nx

from dwave_networkx.generators.chimera import chimera_coordinates

__all__ = ['find_candidates']

def _find_faults(origin_i, origin_j, width, height, Tg):
    faults = 0
    t = Tg.graph['tile']
    for i in range(origin_i, origin_i+width):
        for j in range(origin_j, origin_j+height):
            for k in range(t):
                for u in range(2):
                    chimera_index = (i, j, u, k)
                    if chimera_index not in Tg.nodes:
                        faults +=1
    return faults


def _slide_window(p, q, Tg):
    """ Try all possible origins and both orientations of the embedding,
    """
    m = Tg.graph['columns']
    n = Tg.graph['rows']
    best_origin = None
    lowest_count = p*q
    orientation = None
    # Try both orientations
    for orientation, width, height in [(0, p, q), (1, q, p)]:
        if width <= m and height <= n:
            for i in range(m-width+1):
                for j in range(n-height+1):
                    count_faults = _find_faults(i, j, width, height, Tg)
                    if count_faults < lowest_count:
                        best_origin = i, j
                        lowest_count = count_faults
                        best_orientation = orientation
    return best_origin, best_orientation




def find_candidates(S, Tg, **params):
    """ find_candidates(S, Tg, **params)
    Given a complete complete bipartite source graph and a target chimera
    graph of dimensions (m,n,t). Systematically find the embedding with fewer
    fault qubits in the qubit chains.

        Args:
            S:
                an iterable of label pairs representing the edges in the
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

            **params (optional): see below

        Returns:

            candidates: a dict that maps labels in S to lists of labels in T.

    """

    # Parse Target graph
    # Use coordinates if available, otherwise get chimera indices
    if Tg.graph['family'] != 'chimera':
        warnings.warn("Invalid target graph family. Only valid for 'chimera' graphs")
    if Tg.graph['labels'] == 'coordinate':
        pass
    elif Tg.graph['labels'] == 'int':
        if not Tg.graph['data']:
            warnings.warn("Coordinate data not found. Using mapping from int.")
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


    # Parse Source graph
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
            raise RuntimeError("Input must be either iterable of edges, or tuple of dimensions (p,q)")

    origin, orientation = _slide_window(len(P), len(Q), Tg)




    print('rows')
    print(rows)
    print('columns')
    print(columns)


    #Precompute faults

    """ CompleteBipartite:
      Create functions map start, end tiles -> # faults
      Two functions, horizontal and vertical
   Optional parameters: faults
   Return min
    """


    candidates = {}

    return candidates
