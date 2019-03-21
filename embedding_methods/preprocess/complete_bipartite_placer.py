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
from dwave_networkx.generators.chimera import chimera_coordinates

__all__ = ['find_candidates']

class CompleteBipartitePlacer():

    def __init__(self, S, Tg, **params):
        # Parse parameters
        self.origin = params.pop('origin', None)
        self.show_faults = params.pop('show_faults', False)
        self.coordinates = params.pop('coordinates', False)

        # Parse Target graph
        self.Tg = Tg
        if Tg.graph['family'] != 'chimera':
            raise ValueError("Invalid target graph family. Only valid for 'chimera' graphs")

        self.m = Tg.graph['columns']
        self.n = Tg.graph['rows']
        self.t = Tg.graph['tile']
        self.qubit_cols = self.m * self.t
        self.qubit_rows = self.n * self.t

        self.c2i = chimera_coordinates(self.m,self.n,self.t)

        # Parse Source graph.
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
        self.P = P
        self.Q = Q

        self.cols = len(P)
        self.rows = len(Q)

    def _slide_window(self):
        """ The sliding window method is a naive approach in which for a given size
        of a bipartite graph, it only assigns columns and rows of qubits that are
        immediately adjacent.

        Note: The topology of the graph allows for embeddings that are not all
        immediately adjacent. But naive exploration of these mappings is too
        expensive.
        """
        p = self.cols
        q = self.rows
        qubit_cols = self.qubit_cols
        qubit_rows = self.qubit_rows

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
                        count_faults = self._find_faults((i, j), width, height)
                        if count_faults < lowest_count:
                            best_origin = i, j
                            lowest_count = count_faults
                            best_orientation = orientation

        if best_origin is None:
            raise RuntimeError('Cannot fit problem in target graph.')

        width, height = (q, p) if best_orientation else (p, q)
        candidates, faults = self._assign_window_nodes(best_origin, width, height)
        return candidates, faults

    def _find_faults(self, origin, width, height):
        """ Traverse the target graph, starting at the given "origin" and count
        the number of faults found.
        """
        t = self.t
        Tg =  self.Tg

        faults = 0
        origin_i, origin_j = origin
        for col in range(origin_i, origin_i+width):
            i, k = divmod(col, t)
            j_init = (origin_j) // t
            j_end =  (origin_j+height-1) // t + 1
            for j in range(j_init, j_end):
                chimera_index = (j, i, 0, k)
                if self.coordinates:
                    faults += chimera_index not in Tg
                else:
                    chimera_label = self.c2i.int(chimera_index)
                    faults += chimera_label not in Tg

        for row in range(origin_j, origin_j+height):
            j, k = divmod(row, t)
            i_init = (origin_i) // t
            i_end =  (origin_i+width-1) // t + 1
            for i in range(i_init, i_end):
                chimera_index = (j, i, 1, k)
                if self.coordinates:
                    faults += chimera_index not in Tg
                else:
                    chimera_label = self.c2i.int(chimera_index)
                    faults += chimera_label not in Tg

        return faults

    def _assign_window_nodes(self, origin, width, height):
        """ Traverse the target graph, starting at the given "origin" and assign
        the valid qubits found.
        """
        t = self.t
        Tg = self.Tg

        candidates = {}
        faults = {}

        origin_i, origin_j = origin
        for node, col in enumerate(range(origin_i, origin_i+width)):
            i, k = divmod(col, t)
            j_init = (origin_j) // t
            j_end =  (origin_j+height-1) // t + 1
            candidates.setdefault(node, [])
            for j in range(j_init, j_end):
                chimera_index = (j, i, 0, k)
                if self.coordinates: chimera_label = chimera_index
                else: chimera_label = self.c2i.int(chimera_index)
                # Check qubit
                if chimera_label in Tg:
                    candidates[node].append(chimera_label)
                elif node in faults:
                    faults[node].append(chimera_label)
                else:
                    faults[node] = [chimera_label]

        for node, row in enumerate(range(origin_j, origin_j+height), width):
            j, k = divmod(row, t)
            i_init = (origin_i) // t
            i_end =  (origin_i+width-1) // t + 1
            candidates.setdefault(node, [])
            for i in range(i_init, i_end):
                chimera_index = (j, i, 1, k)
                if self.coordinates: chimera_label = chimera_index
                else: chimera_label = self.c2i.int(chimera_index)
                # Check qubit
                if chimera_label in Tg:
                    candidates[node].append(chimera_label)
                elif node in faults:
                    faults[node].append(chimera_label)
                else:
                    faults[node] = [chimera_label]

        return candidates, faults

    def run(self):

            if self.origin is not None:
                origin = self.origin
                width = self.cols
                height = self.rows
                # Assign window of nodes starting from origin
                mapping, faults = self._assign_window_nodes(origin, width, height)
            else:
                # Use a sliding window to test all immediately adjacent columns and rows
                # and find lowest number of faults
                mapping, faults = self._slide_window()

            # To recover original graph node names
            names_list = list(self.P|self.Q)

            # Mapping uses chimera indices
            candidates = { names_list[v]:qubits for v, qubits in mapping.items()}

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
                coordinates: bool (default False)
                    If True, node labels are 4-tuples, equivalent to the chimera_index
                    attribute as above.  In this case, the `data` parameter controls the
                    existence of a `linear_index attribute`, which is an int
                show_faults: bool (default False)
                    See below. If True return a Tuple of the candidates and
                    faulty qubits.

        Returns:
            candidates: a dict that maps labels in S to lists of labels in T.

            (optional) faults: a list of faulty qubits


    """

    placer = CompleteBipartitePlacer(S, Tg, **params)
    candidates, faults = placer.run()

    if placer.show_faults:
        return candidates, faults
    else:
        return candidates
