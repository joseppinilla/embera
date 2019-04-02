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
import random
import networkx as nx
from dwave_networkx.generators.chimera import chimera_coordinates

__all__ = ['find_candidates', 'CompleteBipartitePlacer']

class CompleteBipartitePlacer():
    """ This class can be used to create and transform systematic mappings of
        complete bipartite graphs onto Chimera target graphs.

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
                    If not None, force the rows and columnd to start from
                    origin (row, col).
                orientation: (0, 1, or None) (default None)
                    If not None, this value determines which set (p or q) is
                    assigned rows or columns. If None, both options are
                    explored to find the orientation with fewer faults.
                    0: p=cols q=rows
                    1: p=rows q=cols
    """
    def __init__(self, S, Tg, **params):

        # Parse parameters
        self.origin = params.pop('origin', None)
        self.orientation = params.pop('orientation', None)

        # Parse Target graph
        self.Tg = Tg
        if Tg.graph['family'] != 'chimera':
            raise ValueError("Invalid target graph family. Only valid for 'chimera' graphs")

        self.m = Tg.graph['columns']
        self.n = Tg.graph['rows']
        self.t = Tg.graph['tile']
        self.coordinates = Tg.graph['labels'] == 'coordinate'

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
                raise RuntimeError("Input must be iterable of edges of a "
                                    "complete bipartite graph, or tuple with "
                                    "dimensions (p,q).")
        self.P = {k:[] for k in P}
        self.Q = {k:[] for k in Q}
        self.faults = None

    def _slide_window(self):
        """ The sliding window method is a naive approach in which for a given size
        of a bipartite graph, it only assigns columns and rows of qubits that are
        immediately adjacent.
        """

        # TODO: The topology of the graph allows for mappings that are not all
        # immediately adjacent. But naive exploration of these mappings is too
        # expensive. Can be formulated as Linear Program, or greedy.

        # TODO: A different cost should be number of gaps. Or number of detached
        # nodes.

        p = len(self.P)
        q = len(self.Q)
        origin = self.origin
        qubit_cols = self.qubit_cols
        qubit_rows = self.qubit_rows
        orientation = self.orientation

        # Restrict search space only if orientation is given
        if orientation is None:
            search_space = [(0, q, p), (1, p, q)]
        elif orientation in (0,1):
            search_space = [(1, p, q)] if orientation==1 else [(0, q, p)]
        else:
            raise ValueError("Orientation must be in (0,1).")

        # Restrict search space if origin is given, or size of problem
        if origin is not None: # Don't slide
            (j, i) = origin
            origins_i = lambda _: [i]
            origins_j = lambda _: [j]
        else: # Slide while (i+width) <= qubit_cols & (j+height) <= qubit_rows
            origins_i = lambda width: range(qubit_cols-width+1)
            origins_j = lambda height: range(qubit_rows-height+1)

        best_origin = None
        best_count = p*q
        # Sliding window through search space
        for orientation, height, width in search_space:
            for j in origins_j(height):
                for i in origins_i(width):
                    if (j+height) <= qubit_rows and (i+width) <= qubit_cols:
                        count_faults = self._find_faults((j, i), width, height)
                        if count_faults < best_count:
                            best_origin = j, i
                            best_count = count_faults
                            best_orientation = orientation

        if best_origin is None:
            raise RuntimeError('Cannot fit problem in target graph.')

        self.orientation = best_orientation
        self.origin = best_origin

        (rows, cols), faults = self._assign_window_nodes()
        return (rows, cols), best_orientation, faults

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

    def _assign_window_nodes(self):
        """ Traverse the target graph, starting at the given "origin" and assign
        the valid qubits found.
            Args:
                origin: Tuple(int, int)
                    Coordinates of the initial assignment of (row, col)

                width: int
                    Number of columns being assigned.

                height: int
                    Number of rows being assigned.
        """

        p = len(self.P)
        q = len(self.Q)

        t = self.t
        Tg = self.Tg
        origin = self.origin
        orientation = self.orientation

        width, height = (q, p) if orientation else (p, q)

        rows = {}
        cols = {}
        faults = {}

        origin_j, origin_i = origin
        for node, col in enumerate(range(origin_i, origin_i+width)):
            i, k = divmod(col, t)
            j_init = (origin_j) // t
            j_end =  (origin_j+height-1) // t + 1
            cols.setdefault(node, [])
            for j in range(j_init, j_end):
                chimera_index = (j, i, 0, k)
                if self.coordinates: chimera_label = chimera_index
                else: chimera_label = self.c2i.int(chimera_index)
                # Check qubit
                if chimera_label in Tg:
                    cols[node].append(chimera_label)
                elif node in faults:
                    faults[node].append(chimera_label)
                else:
                    faults[node] = [chimera_label]
            if not cols[node]:
                raise RuntimeError('Column %s is empty.' % col)

        for node, row in enumerate(range(origin_j, origin_j+height)):
            j, k = divmod(row, t)
            i_init = (origin_i) // t
            i_end =  (origin_i+width-1) // t + 1
            rows.setdefault(node, [])
            for i in range(i_init, i_end):
                chimera_index = (j, i, 1, k)
                if self.coordinates: chimera_label = chimera_index
                else: chimera_label = self.c2i.int(chimera_index)
                # Check qubit
                if chimera_label in Tg:
                    rows[node].append(chimera_label)
                elif node in faults:
                    faults[node].append(chimera_label)
                else:
                    faults[node] = [chimera_label]
            if not rows[node]:
                raise RuntimeError('Row %s is empty.' % row)

        return (rows, cols), faults

    def sort(self, axis=None):
        """ Sort the order of nodes assigned to each row or column.
            Args: (optional)
                axis: (0,1,or None)
                    If None, both axes are sorted.
        """

        # If it has been ran, faults is a dictionary
        if self.faults is None:
            self.run()

        # Determine which axis to sort, or both
        if axis is None:
            sort_p = True; sort_q = True
        elif axis==0:
            sort_p = True; sort_q = False
        elif axis==1:
            sort_p = False; sort_q = True
        else:
            raise ValueError('Value must be 0 or 1, or None for both.')

        def sort_qubit(q):
            # u==0 use column i, u==1 use row j
            (j,i,u,k) = q if self.coordinates else self.c2i.tuple(q)
            return i*self.t+k if u==0 else j*self.t+k

        if sort_p:
            keys_p = list(self.P.keys())
            keys_p.sort()
            chains_p = []
            for v in  keys_p:
                chain = self.P[v]
                chain.sort()
                chains_p.append(chain)
            chains_p.sort(key=lambda x: sort_qubit(x[0]))
            self.P = {k: chains_p[i] for i, k in enumerate(keys_p)}

        if sort_q:
            keys_q = list(self.Q.keys())
            keys_q.sort()
            chains_q = []
            for v in  keys_q:
                chain = self.Q[v]
                chain.sort()
                chains_q.append(chain)
            chains_q.sort(key=lambda x: sort_qubit(x[0]))
            self.Q = {k: chains_q[i] for i, k in enumerate(keys_q)}

    def shuffle(self, axis=None):
        """ Shuffle the order of nodes assigned each row or column.
            Args: (optional)
                axis: (0,1,or None)
                    If None, both axes are shuffled.
        """

        # If it has been ran, faults is a dictionary
        if self.faults is None:
            self.run()

        # Determine which axis to shuffle, or both
        if axis is None:
            shuffle_p = True; shuffle_q = True
        elif axis==0:
            shuffle_p = True; shuffle_q = False
        elif axis==1:
            shuffle_p = False; shuffle_q = True
        else:
            raise ValueError('Value must be 0 or 1, or None for both.')

        if shuffle_p:
            keys_p = list(self.P.keys())
            random.shuffle(keys_p)
            self.P = {keys_p[i]: v for i, (k, v) in enumerate(self.P.items())}

        if shuffle_q:
            keys_q = list(self.Q.keys())
            random.shuffle(keys_q)
            self.Q = {keys_q[i]: v for i, (k, v) in enumerate(self.Q.items())}

    def rotate(self):
        """ Assign chains of s nodes to p nodes and viceversa, by running the
            placer again, except the orientation is the opposite of what was
            obtained. However, if it's a symmetrical graph len(P)==len(Q), then
            it's faster, and more meaningful to just swap chain assignments.
        """
        if self.faults is None:
            self.run()

        self.orientation = self.orientation ^ 1
        if len(self.P)!=len(self.Q):
            _ = self.run()
        else:
            zipped_items = zip(self.P.items(), self.Q.items())
            for (k_p, v_p), (k_q, v_q) in zipped_items:
                self.P[k_p] = v_q
                self.Q[k_q] = v_p

    def get_candidates(self):
        """ Returns merged dictionary of both shores.
        """
        if self.faults is None:
            _ = self.run()
        return {**self.P, **self.Q}

    @classmethod
    def from_candidates(cls, S, Tg, candidates):
        """ Populate attributes from given candidates.
            Args:
                P: dict of lists
                    dictionary keyed by nodes in one shore of the bipartite
                    graph, and values for each chain.

                Q: dict of lists
                    dictionary keyed by nodes in the other shore of the bipartite
                    graph, and values for each chain.

            Returns:
                K_pq: CompleteBipartitePlacer object
                    object had been initialized with the provided/found values.
        """

        K_pq = cls(S, Tg)
        P = { p:candidates[p] for p in K_pq.P }
        Q = { q:candidates[q] for q in K_pq.Q }

        K_pq.P, K_pq.Q = P,Q

        # Determine orientation and origin from candidates
        origin_j, origin_i = (K_pq.qubit_rows, K_pq.qubit_cols)
        orientation = None

        # Note: qubit index is (row, col, shore, index). I preserve my
        # notation of i for cols, and j for rows. Therefore (j,i,u,k)

        # Parse P shore
        for v, chain in P.items():
            if not K_pq.coordinates:
                chain = K_pq.c2i.tuples(chain)
            for (j,i,u,k) in chain:
                if u==0:
                    col = i*K_pq.t + k
                    if col < origin_i:
                        origin_i = col
                else:
                    row = j*K_pq.t + k
                    if row < origin_j:
                        origin_j = row
                if orientation is None:
                    orientation = u
                elif u!=orientation:
                    raise ValueError('Same shore nodes are in rows and cols.')

        # Parse Q shore
        for v, chain in Q.items():
            if not K_pq.coordinates:
                chain = K_pq.c2i.tuples(chain)
            for (j,i,u,k) in chain:
                if u==0:
                    col = i*K_pq.t + k
                    if col < origin_i:
                        origin_i = col
                else:
                    row = j*K_pq.t + k
                    if row < origin_j:
                        origin_j = row
                if u!=orientation^1:
                    raise ValueError('Same shore nodes are in rows and cols.')

        K_pq.orientation = orientation
        K_pq.origin = (origin_j, origin_i)
        _, faults = K_pq._assign_window_nodes()

        # To recover original graph node names in faults
        names_p = list(K_pq.P.keys())
        names_q = list(K_pq.Q.keys())
        names_list = list(names_p + names_q)
        K_pq.faults = {names_list[v]:qubits for v, qubits in faults.items()}

        return K_pq


    def run(self):
        """ Run the complete bipartite placer.

                Returns:
                    candidates: a tuple of dictionaries that map labels in
                        S to lists of labels in T.

                    faults: a list of faulty qubits
        """

        # Use sliding window greedy method to find best mapping
        (rows, cols), orientation, faults = self._slide_window()

        # Assign columns and rows to specified nodes
        for i, (k, v) in enumerate(self.P.items()):
            self.P[k] = rows[i] if orientation==1 else cols[i]
        for j, (k, v) in enumerate(self.Q.items()):
            self.Q[k] = cols[j] if orientation==1 else rows[j]

        # To recover original graph node names
        names_p = list(self.P.keys())
        names_q = list(self.Q.keys())
        names_list = list(names_p + names_q)
        self.faults = {names_list[v]:qubits for v, qubits in faults.items()}

        return (self.P, self.Q), self.faults

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
                    If not None, force the rows and columnd to start from
                    origin (row, col).
                orientation: (0, 1, or None) (default None)
                    If not None, this value determines which set (p or q) is
                    assigned rows or columns. If None, both options are
                    explored to find the orientation with fewer faults.
                    0: p=cols q=rows
                    1: p=rows q=cols
                shores: bool (default False)
                    See below. If True, return a tuple of dictionaries
                show_faults: bool (default False)
                    See below. If True, return a dictionary of the source nodes
                    and their faulty qubits.

        Returns:
            candidates: a tuple of dictionaries that map labels in S to
                lists of labels in T.

            (optional) faults: a list of faulty qubits
    """

    shores = params.pop('shores', False)
    show_faults = params.pop('show_faults', False)

    placer = CompleteBipartitePlacer(S, Tg, **params)
    (P, Q), faults = placer.run()
    if shores:
        candidates = (P, Q)
    else:
        candidates = {**P, **Q}

    if show_faults:
        return candidates, faults
    else:
        return candidates
