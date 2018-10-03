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

import pulp
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

def _parse_target(Tg):
    """ Parse Target graph
     Use coordinates if available, otherwise get chimera indices
     """
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

    faults_row = {}
    faults_col = {}
    qubits_row = {}
    qubits_col = {}
    t = Tg.graph['tile']

    for i in range(Tg.graph['columns']):
        for j in range(Tg.graph['rows']):
            for k in range(t):
                qubit_col = i*t + k
                qubit_row = j*t + k
                faults_col.setdefault(qubit_col, 0)
                faults_row.setdefault(qubit_row, 0)
                qubits_row.setdefault(qubit_row, set())
                qubits_col.setdefault(qubit_col, set())
                for u in range(2):
                    chimera_index = (j, i, u, k)
                    if u==0:
                        if chimera_index not in Tg.nodes:
                            faults_col[qubit_col] += 1
                        else:
                            qubits_col[qubit_col].add(chimera_index)
                    else:
                        if chimera_index not in Tg.nodes:
                            faults_row[qubit_row] += 1
                        else:
                            qubits_row[qubit_row].add(chimera_index)

    return faults_row, faults_col, qubits_row, qubits_col


def _parse_source(S):
    """ Parse Source graph
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
            raise RuntimeError("Input must be either iterable of edges, or tuple of dimensions (p,q)")
    return P, Q


def _setup_lp(p, q, faults_row, faults_col):
    """ Setup linear Programming Problem
    Notes:
        -Constraint names need to start with letters

    Example: K3,4 embedding
        >>  Solve Chains
        >>  Minimize
        >>  OBJ: Z
        >>  Subject To
        >>  ZeroSum12: var112 + var212 = 1
        >>  c_0: Z >= 1
        >>  c_1: Z - var112 >= 1
        >>  c_2: Z - var212 >= 1
        >>  Bounds
        >>  0 <= Z
        >>  0 <= var112
        >>  0 <= var212
        >>  Generals
        >>  Z
        >>  var112
        >>  var212
        >>  End
    """
    lp = pulp.LpProblem("Solve ", pulp.LpMinimize)

    row_vars = pulp.LpVariable.dicts('r', faults_row, lowBound=0, cat='Binary')
    col_vars = pulp.LpVariable.dicts('c', faults_col, lowBound=0, cat='Binary')

    row_expr = [ faults_row[i]*row_vars[i] for i in faults_row ]

    col_expr =  [ faults_col[i]*col_vars[i] for i in faults_col ]
    lp += pulp.lpSum(row_expr + col_expr), 'OBJ'

    lp += pulp.lpSum( [row_vars[i] for i in faults_row] ) == p, 'rows'
    lp += pulp.lpSum( [col_vars[i] for i in faults_col] ) == q, 'cols'


    lp.writeLP("COMPLETEBIPARTITE.lp")

    lp.solve(solver=pulp.GLPK_CMD())

    lp_sol = {}
    for v in  lp.variables():
        lp_sol[v.name] = v.varValue


    return lp_sol

def _assign_nodes(p, q, qubits_row, qubits_col, lp_sol):

    embedding = {}
    row_node = 0
    for row, qubits in qubits_row.items():
        row_name = 'r_' + str(row)
        if lp_sol[row_name]:
            embedding[row_node] = qubits
            row_node += 1


    col_node = p
    for col, qubits in qubits_col.items():
        col_name = 'c_' + str(col)
        if lp_sol[col_name]:
            embedding[col_node] = qubits
            col_node += 1

    return embedding

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

    faults_row, faults_col, qubits_row, qubits_col = _parse_target(Tg)
    P, Q = _parse_source(S)

    p = len(P)
    q = len(Q)

    lp_sol = _setup_lp(p, q, faults_row, faults_col)
    candidates = _assign_nodes(p, q, qubits_row, qubits_col, lp_sol)

    return candidates




#TEMP: Example

import networkx as nx
import matplotlib.pyplot as plt
from embedding_methods.utilities.architectures import drawing, generators

# A 2x2 grid problem graph
p, q = 4, 3
Sg = nx.complete_bipartite_graph(p,q)
S_edgelist = list(Sg.edges())

# The corresponding graph of the D-Wave C4 annealer with 0.95 qubit yield
Tg = generators.faulty_arch(generators.rainier_graph, arch_yield=0.9)()
T_edgelist = list(Tg.edges())

# Systematically find the best candidates
candidates = find_candidates(S_edgelist, Tg)
#candidates = find_candidates(S_edgelist, Tg)

drawing.draw_architecture_embedding(Tg, candidates, node_size=40)
plt.title('Bipartite Embedding')
plt.show()
