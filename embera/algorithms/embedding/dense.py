""" Dense Placement Embedding Algorithm

It greedely attempts to place every node in the source connectivity graph onto a
corresponding assigned qubit in the Chimera graph. In the case that there is no
suitable qubit available, the translational symmetry of the chimera graph allows
an avenue of qubits and couplers to be “opened” by shifting all qubits after or
before a row or column.
[1] https://arxiv.org/abs/1709.04972

NOTE: This embedding method is limited to the Chimera architecture
"""
import warnings

__all__ = ["find_embedding"]


DEFAULT_CHIMERA = {'family': 'chimera', 'rows': 12, 'columns': 12, 'tile': 4}

def find_embedding(S, T, graph_dict=DEFAULT_CHIMERA, **params):
    """ find_embedding(S, T, T_dict, **params)
    Heuristically attempt to find a minor-embedding of a graph, representing an
    Ising/QUBO, into a target graph.

    Args:

        S: an iterable of label pairs representing the edges in the source graph

        T: an iterable of label pairs representing the edges in the Chimera graph

        graph_dict: a dictionary of Chimera graph construction parameters
        (default: C12 Vesuvius graph)

        **params (optional): see RouterOptions_

    Returns:

        embedding: a dict that maps labels in S to lists of labels in T

    """

    warnings.warn('Work in Progress.')

    embedding = {}

    return embedding
