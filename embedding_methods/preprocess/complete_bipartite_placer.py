"""


[1] https://arxiv.org/abs/1510.06356

NOTE: Because this systematic node mapping does not guarantee a valid
embedding, these assignments are deemed candidates.
"""

import warnings



__all__ = ['find_candidates']


def find_candidates(S, Tg, **params):
    """ find_candidates(S, Tg, **params)
    Given a complete complete bipartite source graph and a target chimera
    graph. Systematically find the embedding with fewer fault qubits in the
    qubit chains.

        Args:
            S: an iterable of label pairs representing the edges in the
                source graph

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

    warnings.warn('Work in progress.')

    candidates = {}

    return candidates
