"""
Utility methods to measure statistics of the resulting embeddings.

    Chains:
        Number of qubits associated to each source-model variable

    Interactions:
        Number of edges between adjacent qubits that are adjacent
        source-model variables.

"""

import math

__all__ = ["get_chain_stats", "get_interactions_stats"]

def get_chain_stats(embedding):
    """ Embedded chains account for the number of qubits used per
    variable in the source.

    Arg:
        embedding (dict):
            Mapping from source graph to target graph as a dict of form
            {s: {t, ...}, ...}, where s is a source-model variable and t
            is a target-model variable.
    Returns:
        *stats:
            Max, min, tota, average, standard deviation

    """
    total = 0
    max_chain = 0
    N = len(embedding)
    min_chain = float('inf')

    # Get max min and total
    for chain in embedding.values():
        chain_len = len(chain)
        total += chain_len
        if chain_len > max_chain:
            max_chain = chain_len
        if chain_len < min_chain:
            min_chain =  chain_len

    # Get avg and standard deviation
    avg_chain = total/N
    sum_deviations = 0
    for chain in embedding.values():
        chain_len = len(chain)
        deviation = (chain_len - avg_chain)**2
        sum_deviations += deviation
    std_dev = math.sqrt(sum_deviations/N)

    return max_chain, min_chain, total, avg_chain, std_dev


def get_interactions_stats(bqm, embedding, target_adjacency):
    """ Interactions are edges between chains that are connected in
    the source adjacency.

    Args:
        bqm (:obj:`dimod.BinaryQuadraticModel`):
            Binary quadratic model to be sampled from.

        embedding (dict):
            Mapping from source graph to target graph as a dict of form
            {s: {t, ...}, ...}, where s is a source-model variable and t
            is a target-model variable.
        target_adjacency (dict/:class:`networkx.Graph`):
            Adjacency of the target graph as a dict of form {t: Nt, ...}, where
            t is a variable in the target graph and Nt is its set of neighbours.

    Returns:
        *stats:
            Max, min, tota, average, standard deviation
    """
    total = 0
    max_inters = 0
    N = len(embedding)
    min_inters = float('inf')

    # Get max min and total
    interactions_dict = {}
    for edge, _ in bqm.quadratic.items():
        (u, v) = edge
        available_interactions = {(s, t) for s in embedding[u] for t in embedding[v] if s in target_adjacency[t]}
        if not available_interactions:
            raise ValueError("no edges in target graph between source variables {}, {}".format(u, v))

        num_interactions = len(available_interactions)
        interactions_dict[(u,v)] = num_interactions
        total += num_interactions
        if num_interactions > max_inters:
            max_inters = num_interactions
        if num_interactions < min_inters:
            min_inters =  num_interactions

    # Get avg and standard deviation
    avg_inters = total/N
    sum_deviations = 0
    for (u, v), num_interactions in interactions_dict.items():
        deviation = (num_interactions - avg_inters)**2
        sum_deviations += deviation
    std_dev = math.sqrt(sum_deviations/N)

    return max_inters, min_inters, total, avg_inters, std_dev
