
__all__ = ["find_embedding"]

def _simulated_annealing_placement():

    return init_loc

def _global_placement():

    return global_loc

def _scale():

    return scale_loc

def _migrate():
    
    return node_loc

def _route():

    return chains

def find_embedding(S, T, **params):
    """
    Heuristically attempt to find a minor-embedding of a graph representing an
    Ising/QUBO into a target graph.

    Args:

        S: an iterable of label pairs representing the edges in the source graph

        T: an iterable of label pairs representing the edges in the target graph

        **params (optional): see below
    Returns:

        embedding: a dict that maps labels in S to lists of labels in T

    Optional parameters:

        topology ({<node>:(<x>,<y>),...}):
            Dict of 2D positions assigned to the source graph nodes.

        verbose: (bool) enable verbosity

    """

    # If a topology of the graph is not provided



    return embedding
