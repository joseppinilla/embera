"""
Tools to visualize graphs of architectures.
"""

import warnings
import networkx as nx
import dwave_networkx as dnx


__all__ = ['draw_architecture','draw_architecture_embedding']

def draw_architecture(G, **kwargs):
    """ Draws graph G according to G's family layout.
    """
    family = G.graph['family']

    if family == 'chimera':
        dnx.draw_chimera(G, **kwargs)

    elif family == 'pegasus':
        dnx.draw_pegasus(G, **kwargs)

    else:
        nx.draw_spring(G)
        warnings.warn("Graph family not available. Using NetworkX spring layout")

def draw_architecture_embedding(G,  *args, **kwargs):
    """ Draws an embedding onto the target graph G,
        according to G's family layout.
    """
    family = G.graph['family']

    if family == 'chimera':
        dnx.draw_chimera_embedding(G, embedding, **kwargs)

    elif family == 'pegasus':
        dnx.draw_pegasus_embedding(G, embedding, **kwargs)

    else:
        layout = nx.spring_layout(G)
        dnx.drawing.qubit_layout.draw_embedding(G, layout,  *args, **kwargs)
        warnings.warn("Graph family not available. Using NetworkX spring layout")
