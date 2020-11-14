"""
Tools to visualize graphs of architectures.
"""

import warnings
import networkx as nx
import dwave_networkx as dnx
import matplotlib.pyplot as plt

__all__ = ['draw_architecture', 'draw_architecture_embedding', 'draw_architecture_yield']

def draw_architecture(target_graph, **kwargs):
    """ Draws graph G according to G's family layout.
    """
    family = target_graph.graph['family']

    if family == 'chimera':
        dnx.draw_chimera(target_graph, **kwargs)
    elif family == 'pegasus':
        dnx.draw_pegasus(target_graph, **kwargs)
    else:
        nx.draw_spring(target_graph)
        warnings.warn("Graph family not available. Using NetworkX spring layout")

def draw_architecture_yield(target_graph, **kwargs):
    """ Draws graph G according to G's family layout and highlights
        faulty qubits.
    """
    family = target_graph.graph['family']

    try:
        m = target_graph.graph['columns']
        n = target_graph.graph['rows']
        t = target_graph.graph['tile']
        coordinates = target_graph.graph['labels'] == 'coordinate'
    except:
        raise ValueError("Target graph needs to have columns, rows, and tile \
        attributes to identify faulty qubits.")

    if family == 'chimera':
        dnx.draw_chimera_yield(target_graph, **kwargs)
    elif family == 'pegasus':
        dnx.draw_pegasus_yield(target_graph, **kwargs)
    else:
        nx.draw_spring(target_graph)
        warnings.warn("Graph family not available. Using NetworkX spring layout")

def draw_architecture_embedding(target_graph, *args, **kwargs):
    """ Draws an embedding onto the target graph G,
        according to G's family layout.
    """
    family = target_graph.graph.get('family')

    if family == 'chimera':
        dnx.draw_chimera_embedding(target_graph, *args, **kwargs)
    elif family == 'pegasus':
        dnx.draw_pegasus_embedding(target_graph, *args, **kwargs)
    else:
        layout = nx.spring_layout(target_graph)
        dnx.drawing.qubit_layout.draw_embedding(target_graph, layout, *args, **kwargs)
        warnings.warn("Graph family not available. Using NetworkX spring layout")
