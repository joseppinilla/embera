""" Generators for architecture graphs.

    All architecture graphs use the same parameters. Additional parameters
    for the underlying generators are allowed but discouraged.

    Parameters
    ----------
    data : bool (optional, default True)
        If True, each node has a `<family>_index attribute`
    coordinates : bool (optional, default False)
        If True, node labels are tuples, equivalent to the <family>_index
        attribute as above.  In this case, the `data` parameter controls the
        existence of a `linear_index attribute`, which is an int

    Returns
    -------
    G : NetworkX Graph of the chosen architecture.
        Nodes are labeled by integers.

"""
import networkx as nx
import dwave_networkx as dnx

__all__ = ['rainier_graph', 'vesuvius_graph', 'dw2x_graph',
            'dw2000q_graph', 'p6_graph', 'p16_graph', 'h20k_graph']


def rainier_graph(**kwargs):
    """ D-Wave One 'Rainier' Quantum Annealer graph
        https://en.wikipedia.org/wiki/D-Wave_Systems
    """
    G = dnx.generators.chimera_graph(4, 4, 4, **kwargs)
    G.name = 'Rainier'
    return G

def vesuvius_graph(**kwargs):
    """ D-Wave Two 'Vesuvius' Quantum Annealer graph
        https://en.wikipedia.org/wiki/D-Wave_Systems
    """
    G = dnx.generators.chimera_graph(8, 8, 4, **kwargs)
    G.name = 'Vesuvius'
    return G

def dw2x_graph(**kwargs):
    """ D-Wave 2X Quantum Annealer graph
        https://en.wikipedia.org/wiki/D-Wave_Systems
    """
    G = dnx.generators.chimera_graph(12, 12, 4, **kwargs)
    G.name = 'DW2X'
    return G

def dw2000q_graph(**kwargs):
    """ D-Wave 2000Q Quantum Annealer graph
        https://en.wikipedia.org/wiki/D-Wave_Systems
    """
    G = dnx.generators.chimera_graph(16, 16, 4, **kwargs)
    G.name = 'DW2000Q'
    return G

def p6_graph(**kwargs):
    """ Pegasus 6 graph
        https://www.dwavesys.com/sites/default/files/mwj_dwave_qubits2018.pdf
    """
    G = dnx.generators.pegasus_graph(6, **kwargs)
    G.name = 'P6'
    return G

def p16_graph(**kwargs):
    """ Pegasus 16 graph
        https://www.dwavesys.com/sites/default/files/mwj_dwave_qubits2018.pdf
    """
    G = dnx.generators.pegasus_graph(16, **kwargs)
    G.name = 'P16'
    return G

def h20k_graph(data=True, coordinates=False):
    """ HITACHI 20k-Spin CMOS digital annealer graph.
        https://ieeexplore.ieee.org/document/7350099/
    """
    n, m, t = 128, 80, 2

    G = nx.grid_graph(dim=[t,m,n])

    G.name = 'HITACHI 20k'
    construction = (("family", "hitachi"),
                    ("rows", 5), ("columns", 4),
                    ("data", data),
                    ("labels", "coordinate" if coordinates else "int"))

    G.graph.update(construction)

    if coordinates:
        if data:
            for q in G:
                (z, y, x) = q
                G.node[q]['linear_index'] = x + n*(y + m*z)
    else:
        coordinate_labels = { (x,y,z) : x + n*(y + m*z) for (x, y, z) in G }
        if data:
            for q in G:
                G.node[q]['grid_index'] = q
        G = nx.relabel_nodes(G, coordinate_labels)

    return G
