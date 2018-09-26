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
import random
import networkx as nx
import dwave_networkx as dnx


__all__ = ['faulty_arch', 'rainier_graph', 'vesuvius_graph', 'dw2x_graph',
           'dw2000q_graph', 'p6_graph', 'p16_graph', 'h20k_graph']


def rainier_graph(**kwargs):
    """ D-Wave One 'Rainier' Quantum Annealer graph
        https://en.wikipedia.org/wiki/D-Wave_Systems
    """
    target_graph = dnx.generators.chimera_graph(4, 4, 4, **kwargs)
    target_graph.name = 'Rainier'
    return target_graph

def vesuvius_graph(**kwargs):
    """ D-Wave Two 'Vesuvius' Quantum Annealer graph
        https://en.wikipedia.org/wiki/D-Wave_Systems
    """
    target_graph = dnx.generators.chimera_graph(8, 8, 4, **kwargs)
    target_graph.name = 'Vesuvius'
    return target_graph

def dw2x_graph(**kwargs):
    """ D-Wave 2X Quantum Annealer graph
        https://en.wikipedia.org/wiki/D-Wave_Systems
    """
    target_graph = dnx.generators.chimera_graph(12, 12, 4, **kwargs)
    target_graph.name = 'DW2X'
    return target_graph

def dw2000q_graph(**kwargs):
    """ D-Wave 2000Q Quantum Annealer graph
        https://en.wikipedia.org/wiki/D-Wave_Systems
    """
    target_graph = dnx.generators.chimera_graph(16, 16, 4, **kwargs)
    target_graph.name = 'DW2000Q'
    return target_graph

def p6_graph(**kwargs):
    """ Pegasus 6 graph
        https://www.dwavesys.com/sites/default/files/mwj_dwave_qubits2018.pdf
    """
    target_graph = dnx.generators.pegasus_graph(6, **kwargs)
    target_graph.name = 'P6'
    return target_graph

def p16_graph(**kwargs):
    """ Pegasus 16 graph
        https://www.dwavesys.com/sites/default/files/mwj_dwave_qubits2018.pdf
    """
    target_graph = dnx.generators.pegasus_graph(16, **kwargs)
    target_graph.name = 'P16'
    return target_graph

def h20k_graph(data=True, coordinates=False):
    """ HITACHI 20k-Spin CMOS digital annealer graph.
        https://ieeexplore.ieee.org/document/7350099/
    """
    n, m, t = 128, 80, 2

    target_graph = nx.grid_graph(dim=[t, m, n])

    target_graph.name = 'HITACHI 20k'
    construction = (("family", "hitachi"),
                    ("rows", 5), ("columns", 4),
                    ("data", data),
                    ("labels", "coordinate" if coordinates else "int"))

    target_graph.graph.update(construction)

    if coordinates:
        if data:
            for t_node in target_graph:
                (z_coord, y_coord, x_coord) = t_node
                linear = x_coord + n*(y_coord + m*z_coord)
                target_graph.node[t_node]['linear_index'] = linear
    else:
        coordinate_labels = {(x, y, z):x+n*(y+m*z) for (x, y, z) in target_graph}
        if data:
            for t_node in target_graph:
                target_graph.node[t_node]['grid_index'] = t_node
        target_graph = nx.relabel_nodes(target_graph, coordinate_labels)

    return target_graph

def _faults(arch_method, arch_yield, **kwargs):
    """ Generate a graph of the given architecture with
    (size*yield) of the original nodes.
    Args:
        arch_method (function):
            One of the graph generator function

        arch_yield (optional, float, default=0.95):
            Ratio of nodes over original size.
            i.e. The original graph has yield=1.0
    """
    target_graph =  arch_method(**kwargs)
    node_set = set(target_graph.nodes)
    target_size = len(target_graph)
    num_faults = target_size - round(target_size * arch_yield)
    randnodes = random.sample(node_set, int(num_faults))
    target_graph.remove_nodes_from(randnodes)
    target_graph.name = target_graph.name + '%s' % arch_yield
    return target_graph
def faulty_arch(arch_method, arch_yield=0.95):
    arch_gen = lambda **kwargs: _faults(arch_method, arch_yield, **kwargs)
    arch_gen.__name__ = arch_method.__name__ + str(arch_yield)
    return arch_gen
