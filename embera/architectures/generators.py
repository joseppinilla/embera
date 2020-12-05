import os
import re
import json
import tarfile
import requests
import dwave.system
import networkx as nx
import dwave_networkx as dnx

__all__ = ['dwave_collection',
           'rainier_graph', 'vesuvius_graph', 'dw2x_graph', 'dw2000q_graph',
           'p6_graph', 'p16_graph',
           'h20k_graph',
           ]

""" ========================== D-Wave Solver Solutions ===================== """

def dwave_collection(chip_id=None, chip_id__regex=None):
    """ List of architecture graphs from current and legacy D-Wave devices. All
        graphs use 'int' labels for qubits and couplers.

        Note: The "collection.tar.gz" file isn't provided with `embera` to avoid
        breaching export control regulations or Copyright. However, the information
        neccesary to replicate it may be obtained from these public sources:

        [1] D-Wave devices - https://support.dwavesys.com/hc/en-us/articles/360005268633-QPU-Specific-Physical-Properties
        [2] LANL DW-2000Q - https://arxiv.org/pdf/2009.00111.pdf
        [3] QuAIL DW-2000Q - https://arxiv.org/pdf/1810.05881.pdf
        [4] DW2X-SYS - https://www.nature.com/articles/s41598-018-22763-2
        [5] LANL DW2X - https://link.springer.com/chapter/10.1007/978-3-030-14082-3_2

        Each device is stored as a `dwave_networkx` graph in a JSON file produced
        with `networkx.adjacency_data()` with the only addition of the 'chip_id'
        value. Contact the author if you have any questions.

            | name                   | chip_id             | nodes | edges |
            | ---------------------- | ------------------- |:-----:| -----:|
            | pegasus_graph(16)      | Advantage_system1.1 | 5436  | 37440 |
            | chimera_graph(16,16,4) | DW_2000Q_6          | 2041  | 5974  |
            | chimera_graph(16,16,4) | DW_2000Q_5          | 2030  | 5909  |
            | chimera_graph(16,16,4) | DW_2000Q_2_1        | 2038  | 5955  |
            | chimera_graph(16,16,4) | DW_2000Q_QuAIL      | 2031  | 5919  |
            | chimera_graph(12,12,4) | DW_2X_LANL          | 1141  | 3298  |

        Optional Arguments:
            chip_id: (string, default=None)
                If provided, only devices matching the exact `chip_id` are
                returned.

            chip_id__regex: (string, default=None)
                If provided, devices with partial/regex-based matches are
                returned.

        Returns:
            graph_list: (iterable of networkx.Graph)
                Each graph in the list has:
                    >>> G.graph = {'columns': <int>,
                                   'data': bool,
                                   'family': <string>,
                                   'labels': 'int',
                                   'name': <string>,
                                   'rows': <int>,
                                   'tile': <int>,
                                   'chip_id': <string>,
                                   # Only for Pegasus graphs
                                   'horizontal_offsets': <list>,
                                   'vertical_offsets': <list>}
    """
    graph_list = []
    path = "./collection.tar.gz"

    if not os.path.isfile(path):
        raise RuntimeError(f"Collection file {path} not found.")

    conditions = []
    if chip_id is not None:
        conditions.append(lambda x: x==chip_id)
    elif chip_id__regex is not None:
        chip_id_re = re.compile(chip_id__regex)
        conditions.append(lambda x: chip_id_re.search(x))

    # Unzip, untar, unpickle
    with tarfile.open(path) as contents:
        for member in contents.getmembers():
            # Filenames are <chip_id>.json
            root, ext = os.path.splitext(member.name)
            if not all(cond(root) for cond in conditions): continue
            # Extract and parse
            f = contents.extractfile(member)
            G = nx.adjacency_graph(json.load(f))
            graph_list.append(G)

    return graph_list

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
    G : NetworkX Graph of the chosen architecture. All graphs use 'int' labels
    for qubits and couplers.
"""

""" =========================== D-Wave Architectures ======================= """

def rainier_graph(**kwargs):
    """ D-Wave One 'Rainier' Quantum Annealer graph
        https://en.wikipedia.org/wiki/D-Wave_Systems
    """
    target_graph = dnx.generators.chimera_graph(4, 4, 4, **kwargs)
    target_graph.graph['chip_id'] = 'Rainier'
    return target_graph

def vesuvius_graph(**kwargs):
    """ D-Wave Two 'Vesuvius' Quantum Annealer graph
        https://en.wikipedia.org/wiki/D-Wave_Systems
    """
    target_graph = dnx.generators.chimera_graph(8, 8, 4, **kwargs)
    target_graph.graph['chip_id'] = 'Vesuvius'
    return target_graph

def dw2x_graph(**kwargs):
    """ D-Wave 2X Quantum Annealer graph
        https://en.wikipedia.org/wiki/D-Wave_Systems
    """
    target_graph = dnx.generators.chimera_graph(12, 12, 4, **kwargs)
    target_graph.graph['chip_id'] = 'DW_2X'
    return target_graph

def dw2000q_graph(**kwargs):
    """ D-Wave 2000Q Quantum Annealer graph
        https://en.wikipedia.org/wiki/D-Wave_Systems
    """
    target_graph = dnx.generators.chimera_graph(16, 16, 4, **kwargs)
    target_graph.graph['chip_id'] = 'DW_2000Q'
    return target_graph

def p6_graph(**kwargs):
    """ Pegasus 6 graph
        https://www.dwavesys.com/sites/default/files/mwj_dwave_qubits2018.pdf
    """
    target_graph = dnx.generators.pegasus_graph(6, **kwargs)
    target_graph.graph['chip_id'] = 'P6'
    return target_graph

def p16_graph(**kwargs):
    """ Pegasus 16 graph
        https://www.dwavesys.com/sites/default/files/mwj_dwave_qubits2018.pdf
    """
    target_graph = dnx.generators.pegasus_graph(16, **kwargs)
    target_graph.graph['chip_id'] = 'P16'
    return target_graph

""" ============================== Miscellaneous =========================== """

def h20k_graph(data=True, coordinates=False):
    """ HITACHI 20k-Spin CMOS digital annealer graph.
        https://ieeexplore.ieee.org/document/7350099/
    """
    n, m, t = 128, 80, 2

    target_graph = nx.grid_graph(dim=[t, m, n])

    target_graph.name = 'hitachi_graph(128,80,2)'
    target_graph.graph['chip_id'] = 'HITACHI 20k'
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
                target_graph.nodes[t_node]['linear_index'] = linear
    else:
        coordinate_labels = {(x, y, z):x+n*(y+m*z) for (x, y, z) in target_graph}
        if data:
            for t_node in target_graph:
                target_graph.nodes[t_node]['grid_index'] = t_node
        target_graph = nx.relabel_nodes(target_graph, coordinate_labels)

    return target_graph
