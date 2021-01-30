import random
from embera.utilities.decorators import nx_graph

@nx_graph(0)
def prune(G, node_yield=1.0, edge_yield=1.0):
    # Remove nodes
    node_set = set(G.nodes)
    num_node_faults = len(node_set) - round(len(node_set) * node_yield)
    randnodes = random.sample(node_set, int(num_node_faults))
    G.remove_nodes_from(randnodes)
    # Remove edges
    edge_set = set(G.edges)
    num_edge_faults = len(edge_set) - round(len(edge_set) * edge_yield)
    randedges = random.sample(edge_set, int(num_edge_faults))
    G.remove_edges_from(randedges)
    return G
