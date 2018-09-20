""" Disperse router for multiple disjoint Steiner Tree Search.

This router uses a set of initial chains as candidates to reduce the search
space of multiple Steiner Trees. As a result, the root of each tree or chain
in the resultant embedding is one of the provided candidates.

"""

import pulp
import random
import warnings

import networkx as nx
import matplotlib.pyplot as plt

from heapq import heappop, heappush

__all__ = ["find_embedding"]

# Routing cost scalers
ALPHA_P = 0.0
ALPHA_H = 0.0

def _init_graphs(Sg, Tg, initial_chains, opts):
    """ Assign values to source and target graphs required for
    the tree search.
    """
    for s_node, s_data in Sg.nodes(data=True):
        # Fixed data
        s_data['degree'] = Sg.degree(s_node)
        try:
            s_data['candidates'] = initial_chains[s_node]
        except KeyError:
            raise KeyError('All source graph nodes require an initial'
                            'chain of candidate target nodes.')

    for t_node, t_data in Tg.nodes(data=True):
        # Fixed cost
        t_data['degree'] = Tg.degree(t_node)
        # BFS
        t_data['history'] =  1.0
        t_data['sharing'] = 0.0

def _get_cost(neighbor_name, Tg):
    """ The cost of using one target node is defined to depend on a base cost,
    a present-sharing cost, and a historical-sharing cost.
    """

    next_node = Tg.nodes[neighbor_name]

    sharing_cost = 1.0 + next_node['sharing'] * ALPHA_P

    scope_cost = 0.0 #TODO: higher if different tile

    degree_cost = 0.0 #TODO: Use next_node['degree'] with ARCH max_degree

    base_cost = 1.0 + degree_cost + scope_cost

    history_cost = next_node['history']

    return base_cost * sharing_cost * history_cost

def _bfs(sink_set, visited, visiting, queue, Tg):
    """ Breadth-First Search
    """

    # Pop node out of Priority queue and its cost, parent, and distance.
    node_cost, node = heappop(queue)
    _, node_parent, node_dist = visiting[node]
    found = False
    while (not found):
        neighbor_dist = node_dist + 1
        neighbor_parent = node
        for neighbor in Tg[node]:
            if neighbor not in visited:
                neighbor_cost = node_cost + _get_cost(neighbor, Tg)

                heappush(queue, (neighbor_cost, neighbor))
                neighbor_data = neighbor_cost, neighbor_parent, neighbor_dist

                prev_cost, _, _ = visiting.setdefault(neighbor, neighbor_data)
                if prev_cost > neighbor_cost:
                    visiting[neighbor] = neighbor_data

        # Once all neighbours have been checked
        visited[node] = node_cost, node_parent, node_dist
        node_cost, node = heappop(queue)

        _, node_parent, node_dist = visiting[node]
        found = (node in sink_set) and (node_dist >= 2)

    visited[node] = node_cost, node_parent, node_dist
    return node

def _traceback(source, sink, reached, visited, unassigned, mapped, Tg):
    """ Retrace steps from sink to source and populate target graph nodes
    accordingly. Use head and tail of chain as mapped nodes.
    """
    # If
    if sink not in mapped:
        mapped[sink] = set([reached])
        Tg.nodes[reached]['sharing'] += 1.0

    path = [reached]
    _, node_parent, _ = visited[reached]
    node = node_parent
    while(node not in mapped[source]):
        path.append(node)

        if node in unassigned:
            if source in unassigned[node]:
                del unassigned[node]
                mapped[source].add(node)
            elif sink in unassigned[node]:
                del unassigned[node]
                mapped[sink].add(node)
            else:
                unassigned[node].add(source)
                unassigned[node].add(sink)
        else:
            unassigned[node] = set([source, sink])

        Tg.nodes[node]['sharing'] += 1.0
        _, node_parent, _ = visited[node]
        node = node_parent
    path.append(node)

    return path

def _get_sink_set(sink, mapped, Sg):
    """ Given a node, return either the its associated candidates or
    previously mapped nodes.
    """
    sink_node = Sg.nodes[sink]
    sink_candidates = sink_node['candidates']
    return sink_candidates if sink not in mapped else mapped[sink]

def _init_queue(source, visiting, queue, mapped, Sg, Tg):
    """ Given a source node, expand the search queue over previously
        assigned target nodes with cost 0.0
    """
    if source not in mapped:
        _embed_node(source, mapped, Sg, Tg)

    for node in mapped[source]:
        node_cost = 0.0
        node_parent = source
        node_dist =  1
        visiting[node] = (node_cost, node_parent, node_dist)
        heappush(queue, (node_cost, node))

def _steiner_tree(source, sinks, mapped, unassigned, Sg, Tg):
    """ Steiner Tree Search
    """
    # Resulting tree dictionary keyed by edges and path values.
    tree = {}
    # Breadth-First Search
    for sink in sinks:
        # Breadth-First Search structures.
        visiting = {} # (cost, parent, distance) during search
        visited = {} # (parent, distance) after popped from queue
        # Priority Queue (cost, name)
        queue = []
        # Start search using previously-assigned nodes
        _init_queue(source, visiting, queue, mapped, Sg, Tg)
        # Search for sink candidates, or nodes assigned to sink
        sink_set = _get_sink_set(sink, mapped, Sg)
        # BFS graph traversal
        reached = _bfs(sink_set, visited, visiting, queue, Tg)
        # Retrace steps from sink to source
        path = _traceback(source, sink, reached, visited, unassigned, mapped, Tg)
        # Update tree
        edge = (source,sink)
        tree[edge] = path

    return tree

def _update_costs(mapped, Tg):
    """ Update present-sharing and history-sharing costs.
    If a target node is shared, the embedding is not legal.
    """

    legal = True
    for s_map in mapped.values():
        for t_node in s_map:
            Tg.nodes[t_node]['history'] += ALPHA_H
            sharing =  Tg.nodes[t_node]['sharing']
            if sharing > 1.0:
                legal = False

    return legal

def _embed_node(source, mapped, Sg, Tg, opts={}):
    """ Given a source node and the current mapping from source nodes to target
    nodes. Select the lowest cost target node from the set of candidates assigned
    to the source node.
    """

    s_node = Sg.nodes[source]

    # Get best candidate
    candidates = s_node['candidates']
    t_index = min( candidates, key=lambda t: _get_cost(t, Tg) )

    # Populate target node
    t_node = Tg.nodes[t_index]
    t_node['sharing'] += 1.0
    mapped[source] = set([t_index])

def _rip_up(Tg):
    """ Rip Up current embedding
    paths = { Sg edge : Tg nodes path }
    mapped = { Sg node: set(Tg nodes) }
    unassigned = { Tg node : set(Sg nodes) }
    """
    paths={}
    mapped={}
    unassigned={}

    # BFS
    for t_node in Tg.nodes():
        Tg.nodes[t_node]['sharing'] = 0.0

    return paths, mapped, unassigned

def _get_node(pending_set, pre_sel=[], opts={}):
    """ Next node preferably in pre-selected nodes
    """
    for node in pre_sel:
        if node in pending_set:
            pending_set.remove(node)
            return node
    # Random if not
    return pending_set.pop()

def _route(Sg, Tg, opts):
    """ The disperse router uses a negotiated-congestion scheme, which is widely
    used for FPGA routing, in which overlap of resources is initially allowed
    but the costs of using each target node is recalculated until a legal
    solution is found. A solution is legal when the occupancy of the qubits do
    not have conflicts.
    """
    global ALPHA_P, ALPHA_H

    # Termination criteria
    legal = False
    tries = opts.tries
    # Negotiated Congestion
    while (not legal) and (tries > 0):
        if opts.verbose: print('############# TRIES LEFT: %s' % tries)
        # First node selection
        pending_set = set(Sg)
        source = _get_node(pending_set)
        # Route Rip Up
        paths, mapped, unassigned = _rip_up(Tg)
        _embed_node(source, mapped, Sg, Tg, opts)
        while pending_set:
            sinks = [sink for sink in Sg[source] if sink in pending_set]
            tree = _steiner_tree(source, sinks, mapped, unassigned, Sg, Tg)
            paths.update(tree)
            source = _get_node(pending_set, pre_sel=sinks)
        legal = _update_costs(mapped, Tg)
        ALPHA_P += opts.delta_p
        ALPHA_H += opts.delta_h
        tries -= 1
    return legal, paths, mapped, unassigned


def _setup_lp(paths, mapped, unassigned):
    """ Setup linear Programming Problem
    Notes:
        -Constraint names need to start with letters

    Example: K3 embedding requiring a 2-length Chain
        >>  Solve Chains
        >>  Minimize
        >>  OBJ: Z
        >>  Subject To
        >>  ZeroSum12: var112 + var212 = 1
        >>  c_0: Z >= 1
        >>  c_1: Z - var112 >= 1
        >>  c_2: Z - var212 >= 1
        >>  Bounds
        >>  0 <= Z
        >>  0 <= var112
        >>  0 <= var212
        >>  Generals
        >>  Z
        >>  var112
        >>  var212
        >>  End
    """
    lp = pulp.LpProblem("Solve Chains", pulp.LpMinimize)

    Z = pulp.LpVariable('Z', lowBound=0, cat='Integer')
    lp += Z, "OBJ"

    var_map = {}
    Lpvars = {}

    # Create constraints per source node
    for node, assigned in mapped.items():
        node_name = str(node).replace(" ","")
        lp += Z >= len(assigned), 'c_' + node_name

    # Create variables per path to select unassigned nodes
    for edge, path in paths.items():
        # Nodes in path excluding source and sink
        shared = len(path) - 2
        if shared > 0:
            # PuLP variable names do not support spaces
            s_name, t_name = (str(x).replace(" ","") for x in edge)
            # Name of variables are var<source><edge> and var<sink><edge>
            var_s = "var" + s_name + s_name + t_name
            var_t = "var" + t_name + s_name + t_name

            # create variables
            Lpvars[var_s] = pulp.LpVariable(var_s, lowBound=0, cat='Integer')
            Lpvars[var_t] = pulp.LpVariable(var_t, lowBound=0, cat='Integer')

            var_map[edge] = {}
            var_map[edge][s_name] = var_s
            var_map[edge][t_name] = var_t

            constraint_name = "ZeroSum" + s_name + t_name
            lp += Lpvars[var_s] + Lpvars[var_t] == shared, constraint_name
            lp.constraints['c_' + s_name] -= Lpvars[var_s]
            lp.constraints['c_' + t_name] -= Lpvars[var_t]

    return lp, var_map

def _assign_nodes(paths, lp_sol, var_map, mapped):
    """ Once a solution to the linear program is found, the new mapping
    of nodes is transformed from the resulting number of target nodes to
    add to a source node, into the corresponding target nodes in the target
    graph.
    """
    for edge, path in paths.items():
        # Nodes in path excluding source and sink
        shared = len(path) - 2
        if shared>0:
            source, sink = edge

            var_t = var_map[edge][str(sink).replace(" ","")]

            num_t = lp_sol[var_t]
            # Path from traceback starts from sink
            for i in range(1,shared+1):
                if i > num_t:
                    mapped[source].add(path[i])
                else:
                    mapped[sink].add(path[i])


def _paths_to_chains(legal, paths, mapped, unassigned, opts):
    """ Using a Linear Programming formulation, map the unassigned
    target nodes, so that the maximum length chain in the embedding
    is minimized.

    Linear Programming formulation to solve unassigned nodes.

        Constraints for all edges:
            var<source><source><sink> + var<sink><source><sink> = |path|
        Constraints for all nodes:
            Z - All( var<node><source><sink> ) >= |<mapped['node']>|
        Goal:
            min(Z)
    """

    if not legal:
        raise RuntimeError('Embedding is illegal.')

    if opts.verbose:
        print('Assigned')
        for node, fixed in mapped.items():
            print(str(node) + str(fixed))
        print('Unassigned')
        for node, shared in unassigned.items():
            print(str(node) + str(shared))

    lp, var_map = _setup_lp(paths, mapped, unassigned)

    if opts.verbose==2: lp.writeLP("SHARING.lp") #TEMP change to verbose 3

    lp.solve(solver=pulp.GLPK_CMD(msg=opts.verbose))

    # read solution
    lp_sol = {}
    for v in  lp.variables():
        lp_sol[v.name] = v.varValue

    _assign_nodes(paths, lp_sol, var_map, mapped)

    return mapped

class RouterOptions(object):
    """ Option parser for negotiated congestion based detailed router.
        Optional parameters:

            random_seed (int): Used as an argument for the RNG.

            tries (int): the algorithm iteratively tries to find an embedding

            delta_p (float):

            delta_h (float):

            verbose (int): Verbosity level
                0: Quiet mode
                1: Print statements
                2: Log LP problem
    """
    def __init__(self, **params):

        self.random_seed = params.pop('random_seed', None)
        self.rng = random.Random(self.random_seed)

        self.tries =  params.pop('tries', 10)

        self.delta_p =  params.pop('delta_p', 0.45)
        self.delta_h =  params.pop('delta_h', 0.10)

        self.verbose =  params.pop('verbose', 0)

        for name in params:
            raise ValueError("%s is not a valid parameter." % name)

def find_embedding(S, T, initial_chains, **params):
    """ find_embedding(S, T, **params)
    Heuristically attempt to find a minor-embedding of a graph, representing an
    Ising/QUBO, into a target graph.

    Args:

        S: an iterable of label pairs representing the edges in the source graph

        T: an iterable of label pairs representing the edges in the target graph
            The node labels for the different target architectures should be either
            node indices or coordinates as given from dwave_networkx_.

        initial_chains: a dictionary, where initial_chains[i] is a list of
            target graph nodes to use as candidates.


        **params (optional): see RouterOptions_

    Returns:

        embedding: a dict that maps labels in S to lists of labels in T

    """

    opts = RouterOptions(**params)

    Sg = nx.Graph(S)

    Tg = nx.Graph(T)

    _init_graphs(Sg, Tg, initial_chains, opts)

    legal, paths, mapped, unassigned = _route(Sg, Tg, opts)

    embedding = _paths_to_chains(legal, paths, mapped, unassigned, opts)

    return embedding
