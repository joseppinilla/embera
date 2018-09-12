import sys
import pulp
import time
import random
import traceback

import networkx as nx
import matplotlib.pyplot as plt

from math import floor, sqrt
from heapq import heappop, heappush

__all__ = ["find_embedding", "find_candidates"]

"""
Option parser for diffusion-based migration of a graph topology
"""
class DiffusionOptions(object):
    def __init__(self, **params):

        self.random_seed =      params.pop('random_seed', None)
        self.rng =              random.Random(self.random_seed)

        self.tries =            params.pop('tries', 10)
        self.verbose =          params.pop('verbose', 0)

        # If a topology of the graph is not provided, one is generated
        self.topology =         params.pop('topology', None)
        # Diffusion hyperparameters
        self.enable_migration = params.pop('enable_migration', True)
        self.vicinity =         params.pop('vicinity', 0)
        self.delta_t =          params.pop('delta_t', 0.20)
        self.d_lim =            params.pop('d_lim', 0.75)
        self.viscosity =        params.pop('viscosity', 0.00)

        for name in params:
            raise ValueError("%s is not a valid parameter." % name)

"""
Tile Class
"""
class Tile:
    """ Tile for migration stage
    """
    def __init__(self, Tg, i, j, opts):

        m = Tg.graph['rows']
        n = Tg.graph['columns']
        t = Tg.graph['tile']
        family = Tg.graph['family']
        index = j*n + i
        self.name = (i,j)
        self.index = index
        self.nodes = set()
        self.neighbors = self._get_neighbors(i, j, n, m, index)

        if family=='chimera':
            self.supply = self._get_chimera_qubits(Tg, t, i, j)
        elif family=='pegasus':
            self.supply = self._get_pegasus_qubits(Tg, t, i, j)

        if self.supply == 0.0:
            self.concentration = 1.0
        else:
            self.concentration = 0.0

    def _i2c(self, index, n):
        """ Convert tile array index to coordinate
        """
        j, i = divmod(index,n)
        return i, j

    def _get_neighbors(self, i, j, n, m, index):
        """ Calculate indices and names of negihbouring tiles to use recurrently
            during migration and routing.
            The vicinity parameter is later used to prune out the neighbors of
            interest.
            Uses cardinal notation north, south, west, east
        """
        north = self._i2c(index - n, n)     if (j > 0)      else   None
        south = self._i2c(index + n, n)     if (j < m-1)    else   None
        west =  self._i2c(index - 1, n)     if (i > 0)      else   None
        east =  self._i2c(index + 1, n)     if (i < n-1)    else   None

        nw = self._i2c(index - n - 1, n)  if (j > 0    and i > 0)    else None
        ne = self._i2c(index - n + 1, n)  if (j > 0    and i < n-1)  else None
        se = self._i2c(index + n + 1, n)  if (j < m-1  and i < n-1)  else None
        sw = self._i2c(index + n - 1, n)  if (j < m-1  and i > 0)    else None

        return (north, south, west, east, nw, ne, se, sw)

    def _get_chimera_qubits(self, Tg, t, i, j):
        """ Finds the available qubits associated to tile (i,j) of the Chimera
            Graph and returns the supply or number of qubits found.

            The notation (i, j, u, k) is called the chimera index:
                i : indexes the row of the Chimera tile from 0 to m inclusive
                j : indexes the column of the Chimera tile from 0 to n inclusive
                u : qubit orientation (0 = left-hand nodes, 1 = right-hand nodes)
                k : indexes the qubit within either the left- or right-hand shore
                    from 0 to t inclusive
        """
        self.qubits = set()
        v = 0.0
        for u in range(2):
            for k in range(t):
                chimera_index = (i, j, u, k)
                if chimera_index in Tg.nodes:
                    self.qubits.add(chimera_index)
                    v += 1.0
        return v


    def _get_pegasus_qubits(self, Tg, t, i, j):
        """ Finds the avilable qubits associated to tile (i,j) of the Pegasus
            Graph and returns the supply or number of qubits found.

            The notation (u, w, k, z) is called the pegasus index:
                u : qubit orientation (0 = vertical, 1 = horizontal)
                w : orthogonal major offset
                k : orthogonal minor offset
                z : parallel offset
        """
        self.qubits = set()
        v=0.0
        for u in range(2):
            for k in range(t):
                pegasus_index = (u, j, k, i)
                if pegasus_index in Tg.nodes:
                    self.qubits.add(pegasus_index)
                    v += 1.0
        return v

class DummyTile:
    def __init__(self):
        # Keyed in tile dictionary as None
        self.name = None
        # Treat as a fully occupied tile
        self.supply = 0.0
        self.concentration = 1.0
        # Dummy empty set to skip calculations
        self.nodes = set()

class Tiling:
    """Tiling for migration stage
    """
    def __init__(self, Tg, opts):
        # Support for different target architectures
        family = Tg.graph['family']
        # Maximum degree of qubits
        if family=='chimera':
            self.max_degree = 6
        elif family=='pegasus':
            self.max_degree = 15
            # TEMP: When tiling Pegasus graph, column is out of range
            Tg.graph['columns'] -= 1

        n = Tg.graph['columns']
        m = Tg.graph['rows']
        t = Tg.graph['tile']
        self.m = m
        self.n = n
        self.t = t
        self.qubits = float(len(Tg))
        # Mapping of source nodes to tile
        self.mapping = {}
        # Add Tile objects
        self.tiles = {}
        for i in range(n):
            for j in range(m):
                tile = (i,j)
                self.tiles[tile] = Tile(Tg, i, j, opts)
        # Dummy tile to represent boundaries
        self.tiles[None] = DummyTile()
        # Dispersion cost accumulator for termination
        self.dispersion_accum = None

"""
"""

def _scale(tiling, opts):
    """ Assign node locations to in-scale values of the dimension
    of the target graph.
    """
    m = tiling.m
    n = tiling.n
    topology = opts.topology
    P = len(topology)

    ###### Find dimensions of source graph S
    Sx_min = Sy_min = float("inf")
    Sx_max = Sy_max = 0.0
    # Loop through all source graph nodes to find dimensions
    for s_node, (sx, sy) in topology.items():
        Sx_min = min(sx, Sx_min)
        Sx_max = max(sx, Sx_max)
        Sy_min = min(sy, Sy_min)
        Sy_max = max(sy, Sy_max)
    # Source graph width
    Swidth =  (Sx_max - Sx_min)
    Sheight = (Sx_max - Sx_min)

    center_x, center_y = n/2.0, m/2.0
    dist_accum = 0.0
    ###### Normalize and scale
    for name, (x, y) in topology.items():
        norm_x = (x-Sx_min) / Swidth
        norm_y = (y-Sy_min) / Sheight
        scaled_x = norm_x * (n-1) + 0.5
        scaled_y = norm_y * (m-1) + 0.5
        topology[name] = (scaled_x, scaled_y)
        tile = min(floor(scaled_x), n-1), min(floor(scaled_y), m-1)
        tiling.mapping[name] = tile
        tiling.tiles[tile].nodes.add(name)
        dist_accum += (scaled_x-center_x)**2 + (scaled_y-center_y)**2

    # Initial dispersion
    dispersion = dist_accum/P
    tiling.dispersion_accum = [dispersion] * 3

def _get_attractors(tiling, i, j):

    n, s, w, e, nw, ne, se, sw = tiling.tiles[(i,j)].neighbors
    lh = (i >= 0.5*tiling.n)
    lv = (j >= 0.5*tiling.m)

    if lh:
        return (w, n, nw) if lv else (w, s, sw)
    # else
    return (e, n, ne) if lv else (e, s, se)

def _get_gradient(tile, tiling, opts):
    d_lim = opts.d_lim

    d_ij = tile.concentration
    if d_ij == 0.0 or tile.name == None:
        return 0.0, 0.0
    h, v, hv = _get_attractors(tiling, *tile.name)
    d_h = tiling.tiles[h].concentration
    d_v = tiling.tiles[v].concentration
    d_hv = tiling.tiles[hv].concentration
    del_x = - (d_lim - (d_h + 0.5*d_hv)) / (2.0 * d_ij)
    del_y = - (d_lim - (d_v + 0.5*d_hv)) / (2.0 * d_ij)
    return del_x, del_y


def _step(tiling, opts):
    """ Discrete Diffusion Step
    """

    # Problem size
    # Number of Qubits
    Q = tiling.qubits
    m = tiling.m
    n = tiling.n
    delta_t = opts.delta_t
    topology = opts.topology
    viscosity = opts.viscosity

    center_x, center_y = n/2.0, m/2.0
    dist_accum = 0.0

    # Problem size
    P = float(len(topology))
    # Diffusivity
    D = min((viscosity*P) / Q, 1.0)

    # Iterate over tiles
    for tile in tiling.tiles.values():
        gradient_x, gradient_y = _get_gradient(tile, tiling, opts)
        # Iterate over nodes in tile and migrate
        for node in tile.nodes:
            x, y = topology[node]
            l_x = (2.0*x/n)-1.0
            l_y = (2.0*y/m)-1.0
            v_x = l_x * gradient_x
            v_y = l_y * gradient_y
            x_1 = x + (1.0 - D) * v_x * delta_t
            y_1 = y + (1.0 - D) * v_y * delta_t
            topology[node] = (x_1, y_1)
            dist_accum += (x_1-center_x)**2 + (y_1-center_y)**2

    dispersion = dist_accum/P
    return dispersion

def _get_demand(tiling, opts):

    m = tiling.m
    n = tiling.n
    topology = opts.topology

    for s_node, (x, y) in topology.items():
        tile = tiling.mapping[s_node]
        i = min(floor(x), n-1)
        j = min(floor(y), m-1)
        new_tile = (i,j)
        tiling.tiles[tile].nodes.remove(s_node)
        tiling.tiles[new_tile].nodes.add(s_node)
        tiling.mapping[s_node] = new_tile

    for tile in tiling.tiles.values():
        if tile.supply:
            tile.concentration = len(tile.nodes)/tile.supply


    if opts.verbose==4:
        concentrations = {name : "d=%s"%tile.concentration
                        for name, tile in tiling.tiles.items() if name!=None}
        G = nx.Graph()
        G.add_nodes_from(topology.keys())
        draw_tiled_graph(G, n, m, tile_labels=concentrations, layout=topology)
        plt.show()

def _condition(tiling, dispersion):
    """ The algorithm iterates until the dispersion, or average distance of
    the cells from the centre of the tile array, increases or has a cumulative
    variance lower than 1%
    """
    tiling.dispersion_accum.pop(0)
    tiling.dispersion_accum.append(dispersion)
    mean = sum(tiling.dispersion_accum) / 3.0
    prev_val = 0.0
    diff_accum = 0.0
    increasing = True
    for value in tiling.dispersion_accum:
        sq_diff = (value-mean)**2
        diff_accum = diff_accum + sq_diff
        if (value<=prev_val):
            increasing = False
        prev_val = value
    variance = (diff_accum/3.0)
    spread = variance > 0.01
    return spread and not increasing

def _migrate(tiling, opts):
    """
    """
    migrating = opts.enable_migration
    while migrating:
        _get_demand(tiling, opts)
        dispersion = _step(tiling, opts)
        migrating = _condition(tiling, dispersion)

def _assign_candidates(tiling, opts):
    """ Use tiling to create the sets of target
        nodes assigned to each source node.
        #TODO: vicinity
    """

    candidates = {}

    for s_node, s_tile in tiling.mapping.items():
        # Fixed data
        candidates[s_node] = tiling.tiles[s_tile].qubits

    return candidates

def _place(S, tiling, opts):
    """

    """
    if opts.topology:
        _scale(tiling, opts)
        _migrate(tiling, opts)
    else:
        _simulated_annealing(S, tiling, opts)

    candidates = _assign_candidates(tiling, opts)

    return candidates

def _simulated_annealing(S, tiling, opts):
    rng = opts.rng
    m = tiling.m
    n = tiling.n

    init_loc = {}
    for node in S:
        init_loc[node] = ( rng.randint(0, n), rng.randint(0, m) )

    #TODO: Simulated Annealing placement
    opts.enable_migration = False
    return init_loc

def find_candidates(S, Tg, **params):
    """

        Args:

            S: an iterable of label pairs representing the edges in the source graph

            Tg: a NetworkX Graph with construction parameters dictionary:
                    family : {'chimera','pegasus', ...}
                    rows : (int)
                    columns : (int)
                    labels : {'coordinate', 'int'}
                    data : (bool)
                    **family_parameters

            **params (optional): see below

        Returns:

            candidates: a dict that maps labels in S to lists of labels in T

        Optional parameters:
            topology ({<node>:(<x>,<y>),...}):
                Dict of 2D positions assigned to the source graph nodes.

            vicinity (int): Granularity of the candidate assignment.
                0: Single tile
                1: Immediate neighbors = (north, south, east, west)
                2: Extended neighbors = (Immediate) + diagonals
                3: Directed  = (Single) + 3 tiles closest to the node
    """

    opts = DiffusionOptions(**params)

    tiling = Tiling(Tg, opts)

    candidates = _place(S, tiling, opts)

    return candidates

""" Negotiated-congestion based router for multiple
    disjoint Steiner Tree Search.
    Args:


    Returns:
        paths: Dictionary keyed by Problem graph edges with assigned nodes
        unassigned: Dictionary keyed by node from Tg and problem node values

"""

# Routing cost scalers
__alpha_p = 0.0
__alpha_h = 0.0

def _init_graphs(Sg, Tg, opts):

    for s_node, s_data in Sg.nodes(data=True):
        # Fixed data
        s_data['degree'] = Sg.degree(s_node)
        s_data['candidates'] = opts.initial_chains[s_node]

    for t_node, t_data in Tg.nodes(data=True):
        # Fixed cost
        t_data['degree'] = Tg.degree(t_node)
        # BFS
        t_data['history'] =  1.0
        t_data['sharing'] = 0.0

def _get_cost(neighbor_name, Tg):

    global __alpha_p

    next_node = Tg.nodes[neighbor_name]

    sharing_cost = 1.0 + next_node['sharing'] * __alpha_p

    scope_cost = 0.0 #TODO: higher if different tile

    degree_cost = 0.0 #TODO: Use next_node['degree'] with ARCH max_degree

    base_cost = 1.0 + degree_cost + scope_cost

    history_cost = next_node['history']

    return base_cost * sharing_cost * history_cost

def _bfs(target_set, visited, visiting, queue, Tg):
    """ Breadth-First Search
        Args:
            queue: Breadth-First Search Priority Queue

    """

    # Pop node out of Priority queue and its cost, parent, and distance.
    node_cost, node = heappop(queue)
    _, node_parent, node_dist = visiting[node]
    found = False
    while (not found):
        #print("Node: %s"%str(node))
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
        found = (node in target_set) and (node_dist >= 2)

    print( 'Found target %s at %s cost %s' % (str(node), str(node_dist), str(node_cost)) )
    visited[node] = node_cost, node_parent, node_dist
    return node

def _traceback(source, target, reached, visited, unassigned, mapped, Tg):

    _, node_parent, _ = visited[reached]

    if target not in mapped:
        mapped[target] = set([reached])
        Tg.nodes[reached]['sharing'] += 1.0

    path = [reached]
    node = node_parent
    while(node not in mapped[source]):
        print('Node:' + str(node))
        path.append(node)

        if node in unassigned:
            if source in unassigned[node]:
                del unassigned[node]
                mapped[source].add(node)
            elif target in unassigned[node]:
                del unassigned[node]
                mapped[target].add(node)
            else:
                unassigned[node].add(source)
                unassigned[node].add(target)
        else:
            unassigned[node] = set([source, target])

        Tg.nodes[node]['sharing'] += 1.0
        _, node_parent, _ = visited[node]
        node = node_parent
    path.append(node)

    print("Path:" + str(path))
    return path


def _get_target_set(target, mapped, Sg):
    target_node = Sg.nodes[target]
    target_candidates = target_node['candidates']
    return target_candidates if target not in mapped else mapped[target]

def _get_target_dict(targets, mapped, Sg):
    target_dict = {}
    for target in targets:
        target_node = Sg.nodes[target]
        target_candts = target_node['candidates']
        target_set = target_candts if target not in mapped else mapped[target]
        for t_node in target_set:
            if t_node not in target_dict:
                target_dict[t_node] = set([target])
            else:
                target_dict[t_node].add(target)

    return target_dict

def _init_queue(source, visiting, queue, mapped, Sg, Tg):
    if source not in mapped:
        _embed_node(source, mapped, Sg, Tg)

    for node in mapped[source]:
        node_cost = 0.0
        node_parent = source
        node_dist =  1
        visiting[node] = (node_cost, node_parent, node_dist)
        heappush(queue, (node_cost, node))

    if verbose:
        queue_str = str(["%0.3f %s" % (c,str(n)) for c, n in queue])
        print('Init Queue:' + queue_str)

def _steiner_tree(source, targets, mapped, unassigned, Sg, Tg):
    """ Steiner Tree Search
    """
    print('\n New Source:' + str(source))
    # Resulting tree dictionary keyed by edges and path values.
    tree = {}

    # Breadth-First Search
    for target in targets:
        print('Target:' + str(target))
        # Breadth-First Search structures.
        visiting = {} # (cost, parent, distance) during search
        visited = {} # (parent, distance) after popped from queue
        # Priority Queue (cost, name)
        queue = []
        # Start search using previously-assigned nodes
        _init_queue(source, visiting, queue, mapped, Sg, Tg)
        # Search for target candidates, or nodes assigned to target
        target_set = _get_target_set(target, mapped, Sg)
        # BFS graph traversal
        reached = _bfs(target_set, visited, visiting, queue, Tg)
        # Retrace steps from target to source
        path = _traceback(source, target, reached, visited, unassigned, mapped, Tg)
        # Update tree
        edge = (source,target)
        tree[edge] = path

    return tree

def _update_costs(mapped, Tg):
    """ Update present-sharing and history-sharing costs

    """

    global __alpha_h

    legal = True
    print("Mapped")
    for t_nodes in mapped.values():
        for t_node in t_nodes:
            Tg.nodes[t_node]['history'] += __alpha_h
            sharing =  Tg.nodes[t_node]['sharing']
            if sharing > 1.0:
                legal = False
            print(t_node, sharing)

    return legal

def _embed_node(source, mapped, Sg, Tg, opts={}):

    if source in mapped: return

    s_node = Sg.nodes[source]

    # Get best candidate
    candidates = s_node['candidates']
    t_index = min( candidates, key=lambda t: _get_cost(t, Tg) )

    # Populate target node
    t_node = Tg.nodes[t_index]
    t_node['sharing'] += 1.0
    mapped[source] = set([t_index])

def _rip_up(Tg):
    """
    Search Structures
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
    # Next node preferably in pre-selected nodes
    for node in pre_sel:
        if node in pending_set:
            pending_set.remove(node)
            return node
    # Random if not
    return pending_set.pop()

def _route(Sg, Tg, opts):
    """ Negotiated Congestion

    """
    global __alpha_p, __alpha_h

    # Termination criteria
    legal = False
    tries = opts.tries
    # Negotiated Congestion
    while (not legal) and (tries > 0):
        print('############# TRIES LEFT: %s' % tries)
        # First node selection
        pending_set = set(Sg)
        source = _get_node(pending_set)
        # Route Rip Up
        paths, mapped, unassigned = _rip_up(Tg)
        _embed_node(source, mapped, Sg, Tg, opts)
        while pending_set:
            targets = [target for target in Sg[source] if target in pending_set]
            tree = _steiner_tree(source, targets, mapped, unassigned, Sg, Tg)
            paths.update(tree)
            source = _get_node(pending_set, pre_sel=targets)
        legal = _update_costs(mapped, Tg)
        __alpha_p += opts.delta_p
        __alpha_h += opts.delta_h
        tries -= 1
    return legal, paths, mapped, unassigned


""" Linear Programming formulation to solve unassigned nodes.

    Constraints for all edges:
        var<source><source><target> + var<target><source><target> = |path|
    Constraints for all nodes:
        Z - All( var<node><source><target> ) >= |<mapped['node']>|
    Goal:
        min(Z)
"""

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
        # Nodes in path excluding source and target
        shared = len(path) - 2
        if shared > 0:
            # PuLP variable names do not support spaces
            s_name, t_name = (str(x).replace(" ","") for x in edge)
            # Name of variables are var<source><edge> and var<target><edge>
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

    for edge, path in paths.items():
        # Nodes in path excluding source and target
        shared = len(path) - 2
        if shared>0:
            source, target = edge

            var_t = var_map[edge][str(target).replace(" ","")]

            num_t = lp_sol[var_t]
            # Path from traceback starts from target
            for i in range(1,shared+1):
                if i > num_t:
                    mapped[source].add(path[i])
                else:
                    mapped[target].add(path[i])


def _paths_to_chains(legal, paths, mapped, unassigned):

    if not legal: print ('##########\n\n\n\nNOT LEGAL\n\n\n\n##########')

    print('Assigned')
    for node, fixed in mapped.items():
        print(str(node) + str(fixed))

    print('Unassigned')
    for node, shared in unassigned.items():
        print(str(node) + str(shared))

    lp, var_map = _setup_lp(paths, mapped, unassigned)

    if verbose>0: lp.writeLP("SHARING.lp") #TEMP change to verbose 3

    lp.solve(solver=pulp.GLPK_CMD(msg=verbose))

    # read solution
    lp_sol = {}
    for v in  lp.variables():
        lp_sol[v.name] = v.varValue

    print(lp_sol)

    _assign_nodes(paths, lp_sol, var_map, mapped)

    return mapped

"""
Option parser for diffusion-based migration of the graph topology
"""
class RouterOptions(object):
    def __init__(self, **params):

        self.random_seed = params.pop('random_seed', None)
        self.rng = random.Random(self.random_seed)

        self.tries =  params.pop('tries', 10)
        self.verbose =  params.pop('verbose', 0)

        self.delta_p =  params.pop('delta_p', 0.45)
        self.delta_h =  params.pop('delta_h', 0.10)

        self.initial_chains = params.pop('initial_chains', None)

        for name in params:
            raise ValueError("%s is not a valid parameter." % name)

def find_embedding(S, T, **params):
    """
    Heuristically attempt to find a minor-embedding of a graph representing an
    Ising/QUBO into a target graph.

    Args:

        S: an iterable of label pairs representing the edges in the source graph

        T: an iterable of label pairs representing the edges in the target graph
            The node labels for the different target archictures should be either
            node indices or coordinates as given from dwave_networkx_.


        **params (optional): see below
    Returns:

        embedding: a dict that maps labels in S to lists of labels in T

    Optional parameters:

        random_seed (int):

        tries (int):

        verbose (int): Verbosity level
            0: Quiet mode
            1: Print statements
            2: NetworkX graph drawings
            3: Migration process

    """

    opts = RouterOptions(**params)

    Sg = nx.Graph(S)

    Tg = nx.Graph(T)

    _init_graphs(Sg, Tg, opts)

    legal, paths, mapped, unassigned = _route(Sg, Tg, opts)

    embedding = _paths_to_chains(legal, paths, mapped, unassigned)

    return embedding

#TEMP: standalone test


import dwave_networkx as dnx

def draw_tiled_graph(G, n, m, tile_labels={}, layout={}, **kwargs):
    dnx.drawing.qubit_layout.draw_qubit_graph(G, layout,**kwargs)
    plt.grid('on')
    plt.axis('on')
    plt.axis([0,n,0,m])
    x_ticks = range(0, n) # steps are width/width = 1 without scaling
    y_ticks = range(0, m)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    # Label tiles
    for (i,j), label in tile_labels.items():
        plt.text(i, j, label)

def get_stats(embedding):
    max_chain = 0
    min_chain = sys.maxsize
    total = 0
    N = len(embedding)
    for chain in embedding.values():
        chain_len = len(chain)
        total += chain_len
        if chain_len > max_chain:
            max_chain = chain_len
        if chain_len < min_chain:
            min_chain =  chain_len
    avg_chain = total/N
    sum_deviations = 0
    for chain in embedding.values():
        chain_len = len(chain)
        deviation = (chain_len - avg_chain)**2
        sum_deviations += deviation
    std_dev = sqrt(sum_deviations/N)

    return max_chain, min_chain, total, avg_chain, std_dev

if __name__ == "__main__":

    verbose = 4

    p = 2
    Sg = nx.grid_2d_graph(p, p)
    topology = {v:v for v in Sg}

    #S = nx.cycle_graph(p)
    #topology = nx.circular_layout(S)

    #S = nx.complete_graph(p)
    #topology = nx.spring_layout(S)

    m = 8
    Tg = dnx.chimera_graph(m, coordinates=True) #TODO: Needs coordinates?
    #T = dnx.pegasus_graph(m, coordinates=True)


    S_edgelist = list(Sg.edges())
    T_edgelist = list(Tg.edges())

    try:
        candidates = find_candidates(S_edgelist, Tg,
                                     topology=topology,
                                     enable_migration=True,
                                     verbose=verbose)
        embedding = find_embedding(S_edgelist, T_edgelist,
                                    initial_chains=candidates,
                                    verbose=verbose)
        print('Layout:\n%s' % str(get_stats(embedding)))
    except:
        traceback.print_exc()

    # import minorminer
    #
    # t_start = time.time()
    # mm_embedding = minorminer.find_embedding( S_edgelist, T_edgelist,
    #                                 initial_chains=candidates,
    #                                 verbose=verbose)
    # t_end = time.time()
    # t_elap = t_end-t_start
    # print('MinorMiner:\n%s in %s' % (str(get_stats(mm_embedding)), t_elap) )
    #
    # t_start = time.time()
    # mm_embedding = minorminer.find_embedding( S_edgelist, T_edgelist,
    #                                 #initial_chains=candidates,
    #                                 verbose=verbose)
    # t_end = time.time()
    # t_elap = t_end-t_start
    # print('MinorMiner:\n%s in %s' % (str(get_stats(mm_embedding)), t_elap) )




    plt.clf()
    dnx.draw_chimera_embedding(Tg, embedding)
    #dnx.draw_pegasus_embedding(T, embedding)
    plt.show()
