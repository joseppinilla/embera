import pulp
import traceback
import networkx as nx
import dwave_networkx as dnx
from heapq import heapify, heappop, heappush
from math import floor, sqrt
import matplotlib.pyplot as plt
from embedding_methods.utilities import *


# Concentration limit
__d_lim__ = 0.75

__all__ = ["find_embedding"]

def _get_neighbors(i, j, n, m, index):
    """ Calculate indices and names of negihbouring tiles to use recurrently
        during migration and routing.
        The vicinity parameter is later used to prune out the neighbors of
        interest.
        Uses cardinal notation north, south, ...
    """
    north = i2c(index - n, n)     if (j > 0)      else   None
    south = i2c(index + n, n)     if (j < m-1)    else   None
    west =  i2c(index - 1, n)     if (i > 0)      else   None
    east =  i2c(index + 1, n)     if (i < n-1)    else   None

    nw = i2c(index - n - 1, n)  if (j > 0    and i > 0)    else None
    ne = i2c(index - n + 1, n)  if (j > 0    and i < n-1)  else None
    se = i2c(index + n + 1, n)  if (j < m-1  and i < n-1)  else None
    sw = i2c(index + n - 1, n)  if (j < m-1  and i > 0)    else None

    return (north,south,west,east,nw,ne,se,sw)

class DummyTile:
    def __init__(self):
        # Keyed in tile dictionary as None
        self.name = None
        # Treat as a fully occupied tile
        self.concentration = 1.0
        # Dummy empty set to skip calculations
        self.nodes = set()

class Tile:
    """ Tile for migration stage
    """
    def __init__(self, Tg, i, j, opts):

        m = opts.construction['rows']
        n = opts.construction['columns']
        t = opts.construction['tile']
        family = opts.construction['family']
        index = j*n + i
        self.name = (i,j)
        self.index = index
        self.nodes = set()
        self.neighbors = _get_neighbors(i, j, n, m, index)

        if family=='chimera':
            self.supply = self._get_chimera_qubits(Tg, t, i, j)
        elif family=='pegasus':
            self.supply = self._get_pegasus_qubits(Tg, t, i, j)

        if self.supply == 0.0:
            self.concentration = 1.0
        else:
            self.concentration = 0.0

    def add_node(self, node):
        self.nodes.add(node)

    def remove_node(self, node):
        self.nodes.remove(node)

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
                    Tg.nodes[chimera_index]['tile'] = (i,j)
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
                    Tg.nodes[pegasus_index]['tile'] = (i,j)
                    v += 1.0
        return v

class Tiling:
    """Tiling for migration stage
    """
    def __init__(self, Tg, opts):
        # Support for different target architectures
        family = opts.construction['family']
        # Maximum degree of qubits
        if family=='chimera':
            self.max_degree = 6
        elif family=='pegasus':
            self.max_degree = 15
            opts.construction['columns'] -= 1

        n = opts.construction['columns']
        m = opts.construction['rows']
        t = opts.construction['tile']
        self.m = m
        self.n = n
        self.t = t
        self.qubits = 1.0*len(Tg)
        self.family = family
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

def _scale(Sg, tiling, opts):
    """ Assign node locations to in-scale values of the dimension
    of the target graph.
    """
    P = len(Sg)
    m = opts.construction['rows']
    n = opts.construction['columns']
    topology = opts.topology

    ###### Find dimensions of source graph S
    Sx_min = Sy_min = float("inf")
    Sx_max = Sy_max = 0.0
    # Loop through all source graph nodes to find dimensions
    for name, node in Sg.nodes(data=True):
        sx,sy = topology[name]
        Sx_min = min(sx,Sx_min)
        Sx_max = max(sx,Sx_max)
        Sy_min = min(sy,Sy_min)
        Sy_max = max(sy,Sy_max)
    # Source graph width
    Swidth =  (Sx_max - Sx_min)
    Sheight = (Sx_max - Sx_min)

    center_x, center_y = n/2.0, m/2.0
    dist_accum = 0.0
    ###### Normalize and scale
    for name, node in Sg.nodes(data=True):
        x,y = topology[name]
        norm_x = (x-Sx_min) / Swidth
        norm_y = (y-Sy_min) / Sheight
        scaled_x = norm_x * (n-1) + 0.5
        scaled_y = norm_y * (m-1) + 0.5
        node['coordinate'] = (scaled_x, scaled_y)
        tile = min(floor(scaled_x), n-1), min(floor(scaled_y), m-1)
        node['tile'] = tile
        tiling.tiles[tile].nodes.add(name)
        dist_accum += (scaled_x-center_x)**2 + (scaled_y-center_x)**2

    # Initial dispersion
    dispersion = dist_accum/P
    tiling.dispersion_accum = [dispersion] * 3

def _get_attractors(tiling, i, j):

    n,s,w,e,nw,ne,se,sw = tiling.tiles[(i,j)].neighbors
    lh = (i >= 0.5*tiling.n)
    lv = (j >= 0.5*tiling.m)

    if (lh):
        if (lv):    return w,n,nw
        else:       return w,s,sw
    else:
        if (lv):    return e,n,ne
        else:       return e,s,se

def _get_gradient(tile, tiling):

    d_ij = tile.concentration
    if d_ij == 0.0 or tile.name==None: return 0.0, 0.0
    h, v, hv = _get_attractors(tiling, *tile.name)
    d_h = tiling.tiles[h].concentration
    d_v = tiling.tiles[v].concentration
    d_hv = tiling.tiles[hv].concentration
    del_x = - (__d_lim__ - (d_h + 0.5*d_hv)) / (2.0 * d_ij)
    del_y = - (__d_lim__ - (d_v + 0.5*d_hv)) / (2.0 * d_ij)
    return del_x, del_y


def _step(Sg, tiling, opts):

    # Problem size
    P = len(Sg)
    # Number of Qubits
    Q = tiling.qubits
    m = opts.construction['rows']
    n = opts.construction['columns']
    delta_t = opts.delta_t
    viscosity = opts.viscosity

    center_x, center_y = n/2.0, m/2.0
    dist_accum = 0.0

    # Diffusivity
    D = min( (viscosity*P) / Q, 1.0)

    # Iterate over tiles
    for name, tile in tiling.tiles.items():

        del_x, del_y = _get_gradient(tile, tiling)
        # Iterate over nodes in tile and migrate
        for node in tile.nodes:
            x, y = Sg.nodes[node]['coordinate']
            l_x = (2.0*x/n)-1.0
            l_y = (2.0*y/m)-1.0
            v_x = l_x * del_x
            v_y = l_y * del_y
            x_1 = x + (1.0 - D) * v_x * delta_t
            y_1 = y + (1.0 - D) * v_y * delta_t
            Sg.nodes[node]['coordinate'] = (x_1, y_1)
            dist_accum += (x_1-center_x)**2 + (y_1-center_y)**2

    dispersion = dist_accum/P
    return dispersion

def _get_demand(Sg, tiling, opts):

    m = opts.construction['rows']
    n = opts.construction['columns']

    for name, node in Sg.nodes(data=True):
        x,y = node['coordinate']
        tile = node['tile']
        i = min(floor(x), n-1)
        j = min(floor(y), m-1)
        new_tile = (i,j)
        tiling.tiles[tile].nodes.remove(name)
        tiling.tiles[new_tile].nodes.add(name)
        node['tile'] = new_tile

    for name, tile in tiling.tiles.items():
        if name!=None and tile.supply!=0:
            tile.concentration = len(tile.nodes)/tile.supply


    if opts.verbose==3:
        concentrations = {name : "d=%s"%tile.concentration
                        for name, tile in tiling.tiles.items() if name!=None}

        draw_tiled_graph(Sg,n,m,concentrations)
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



def _migrate(Sg, tiling, opts):
    """
    """
    m = opts.construction['rows']
    n = opts.construction['columns']
    familiy = opts.construction['family']

    migrating = opts.enable_migration
    while migrating:
        _get_demand(Sg, tiling, opts)
        dispersion = _step(Sg, tiling, opts)
        migrating = _condition(tiling, dispersion)

    return tiling

def _place(Sg, tiling, opts):
    """

    """
    if opts.topology:
        _scale(Sg, tiling, opts)
        _migrate(Sg, tiling, opts)
    else:
        opts.topology = nx.spring_layout(Sg, center=(1.0,1.0))
        _scale(Sg, tiling, opts)
        _migrate(Sg, tiling, opts)

    #TODO: Plugin different placement methods
    #elif:
    #_simulated_annealing(Sg, tiling, opts)

    print("Placement:")
    for name, node in Sg.nodes(data=True):
        print(str(name) + str(node['tile']))
        print('qubits:' + str(tiling.tiles[node['tile']].qubits))


def _simulated_annealing(Sg, tiling, opts):
    rng = opts.rng
    m = opts.construction['rows']
    n = opts.construction['columns']
    family = opts.construction['family']

    init_loc = {}
    for node in S:
        init_loc[node] = (rng.randint(0,n),rng.randint(0,m))

    #TODO: Simulated Annealing placement
    opts.enable_migration = False
    return init_loc


""" Negotiated-congestion based router for multiple
    disjoint Steiner Tree Search.
    Args:


    Returns:
        paths: Dictionary keyed by Problem graph edges with assigned nodes
        unassigned: Dictionary keyed by node from Tg and problem node values

"""

def _init_graphs(Sg, Tg, tiling, opts):

    for name, node in Sg.nodes(data=True):
        # Fixed data
        node_tile = node['tile']
        node['degree'] = Sg.degree(name)
        node['candidates'] = tiling.tiles[node_tile].qubits #TODO: Granularity
        # Mapping
        node['assigned'] = set()

    for name, qubit in Tg.nodes(data=True):
        # Fixed cost
        qubit['degree'] = 1.0 - ( Tg.degree(name)/tiling.max_degree )
        # BFS
        qubit['history'] =  1.0
        qubit['sharing'] = 0.0

def _rip_up(Sg, Tg):

    for name,node in Sg.nodes(data=True):
        # No qubits are assigned to the source graph node
        node['assigned'] = set()

    for name, qubit  in Tg.nodes(data=True):
        # BFS
        qubit['sharing'] = 0.0

def _get_cost(node_tile, neighbor_name, Tg):

    next_node = Tg.nodes[neighbor_name]

    sharing_cost = 1.0 + next_node['sharing']

    scope_cost = 0.0 if node_tile==next_node['tile'] else 1.0

    degree_cost = next_node['degree']

    base_cost = 1.0 + degree_cost + scope_cost

    history_cost = next_node['history']

    return base_cost * sharing_cost * history_cost

def _pre_search(target_set, queue, visited, visiting):

    for tgt in target_set:
        if tgt in visited:
            tgt_cost, tgt_parent, tgt_dist = visited[tgt]
            # If newly occupied, node cost has been increased
            heappush(queue, (tgt_cost, tgt))
            visiting[tgt] = tgt_cost, tgt_parent, tgt_dist
        elif tgt in visiting:
            tgt_cost, tgt_parent, tgt_dist = visiting[tgt]
            # Decrease cost in PQ
            heappush(queue, (tgt_cost-1.0, tgt))
            visiting[tgt] = tgt_cost-1.0, tgt_parent, tgt_dist

def _bfs(target_set, visited, visiting, queue, Sg, Tg):
    """ Breadth-First Search
        Args:
            queue: Breadth-First Search Priority Queue

    """

    # If target has been reached in current tree search
    _pre_search(target_set, queue, visited, visiting)

    # Pop node out of Priority queue and its cost, parent, and distance.
    node_cost, node = heappop(queue)
    _, node_parent, node_dist = visiting[node]
    found = False
    while (not found):
        #print("Node: %s"%str(node))
        node_tile = Tg.nodes[node]['tile']
        neighbor_dist = node_dist + 1
        neighbor_parent = node
        for neighbor in Tg[node]:
            if neighbor not in visited:
                if neighbor in target_set:
                    # Queue target without added cost
                    neighbor_cost = node_cost
                else:
                    neighbor_cost = node_cost + _get_cost(node_tile, neighbor, Tg)

                heappush(queue, (neighbor_cost, neighbor))
                neighbor_data = neighbor_cost, neighbor_parent, neighbor_dist

                prev_cost, _, _ = visiting.setdefault(neighbor, neighbor_data)
                if prev_cost > neighbor_cost:
                    visiting[neighbor] = neighbor_data

                #print('Queue:' + str(queue))
        # Once all neighbours have been checked
        visited[node] = node_cost, node_parent, node_dist
        node_cost, node = heappop(queue)
        _, node_parent, node_dist = visiting[node]
        found = (node in target_set) and (node_dist >= 2)

    print('Found target' + str(node) + str(node_dist))
    visited[node] = node_cost, node_parent, node_dist
    return node

def _traceback(source, target, reached, visited, unassigned, mapped, Sg, Tg):

    node_cost, node_parent, node_dist = visited[reached]

    source_node = Sg.nodes[source]
    target_node = Sg.nodes[target]

    if reached not in target_node['assigned']:
        target_node['assigned'].add(reached)
        mapped[target] = set([reached])
        Tg.nodes[reached]['sharing'] += 1.0
        visited[reached] = node_cost + 1.0, node_parent, node_dist


    path = [reached]
    node = node_parent
    while(node not in source_node['assigned']):
        print('Node:' + str(node))
        path.append(node)

        if node in unassigned:
            if source in unassigned[node]:
                del unassigned[node]
                Sg.nodes[source]['assigned'].add(node)
                mapped[source].add(node)
            elif target in unassigned[node]:
                del unassigned[node]
                Sg.nodes[target]['assigned'].add(node)
                mapped[target].add(node)
            else:
                unassigned[node].add(source)
                unassigned[node].add(target)
        else:
            unassigned[node] = set([source, target])

        Tg.nodes[node]['sharing'] += 1.0
        node_cost, node_parent, node_dist = visited[node]
        visited[node] = node_cost + 1.0, node_parent, node_dist
        node = node_parent
    path.append(node)

    print("Path:" + str(path))
    return path


def _init_queue(source, visiting, queue, Sg):
    source_node = Sg.nodes[source]
    for node in source_node['assigned']:
        node_cost = 0.0
        node_parent = source
        node_dist =  1 if node in source_node['candidates'] else 2
        visiting[node] = (node_cost, node_parent, node_dist)
        heappush(queue, (node_cost, node))

    if verbose:
        queue_str = str(["%0.3f %s" % (c,str(n)) for c,n in queue])
        print('Init Queue:' + queue_str)

def _get_targets(target, Sg):
    target_node = Sg.nodes[target]
    target_assigned = target_node['assigned']
    target_candidates = target_node['candidates']
    return target_candidates if not target_assigned else target_assigned

def _steiner_tree(source, targets, mapped, unassigned, Sg, Tg):
    """ Steiner Tree Search
    """
    print('\n New Source:' + str(source))
    # Resulting tree dictionary keyed by edges and path values.
    tree = {}
    # Breadth-First Search structures.
    visiting = {} # (cost, parent, distance) during search
    visited = {} # (parent, distance) after popped from queue
    # Priority Queue (cost, name)
    queue = []

    # Start search using previously-assigned nodes
    _init_queue(source, visiting, queue, Sg)

    # Breadth-First Search
    for target in targets:
        print('Target:' + str(target))
        # Search for target candidates, or nodes assigned to target
        target_set = _get_targets(target, Sg)
        # Incremental BFS graph traversal
        reached = _bfs(target_set, visited, visiting, queue, Sg, Tg)
        # Retrace steps from target to source
        path = _traceback(source, target, reached, visited, unassigned, mapped, Sg, Tg)
        # Update tree
        edge = frozenset((source,target))
        tree[edge] = path

    return tree

def _update_costs(paths, Sg, Tg):
    """ Update present-sharing and history-sharing costs

    """
    legal = True
    print("Paths:")
    for (u,v), path in paths.items():
        print(u,v,path)
        for qubit in path:
            print(Tg.nodes[qubit]['sharing'])
            if Tg.nodes[qubit]['sharing'] > 1:
                legal = False

    return legal

def _get_node(node_list, mapped, opts):
    # Next non-mapped node
    for node in node_list:
        if node not in mapped:
            return node
    # If list is empty
    return None

def _embed_node(node_name, mapped, Sg, Tg, opts):

    node = Sg.nodes[node_name]

    if node['assigned']: return

    # Get best candidate
    candidates = node['candidates']
    q_index = min( candidates, key=lambda q: _get_cost(node['tile'], q, Tg) )

    # Populate qubit
    qubit = Tg.nodes[q_index]
    qubit['sharing'] += 1.0
    # Assign qubit to node
    node['assigned'].add(q_index)
    mapped[node_name] = set([q_index])

def _route(Sg, Tg, opts):
    """ Negotiated Congestion algorithm

    """

    # Search Structures
    paths = {} # { Sg edge : Tg nodes path }
    mapped = {} # { node: len(nodes_assigned) }
    assigned = {} # { Sg node : set(Tg nodes) }
    unassigned = {} # { Tg node : set(Sg nodes) }

    # Operator getting unrouted problem nodes
    unrouted = lambda u: [v for v in Sg[u] if frozenset((u,v)) not in paths]

    # Termination criteria
    legal = False
    tries = opts.tries
    source = _get_node(list(Sg.nodes()), mapped, opts)
    #node_list = list(Sg.nodes())
    #source = node_list.pop()
    done = set()
    while (not legal) and (tries > 0):
        _rip_up(Sg, Tg)
        _embed_node(source, mapped, Sg, Tg, opts)
        while source is not None:
            targets = unrouted(source)
            tree = _steiner_tree(source, targets, mapped, unassigned, Sg, Tg)
            paths.update(tree)
            done.add(source)
            source = _get_node(targets, done, opts)
        legal = _update_costs(paths, Sg, Tg)
        tries -= 1
    return paths, mapped, unassigned


""" Linear Programming formulation to solve unassigned nodes.

    Constraints for all edges:
        var<source><source><target> + var<target><source><target> = |path|
    Constraints for all nodes:
        Z - All( var<node><source><target> ) >= |<node['assigned']>|
    Goal:
        min(Z)
"""

def _setup_lp(paths, unassigned, mapped):
    """ Setup linear Programming Problem
        Goal: Minimize

    """
    lp = pulp.LpProblem("Solve Chains",pulp.LpMinimize)

    Z = pulp.LpVariable('Z',lowBound=0,cat='Integer')
    lp += Z, "OBJ"

    var_map = {}
    Lpvars = {}
    chain = {}

    for node, assigned in mapped.items():
        node_name = str(node).replace(" ","")
        lp += Z >= len(assigned), node_name

    for edge, path in paths.items():
        # Nodes in path excluding source and target
        shared = len(path) - 2
        if shared>0:
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
            lp.constraints[s_name] -= Lpvars[var_s]
            lp.constraints[t_name] -= Lpvars[var_t]

    return lp, var_map

def _assign_nodes(paths, lp_sol, var_map, mapped):

    for edge, path in paths.items():
        # Nodes in path excluding source and target
        shared = len(path) - 2
        if shared>0:
            u,v = edge
            head = path[0]
            if head in mapped[u]:
                source = v
                target = u
            else:
                source = u
                target = v

            var_s = var_map[edge][str(source).replace(" ","")]
            var_t = var_map[edge][str(target).replace(" ","")]

            num_s = lp_sol[var_s]
            num_t = lp_sol[var_t]
            # Path from traceback starts from target
            for i in range(1,shared+1):
                if i > num_t:
                    mapped[source].add(path[i])
                else:
                    mapped[target].add(path[i])


def _paths_to_chains(paths, unassigned, mapped):

    print('Assigned')
    for node, fixed in mapped.items():
        print(str(node) + str(fixed))

    print('Unassigned')
    for node, shared in unassigned.items():
        print(str(node) + str(shared))

    lp, var_map = _setup_lp(paths, unassigned, mapped)

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

"""
class TopologicalOptions(EmbedderOptions):
    def __init__(self, **params):
        EmbedderOptions.__init__(self, **params)
        # Parse optional parameters
        self.names.update({ "topology",
                            "enable_migration",
                            "vicinity",
                            "delta_t",
                            "viscosity",
                            "verbose"})

        for name in params:
            if name not in self.names:
                raise ValueError("%s is not a valid parameter for \
                                    topological find_embedding"%name)

        # If a topology of the graph is not provided, generate one
        try: self.topology =  params['topology']
        except KeyError: self.topology = None

        try: self.enable_migration = params['enable_migration']
        except: self.enable_migration = True

        try: self.vicinity = params['vicinity']
        except: self.vicinity = 0

        try: self.delta_t = params['delta_t']
        except: self.delta_t = 0.2

        try: self.viscosity = params['viscosity']
        except: self.viscosity = 0.0

        try: self.verbose = params['verbose']
        except: self.verbose = 0

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

        topology ({<node>:(<x>,<y>),...}):
            Dict of 2D positions assigned to the source graph nodes.

        vicinity (int): Configuration of the set of tiles that will provide the
            candidate qubits.
            0: Single tile
            1: Immediate neighbors (north, south, east, west)
            2: Extended neighbors (Immediate) + diagonals
            3: Directed (Single) + 3 tiles closest to the node


        random_seed (int):

        tries (int):

        construction (dict of construction parameters of the graph):
            family {'chimera','pegasus'}: Target graph architecture family
            rows (int)
            columns (int)
            labels {'coordinate', 'int'}
            data (bool)
            **family_parameters:


        verbose (int): Verbosity level
            0: Quiet mode
            1: Print statements
            2: NetworkX graph drawings
            3: Migration process

    """

    opts = TopologicalOptions(**params)

    Sg = read_source_graph(S, opts)

    Tg = read_target_graph(T, opts)

    tiling = Tiling(Tg, opts)

    _place(Sg, tiling, opts)

    _init_graphs(Sg, Tg, tiling, opts)

    paths, mapped, unassigned = _route(Sg, Tg, opts)

    embedding = _paths_to_chains(paths, unassigned, mapped)

    return embedding


#TEMP: standalone test
if __name__== "__main__":

    verbose = 3

    p = 2
    S = nx.grid_2d_graph(p,p)
    topology = {v:v for v in S}

    #S = nx.cycle_graph(p)
    #topology = nx.circular_layout(S)

    #S = nx.complete_graph(p)
    #S = nx.relabel_nodes(S, {0:'A',1:'B',2:'C'})
    #topology = nx.spring_layout(S)

    m = 2
    T = dnx.chimera_graph(m, coordinates=True)
    #T = dnx.pegasus_graph(m, coordinates=True)

    S_edgelist = list(S.edges())
    T_edgelist = list(T.edges())

    try:
        #find_embedding(S_edgelist, T_edgelist, topology=topology, construction=T.graph, verbose=verbose)
        embedding = find_embedding(S_edgelist, T_edgelist, topology=topology, construction=T.graph, enable_migration=False, verbose=verbose)
        #find_embedding(S_edgelist, T_edgelist, construction=T.graph, verbose=verbose)
    except:
        traceback.print_exc()

    print('Embedding:' + str(embedding))
    plt.clf()
    dnx.draw_chimera_embedding(T, embedding)
    #dnx.draw_pegasus_embedding(T, embedding)
    plt.show()
