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


    if verbose==3:
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


"""

"""

def _routing_graph(Sg, Tg, tiling, opts):

    for name, qubit in Tg.nodes(data=True):
        # BFS
        qubit['history'] =  1.0
        qubit['degree'] = 1.0 - ( Tg.degree(name)/tiling.max_degree )
        qubit['sharing'] = 0.0
        # Mapping
        qubit['nodes'] = set()
        qubit['paths'] = set()

    for name, node in Sg.nodes(data=True):
        # BFS. Dummy values to calculate costs
        node['degree'] = Sg.degree(name)
        node['sharing'] = 0.0
        node['history'] = 1.0
        # Mapping
        node['assigned'] = set()

    Rg =  Tg.to_directed()
    Rg.add_nodes_from(Sg.nodes(data=True))
    for name, tile in tiling.tiles.items():
        if name!=None:
            qubits =  tile.qubits
            for node in tile.nodes:
                for qubit in qubits:
                    Rg.add_edge(qubit, node)
    return Rg


def _rip_up(Sg, Tg, Rg):

    for u,v,edge in Sg.edges.data():
        edge['routed'] =  False

    for name,node in Sg.nodes(data=True):
        # No qubits are assigned to the source graph node
        node['assigned'] = set()


    for name, qubit  in Tg.nodes(data=True):
        qubit['sharing'] = 0.0

        qubit['nodes'].clear()
        qubit['paths'].clear()

def _get_cost(node_name, neighbor_name, Rg):

    node = Rg.nodes[node_name]
    next_node = Rg.nodes[neighbor_name]

    #print('Next:' + str(next_node))

    sharing_cost = 1.0 + next_node['sharing']

    scope_cost = 0.0 if node['tile']==next_node['tile'] else 1.0

    degree_cost = next_node['degree']

    base_cost = 1.0 + degree_cost + scope_cost

    history_cost = next_node['history']

    return base_cost * sharing_cost * history_cost

def _embed_first(Sg, Tg, Rg, tiling, opts):
    # Pick first node
    node_list = list(Sg.nodes(data=True))
    first, first_node = node_list.pop() #TODO: Pops random. Allow pop priority.

    # Get best candidate
    tile = first_node['tile']
    candidates = tiling.tiles[tile].qubits #TODO: Consider granularity of candidates
    q_index = min( candidates, key=lambda q: _get_cost(first, q, Rg) )

    # Populate qubit
    qubit = Rg.nodes[q_index]
    qubit['sharing'] += 1.0
    qubit['nodes'].add(first)

    # Assign qubit to node
    first_node['assigned'].add(q_index)

    return (first,first_node), node_list

def _unrouted_edges(source, Sg):

    unrouted = [neighbor for neighbor in Sg[source]
                if Sg.edges[source,neighbor]['routed']==False]

    return unrouted

def _bfs(target_set, visited, parents, cost, distance, queue, Rg):
    """ Breadth-First Search
        Args:
            source:
            target:
            visited:
            parents:
            Rg: Routing Graph

    """

    # Don't continue BFS expansion if target has been reached in len>1 path
    for tgt in target_set:
        if tgt in visited:
            if distance[tgt] > 2:
                # Returns first encounter of valid target
                return tgt

    #reached = next((distance[tgt] > 1 for tgt in target_set if tgt in visited), False)
    #if reached: return reached

    # Breadth-First Search Priority Queue
    node_cost, node = heappop(queue)
    node_dist = distance[node]
    found = False
    while (not found):
        #print("Node: %s"%str(node))
        neighbor_dist = node_dist + 1
        for neighbor in Rg[node]:
            if neighbor in target_set:
                if neighbor_dist <= 2:
                    print('Target found but not queuing')
                    continue
            if neighbor not in visited:
                if neighbor in target_set:
                    heappush(queue, (node_cost, neighbor))
                    parents[neighbor] = node
                    distance[neighbor] = neighbor_dist
                    cost[neighbor] = node_cost
                else:
                    neighbor_cost = node_cost + _get_cost(node, neighbor, Rg)
                    heappush(queue, (neighbor_cost, neighbor))
                    # Updates cost and parent if not-visited or lower/same cost
                    # if cost.setdefault(neighbor, neighbor_cost) >= neighbor_cost:
                    #     parents[neighbor] = node
                    #     cost[neighbor] = neighbor_cost
                    #     distance[neighbor] = neighbor_dist
                    # TODO: Merge cost, parents, and distance. Maybe use setdefault
                    if neighbor in cost:
                        if cost[neighbor] > neighbor_cost:
                            parents[neighbor] = node
                            cost[neighbor] = neighbor_cost
                            distance[neighbor] = neighbor_dist
                    else:
                        parents[neighbor] = node
                        cost[neighbor] = neighbor_cost
                        distance[neighbor] = neighbor_dist


                #print('Queue:' + str(queue))
        # Once all neighbours have been checked
        visited[node] = node_cost
        node_cost, node = heappop(queue)
        node_dist =  distance[node]
        found = (node in target_set) and (distance[node] > 2)


    print('Found target' + str(node))
    return node

def _traceback(source, target, reached, parents, unassigned, Sg, Rg):

    target_node = Sg.nodes[target]
    source_node = Sg.nodes[source]

    if reached in Sg:
        reached = parents[target]
        target_node['assigned'].add(reached)
        Rg.nodes[reached]['sharing'] += 1.0
        Rg.nodes[reached]['nodes'].add(target)



    path = [reached]
    node = parents[reached]
    while(node not in source_node['assigned']):
        print('Node:' + str(node))
        path.append(node)

        # Node is only reached if path len=1
        if node == source:
            print('This')

        if node in unassigned:
            if source in unassigned[node]:
                del unassigned[node]
                Sg.nodes[source]['assigned'].add(node)
                Rg.nodes[node]['nodes'].add(source)
            elif target in unassigned[node]:
                del unassigned[node]
                Sg.nodes[target]['assigned'].add(node)
                Rg.nodes[node]['nodes'].add(target)
            else:
                unassigned[node].add(source)
                unassigned[node].add(target)
                Rg.nodes[node]['nodes'].add(source)
                Rg.nodes[node]['nodes'].add(target)
        else:
            unassigned[node] = set([source, target])
            Rg.nodes[node]['nodes'].add(source)
            Rg.nodes[node]['nodes'].add(target)

        Rg.nodes[node]['sharing'] += 1.0
        node = parents[node]
    path.append(node)

    print("Path:" + str(path))
    return path


def _steiner_tree(source, targets, unassigned, Sg, Rg):
    """ Steiner Tree Search
    """
    print('Soruce:' + str(source))
    # Breadth-First Search structures. TODO: Merge cost, parents and distance.
    # TODO: Make visited a set
    cost = {}
    parents = {}
    distance = {}
    visited = {}
    # Resulting tree dictionary keyed by edges and path values.
    tree = {}

    # Priority Queue
    queue = []
    # Start search using previously-assigned nodes
    for node in Sg.nodes[source]['assigned']:
        parents[node] = source
        if source in Rg[node]:
            distance[node] = 1
        else:
            distance[node] = 2
        heappush(queue, (0.0, node))

    print('Init Queue:' + str(queue))

    for target in targets:
        print('Target:' + str(target))
        # Search for target node, or nodes pre-assigned to target
        target_node = Sg.nodes[target]
        target_assigned = target_node['assigned']
        target_set = set([target]) if not target_assigned else target_assigned

        # Incremental BFS graph traversal
        reached = _bfs(target_set, visited, parents, cost, distance, queue, Rg)

        # Retrace steps from target to source
        path = _traceback(source, target, reached, parents, unassigned, Sg, Rg)

        edge = (source,target)
        tree.update({edge:path})

        Sg.edges[source,target]['routed'] = True

    return tree

def _update_costs(paths, Sg, Tg, Rg):
    """ Update present-sharing and history-sharing costs

    """
    print("Update Costs:")
    for name, node in Rg.nodes(data=True):
        if 'nodes' in node:
            nodes = node['nodes']
            print(nodes)

    print("Paths:")
    for (u,v), path in paths.items():
        print(u,v,path)
        for qubit in path:
            print(Rg.nodes[qubit]['sharing'])
    return True

def _route(Sg, Tg, Rg, tiling, opts):
    """ Negotiated-congestion based router for multiple
        disjoint Steiner Tree Search.

    """
    tries = opts.tries
    paths = {}
    unassigned = {}

    legal = False
    while (not legal) and (tries > 0):
        _rip_up(Sg, Tg, Rg)
        (source, node), node_list = _embed_first(Sg, Tg, Rg, tiling, opts)
        while node_list:
            print('Unassigned:' + str(node_list))
            targets = _unrouted_edges(source, Sg)
            tree = _steiner_tree(source, targets, unassigned, Sg, Rg)
            paths.update(tree)
            source,node = node_list.pop()
        legal = _update_costs(paths, Sg, Tg, Rg)
        tries -= 1
    return paths, unassigned


""" Linear Programming formulation to solve unassigned nodes.

    Constraints for all edges:
        var<source><source><target> + var<target><source><target> = |path|
    Constraints for all nodes:
        Z - All( var<node><source><target> ) >= |<node['assigned']>|
    Goal:
        min(Z)
"""

def _setup_lp(paths, unassigned, Sg, Rg):
    """ Setup linear Programming Problem
        Goal: Minimize

    """
    lp = pulp.LpProblem("Solve Chains",pulp.LpMinimize)

    Z = pulp.LpVariable('Z',lowBound=0,cat='Integer')
    lp += Z, "OBJ"

    var_map = {}
    Lpvars = {}
    chain = {}

    for node in Sg.nodes():
        node_name = str(node).replace(" ","")
        lp += Z >= len(Sg.nodes[node]['assigned']), node_name

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

def _assign_nodes(paths, lp_sol, var_map, Sg):

    for edge, path in paths.items():
        # Nodes in path excluding source and target
        shared = len(path) - 2
        if shared>0:

            source, target = edge
            (s_name, var_s), (t_name, var_t) = var_map[edge].items()
            num_s = lp_sol[var_s]
            num_t = lp_sol[var_t]
            # Path from traceback starts from target
            for i in range(1,shared+1):
                if i > num_t:
                    Sg.nodes[source]['assigned'].add(path[i])
                else:
                    Sg.nodes[target]['assigned'].add(path[i])


def _paths_to_chains(paths, unassigned, Sg, Rg):

    print('Assigned')
    for name, node in Sg.nodes(data=True):
        print(str(name) + str(node['assigned']))

    print('Unassigned')
    for node, shared in unassigned.items():
        print(str(node) + str(shared))

    lp, var_map = _setup_lp(paths, unassigned, Sg, Rg)

    if verbose==0: lp.writeLP("SHARING.lp") #TEMP change to verbose 3

    lp.solve(solver=pulp.GLPK_CMD(msg=verbose))

    # read solution
    lp_sol = {}
    for v in  lp.variables():
        lp_sol[v.name] = v.varValue

    print(lp_sol)

    embedding = _assign_nodes(paths, lp_sol, var_map, Sg)

    print('Assigned')
    embedding = {}
    for name, node in Sg.nodes(data=True):
        print(str(name) + str(node['assigned']))
        embedding[name] = node['assigned']

    return embedding

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
                            "viscosity"})

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

    Rg = _routing_graph(Sg, Tg, tiling, opts)

    paths, unassigned = _route(Sg, Tg, Rg, tiling, opts)

    embedding = _paths_to_chains(paths, unassigned, Sg, Rg)

    return embedding


#TEMP: standalone test
if __name__== "__main__":

    verbose = 3

    p = 2
    S = nx.grid_2d_graph(p,p)
    topology = {v:v for v in S}

    #S = nx.cycle_graph(p)
    #topology = nx.circular_layout(S)

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
    dnx.draw_chimera_embedding(T, embedding)
    #dnx.draw_pegasus_embedding(T, embedding)
    plt.show()
