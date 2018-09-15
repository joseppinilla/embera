import pulp
import random
import warnings

import networkx as nx
import matplotlib.pyplot as plt

from math import floor, sqrt
from heapq import heappop, heappush

from embedding_methods.architectures.tiling import Tiling
from embedding_methods.architectures.drawing import draw_tiled_graph

__all__ = ["find_embedding", "find_candidates"]

class Placer(Tiling):
    """ Placement general attributes and methods
    """
    def __init__(self, Tg, **params):
        Tiling.__init__(self, Tg)

        self.tries = params.pop('tries', 1)
        self.verbose = params.pop('verbose', 0)

        # Random Number Generator Configuration
        self.random_seed = params.pop('random_seed', None)
        self.rng = random.Random(self.random_seed)

        # Choice of vicinity. See below.
        self.vicinity = params.pop('vicinity', 0)

        for name in params:
            raise ValueError("%s is not a valid parameter." % name)

    def _assign_candidates(self):
        """ Use tiling to create the sets of target
            nodes assigned to each source node.
                0: Single tile
                1: Immediate neighbors = (north, south, east, west)
                2: Extended neighbors = (Immediate) + diagonals
                3: Directed  = (Single) + 3 tiles closest to the node
        """

        candidates = {}

        for s_node, s_tile in self.mapping.items():
            if self.vicinity == 0:
                # Single tile
                candidates[s_node] = self.tiles[s_tile].qubits
            else:
                # Neighbouring tiles (N, S, W, E, NW, NE, SE, SW)
                neighbors = self.tiles[s_tile].neighbors
                if self.vicinity == 1:
                    # Immediate neighbors
                    candidates[s_node] = self.tiles[s_tile].qubits
                    for tile in neighbors[0:3]:
                        candidates[s_node].update(self.tiles[tile].qubits)
                elif self.vicinity == 2:
                    # Extended neighbors
                    candidates[s_node] = self.tiles[s_tile].qubits
                    for tile in neighbors:
                        candidates[s_node].update(self.tiles[tile].qubits)
                elif self.vicinity == 3:
                    #TODO:# Directed  = (Single) + 3 tiles closest to the node
                    warnings.warn('Not implemented. Using [0] Single vicinity.')
                    candidates[s_node] = self.tiles[s_tile].qubits
                else:
                    raise ValueError("vicinity %s not valid [0-3]." % self.vicinity)

        return candidates

class DiffusionPlacer(Placer):
    """ Diffusion-based migration of a graph topology
    """
    def __init__(self, S, Tg, **params):

        # Diffusion hyperparameters
        self.enable_migration = params.pop('enable_migration', True)
        self.delta_t = params.pop('delta_t', 0.20)
        self.d_lim = params.pop('d_lim', 0.75)
        self.viscosity = params.pop('viscosity', 0.00)

        # Source graph topology
        try: self.topology = params.pop('topology')
        except KeyError:
            self.topology = nx.spring_layout(nx.Graph(S))
            warnings.warn('A spring layout was generated using NetworkX.')

        Placer.__init__(self, Tg, **params)

    def _scale(self):
        """ Assign node locations to in-scale values of the dimension
        of the target graph.
        """
        m = self.m
        n = self.n
        topology = self.topology
        P = len(topology)

        # Find dimensions of source graph S
        Sx_min = Sy_min = float("inf")
        Sx_max = Sy_max = 0.0
        # Loop through all source graph nodes to find dimensions
        for s_node, (sx, sy) in topology.items():
            Sx_min = min(sx, Sx_min)
            Sx_max = max(sx, Sx_max)
            Sy_min = min(sy, Sy_min)
            Sy_max = max(sy, Sy_max)
        s_width =  (Sx_max - Sx_min)
        s_height = (Sx_max - Sx_min)

        center_x, center_y = n/2.0, m/2.0
        dist_accum = 0.0
        # Normalize, scale and accumulate initial distances
        for s_node, (sx, sy) in topology.items():
            norm_x = (sx-Sx_min) / s_width
            norm_y = (sy-Sy_min) / s_height
            scaled_x = norm_x * (n-1) + 0.5
            scaled_y = norm_y * (m-1) + 0.5
            topology[s_node] = (scaled_x, scaled_y)
            tile = min(floor(scaled_x), n-1), min(floor(scaled_y), m-1)
            self.mapping[s_node] = tile
            self.tiles[tile].nodes.add(s_node)
            dist_accum += (scaled_x-center_x)**2 + (scaled_y-center_y)**2

        # Initial dispersion
        dispersion = dist_accum/P
        self.dispersion_accum = [dispersion] * 3

    def _get_attractors(self, i, j):
        """ Get three neighboring tiles that are in the direction
            of the center of the tile array.
        """
        n, s, w, e, nw, ne, se, sw = self.tiles[(i,j)].neighbors
        lh = (i >= 0.5*self.n)
        lv = (j >= 0.5*self.m)

        if lh:
            return (w, n, nw) if lv else (w, s, sw)
        # else
        return (e, n, ne) if lv else (e, s, se)

    def _get_gradient(self, tile):
        """ Get the x and y gradient from the concentration of Nodes
            in neighboring tiles. The gradient is calculated against
            tiles with concentration at limit value d_lim, in order to
            force displacement of the nodes to the center of the tile array.
        """
        d_lim = self.d_lim
        d_ij = tile.concentration

        if d_ij == 0.0 or tile.name == None:
            return 0.0, 0.0
        h, v, hv = self._get_attractors(*tile.name)
        d_h = self.tiles[h].concentration
        d_v = self.tiles[v].concentration
        d_hv = self.tiles[hv].concentration
        gradient_x = - (d_lim - (d_h + 0.5*d_hv)) / (2.0 * d_ij)
        gradient_y = - (d_lim - (d_v + 0.5*d_hv)) / (2.0 * d_ij)

        return gradient_x, gradient_y


    def _step(self):
        """ Discrete Diffusion Step
        """

        # Problem size
        # Number of Qubits
        Q = self.qubits
        m = self.m
        n = self.n
        delta_t = self.delta_t
        topology = self.topology
        viscosity = self.viscosity

        center_x, center_y = n/2.0, m/2.0
        dist_accum = 0.0

        # Problem size
        P = float(len(topology))
        # Diffusivity
        D = min((viscosity*P) / Q, 1.0)

        # Iterate over tiles
        for tile in self.tiles.values():
            gradient_x, gradient_y = self._get_gradient(tile)
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

    def _map_tiles(self):
        """ Use source nodes topology to determine tile mapping.
            Then use new populations of tiles to calculate tile
            concentrations.
            Using verbose==4, a call to draw_tiled_graph() plots
            source nodes over a tile grid.
        """
        m = self.m
        n = self.n
        topology = self.topology

        for s_node, (x, y) in topology.items():
            tile = self.mapping[s_node]
            i = min(floor(x), n-1)
            j = min(floor(y), m-1)
            new_tile = (i,j)
            self.tiles[tile].nodes.remove(s_node)
            self.tiles[new_tile].nodes.add(s_node)
            self.mapping[s_node] = new_tile

        for tile in self.tiles.values():
            if tile.supply:
                tile.concentration = len(tile.nodes)/tile.supply

        if self.verbose==4:
            draw_tiled_graph(self.m, self.n, self.tiles, self.topology)
            plt.show()

    def _condition(self, dispersion):
        """ The algorithm iterates until the dispersion, or average distance of
            the nodes from the centre of the tile array, increases or has a
            cumulative variance lower than 1%
        """
        self.dispersion_accum.pop(0)
        self.dispersion_accum.append(dispersion)
        mean = sum(self.dispersion_accum) / 3.0
        prev_val = 0.0
        diff_accum = 0.0
        increasing = True
        for value in self.dispersion_accum:
            diff_accum = diff_accum + (value-mean)**2
            increasing = value > prev_val
            prev_val = value
        variance = (diff_accum/3.0)
        spread = variance > 0.01
        return spread and not increasing

    def run(self):
        """ Run two-stage global placement.
        """
        self._scale()
        migrating = self.enable_migration
        while migrating:
            self._map_tiles()
            dispersion = self._step()
            migrating = self._condition(dispersion)
        candidates = self._assign_candidates()
        return candidates


class SimulatedAnnealingPlacer(Tiling):
    """ A simulated annealing based global placement
    """
    def __init__(self, S, T, **params):
        Placer.__init__(self)
        Tiling.__init__(self, Tg)

        rng = self.rng
        m = self.m
        n = self.n

        init_loc = {}
        for s_node in S:
            self.mapping[node] = ( rng.randint(0, n), rng.randint(0, m) )

    def run():
        #TODO: Simulated Annealing placement
        candidates = self._assign_candidates()
        return candidates

def find_candidates(S, Tg, **params):
    """ find_candidates(S, Tg, **params)
    Given an arbitrary source graph and a target graph belonging to a
    tiled architecture (i.e. Chimera Graph), find a mapping from source
    nodes to target nodes, so that this mapping assists in a subsequent
    minor embedding.

    If a topology is given, the chosen method to find candidates is
    the DiffusionPlacer_ approach. If no topology is given, the
    SimulatedAnnealingPlacer_ is used.

        Args:
            S: an iterable of label pairs representing the edges in the
                source graph

            Tg: a NetworkX Graph with construction parameters such as those
                generated using dwave_networkx_:
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
                3: Directed  = (Single) + 3 tiles closest to the node coordinates

    """

    diffuse = params.pop('diffuse', True)

    if diffuse:
        placer = DiffusionPlacer(S, Tg, **params)
    else:
        placer = SimulatedAnnealingPlacer(S, Tg, **params)

    candidates = placer.run()

    return candidates

""" Negotiated-congestion based router for multiple disjoint Steiner Tree Search.

This router uses a negotiated-congestion scheme, which is widely-used for FPGA
routing, in which overlap of resources is initially allowed but the costs of
using each qubit is recalculated until a legal solution is found. A solution is
legal when the occupancy of the qubits do not have conflicts. The cost of using
one qubit is defined to depend on a base cost, a present-sharing cost, and a
historical-sharing cost.

"""

# Routing cost scalers
ALPHA_P = 0.0
ALPHA_H = 0.0

def _init_graphs(Sg, Tg, opts):
    """ Assign values to source and target graphs required for
        the tree search.
    """
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

    global ALPHA_P

    next_node = Tg.nodes[neighbor_name]

    sharing_cost = 1.0 + next_node['sharing'] * ALPHA_P

    scope_cost = 0.0 #TODO: higher if different tile

    degree_cost = 0.0 #TODO: Use next_node['degree'] with ARCH max_degree

    base_cost = 1.0 + degree_cost + scope_cost

    history_cost = next_node['history']

    return base_cost * sharing_cost * history_cost

def _bfs(target_set, visited, visiting, queue, Tg):
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
        found = (node in target_set) and (node_dist >= 2)

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

def _steiner_tree(source, targets, mapped, unassigned, Sg, Tg):
    """ Steiner Tree Search
    """
    # Resulting tree dictionary keyed by edges and path values.
    tree = {}

    # Breadth-First Search
    for target in targets:
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

    #if source in mapped: return

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
            targets = [target for target in Sg[source] if target in pending_set]
            tree = _steiner_tree(source, targets, mapped, unassigned, Sg, Tg)
            paths.update(tree)
            source = _get_node(pending_set, pre_sel=targets)
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
    """ Once a solution to the linear program is found, the new mapping
    of nodes is transformed from the resulting number of target nodes to
    add to a source node, into the corresponding target nodes in the target
    graph.
    """
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


def _paths_to_chains(legal, paths, mapped, unassigned, opts):
    """ Using a Linear Programming formulation, map the unassigned
    target nodes, so that the maximum length chain in the embedding
    is minimized.

    Linear Programming formulation to solve unassigned nodes.

        Constraints for all edges:
            var<source><source><target> + var<target><source><target> = |path|
        Constraints for all nodes:
            Z - All( var<node><source><target> ) >= |<mapped['node']>|
        Goal:
            min(Z)
    """

    if not legal:
        warnings.warn('Embedding is illegal.')

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

            tries (int):

            verbose (int): Verbosity level
                0: Quiet mode
                1: Print statements
                2: Log LP problem
    """
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
    """ find_embedding(S, T, **params)
    Heuristically attempt to find a minor-embedding of a graph, representing an
    Ising/QUBO, into a target graph.

    Args:

        S: an iterable of label pairs representing the edges in the source graph

        T: an iterable of label pairs representing the edges in the target graph
            The node labels for the different target archictures should be either
            node indices or coordinates as given from dwave_networkx_.

        **params (optional): see RouterOptions_

    Returns:

        embedding: a dict that maps labels in S to lists of labels in T

    """

    opts = RouterOptions(**params)

    Sg = nx.Graph(S)

    Tg = nx.Graph(T)

    _init_graphs(Sg, Tg, opts)

    legal, paths, mapped, unassigned = _route(Sg, Tg, opts)

    embedding = _paths_to_chains(legal, paths, mapped, unassigned, opts)

    return embedding
