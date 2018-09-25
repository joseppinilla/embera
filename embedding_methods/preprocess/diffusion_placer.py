import math
import random
import warnings
import networkx as nx
import matplotlib.pyplot as plt

from embedding_methods.utilities.architectures.tiling import Tiling
from embedding_methods.utilities.architectures.drawing import draw_tiled_graph

__all__ = ['find_candidates']

class DiffusionPlacer(Tiling):
    """ Diffusion-based migration of a graph layout
    """
    def __init__(self, S, Tg, **params):
        Tiling.__init__(self, Tg)

        self.tries = params.pop('tries', 1)
        self.verbose = params.pop('verbose', 0)

        # Random Number Generator Configuration
        self.random_seed = params.pop('random_seed', None)
        self.rng = random.Random(self.random_seed)

        # Choice of vicinity. See below.
        self.vicinity = params.pop('vicinity', 0)

        # Source graph layout
        try: self.layout = params.pop('layout')
        except KeyError:
            self.layout = nx.spring_layout(nx.Graph(S))
            warnings.warn('A spring layout was generated using NetworkX.')

        # Diffusion hyperparameters
        self.enable_migration = params.pop('enable_migration', True)
        self.delta_t = params.pop('delta_t', 0.20)
        self.d_lim = params.pop('d_lim', 0.75)
        self.viscosity = params.pop('viscosity', 0.00)

        # Check if all parameters have been parsed.
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
                    #TODO: Not implemented.
                    # Directed  = (Single) + 3 tiles closest to the node coordinates
                    warnings.warn('Not implemented. Using [0] Single vicinity.')
                    candidates[s_node] = self.tiles[s_tile].qubits
                else:
                    raise ValueError("vicinity %s not valid [0-3]." % self.vicinity)

        return candidates

    def _scale(self):
        """ Assign node locations to in-scale values of the dimension
        of the target graph.
        """
        m = self.m
        n = self.n
        layout = self.layout
        P = len(layout)

        # Find dimensions of source graph S
        Sx_min = Sy_min = float("inf")
        Sx_max = Sy_max = 0.0
        # Loop through all source graph nodes to find dimensions
        for sx, sy in layout.values():
            Sx_min = min(sx, Sx_min)
            Sx_max = max(sx, Sx_max)
            Sy_min = min(sy, Sy_min)
            Sy_max = max(sy, Sy_max)
        s_width =  (Sx_max - Sx_min)
        s_height = (Sx_max - Sx_min)

        center_x, center_y = n/2.0, m/2.0
        dist_accum = 0.0
        # Normalize, scale and accumulate initial distances
        for s_node, s_coords in layout.items():
            (sx, sy) = s_coords
            norm_x = (sx-Sx_min) / s_width
            norm_y = (sy-Sy_min) / s_height
            scaled_x = norm_x * (n-1) + 0.5
            scaled_y = norm_y * (m-1) + 0.5
            layout[s_node] = (scaled_x, scaled_y)
            tile = self._coords_to_tile(scaled_x, scaled_y)
            self.mapping[s_node] = tile
            self.tiles[tile].nodes.add(s_node)
            dist_accum += (scaled_x-center_x)**2 + (scaled_y-center_y)**2

        # Initial dispersion
        dispersion = dist_accum/P
        self.dispersion_accum = [dispersion] * 3

    def _coords_to_tile(self, x_coord, y_coord):
        """ Tile values are restricted.
        Horizontallly 0<=i<=n
        Vertically 0<=j<=m

        """
        i = max(min(round(x_coord), self.n-1), 0)
        j = max(min(round(y_coord), self.m-1), 0)
        tile = (i,j)
        return tile


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
        # Target graph dimensions
        m = self.m
        n = self.n
        t_size = self.size
        # Migration hyperparameters
        delta_t = self.delta_t
        viscosity = self.viscosity

        # Problem size
        layout = self.layout
        P = float(len(layout))

        # Diffusivity
        D = min((viscosity*P) / t_size, 1.0)

        # Iterate over tiles
        center_x, center_y = n/2.0, m/2.0
        dist_accum = 0.0
        for tile in self.tiles.values():
            gradient_x, gradient_y = self._get_gradient(tile)
            # Iterate over nodes in tile and migrate
            for node in tile.nodes:
                x, y = layout[node]
                l_x = (2.0*x/n)-1.0
                l_y = (2.0*y/m)-1.0
                v_x = l_x * gradient_x
                v_y = l_y * gradient_y
                x_1 = x + (1.0 - D) * v_x * delta_t
                y_1 = y + (1.0 - D) * v_y * delta_t
                layout[node] = (x_1, y_1)
                dist_accum += (x_1-center_x)**2 + (y_1-center_y)**2

        dispersion = dist_accum/P
        return dispersion

    def _map_tiles(self):
        """ Use source nodes layout to determine tile mapping.
            Then use new populations of tiles to calculate tile
            concentrations.
            Using verbose==4, a call to draw_tiled_graph() plots
            source nodes over a tile grid.
        """
        m = self.m
        n = self.n
        layout = self.layout

        for s_node, s_coords in layout.items():
            tile = self.mapping[s_node]
            new_tile = self._coords_to_tile(*s_coords)
            self.tiles[tile].nodes.remove(s_node)
            self.tiles[new_tile].nodes.add(s_node)
            self.mapping[s_node] = new_tile

        for tile in self.tiles.values():
            if tile.supply:
                tile.concentration = len(tile.nodes)/tile.supply

        if self.verbose==4:
            draw_tiled_graph(self.m, self.n, self.tiles, self.layout)
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


def find_candidates(S, Tg, **params):
    """ find_candidates(S, Tg, **params)
    Given an arbitrary source graph and a target graph belonging to a
    tiled architecture (i.e. Chimera Graph), find a mapping from source
    nodes to target nodes, so that this mapping assists in a subsequent
    minor embedding.

    If a layout is given, the chosen method to find candidates is
    the DiffusionPlacer_ approach. If no layout is given, the
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
            verbose (int): Verbosity level
                0: Quiet mode
                1: Print statements
                4: Tile drawings with concentration

            layout ({<node>:(<x>,<y>),...}):
                Dict of 2D positions assigned to the source graph nodes.

            vicinity (int): Granularity of the candidate assignment.
                0: Single tile
                1: Immediate neighbors = (north, south, east, west)
                2: Extended neighbors = (Immediate) + diagonals
                3: Directed  = (Single) + 3 tiles closest to the node coordinates

    """

    placer = DiffusionPlacer(S, Tg, **params)
    candidates = placer.run()

    return candidates
