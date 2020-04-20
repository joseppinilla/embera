import warnings

from embera.preprocess.tiling_parser import DWaveNetworkXTiling

__all__ = ['find_candidates', 'SimulatedAnnealingPlacer']

class SimulatedAnnealingPlacer(DWaveNetworkXTiling):
    """ A simulated annealing based global placement
    """
    def __init__(self, S, T, **params):
        DWaveNetworkXTiling.__init__(self, Tg)

        self.tries = params.pop('tries', 1)
        self.verbose = params.pop('verbose', 0)

        # Choice of vicinity. See below.
        self.vicinity = params.pop('vicinity', 0)

        # Check if all parameters have been parsed.
        for name in params:
            raise ValueError("%s is not a valid parameter." % name)

    def _assign_candidates(self):
        """ Use tiling to create the sets of target nodes assigned
        to each source node.
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
                else:
                    raise ValueError("vicinity %s not valid [0-3]." % self.vicinity)

        return candidates

    def run():
        #TODO: Simulated Annealing placement
        warnings.warn('Work in progress.')
        init_loc = {}
        # for s_node in S:
            # i = randint(0, self.n)
            # j = randint(0, self.m)
            # self.mapping[node] = (i, j)

        candidates = self._assign_candidates()
        return candidates

def find_candidates(S, Tg, **params):
    """ find_candidates(S, Tg, **params)
    Given an arbitrary source graph and a target graph belonging to a
    tiled architecture (i.e. Chimera Graph), find a mapping from source
    nodes to target nodes, so that this mapping assists in a subsequent
    minor embedding.

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
    """

    placer = SimulatedAnnealingPlacer(S, Tg, **params)
    candidates = placer.run()

    return candidates
