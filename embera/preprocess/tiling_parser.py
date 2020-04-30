import embera

__all__ = ['DWaveNetworkXTiling']

class DWaveNetworkXTiling:
    """ Generate tiling from architecture graph construction. According to
        the architecture family, create a grid of Tile objects.
    """
    def __init__(self, Tg):
        # Graph elements
        self.qubits = list(Tg.nodes)
        self.couplers = list(Tg.edges)
        # Graph dimensions
        m = Tg.graph["columns"]
        n = Tg.graph["rows"]
        t = Tg.graph["tile"]
        # Graph type
        family = Tg.graph['family']
        if family=='chimera':
            self.shape = (m,n)
        elif family=='pegasus':
            self.shape = (3,m,n)
        else:
            raise ValueError("Invalid family. {'chimera', 'pegasus'}")
        # Graph cooordinates
        dim = len(self.shape)
        labels = Tg.graph['labels']
        if labels is 'int':
            converter = embera.dwave_coordinates.from_graph_dict(Tg.graph)
            self._get_tile = lambda q: converter.linear_to_nice(q)[3-dim:3]
        elif labels is 'coordinate':
            converter = embera.dwave_coordinates.from_graph_dict(Tg.graph)
            self._get_tile = lambda q: converter.coordinate_to_nice(q)[3-dim:3]
        elif labels is 'nice':
            self._get_tile = lambda q: q[3-dim:3]
        # Add Tile objects
        self.tiles = {}
        for q in self.qubits:
            tile = self.get_tile(q)
            if tile in self.tiles:
                self.tiles[tile].qubits.append(q)
            else:
                self.tiles[tile] = Tile(tile, self.shape, [q])

    def __iter__(self):
        return self.tiles

    def __getitem__(self, key):
        return self.tiles[key]

    def __delitem__(self, key):
        del self.tiles[key]

    def items(self):
        return self.tiles.items()

    def get_tile(self, q):
        return self._get_tile(q)

    def get_tile_neighbors(self,tile):
        neighbors = set()
        for i, d in enumerate(tile):
            neg = tile[0:i] + (d-1,) + tile[i+1:]
            neighbors.add(neg)
            pos = tile[0:i] + (d+1,) + tile[i+1:]
            neighbors.add(pos)
        return [tile for tile in neighbors if tile in self.tiles]

class Tile:
    """ Tile Class """
    def __init__(self, index, shape, qubits):
        self.index = index
        self.qubits = qubits

    @property
    def supply(self):
        return self.qubits

    def links(self, tile, edges):
        for q in self.qubits:
            for p in tile.qubits:
                if (q,p) in edges:
                    yield (q,p)

    def is_connected(self, tile, edge_list):
        return any(self.links(tile,edge_list))

    def __repr__(self):
        return str(self.qubits)

    def __str__(self):
        return str(self.index)
