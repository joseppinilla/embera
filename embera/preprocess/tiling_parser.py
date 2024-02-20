import embera

__all__ = ['DWaveNetworkXTiling']

class DWaveNetworkXTiling:
    """ Generate tiling from architecture graph construction. According to
        the architecture family, create a grid of Tile objects.
    """
    def __init__(self, Tg):
        # Graph elements
        self.graph = Tg.graph
        self.qubits = list(Tg.nodes)
        self.couplers = list(Tg.edges)
        # Graph dimensions
        self.m = self.graph["rows"]
        self.n = self.graph["columns"]
        # Graph type
        family = self.graph['family']
        if family=='chimera':
            self.shape = (self.m,self.n)
        elif family=='pegasus':
            self.shape = (3,self.m,self.n)
        else:
            raise ValueError("Invalid family. {'chimera', 'pegasus'}")
        # Graph cooordinates
        dim = len(self.shape)
        labels = self.graph['labels']
        converter = embera.dwave_coordinates.from_graph_dict(self.graph)
        if labels == 'int':
            self.to_nice = converter.linear_to_nice
            self.from_nice = converter.nice_to_linear
        elif labels == 'coordinate':
            self.to_nice = converter.coordinate_to_nice
            self.from_nice = converter.nice_to_coordinate
        elif labels == 'nice':
            self.to_nice = lambda n: n
            self.from_nice = lambda n: n
        # Add Tile objects
        self.tiles = {}
        for q in self.qubits:
            tile = self.get_tile(q)
            if tile in self.tiles:
                self.tiles[tile].qubits.append(q)
            else:
                self.tiles[tile] = Tile(tile, self.shape, [q])
        
        for tile in self.tiles.values():
            tile.neighbors = self.get_tile_neighbors(tile.index)

    def __iter__(self):
        return self.tiles

    def __getitem__(self, key):
        return self.tiles[key]

    def __delitem__(self, key):
        del self.tiles[key]

    def items(self):
        return self.tiles.items()

    def get_tile(self, x):
        t,i,j,u,k = self.to_nice(x)
        return (t,i,j)[-len(self.shape):]

    def set_tile(self, x, tile):
        _,_,_,u,k = self.to_nice(x)
        return self.from_nice((0,)*(3-len(tile)) + tile + (u,k))

    def get_shore(self, x):
        _,_,_,u,_ = self.to_nice(x)
        return u

    def set_shore(self, x, u):
        t,i,j,_,k = self.to_nice(x)
        return self.from_nice((t,i,j,u,k))

    def get_k(self, x):
        _,_,_,_,k = self.to_nice(x)
        return k

    def set_k(self, x, k):
        t,i,j,u,_ = self.to_nice(x)
        return self.from_nice((t,i,j,u,k))

    def get_qubits(self, tile, shore=None, k=None):
        shores = (0,1) if shore is None else (shore,)
        indices = range(self.graph['tile']) if k is None else (k,)
        nice_tile = (0,)+tile if len(tile)==2 else tile
        for u in shores:
            for k in indices:
                n = nice_tile + (u,) + (k,)
                yield self.from_nice(n)

    def get_tile_neighbors(self, tile):
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
        self.nodes  = set()
        self.neighbors = []

    @property
    def supply(self):
        return len(self.qubits)
    
    @property
    def concentration(self):
        if (self.supply):            
            return len(list(self.nodes)) / self.supply
        return None

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
