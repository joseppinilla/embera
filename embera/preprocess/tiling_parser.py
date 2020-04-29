import embera
import numpy as np

__all__ = ['DWaveNetworkXTiling']

class DWaveNetworkXTiling:
    """ Generate tiling from architecture graph construction.
        According to the architecture family, create a grid of Tile
        objects.
    """
    def __init__(self, Tg):
        self.qubits = Tg.nodes
        self.couplers = Tg.edges
        # Coordinate converter
        self.converter = embera.dwave_coordinates.from_graph_dict(Tg.graph)
        # Graph attributes
        self.labels = Tg.graph['labels']
        self.family = Tg.graph['family']

        # Graph dimensions
        m = Tg.graph["columns"]
        n = Tg.graph["rows"]
        t = Tg.graph["tile"]
        if self.family=='chimera':
            p = 1
            self.shores = 2
        elif self.family=='pegasus':
            p = 3
            self.shores = 2
        else:
            raise ValueError("Invalid family. {'chimera', 'pegasus'}")
        # Add Tile objects
        self.t = t
        self.shape = (p,m,n)
        self.tiles = np.empty(self.shape, dtype=object)
        for t in range(p):
            for j in range(n):
                for i in range(m):
                    tile = (t,i,j)
                    qubits = self.get_tile_qubits(tile)
                    self.tiles[tile] = Tile(tile, self.shape, qubits)

    def get_tile_qubits(self, tile):
        """ Finds the available qubits associated to tile (t,i,j) of the
            Chimera or Pegasus Graph and returns the supply or qubits found.

            The notation (i, j, u, k) is called the chimera coordinates index:
                i : indexes the row of the Chimera tile from 0 to m inclusive
                j : indexes the column of the Chimera tile from 0 to n inclusive
                u : qubit orientation (0 = left-hand nodes, 1 = right-hand nodes)
                k : indexes the qubit within either the left- or right-hand shore
                    from 0 to t inclusive
            The notation (u, w, k, z) is called the pegasus coordinates index:
                u : qubit orientation (0 = vertical, 1 = horizontal)
                w : orthogonal major offset
                k : orthogonal minor offset
                z : parallel offset
            The notation (t, i, j, u, k) is called the nice coordinates index:
                t : indexes the Chimera subgraph. Chimera t=0. Pegasus 0 <= t < 3
                i : indexes the row of the Chimera tile from 0 to m inclusive
                j : indexes the column of the Chimera tile from 0 to n inclusive
                u : qubit orientation (0 = left-hand nodes, 1 = right-hand nodes)
                k : indexes the qubit within either the left- or right-hand shore
                    from 0 to t inclusive
        """
        t,i,j = tile
        qubits = set()
        for u in range(self.shores):
            for k in range(self.t):
                nice_index = (t, i, j, u, k)
                if self.labels == 'nice':
                    label = nice_index
                elif self.labels == 'int':
                    label = self.converter.nice_to_linear(nice_index)
                elif self.labels == 'coordinate':
                    label = self.converter.nice_to_coordinate(nice_index)
                else:
                    raise Exception("Label not in {'nice','int','coordinate'}")

                if label in self.qubits:
                    qubits.add(label)

        return qubits

class Tile:
    """ Tile Class """
    def __init__(self, index, shape, qubits):
        t,i,j = index
        m,n,p = shape
        self.index = index
        self.qubits = qubits

    @property
    def supply(self):
        return self.qubits

    def links(self, tile, edge_list):
        for q in self.qubits:
            for p in tile.qubits:
                if (q,p) in edge_list:
                    yield (q,p)

    def is_connected(self, tile, edge_list):
        return any(self.links(tile,edge_list))

    def __repr__(self):
        return str(self.qubits)

    def __str__(self):
        return str(self.name)
