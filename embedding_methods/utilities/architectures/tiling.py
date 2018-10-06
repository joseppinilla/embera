""" Architecture specific tilings for target graphs

"""

__all__ = ['Tiling']

import dwave_networkx as dnx
from dwave_networkx.generators.chimera import chimera_coordinates
from dwave_networkx.generators.pegasus import pegasus_coordinates

class Tiling:
    """ Generate tiling from architecture graph construction.
    According to the architecture family, create a grid of Tile
    objects.
    """
    def __init__(self, Tg):
        # Support for different target architectures
        self.family = Tg.graph['family']
        self.size = float(len(Tg))
        # Mapping of source nodes to tile
        self.mapping = {}

        if self.family=='chimera':
            self.max_degree = 6
            TileClass = ChimeraTile
        elif self.family=='pegasus':
            Tg.graph['columns'] = Tg.graph['columns']*3
            self.max_degree = 15
            TileClass = PegasusTile

        self.n = Tg.graph['columns']
        self.m = Tg.graph['rows']
        self.t = Tg.graph['tile']
        # Add Tile objects
        self.tiles = {}
        for i in range(self.n):
            for j in range(self.m):
                tile = (i,j)
                self.tiles[tile] = TileClass(Tg, i, j)

        # Dummy tile to represent boundaries
        self.tiles[None] = DummyTile()

class Tile:
    """ Tile Class
    """
    def __init__(self, Tg, i, j):
        self.n = Tg.graph['columns']
        self.m = Tg.graph['rows']
        self.t = Tg.graph['tile']
        self.labels = Tg.graph['labels']
        self.name = (i,j)
        self.nodes = set()
        index = j*self.n + i
        self.index = index
        self.neighbors = self._get_neighbors(i, j, index)

    def _i2c(self, index, n):
        """ Convert tile array index to coordinate
        """
        j, i = divmod(index,n)
        return i, j

    def _get_neighbors(self, i, j, index):
        """ Calculate indices and names of negihbouring tiles to use recurrently
            during migration and routing.
            The vicinity parameter is later used to prune out the neighbors of
            interest.
            Uses cardinal notation north, south, west, east
        """
        n = self.n
        m = self.m

        north = self._i2c(index - n, n)
        south = self._i2c(index + n, n)
        west =  self._i2c(index - 1, n)
        east =  self._i2c(index + 1, n)

        nw = self._i2c(index - n - 1, n)
        ne = self._i2c(index - n + 1, n)
        se = self._i2c(index + n + 1, n)
        sw = self._i2c(index + n - 1, n)

        neighbors = []
        for tile in [north, south, west, east, nw, ne, se, sw]:
            (i,j) = tile
            if (i >= 0 and i < n) and (j >= 0 and j < m):
                neighbors.append(tile)
            else:
                neighbors.append(None)
        return neighbors

class DummyTile:
    """ Dummy Tile Class to use as boundaries
    """
    def __init__(self):
        # Keyed in tile dictionary as None
        self.name = None
        # Treat as a fully occupied tile
        self.supply = 0.0
        self.concentration = 1.0
        # Dummy empty sets
        self.nodes = set()
        self.qubits = set()

class ChimeraTile(Tile):
    """ Tile configuration for Chimera Architecture
    """
    def __init__(self, Tg, i, j):
        Tile.__init__(self, Tg, i, j)
        self.converter = chimera_coordinates(self.m, self.n, self.t)
        self.supply = self._get_chimera_qubits(Tg, i, j)
        self.concentration = 1.0 if not self.supply else 0.0

    def _get_chimera_qubits(self, Tg, i, j):
        """ Finds the available qubits associated to tile (i,j) of the Chimera
            Graph and returns the supply or number of qubits found.

            The notation (i, j, u, k) is called the chimera index:
                i : indexes the row of the Chimera tile from 0 to m inclusive
                j : indexes the column of the Chimera tile from 0 to n inclusive
                u : qubit orientation (0 = left-hand nodes, 1 = right-hand nodes)
                k : indexes the qubit within either the left- or right-hand shore
                    from 0 to t inclusive
        """
        t = self.t
        self.qubits = set()
        v = 0.0
        for u in range(2):
            for k in range(t):
                chimera_index = (i, j, u, k)
                if self.labels == 'coordinate':
                    chimera_label = chimera_index
                elif self.labels == 'int':
                    chimera_label = self.converter.int(chimera_index)
                else:
                    raise Exception("Invalid labeling. {'coordinate', 'int'}")
                if chimera_label in Tg.nodes:
                    self.qubits.add(chimera_label)
                    v += 1.0
        return v

class PegasusTile(Tile):
    """ Tile configuration for Pegasus Architecture
    """
    def __init__(self, Tg, i, j):
        Tile.__init__(self, Tg, i, j)
        self.converter = pegasus_coordinates(self.m)
        self.supply = self._get_pegasus_qubits(Tg, i, j)
        self.concentration = 1.0 if not self.supply else 0.0

    def _get_pegasus_qubits(self, Tg, i, j):
        """ Finds the avilable qubits associated to tile (i,j) of the Pegasus
            Graph and returns the supply or number of qubits found.

            The notation (u, w, k, z) is called the pegasus index:
                u : qubit orientation (0 = vertical, 1 = horizontal)
                w : orthogonal major offset
                k : orthogonal minor offset
                z : parallel offset
        """
        v=0.0
        qubit_indices = []

        k_start = (i%3)*4
        for k in range(k_start, k_start+4):
            w = i//3
            z = j if i%3==0 else j-1
            pegasus_index = (0, w, k, z)
            qubit_indices.append(pegasus_index)

        k_start = (2-i%3)*4
        for k in range(k_start, k_start+4):
            w = j
            z = (i-1)//3
            pegasus_index = (1, w, k, z)
            qubit_indices.append(pegasus_index)

        self.qubits = set()
        if self.labels == 'coordinate':
            pegasus_labels = qubit_indices
        elif self.labels == 'int':
            pegasus_labels = self.converter.ints(qubit_indices)
        else:
            raise Exception("Invalid labeling. {'coordinate', 'int'}")
        for label in list(pegasus_labels):
            if label in Tg.nodes:
                self.qubits.add(label)
                v += 1.0

        return v
