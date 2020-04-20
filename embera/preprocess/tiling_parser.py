""" Architecture specific tilings for target graphs """

__all__ = ['DWaveNetworkXTiling']

import dwave_networkx as dnx
from dwave_networkx.generators.chimera import chimera_coordinates
from dwave_networkx.generators.pegasus import pegasus_coordinates

class DWaveNetworkXTiling:
    """ Generate tiling from architecture graph construction.
        According to the architecture family, create a grid of Tile
        objects.
    """
    def __init__(self, Tg):
        # Support for different target architectures
        self.family = Tg.graph['family']
        self.m = Tg.graph['rows']
        self.t = Tg.graph['tile']
        self.labels = Tg.graph['labels']
        self.t_size = float(len(Tg))
        # Mapping of source nodes to tile
        self.mapping = {}

        if self.family=='chimera':
            self.n = Tg.graph['columns']
            self.max_degree = 6
            TileClass = ChimeraTile
            self.converter = chimera_coordinates(self.m, self.n, self.t )
        elif self.family=='pegasus':
            self.n = Tg.graph['columns']*3
            self.max_degree = 15
            TileClass = PegasusTile
            self.converter = pegasus_coordinates(self.m)
        else:
            raise ValueError("Invalid family. {'chimera', 'pegasus'}")

        # Add Tile objects
        self.tiles = {}
        for j in range(self.n):
            for i in range(self.m):
                tile = (i,j)
                self.tiles[tile] = TileClass(Tg, i, j, self.m, self.n)

        # Dummy tile to represent boundaries
        self.tiles[None] = DummyTile()

    def get_tile(self, qubit_label):
        """ Given a qubit tuple, return the i and j values of the
        corresponding tile.
        """
        if self.family=='chimera':
            if self.labels=='coordinate':
                (i,j,_,_) = qubit_label
            elif self.labels=='int':
                (i,j,_,_) = self.converter.tuple(qubit_label)

        elif self.family=='pegasus':
            if self.labels=='coordinate':
                (u,w,k,z) = qubit_label
            elif self.labels=='int':
                (u,w,k,z) = self.converter.tuple(qubit_label)
            if u ==0:
                q = w
                r = k//4
                j = q*3+r
                i = z+1 if j%3 else z
            else:
                i = w
                r = (2-k//4)
                q = z+1 if r==0 else z
                j = q*3+r
        return i,j

class Tile:
    """ Tile Class
    """
    def __init__(self, Tg, i, j, m, n):
        self.m = m
        self.n = n
        self.t = Tg.graph['tile']
        self.labels = Tg.graph['labels']
        self.name = (i,j)
        self.nodes = set()
        index = i*self.n + j
        self.index = index
        self.neighbors = self._get_neighbors(i, j, index)

    def _i2c(self, index, n):
        """ Convert tile array index to coordinate
        """
        i, j = divmod(index,n)
        return i, j

    def _get_neighbors(self, i, j, index):
        """ Calculate indices and names of negihbouring tiles to use recurrently
            during migration and routing.
            The vicinity parameter is later used to prune out the neighbors of
            interest.
            Uses cardinal notation north, south, west, east
        """
        m = self.m
        n = self.n

        north = self._i2c(index - n, n)   if (i > 0)      else   None
        south = self._i2c(index + n, n)   if (i < m-1)    else   None
        west =  self._i2c(index - 1, n)   if (j > 0)      else   None
        east =  self._i2c(index + 1, n)   if (j < n-1)    else   None

        nw = self._i2c(index - n - 1, n)  if (i > 0    and j > 0)    else None
        ne = self._i2c(index - n + 1, n)  if (i > 0    and j < n-1)  else None
        se = self._i2c(index + n + 1, n)  if (i < m-1  and j < n-1)  else None
        sw = self._i2c(index + n - 1, n)  if (i < m-1  and j > 0)    else None

        neighbors = [north, south, west, east, nw, ne, se, sw]

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
    def __init__(self, Tg, i, j, m, n):
        Tile.__init__(self, Tg, i, j, m, n)
        self.converter = chimera_coordinates(self.m, self.n, self.t )
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
    def __init__(self, Tg, i, j, m, n):
        Tile.__init__(self, Tg, i, j, m, n)
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
        self.qubits = set()

        for u in range(2):
            for k_j in range(4):
                if u==0:
                    k = (j%3)*4 + k_j
                    w = j//3
                    z = i if j%3==0 else i-1
                    pegasus_index = (0, w, k, z)
                else:
                    k = (2-j%3)*4 + k_j
                    w = i
                    z = (j-1)//3
                    pegasus_index = (1, w, k, z)

                if self.labels == 'coordinate':
                    pegasus_label = pegasus_index
                elif self.labels == 'int':
                    pegasus_label = self.converter.int(pegasus_index)
                    if self.converter.tuple(pegasus_label) != pegasus_index:
                        continue
                else:
                    raise Exception("Invalid labeling. {'coordinate', 'int'}")

                if pegasus_label in Tg.nodes:
                    self.qubits.add(pegasus_label)
                    v += 1.0

        return v
