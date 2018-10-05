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
        self.n = Tg.graph['columns']
        self.m = Tg.graph['rows']
        self.t = Tg.graph['tile']
        self.size = float(len(Tg))
        # Mapping of source nodes to tile
        self.mapping = {}

        if self.family=='chimera':
            self.max_degree = 6
            TileClass = ChimeraTile
            # Add Tile objects
            self.tiles = {}
            for i in range(self.n):
                for j in range(self.m):
                    tile = (i,j)
                    self.tiles[tile] = TileClass(Tg, i, j)
        elif self.family=='pegasus':
            self.max_degree = 15
            TileClass = PegasusTile
            # Add Tile objects
            self.tiles = {}
            for i in range(self.n):
                for j in range(self.m*3):
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

        north = self._i2c(index - n, n)   if (j > 0)      else   None
        south = self._i2c(index + n, n)   if (j < m-1)    else   None
        west =  self._i2c(index - 1, n)   if (i > 0)      else   None
        east =  self._i2c(index + 1, n)   if (i < n-1)    else   None

        nw = self._i2c(index - n - 1, n)  if (j > 0    and i > 0)    else None
        ne = self._i2c(index - n + 1, n)  if (j > 0    and i < n-1)  else None
        se = self._i2c(index + n + 1, n)  if (j < m-1  and i < n-1)  else None
        sw = self._i2c(index + n - 1, n)  if (j < m-1  and i > 0)    else None

        return (north, south, west, east, nw, ne, se, sw)

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
        #TODO: Choice of qubits from Pegasus index is naive. Can do better.
        n = self.n
        m = self.m
        l = self.t
        self.qubits = set()
        v=0.0
        h_order = [4, 5, 6, 7, 0, 1, 2, 3, 8, 9, 10, 11]
        for n_i in range(n):  # for each tile offset
            # eliminate vertical qubits:
            for l_i in range(0, l, 2):
                q = l_i // 4
                mod = (q+1) % 3
                for l_v in range(l_i, l_i + 2):
                    for m_i in range(m - 1):  # for each column
                        pegasus_index = (0, n_i, l_v, m_i)
                        offset = 1 if q!=0 else 0
                        tile = (m_i+offset, n_i*3 + q)
                        if (i,j)!=tile: continue
                        if self.labels == 'coordinate':
                            pegasus_label = pegasus_index
                        elif self.labels == 'int':
                            pegasus_label = self.converter.int(pegasus_index)
                        else:
                            raise Exception("Invalid labeling. {'coordinate', 'int'}")
                        if pegasus_label in Tg.nodes:
                            self.qubits.add(pegasus_label)
                            v += 1.0

                # eliminate horizontal qubits:
                if n_i > 0 and not(l_i % 4):
                    # a new set of horizontal qubits have had all their neighbouring vertical qubits eliminated.
                    for m_i in range(m):
                        for l_h in range(h_order[l_i], h_order[l_i] + 4):
                            offset = 0 if q==2 else -1
                            tile = (m_i, (n_i+offset)*3+mod)
                            if (i,j)!=tile: continue
                            pegasus_index = (1, m_i, l_h, n_i - 1)
                            if self.labels == 'coordinate':
                                pegasus_label = pegasus_index
                            elif self.labels == 'int':
                                pegasus_label = self.converter.int(pegasus_index)
                            else:
                                raise Exception("Invalid labeling. {'coordinate', 'int'}")
                            if pegasus_label in Tg.nodes:
                                self.qubits.add(pegasus_label)
                                v += 1.0





        return v
