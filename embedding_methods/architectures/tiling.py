""" Architecture specific tilings for target graphs

"""

__all__ = ['Tiling']

class Tiling:
    """ Generate tiling from architecture graph construction.
    According to the architecture family, create a grid of Tile
    objects.
    """
    def __init__(self, Tg):
        # Support for different target architectures
        self.family = Tg.graph['family']

        if self.family=='chimera':
            self.max_degree = 6
            TileClass = ChimeraTile
        elif self.family=='pegasus':
            self.max_degree = 15
            TileClass = PegasusTile
            Tg.graph['columns'] -= 1 # n is out of range

        self.n = Tg.graph['columns']
        self.m = Tg.graph['rows']
        self.t = Tg.graph['tile']
        self.size = float(len(Tg))
        # Mapping of source nodes to tile
        self.mapping = {}
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
                if chimera_index in Tg.nodes:
                    self.qubits.add(chimera_index)
                    v += 1.0
        return v

class PegasusTile(Tile):
    """ Tile configuration for Pegasus Architecture
    """
    def __init__(self, Tg, i, j):
        Tile.__init__(self, Tg, i, j)
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
        t = self.t
        self.qubits = set()
        v=0.0
        for u in range(2):
            for k in range(self.t):
                pegasus_index = (u, j, k, i)
                if pegasus_index in Tg.nodes:
                    self.qubits.add(pegasus_index)
                    v += 1.0
        return v
