import unittest
import dwave_networkx as dnx

from embera.architectures.tiling import DWaveNetworkXTiling


class TestTilingParser(unittest.TestCase):

    def test_chimera(self):
        T_linear = dnx.chimera_graph(2)
        linear_tiling = DWaveNetworkXTiling(T_linear)
        T_coord = dnx.chimera_graph(2,coordinates=True)
        coord_tiling = DWaveNetworkXTiling(T_coord)
        self.assertEqual(linear_tiling.get_tile(1),
                         coord_tiling.get_tile((0,0,0,1)))

    def test_pegasus(self):
        T_linear = dnx.pegasus_graph(2)
        linear_tiling = DWaveNetworkXTiling(T_linear)
        T_coord = dnx.pegasus_graph(2,coordinates=True)
        coord_tiling = DWaveNetworkXTiling(T_coord)
        T_nice = dnx.pegasus_graph(2,nice_coordinates=True)
        nice_tiling = DWaveNetworkXTiling(T_nice)
        self.assertEqual(linear_tiling.get_tile(10),
                         coord_tiling.get_tile((0,0,10,0)),
                         nice_tiling.get_tile((1,0,0,0,2)))
