import unittest
import dwave_networkx as dnx

from embera.preprocess.tiling_parser import DWaveNetworkXTiling


class TestTilingParser(unittest.TestCase):

    def test_chimera(self):
        T = dnx.chimera_graph(2)
        print(T.graph['family'])
        T_tiling = DWaveNetworkXTiling(T)

    def test_pegasus(self):
        T = dnx.pegasus_graph(2)
        print(T.graph['family'])
        T_tiling = DWaveNetworkXTiling(T)
