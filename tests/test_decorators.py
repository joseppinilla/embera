import unittest
import networkx as nx
import dwave_networkx as dnx

from embera.utilities.decorators import nx_graph, dnx_graph

class TestDecorators(unittest.TestCase):

    def test_nx_graph(self):
        self.mock_nx_graph_method([(1,2),(2,3),(3,1)])
        G = nx.Graph([(1,2),(2,3),(3,1)])
        self.mock_nx_graph_method(G)

    def test_dnx_graph(self):
        # Graph Architecture
        G = dnx.chimera_graph(4)
        self.mock_dnx_graph_method(G)
        G = dnx.pegasus_graph(4)
        self.mock_dnx_graph_method(G)
        # Labels
        G = dnx.pegasus_graph(4,coordinates=True,nice_coordinates=False)
        self.mock_dnx_graph_nice_coordinates(G)

    @nx_graph(1)
    def mock_nx_graph_method(self,G):
        self.assertIsInstance(G, nx.Graph)

    @dnx_graph(1)
    def mock_dnx_graph_method(self,G):
        self.assertIsInstance(G, nx.Graph)
        self.assertTrue(hasattr(G,'graph'))
        self.assertIn('family',G.graph)
        self.assertIn('rows',G.graph)
        self.assertIn('columns',G.graph)
        self.assertIn('tile',G.graph)
        self.assertIn('labels',G.graph)

    @dnx_graph(1,nice_coordinates=True)
    def mock_dnx_graph_nice_coordinates(self, G):
        self.assertEqual(G.graph['labels'],'nice')

    @nx_graph(1)
    @dnx_graph(2)
    def mock_both_graph_method(self,G,H):
        # Assert NX
        self.assertIsInstance(G, nx.Graph)
        # Assert DNX
        self.assertIsInstance(H, nx.Graph)
        self.assertTrue(hasattr(H,'graph'))
        self.assertIn('family',H.graph)
        self.assertIn('rows',H.graph)
        self.assertIn('columns',H.graph)
        self.assertIn('tile',H.graph)
        self.assertIn('labels',H.graph)
