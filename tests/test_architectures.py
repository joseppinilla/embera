import embera
import unittest
import minorminer

from embera.utilities.architectures import generators

GRAPHS = [embera.benchmark.complete_graph(16),
          embera.benchmark.complete_bipartite_graph(8,8),
          embera.benchmark.grid_2d_graph(4,4),
          embera.benchmark.rooks_graph(4,4),
          embera.benchmark.hypercube_graph(n=16)
          ]

class TestArchitectures(unittest.TestCase):

    def setUp(self):
        self.method = minorminer

    def all_graphs(self):
        print('target %s' % self.target)
        target_gen = generators.__dict__[self.target]
        target_graph = target_gen()

        for source_graph in GRAPHS:
            print('graph %s' % source_graph.name)
            embedding = self.method.find_embedding(source_graph, target_graph)
            self.assertEqual(set(source_graph), set(embedding),'Invalid embedding.')
            print('sum: %s' % sum(len(v) for v in embedding.values()))
            print('max: %s' % max(len(v)for v in embedding.values()))

    def test_rainier(self):
        """ Rainier """
        self.target = 'rainier_graph'
        self.all_graphs()

    def test_vesuvius(self):
        """ Vesuvius """
        self.target = 'vesuvius_graph'
        self.all_graphs()

    def test_dw2000q(self):
        """ D-Wave 2000Q """
        self.target = 'dw2000q_graph'
        self.all_graphs()

    def test_dw2x(self):
        """ D-Wave 2X """
        self.target = 'dw2x_graph'
        self.all_graphs()

    def test_p6(self):
        """ D-Wave P6 """
        self.target = 'p6_graph'
        self.all_graphs()

    def test_p16(self):
        """ D-Wave P16 """
        self.target = 'p16_graph'
        self.all_graphs()
