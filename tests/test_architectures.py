import math
import unittest
import minorminer

import networkx as nx

from tests import graph_topologies

from embedding_methods.architectures import generators
from embedding_methods.composites.embedding import EmbeddingComposite

from dimod.reference.composites.structure import StructureComposite
from dimod.reference.samplers.simulated_annealing import SimulatedAnnealingSampler

GRAPH_SIZE = 16

class TestArchitectures(unittest.TestCase):

    def setUp(self):
        self.method = minorminer

    def all_graphs(self):
        print('target %s' % self.target)
        target_gen = generators.__dict__[self.target]
        target_graph = target_gen()

        for name, source_gen in graph_topologies.__dict__.items():
            if callable(source_gen):
                print('graph %s' % name)
                source_graph = source_gen(GRAPH_SIZE)
                source_edgelist = source_graph.edges()
                target_edgelist = target_graph.edges()
                embedding = self.method.find_embedding(source_edgelist, target_edgelist)
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
