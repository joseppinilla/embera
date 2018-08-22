'''
Stress Test of embedding complete graphs

Using example from
http://dw-docs.readthedocs.io/en/latest/examples/multi_gate.html

Modified:
Added EmbeddingComposite to bypass need for D-Wave connection


>> python3 -m unittest tests.test_stress

'''
import math
import json
import time
import numpy
import random
import unittest
import minorminer
import networkx as nx
import dwave_networkx as dnx

from collections import Counter

import matplotlib.pyplot as plt
from embedding_methods.dense import dense
from embedding_methods.topological import topological
from embedding_methods.composites.embedding import EmbeddingComposite

from dimod.reference.samplers.exact_solver import ExactSolver
from dimod.reference.samplers.random_sampler import RandomSampler
from dimod.reference.composites.structure import StructureComposite
from dimod.reference.samplers.simulated_annealing import SimulatedAnnealingSampler

verbose = 0
#tries = 1

""" Target Graph Architecture """
""" D-Wave 2X """
DW2X_GEN = dnx.generators.chimera_graph
DW2X_SPECS = [12,12,4]
""" D-Wave 2000Q """
DW2000Q_GEN = dnx.generators.chimera_graph
DW2000Q_SPECS = [16,16,4]
""" D-Wave P6 """
P6_GEN = dnx.generators.pegasus_graph
P6_SPECS = [6]
""" D-Wave P16 """
P16_GEN = dnx.generators.pegasus_graph
P16_SPECS = [16]


#@unittest.skip("Comprehensive Test!")
class CharacterizeEmbedding(unittest.TestCase):

    def setUp(self):

        self.tries = 10
        self.TARGET = 'C4'
        self.TARGET_GEN = dnx.generators.chimera_graph
        self.TARGET_SPECS = [4,4,4]
        self.T = self.TARGET_GEN(*self.TARGET_SPECS)

        #strucsampler = StructureComposite(ExactSolver(), chimera.nodes, chimera.edges)
        #strucsampler = StructureComposite(RandomSampler(), chimera.nodes, chimera.edges)
        self.structsampler = StructureComposite(SimulatedAnnealingSampler(), self.T.nodes, self.T.edges)


    def chain_length_histo(self, embedding):
        histo = Counter()
        for chain in embedding.values():
            key = str(len(chain))
            histo[key] += 1

        return histo

    def size_bisection(self, s_generator, method=minorminer):

        structsampler = self.structsampler
        T_size = len(structsampler.nodelist)

        sampler = EmbeddingComposite(structsampler, method)

        # Expect maximum size to embed is equal to size of target
        #S_max = T_size
        S_max = 100
        S_min = 0

        while True:
            S_size = math.ceil( S_min + (S_max-S_min)/2 )
            S = s_generator(S_size)
            S_edgelist = list(S.edges())

            results = {}
            possible = False
            for i in range(self.tries):
                t_start = time.time()
                embedding = sampler.get_embedding(S_edgelist, verbose=verbose)
                t_end = time.time()
                t_elap = t_end-t_start

                histo = self.chain_length_histo(embedding)
                if bool(embedding): possible = True
                results[i] = [len(S), len(S_edgelist),
                            t_elap, dict(histo)]

            if (S_max==S_size):
                print()
                break
            elif possible:
                graphstr = '-' + S.name
                sizestr = '-' + str(S_size)
                methodstr = '-' + method.__name__
                timestr = time.strftime("%Y%m%d-%H%M%S")

                filename = self.TARGET + sizestr + methodstr + graphstr
                with open(filename, 'w') as fp:
                    json.dump([ S_size, methodstr, graphstr,
                    self.TARGET, results],
                    fp)

                S_min = S_size
            else:
                S_max = S_size

    def complete_generator(self, size):
        G = nx.complete_graph(size)
        G.name = 'complete'
        return G

    def complete_bipartite_generator(self, size):
        m = n = int(size/2)
        G = nx.complete_bipartite_graph(m,n)
        G.name = 'bipartite'
        return G

    def grid_2d_generator(self, size):
        m = n = int(math.sqrt(size))
        G = nx.grid_2d_graph(m,n)
        G.name = 'grid2d'
        return G

    def test_complete(self):
        self.size_bisection(self.complete_generator)

    def test_bipartite(self):
        self.size_bisection(self.complete_bipartite_generator)

    def test_grid(self):
        self.size_bisection(self.grid_2d_generator)



class CharacterizeArchitecture(CharacterizeEmbedding):

    def test_dw2000q(self):
        """ D-Wave 2000Q """
        self.TARGET = 'DW2000Q'
        self.TARGET_GEN = DW2000Q_GEN
        self.TARGET_SPECS = DW2000Q_SPECS
        self.T = self.TARGET_GEN(*self.TARGET_SPECS)
        self.structsampler = StructureComposite(SimulatedAnnealingSampler(), self.T.nodes, self.T.edges)
        self.test_grid()
        self.test_bipartite()
        self.test_complete()

    def test_dw2x(self):
        """ D-Wave 2X """
        self.TARGET = 'DW2X'
        self.TARGET_GEN = DW2X_GEN
        self.TARGET_SPECS = DW2X_SPECS
        self.T = self.TARGET_GEN(*self.TARGET_SPECS)
        self.structsampler = StructureComposite(SimulatedAnnealingSampler(), self.T.nodes, self.T.edges)
        self.test_grid()
        self.test_bipartite()
        self.test_complete()

    def test_p6(self):
        """ D-Wave P6 """
        self.TARGET = 'P6'
        self.TARGET_GEN = P6_GEN
        self.TARGET_SPECS = P6_SPECS
        self.T = self.TARGET_GEN(*self.TARGET_SPECS)
        self.structsampler = StructureComposite(SimulatedAnnealingSampler(), self.T.nodes, self.T.edges)
        self.test_grid()
        self.test_bipartite()
        self.test_complete()

    def test_p16(self):
        """ D-Wave P16 """
        self.TARGET = 'P16'
        self.TARGET_GEN = P16_GEN
        self.TARGET_SPECS = P16_SPECS
        self.T = self.TARGET_GEN(*self.TARGET_SPECS)
        self.structsampler = StructureComposite(SimulatedAnnealingSampler(), self.T.nodes, self.T.edges)
        self.test_grid()
        self.test_bipartite()
        self.test_complete()
