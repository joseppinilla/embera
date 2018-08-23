'''
Stress Test of embedding complete graphs

Using example from
http://dw-docs.readthedocs.io/en/latest/examples/multi_gate.html

Modified:
Added EmbeddingComposite to bypass need for D-Wave connection


>> python3 -m unittest tests.test_stress

'''
import os
import sys
import math
import json
import time
import numpy
import pickle
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

""" Target Graph Architecture """
""" C4 """
C4_GEN = dnx.generators.chimera_graph
C4_DRAW = dnx.draw_chimera_embedding
C4_SPECS = [4,4,4]
""" D-Wave 2X """
DW2X_GEN = dnx.generators.chimera_graph
DW2X_DRAW = dnx.draw_chimera_embedding
DW2X_SPECS = [12,12,4]
""" D-Wave 2000Q """
DW2000Q_GEN = dnx.generators.chimera_graph
DW2000Q_DRAW = dnx.draw_chimera_embedding
DW2000Q_SPECS = [16,16,4]
""" D-Wave P6 """
P6_GEN = dnx.generators.pegasus_graph
P6_DRAW = dnx.draw_pegasus_embedding
P6_SPECS = [6]
""" D-Wave P16 """
P16_GEN = dnx.generators.pegasus_graph
P16_DRAW = dnx.draw_pegasus_embedding
P16_SPECS = [16]

ARCHS = {'C4':      ( C4_GEN, C4_DRAW, C4_SPECS ),
        'DW2X':     ( DW2X_GEN, DW2X_DRAW, DW2X_SPECS ),
        'DW2000Q':  ( DW2000Q_GEN, DW2000Q_DRAW, DW2000Q_SPECS ),
        'P6':       ( P6_GEN, P6_DRAW, P6_SPECS ),
        'P16':      ( P16_GEN, P16_DRAW, P16_SPECS )}

filedir = "../results"

""" Process Data """
def chain_length_histo(embedding):
    histo = Counter()
    for chain in embedding.values():
        key = len(chain)
        histo[key] += 1
    return histo

def read_log_json(filename):
    fp = open(filename, 'r')
    data = json.load(fp)
    fp.close()
    return data

def read_log_pickle(filename):
    fp = open(filename, 'rb')
    data = pickle.load(fp)
    fp.close()
    return data

def read_logs():
    for file in os.listdir(filedir):
        filename = os.path.join(filedir, file)
        base, ext = os.path.splitext(file)
        if ext=='.pkl':
            results = read_log_pickle(filename)
        elif ext=='.json':
            results = read_log_json(filename)
        arch, graph, size, method = base.split('-')
        gen, draw, specs = ARCHS[arch]
        T = gen(*specs)
        for i, result in results.items():
            time, embedding = result
            if embedding:
                histo = chain_length_histo(embedding)

                plt.bar(list(histo.keys()), histo.values())
                plt.xticks(list(histo.keys()))
                plt.title(base)
                plt.show()
                #draw(T, embedding)







class CharacterizeEmbedding(unittest.TestCase):

    def setUp(self):

        self.tries = 3
        self.TARGET = 'C4'
        self.TARGET_GEN = dnx.generators.chimera_graph
        self.TARGET_SPECS = [4,4,4]
        self.T = self.TARGET_GEN(*self.TARGET_SPECS)

        #strucsampler = StructureComposite(ExactSolver(), chimera.nodes, chimera.edges)
        #strucsampler = StructureComposite(RandomSampler(), chimera.nodes, chimera.edges)
        self.structsampler = StructureComposite(SimulatedAnnealingSampler(), self.T.nodes, self.T.edges)

        self.method = minorminer
        self.sampler = EmbeddingComposite(self.structsampler, self.method)


    def log(self, S, obj):
        graphstr = '-' + S.name
        sizestr = '-' + str(len(S))
        methodstr = '-' + self.method.__name__
        timestr = time.strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        filename = self.TARGET + graphstr + sizestr + methodstr + ".pkl"
        self.log_pickle(obj, filedir + filename)
        #filename = self.TARGET + graphstr + sizestr + methodstr + ".json"
        #self.log_json(obj, filedir + filename)

    def log_json(self, obj, filename):
        """ JSON doesn't allow dumping non-string keyed dictionaries """
        with open(filename, 'w') as fp:
            pickle.dump(obj, fp)

    def log_pickle(self, obj, filename):
        """ Pickle allows dumping non-string keyed dictionaries """
        with open(filename, 'wb') as fp:
            pickle.dump(obj, fp)

    def embed(self, S):
        S_edgelist = list(S.edges())

        results = {}
        valid = False
        for i in range(self.tries):
            t_start = time.time()
            embedding = sampler.get_embedding(S_edgelist,
                                get_new = True,
                                timeout = 20,
                                verbose=verbose)
            t_end = time.time()
            t_elap = t_end-t_start

            if bool(embedding): valid = True
            results[i] = [ t_elap, embedding ]

        return valid, results

    def size_bisection(self, s_generator):

        sampler = self.sampler
        T_size = len(structsampler.nodelist)

        # Expect maximum size to embed is equal to size of target
        S_max = T_size
        S_size = math.ceil(T_size/2)
        S_min = 0

        # Bisection method to find largest valid problem embedding
        while (S_max!=S_size):
            S = s_generator(S_size)
            valid, results = self.embed(S)
            if valid:
                self.log(results)
                S_min = S_size
            else:
                S_max = S_size
            # Next data point
            S_size = math.ceil( S_min + (S_max-S_min)/2 )


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
