'''

>> python3 -m unittest tests.test_greedy

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
import matplotlib.pyplot as plt

from collections import Counter

from embedding_methods.architectures import *

from embedding_methods.dense import dense
from embedding_methods.topological import topological
from embedding_methods.composites.embedding import EmbeddingComposite

from dimod.reference.samplers.exact_solver import ExactSolver
from dimod.reference.samplers.random_sampler import RandomSampler
from dimod.reference.composites.structure import StructureComposite
from dimod.reference.samplers.simulated_annealing import SimulatedAnnealingSampler

verbose = 0
tries = 3

filedir = "./results/"

class CharacterizeEmbedding(unittest.TestCase):

    def setUp(self):

        self.tries = tries
        self.target = 'C4'
        self.method = minorminer
        gen, _, specs = ARCHS['C4']
        T = gen(*specs)
        self.structsampler = StructureComposite(SimulatedAnnealingSampler(), T.nodes, T.edges)
        self.sampler = EmbeddingComposite(self.structsampler, self.method)


    def log(self, S, obj):
        graphstr = '-' + S.name
        sizestr = '-' + str(len(S))
        methodstr = '-' + self.method.__name__
        timestr = time.strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        filename = self.target + graphstr + sizestr + methodstr + ".pkl"
        self.log_pickle(obj, filedir + filename)
        #filename = self.target + graphstr + sizestr + methodstr + ".json"
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
        sampler = self.sampler
        S_edgelist = list(S.edges())

        results = {}
        valid = False
        for i in range(self.tries):
            t_start = time.time()
            embedding = sampler.get_embedding(S_edgelist,
                                get_new = True,
                                verbose=verbose)
            t_end = time.time()
            t_elap = t_end-t_start

            if bool(embedding): valid = True
            results[i] = [ t_elap, embedding ]

        return valid, results

    def size_bisection(self, s_generator):

        sampler = self.sampler
        T_size = len(self.structsampler.nodelist)

        # Expect maximum size to embed is equal to size of target
        S_max = T_size
        S_size = math.ceil(T_size/2)
        S_min = 0

        # Bisection method to find largest valid problem embedding
        while (S_max!=S_size):
            S = s_generator(S_size)
            valid, results = self.embed(S)
            if valid:
                self.log(S, results)
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

    def all_tests(self):
        gen, _, specs = ARCHS[self.target]
        T = gen(*specs)
        self.structsampler = StructureComposite(SimulatedAnnealingSampler(), T.nodes, T.edges)
        self.test_grid()
        self.test_bipartite()
        self.test_complete()

    def test_dw2000q(self):
        """ D-Wave 2000Q """
        self.target = 'DW2000Q'
        all_tests()

    def test_dw2x(self):
        """ D-Wave 2X """
        self.target = 'DW2X'
        all_tests()

    def test_p6(self):
        """ D-Wave P6 """
        self.target = 'P6'
        all_tests()

    def test_p16(self):
        """ D-Wave P16 """
        self.target = 'P16'
        all_tests()
