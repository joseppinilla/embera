'''

>> python3 -m unittest tests.<test_name>.<test_method>

'''
import os
import math
import json
import time
import pickle
import unittest
import minorminer
import networkx as nx
import dwave_networkx as dnx

from embedding_methods.architectures import generators
from embedding_methods.composites.embedding import EmbeddingComposite

from dimod.reference.samplers.exact_solver import ExactSolver
from dimod.reference.samplers.random_sampler import RandomSampler
from dimod.reference.composites.structure import StructureComposite
from dimod.reference.samplers.simulated_annealing import SimulatedAnnealingSampler

verbose = 1
tries = 4

filedir = "./results/"

@unittest.skip("Exhaustive test. Not run if not characterizing a new method")
class CharacterizeEmbedding(unittest.TestCase):

    def log(self, S, obj):
        graphstr = '-' + S.name
        sizestr = '-' + str(len(S))
        methodstr = '-' + self.method.__name__
        timestr = time.strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        filename = self.target + graphstr + sizestr + methodstr + ".pkl"
        if verbose: print(filename)
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
            print('-------------------------Iteration %s' % i)
            t_start = time.time()
            embedding = sampler.get_embedding(S_edgelist,
                                force_embed = True,
                                timeout = 30,
                                verbose=verbose)
            t_end = time.time()
            t_elap = t_end-t_start

            valid = bool(embedding)
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

    def hypercube_generator(self, size):
        n = int(math.log(size,2))
        G = nx.hypercube_graph(n)
        G.name = 'hypercube'
        return G

    def rooks_generator(self, size):
        m = n = int(math.sqrt(size))
        G = nx.complete_graph(n)
        H = nx.complete_graph(m)
        F = nx.cartesian_product(G,H)
        print(len(F))
        F.name = 'rooks'
        return F

    def grid_3d_generator(self, size):
        m = n = t = int(size**(1./3.))
        G = nx.grid_graph(dim=[m,n,t])
        G.name = 'grid3d'
        return G

    def complete(self):
        self.size_bisection(self.complete_generator)

    def bipartite(self):
        self.size_bisection(self.complete_bipartite_generator)

    def grid(self):
        self.size_bisection(self.grid_2d_generator)

    def hypercube(self):
        self.size_bisection(self.hypercube_generator)

    def rooks(self):
        self.size_bisection(self.rooks_generator)

@unittest.skip("Exhaustive test. Not run if not characterizing a new method")
class CharacterizeArchitecture(CharacterizeEmbedding):

    def setUp(self):
        self.tries = tries
        self.method = minorminer

    def all_graphs(self):
        gen = generators.__dict__[self.target]
        T = gen()
        self.structsampler = StructureComposite(SimulatedAnnealingSampler(), T.nodes, T.edges)
        self.sampler = EmbeddingComposite(self.structsampler, self.method)
        self.grid()
        self.bipartite()
        self.complete()
        self.hypercube()

    def all_archs(self):
        for name, gen in generators.__dict__.items:
            if callable(gen):
                print('ARCH %s' % name)
                self.target = name
                T = gen()
                self.structsampler = StructureComposite(SimulatedAnnealingSampler(), T.nodes, T.edges)
                self.sampler = EmbeddingComposite(self.structsampler, self.method)
                self.graph()

    @unittest.skip("Exhaustive test. Not run if not characterizing a new method")
    def test_hypercube(self):
        self.graph = self.hypercube
        self.all_archs()

    def test_rooks(self):
        self.graph = self.rooks
        self.all_archs()

    def test_complete(self):
        self.graph = self.complete
        self.all_archs()

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
