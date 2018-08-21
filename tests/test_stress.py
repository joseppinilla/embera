'''
Stress Test of embedding complete graphs

Using example from
http://dw-docs.readthedocs.io/en/latest/examples/multi_gate.html

Modified:
Added EmbeddingComposite to bypass need for D-Wave connection


>> python3 -m unittest tests.test_stress_complete

'''
import time
import numpy
import random
import unittest
import minorminer
import networkx as nx
import dwave_networkx as dnx

import matplotlib.pyplot as plt
from embedding_methods.dense import dense
from embedding_methods.topological import topological
from embedding_methods.composites.embedding import EmbeddingComposite

from dimod.reference.samplers.exact_solver import ExactSolver
from dimod.reference.samplers.random_sampler import RandomSampler
from dimod.reference.composites.structure import StructureComposite
from dimod.reference.samplers.simulated_annealing import SimulatedAnnealingSampler

verbose = 0
tries = 10

class StressEmbedding(unittest.TestCase):

    def setUp(self):

        # Size of Chimera Graph
        m,n,t = 16,16,4
        T = dnx.generators.chimera.chimera_graph(m,n,t)
        self.T = T

        #strucsampler = StructureComposite(ExactSolver(), chimera.nodes, chimera.edges)
        #strucsampler = StructureComposite(RandomSampler(), chimera.nodes, chimera.edges)
        self.structsampler = StructureComposite(SimulatedAnnealingSampler(), T.nodes, T.edges)

        self.T_edgelist = list(T.edges())

        self.methods = [minorminer]
        #self.methods = [minorminer, topological, dense]

        #self.prune = [0]
        self.prune = [0, 5, 10, 20, 40]

    def get_stats(self, embedding, stats):
        max_chain = 0
        chain_lens = {}
        chain_accum = 0
        for node, chain in embedding.items():
            chain_len = len(chain)
            chain_lens[node] = chain_len
            chain_accum += chain_len
            if chain_len > max_chain: max_chain = chain_len
        if verbose > 1:
            plt.hist(list(chain_lens.values()))
            plt.show()

        return chain_accum, max_chain


    def prune_graph(self, pct, S):
        """
            Args:
                pct: Percentage of edges to be removed
                S: Problem Graph
        """
        size = len(S)
        to_prune = int(size*pct/100)
        print('Size %s PCT %s Prune %s' % (size,pct,to_prune))
        for i in range(to_prune):
            S.pop( random.randrange(size) )
            size-=1


    def embed_with_method(self, S_edgelist, method):
        sampler = EmbeddingComposite(self.structsampler, minorminer)
        t_start = time.time()
        embedding = sampler.get_embedding(S=S_edgelist, T=self.T_edgelist, verbose=verbose)
        t_end = time.time()
        t_elap = t_end-t_start

        total, max = self.get_stats(embedding)


        print(total, max, t_elap)

        if verbose > 1:
            dnx.draw_chimera_embedding(self.T,embedding)
            plt.show()

        return total, max, t_elap

    def stress(self, S):
        self.S = S
        valid = 0
        for val in self.prune:
            S_edgelist = list(S.edges())
            self.prune_graph(val, S_edgelist)
            for method in self.methods:
                acc_total = 0
                acc_max = 0
                acc_valid = 0
                acc_time = 0
                for i in range(tries):
                    total, max, t_elap = self.embed_with_method(S_edgelist, method)
                    acc_time += t_elap
                    acc_total += total
                    acc_max += max
                    acc_valid += bool(max)
            print('AVG: %s %s %s %s' % (acc_total/tries, acc_max/tries, acc_valid/tries, acc_time/tries))





class StressBipartiteEmbedding(StressEmbedding):
    """

    >> python3 -m unittest tests.test_stress_complete.StressBipartiteEmbedding
    """
    def test_bipartite(self):
        print('BIPARTITE')
        sizes = [50,20,10,8,5]
        for p in sizes:
            #print(p)
            S = nx.complete_bipartite_graph(p,p)
            #TODO: self.topology =
            self.stress(S)



class StressGridEmbedding(StressEmbedding):
    """

    >> python3 -m unittest tests.test_stress_complete.StressGridEmbedding
    """
    def test_grid(self):
        print('GRID')
        sizes = [20,15,10,5]
        for p in sizes:
            #print(p)
            S = nx.grid_2d_graph(p,p)
            topology = {v:v for v in S}
            self.stress(S)


class StressCompleteEmbedding(StressEmbedding):
    """

    >> python3 -m unittest tests.test_stress_complete.StressCompleteEmbedding
    """
    def test_complete(self):
        print('COMPLETE')
        sizes = [50,40,30,20,15,10,5]
        for p in sizes:
            #print(p)
            S = nx.complete_graph(p)
            self.S = S
            #TODO: self.topology =
            self.stress(S)
