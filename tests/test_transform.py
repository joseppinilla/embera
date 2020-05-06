import embera
import unittest
import minorminer

import networkx as nx
import dwave_networkx as dnx

class TestTransformEmbedding(unittest.TestCase):

    def setUp(self):
        self.S = nx.complete_graph(11)
        self.T = dnx.chimera_graph(8)
        self.embedding = minorminer.find_embedding(self.S,self.T,random_seed=10)

    def test_translate_chimera(self):
        S = self.S
        T = self.T
        embedding = self.embedding
        new_embedding = embera.transform.embedding.translate(S,T,embedding,(0,0))

    def test_spread_out_chimera(self):
        S = self.S
        T = self.T
        embedding = self.embedding
        new_embedding = embera.transform.embedding.spread_out(S,T,embedding)

    def test_rotate_chimera(self):
        S = self.S
        T = self.T
        embedding = self.embedding
        embedding_90 = embera.transform.embedding.rotate(S,T,embedding,90)
        embedding_180 = embera.transform.embedding.rotate(S,T,embedding,180)
        embedding_270 = embera.transform.embedding.rotate(S,T,embedding,270)

    def test_mirror_chimera(self):
        S = self.S
        T = self.T
        embedding = self.embedding
        embedding_0 = embera.transform.embedding.mirror(S,T,embedding,0)
        embedding_1 = embera.transform.embedding.mirror(S,T,embedding,1)

    def test_greedy_fit(self):
        S = self.S
        T = self.T
        embedding = self.embedding
        new_embedding = embera.transform.embedding.greedy_fit(S,T,embedding)

    def test_open_seam(self):
        S = self.S
        T = self.T
        embedding = self.embedding
        embedding_l = embera.transform.embedding.open_seam(S,T,embedding,2,'left')
        embedding_r = embera.transform.embedding.open_seam(S,T,embedding,2,'right')
        embedding_u = embera.transform.embedding.open_seam(S,T,embedding,2,'up')
        embedding_d = embera.transform.embedding.open_seam(S,T,embedding,2,'down')
