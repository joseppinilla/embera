import dimod
import shutil
import unittest
import minorminer

import networkx as nx

from embera.interfaces.embedding import Embedding
from embera.interfaces.database import EmberaDataBase

class TestDataBase(unittest.TestCase):

    db = None
    source_edgelist = None
    target_edgelist = None

    def setUp(self):
        # Temporary DataBase
        self.db = EmberaDataBase("./TMP_DB")
        # Toy Problem
        self.bqm = dimod.BinaryQuadraticModel(
                   {'a':2,'b':2,'c':2},
                   {('a','b'):-1,('b','c'):-1,('c','a'):-1},
                   0.0,dimod.Vartype.SPIN)
        self.source_edgelist = [('a','b'),('b','c'),('c','a')]
        self.target_edgelist = [(1,2),(2,3),(3,4),(4,1)]
        # Toy Entries
        self.embedding = minorminer.find_embedding(self.source_edgelist,
                                                   self.target_edgelist)
        self.sampleset = dimod.SampleSet.from_samples(
                         [{'a': 0, 'b': 1, 'c': 0},{'a': 0, 'b': 1, 'c': 1}],
                         'BINARY',0)

    def tearDown(self):
        shutil.rmtree("./TMP_DB")

    def test_embedding_graphs(self):
        S = nx.Graph(self.source_edgelist)
        S.name = "TMP_K3"
        T = nx.Graph(self.target_edgelist)
        T.name = "TMP_P4"

        embedding_obj = Embedding(S,T,self.embedding)
        interactions = embedding_obj.interactions_histogram(S,T)
        self.assertEqual(interactions, {1:3})

        self.db.dump_embedding(S,T,embedding_obj,'minorminer')
        embedding_copy = self.db.load_embedding(S,T,tag='minorminer')

        self.assertEqual(embedding_obj, Embedding(S,T,embedding_copy))

    def test_embedding_noname(self):
        embedding = self.embedding
        source_edgelist = self.source_edgelist
        target_edgelist = self.target_edgelist
        self.db.dump_embedding(source_edgelist,target_edgelist,embedding)
        embedding_copy = self.db.load_embedding(source_edgelist,target_edgelist)
        self.assertEqual(embedding,embedding_copy)

    def test_sampleset(self):
        bqm = self.bqm
        sampleset = self.sampleset
        embedding = self.embedding
        source_edgelist = self.source_edgelist
        target_edgelist = self.target_edgelist
        self.db.dump_sampleset(bqm,target_edgelist,sampleset,embedding=embedding)
        sampleset_copy = self.db.load_sampleset(bqm,target_edgelist,embedding=embedding)

    def test_empty(self):
        self.db.dump_embedding([],[],{})

# import json
# import dimod
# from dimod.serialization.json import DimodEncoder, DimodDecoder
#
# bqm = dimod.BinaryQuadraticModel({'a':2,'b':2,'c':2},{('a','b'):-1,('b','c'):-1,('c','a'):-1},0.0,dimod.Vartype.SPIN)
# hash(json.dumps(bqm,cls=DimodEncoder))
#
# list(bqm.linear)
# list(bqm.quadratic)
