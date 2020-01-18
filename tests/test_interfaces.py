import dimod
import shutil
import unittest
import minorminer

import networkx as nx

from embera.interfaces.embedding import Embedding
from embera.interfaces.database import EmberaDataBase

class TestEmbedding(unittest.TestCase):
    def setUp(self):
        self.source_edgelist = [('a','b'),('b','c'),('c','a')]
        self.target_edgelist = [(1,2),(2,3),(3,4),(4,1)]
        self.embedding = minorminer.find_embedding(self.source_edgelist,self.target_edgelist)

    def test_properties(self):
        source = self.source_edgelist
        target = self.target_edgelist
        embedding = self.embedding
        runtime = 1.0
        embedding_obj = Embedding(source,target,embedding,runtime=runtime)
        self.assertEqual(embedding_obj.properties['runtime'],runtime)
        embedding_obj = Embedding(source,target,embedding,**{'test':True})
        assert(embedding_obj.properties['test'])


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
        T = nx.Graph(self.target_edgelist)

        embedding_obj = Embedding(S,T,self.embedding)
        interactions = embedding_obj.interactions_histogram(S,T)
        self.assertEqual(interactions, {1:3})

        self.db.dump_embedding(S,T,embedding_obj,'minorminer')
        embedding_copy = self.db.load_embedding(S,T,tag='minorminer')

        self.assertEqual(embedding_obj, Embedding(S,T,embedding_copy))

    def test_sampleset(self):
        bqm = self.bqm
        sampleset = self.sampleset
        embedding = self.embedding
        source_edgelist = self.source_edgelist
        target_edgelist = self.target_edgelist
        self.db.dump_sampleset(bqm,target_edgelist,sampleset,embedding=embedding)
        sampleset_copy = self.db.load_sampleset(bqm,target_edgelist,embedding=embedding)
        self.assertEqual(sampleset,sampleset_copy)

    def test_id_bqm(self):
        bqm = self.bqm
        bqm_id = self.db.id_bqm(bqm)
        S = bqm.to_networkx_graph()
        graph_id = self.db.id_bqm(S)
        self.assertEqual(bqm_id,graph_id)

        self.db.set_bqm_alias(bqm,'TEST')
        name_id = self.db.id_bqm('TEST')
        self.assertEqual(bqm_id,name_id)

        self.assertRaises(ValueError,self.db.id_bqm,0)

    def test_id_target(self):
        target_edgelist = self.target_edgelist
        edgelist_id = self.db.id_target(target_edgelist)
        T = nx.Graph(target_edgelist)
        graph_id = self.db.id_target(T)
        self.assertEqual(edgelist_id,graph_id)

        self.db.set_target_alias(target_edgelist,'TEST')
        name_id = self.db.id_target('TEST')
        self.assertEqual(edgelist_id,name_id)

        self.assertRaises(ValueError,self.db.id_target,0)

    def test_id_source(self):
        source_edgelist = self.source_edgelist
        edgelist_id = self.db.id_source(source_edgelist)
        S = nx.Graph(source_edgelist)
        graph_id = self.db.id_source(S)
        self.assertEqual(edgelist_id,graph_id)

        bqm_id = self.db.id_source(self.bqm)
        self.assertEqual(edgelist_id,bqm_id)

        self.db.set_source_alias(source_edgelist,'TEST')
        name_id = self.db.id_source('TEST')
        self.assertEqual(edgelist_id,name_id)

        self.assertRaises(ValueError,self.db.id_source,0)

    def test_id_embedding(self):

        embedding = self.embedding
        source = self.source_edgelist
        target = self.target_edgelist
        embedding_obj = Embedding(source,target,embedding)
        embedding_id = self.db.id_embedding([],[],embedding_obj)
        self.assertEqual(embedding_id,embedding_obj.id)

        dict_id = self.db.id_embedding(source,target,embedding)
        self.assertEqual(embedding_id,dict_id)

        self.db.set_embedding_alias(embedding_obj,'TEST')
        name_id = self.db.id_embedding([],[],'TEST')
        self.assertEqual(dict_id,name_id)

        self.assertRaises(ValueError,self.db.id_embedding,[],[],0)

    def test_empty(self):
        self.db.dump_embedding([],[],{})
