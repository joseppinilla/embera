import shutil
import unittest
import minorminer

import networkx as nx

from embera import Embedding
from embera.interfaces.database import EmberaDataBase

class TestDataBase(unittest.TestCase):

    db = None
    source_edgelist = None
    target_edgelist = None

    def setUp(self):
        # Temporary DataBase
        self.db = EmberaDataBase("./TMP_DB")
        #Toy Graphs
        self.source_edgelist = [('a','b'),('b','c'),('c','a')]
        self.target_edgelist = [(1,2),(2,3),(3,4),(4,1)]
        self.embedding = minorminer.find_embedding(self.source_edgelist,self.target_edgelist)

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

    def test_report(self):
        report = {"this":"is","a":"test"}
        embedding = self.embedding
        source_edgelist = self.source_edgelist
        target_edgelist = self.target_edgelist
        self.db.dump_report(source_edgelist,target_edgelist,embedding,report)
        report_copy = self.db.load_report(source_edgelist,target_edgelist)
        self.assertEqual(report,report_copy)
