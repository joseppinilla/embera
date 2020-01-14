import shutil
import unittest
import minorminer

import networkx as nx

from embera import Embedding
from embera.interfaces.database import EmberaDataBase

class TestDataBase(unittest.TestCase):

    def test_embedding(self):
        db = EmberaDataBase("./TMP_DB")
        source_edgelist = [('a','b'),('b','c'),('c','a')]
        target_edgelist = [(1,2),(2,3),(3,4),(4,1)]
        S = nx.Graph(source_edgelist)
        S.name = "TMP_K3"
        T = nx.Graph(target_edgelist)
        T.name = "TMP_P4"
        embedding = minorminer.find_embedding(S,T)

        embedding_obj = Embedding(S,T,embedding)
        interactions = embedding_obj.interactions_histogram(S,T)
        self.assertEqual(interactions, {1:3})

        db.dump_embedding(S,T,embedding,'minorminer')
        embedding_copy = db.load_embedding(S,T,embedding_tag='minorminer')
        self.assertEqual(embedding, embedding_copy)

        shutil.rmtree("./TMP_DB")
