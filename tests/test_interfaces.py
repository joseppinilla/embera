import dimod
import shutil
import unittest
import minorminer

import networkx as nx

from embera.interfaces.embedding import Embedding
from embera.interfaces.database import EmberaDataBase

try: # Pandas isn't required. Tests are done if found.
    import pandas as pd
    _pandas = True
except e:
    _pandas = False

class TestEmbedding(unittest.TestCase):
    def setUp(self):
        self.source_edgelist = [('a','A1.S'),('A1.S',(0,1)),((0,1),'a')]
        self.target_edgelist = [(1,2),(2,3),(3,4),(4,1)]
        self.embedding = minorminer.find_embedding(self.source_edgelist,self.target_edgelist)
        self.sampleset = dimod.SampleSet.from_samples(
                         [{1:-1, 2:1, 3:-1, 4:-1},{1:-1, 2:1, 3:1, 4:1}],
                         'SPIN',0)

    def test_properties(self):
        source = self.source_edgelist
        target = self.target_edgelist
        embedding = self.embedding
        runtime = 1.0
        embedding_obj = Embedding(embedding,runtime=runtime)
        self.assertEqual(embedding_obj.properties['runtime'],runtime)
        embedding_obj = Embedding(embedding,**{'test':True})
        self.assertIn('test',embedding_obj.properties)

    def test_interactions(self):
        S = self.source_edgelist
        T = self.target_edgelist
        embedding_obj = Embedding(self.embedding)
        interactions = embedding_obj.interactions_histogram(S,T)
        self.assertEqual(interactions, {1:3})

    def test_qubit_metrics(self):
        S = self.source_edgelist
        T = self.target_edgelist
        embedding_obj = Embedding(self.embedding)

        qubit_interactions = embedding_obj.qubit_interactions(S,T)
        self.assertEqual(len(qubit_interactions),4)

        qubit_not_interactions = embedding_obj.qubit_interactions(S,T,False)
        self.assertFalse(qubit_not_interactions)

        qubit_connectivity = embedding_obj.qubit_connectivity(S,T)
        for s,c in qubit_connectivity.items():
            self.assertLessEqual(c,1.0)

    def test_chain_breaks(self):
        embedding_obj = Embedding(self.embedding)
        broken = embedding_obj.chain_breaks(self.sampleset)
        for v,b in broken.items():
            if len(embedding_obj[v])==1:
                self.assertEqual(b,0.0)
            else:
                self.assertLessEqual(b,1.0)

    def test_comparison(self):
        emb1 = self.embedding
        emb1_obj = Embedding(emb1)
        emb2 = minorminer.find_embedding(self.source_edgelist,self.target_edgelist)
        emb2_obj = Embedding(emb2)

        self.assertEqual(emb2_obj==emb1_obj, not emb1_obj!=emb2_obj)
        self.assertFalse(emb2_obj > emb1_obj)
        self.assertFalse(emb2_obj < emb1_obj)
        self.assertTrue(emb2_obj >= emb1_obj)
        self.assertTrue(emb2_obj <= emb1_obj)

class TestDataBase(unittest.TestCase):

    db = None
    source_edgelist = None
    target_edgelist = None

    def setUp(self):
        # Temporary DataBase
        self.db = EmberaDataBase("./TMP_DB")
        # Toy Problem
        self.bqm = dimod.BinaryQuadraticModel(
                   {'a':2,'A1.S':2,'(0,1)':2},
                   {('a','A1.S'):-1,('A1.S','(0,1)'):-1,('(0,1)','a'):-1},
                   0.0,dimod.Vartype.SPIN)
        self.source_edgelist = [('a','A1.S'),('A1.S','(0,1)'),('(0,1)','a')]
        self.target_edgelist = [(1,2),(2,3),(3,4),(4,1)]
        # Toy Entries
        self.embedding = minorminer.find_embedding(self.source_edgelist,
                                                   self.target_edgelist)
        self.sampleset = dimod.SampleSet.from_samples(
                         [{1:-1, 2:1, 3:-1, 4:-1},{1:-1, 2:1, 3:1, 4:1}],
                         'SPIN',0)
        self.report = {'emb1':{'a':4,'A1.S':8,'(0,1)':12}}

    def tearDown(self):
        shutil.rmtree("./TMP_DB")

    def test_bqm(self):
        bqm = self.bqm
        bqm_id = self.db.id_bqm(bqm)
        source = self.source_edgelist

        self.db.dump_bqm(bqm)
        bqm_copy = self.db.load_bqm(source)
        self.assertEqual(bqm, bqm_copy)
        copy_id = self.db.id_bqm(bqm_copy)
        self.assertEqual(bqm_id,copy_id)

        self.db.dump_bqm(bqm,tags=["Tag"])
        bqm_copy2 = self.db.load_bqm(source,tags=["Tag"])
        self.assertEqual(bqm,bqm_copy2)
        copy_id2 = self.db.id_bqm(bqm_copy2)
        self.assertEqual(bqm_id,copy_id2)


    def test_embedding(self):
        S = nx.Graph(self.source_edgelist)
        T = nx.Graph(self.target_edgelist)

        embedding_obj = Embedding(self.embedding)

        self.db.dump_embedding(S,T,embedding_obj,tags=['minorminer'])
        embedding_copy = self.db.load_embedding(S,T,tags=['minorminer'])

        self.assertEqual(embedding_obj, Embedding(embedding_copy))

    def test_sampleset(self):
        bqm = self.bqm
        sampleset = self.sampleset
        embedding = self.embedding
        target_edgelist = self.target_edgelist
        self.db.dump_sampleset(bqm,target_edgelist,embedding,sampleset)
        sampleset_copy = self.db.load_sampleset(bqm,target_edgelist,"")
        self.assertEqual(sampleset,sampleset_copy)

    def test_report(self):
        bqm = self.bqm
        report = self.report
        T = nx.Graph(self.target_edgelist)
        self.db.dump_report(bqm,T,report,'mock_metric')
        report_copy = self.db.load_report(bqm,T,'mock_metric')

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
        embedding_obj = Embedding(embedding)
        embedding_id = self.db.id_embedding(embedding_obj)
        self.assertEqual(embedding_id,embedding_obj.id)

        dict_id = self.db.id_embedding(embedding)
        self.assertEqual(embedding_id,dict_id)

        self.db.set_embedding_alias(embedding_obj,'TEST')
        name_id = self.db.id_embedding('TEST')
        self.assertEqual(dict_id,name_id)

        self.assertRaises(ValueError,self.db.id_embedding,0)

    def test_load_embeddings(self):
        embedding = self.embedding
        source = self.source_edgelist
        target = nx.path_graph(8)
        self.db.dump_embedding(source,target,embedding,tags=['tag1'])
        self.db.dump_embedding(source,target,embedding,tags=['tag2'])
        copy1,copy2 = self.db.load_embeddings(source,target)
        # When stored, chains are sorted, so:
        for k in embedding:
            self.assertCountEqual(embedding[k],copy1[k])
            self.assertCountEqual(embedding[k],copy2[k])

    def test_empty(self):
        self.db.dump_embedding([],[],{})

    @unittest.skipUnless(_pandas, "No Pandas package found")
    def test_dataframe(self):
        bqm = self.bqm
        report = self.report
        T = nx.Graph(self.target_edgelist)
        self.db.dump_report(bqm,T,report,'mock_pandas')
        report_df = self.db.load_report(bqm,T,'mock_pandas',dataframe=True)
        self.assertEqual(list(bqm),list(report_df.columns))
