import os
import dimod
import embera
import shutil
import unittest

import networkx as nx


class TestBenchmarkDrawing(unittest.TestCase):
    def setUp(self):
        os.mkdir("./TMP")
        self.source_edgelist = [('a','b'),('b','c'),('c','a')]
        self.target_edgelist = [(1,2),(2,3),(3,4),(4,1)]
        self.embedding = embera.Embedding({'a':[1],'b':[2,3],'c':[4]})
        self.sampleset = dimod.SampleSet.from_samples(
                         [{1:-1, 2:+1, 3:-1, 4:-1},
                          {1:-1, 2:+1, 3:-1, 4:+1},
                          {1:+1, 2:+1, 3:+1, 4:+1},
                          {1:-1, 2:-1, 3:-1, 4:-1}],
                         'SPIN',[0.0,-1.0,-2.0,1.0])

    def tearDown(self):
        shutil.rmtree("./TMP")

    def test_plot_embeddings(self):
        embeddings = [self.embedding]
        T = nx.Graph(self.target_edgelist)
        embera.benchmark.plot_embeddings(embeddings,T,savefig='./TMP/0.png')
        self.assertTrue(os.path.exists('./TMP/0.png'))

    def test_plot_joint_sampleset(self):
        samplesets = [self.sampleset]
        embera.benchmark.plot_joint_samplesets(samplesets,savefig='./TMP/1.png')
        self.assertTrue(os.path.exists('./TMP/1.png'))
        embera.benchmark.plot_joint_samplesets(samplesets,gray=True,savefig='./TMP/2.png')
        self.assertTrue(os.path.exists('./TMP/2.png'))

    def test_plot_chain_metrics(self):
        embeddings = [self.embedding]
        S = nx.Graph(self.source_edgelist)
        T = nx.Graph(self.target_edgelist)
        embera.benchmark.plot_chain_metrics(embeddings,S,T,savefig='./TMP/3.png')
        self.assertTrue(os.path.exists('./TMP/3.png'))
