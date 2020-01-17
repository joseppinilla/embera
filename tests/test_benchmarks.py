import embera
import unittest

class TestBenchmarks(unittest.TestCase):

    def test_dwave(self):
        benchmarks = embera.benchmark.topologies.dwave_bench()
        self.assertTrue(benchmarks)

    def test_qca(self):
        benchmarks = embera.benchmark.topologies.qca_bench()
        self.assertTrue(benchmarks)

    def test_geometry(self):
        benchmarks = embera.benchmark.topologies.geometry_bench()
        self.assertTrue(benchmarks)

    def test_misc(self):
        benchmarks = embera.benchmark.topologies.misc_bench()
        self.assertTrue(benchmarks)

    def test_mnist(self):
        path = "./mnist.pkl"
        url = "http://www.ece.ubc.ca/~jpinilla/resources/embera/misc/mnist.pkl"
        mnist = embera.benchmark.topologies.download_pickle(path,url)
        G = embera.benchmark.topologies.complete_multipartite_graph(32,o=10)
        self.assertEqual(G.edges,mnist.edges)

    def test_hypercube(self):
        G = embera.benchmark.topologies.hypercube_graph()
        self.assertEqual(len(G),0)

    def test_embera(self):
        benchmarks = embera.benchmark.topologies.embera_bench()
        self.assertTrue(benchmarks)

    def test_random(self):
        pvals = embera.random.prob_vector(16)
        samples = embera.random.categorical(16,pvals)
        self.assertEqual(len(samples),16)
        samples = embera.random.bimodal(16)
        self.assertEqual(len(samples),16)
