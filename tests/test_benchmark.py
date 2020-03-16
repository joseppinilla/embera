import dimod
import embera
import unittest

class TestParameters(unittest.TestCase):

    def test_bqm(self):
        G = embera.benchmark.topologies.complete_graph(4)
        csp_bqm = embera.benchmark.parameters.csp(G)
        self.assertIsInstance(csp_bqm,dimod.BinaryQuadraticModel)
        init_bm_bqm = embera.benchmark.parameters.init_bm(G)
        self.assertIsInstance(init_bm_bqm,dimod.BinaryQuadraticModel)
        trained_bm_bqm = embera.benchmark.parameters.trained_bm(G)
        self.assertIsInstance(trained_bm_bqm,dimod.BinaryQuadraticModel)

    def test_marshall(self):
        bqm_list = embera.benchmark.parameters.marshall_bench()
        for bqm in bqm_list:
            self.assertIsInstance(bqm,dimod.BinaryQuadraticModel)

class TestTopologies(unittest.TestCase):

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
