import dimod
import embera
import unittest

class TestParameters(unittest.TestCase):

    def test_bqm(self):
        G = embera.benchmarks.generators.topologies.complete_graph(4)
        csp_bqm = embera.benchmarks.generators.parameters.csp(G)
        self.assertIsInstance(csp_bqm,dimod.BinaryQuadraticModel)
        init_bm_bqm = embera.benchmarks.generators.parameters.init_bm(G)
        self.assertIsInstance(init_bm_bqm,dimod.BinaryQuadraticModel)
        trained_bm_bqm = embera.benchmarks.generators.parameters.trained_bm(G)
        self.assertIsInstance(trained_bm_bqm,dimod.BinaryQuadraticModel)

    def test_frust_loops(self):
        bqm_list = embera.benchmarks.parameters.frust_loops_bench()
        for bqm in bqm_list:
            self.assertIsInstance(bqm,dimod.BinaryQuadraticModel)

class TestTopologies(unittest.TestCase):

    def test_dwave(self):
        benchmarks = embera.benchmarks.topologies.dwave_bench()
        self.assertTrue(benchmarks)

    def test_qca(self):
        benchmarks = embera.benchmarks.topologies.qca_bench()
        self.assertTrue(benchmarks)

    def test_geometry(self):
        benchmarks = embera.benchmarks.topologies.geometry_bench()
        self.assertTrue(benchmarks)

    def test_misc(self):
        benchmarks = embera.benchmarks.topologies.misc_bench()
        self.assertTrue(benchmarks)

    def test_hypercube(self):
        G = embera.benchmarks.generators.topologies.hypercube_graph()
        self.assertEqual(len(G),0)

    def test_embera(self):
        benchmarks = embera.benchmarks.topologies.embera_bench()
        self.assertTrue(benchmarks)
