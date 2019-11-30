import unittest
import dimod.testing as dtest


from embera.utilities.architectures import generators
from embera.composites.minorminer import MinorMinerEmbeddingComposite

from dimod.reference.composites.structure import StructureComposite
from dimod.reference.samplers.simulated_annealing import SimulatedAnnealingSampler

class TestMinorMinerEmbeddingComposite(unittest.TestCase):

    def test_instantiate_pegasus(self):
        # Use the provided architectures
        target_graph = generators.p6_graph()

        # Use any sampler and make structured (i.e. Simulated Annealing, Exact) or use structured sampler if available (i.e. D-Wave machine)
        structsampler = StructureComposite(SimulatedAnnealingSampler(), target_graph.nodes, target_graph.edges)
        sampler = MinorMinerEmbeddingComposite(structsampler)

        dtest.assert_sampler_api(sampler)

    def test_instantiate_chimera(self):
        # Use the provided architectures
        target_graph = generators.dw2x_graph()

        # Use any sampler and make structured (i.e. Simulated Annealing, Exact) or use structured sampler if available (i.e. D-Wave machine)
        structsampler = StructureComposite(SimulatedAnnealingSampler(), target_graph.nodes, target_graph.edges)
        sampler = MinorMinerEmbeddingComposite(structsampler)

        dtest.assert_sampler_api(sampler)
