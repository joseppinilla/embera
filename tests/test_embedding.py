import unittest
import minorminer
import dimod.testing as dtest

from embedding_methods.utilities.architectures import generators
from embedding_methods.composites.embedding import EmbeddingComposite

from dimod.reference.samplers.random_sampler import RandomSampler
from dimod.reference.composites.structure import StructureComposite

class TestEmbeddingComposite(unittest.TestCase):

    def test_instantiate_pegasus(self):
        # Use the provided architectures
        target_graph = generators.p6_graph()

        # Use any sampler and make structured (i.e. Simulated Annealing, Exact) or use structured sampler if available (i.e. D-Wave machine)
        structsampler = StructureComposite(RandomSampler(), target_graph.nodes, target_graph.edges)
        sampler = EmbeddingComposite(structsampler, minorminer)

        dtest.assert_sampler_api(sampler)

    def test_instantiate_chimera(self):
        # Use the provided architectures
        target_graph = generators.dw2x_graph()

        # Use any sampler and make structured (i.e. Simulated Annealing, Exact) or use structured sampler if available (i.e. D-Wave machine)
        structsampler = StructureComposite(RandomSampler(), target_graph.nodes, target_graph.edges)
        sampler = EmbeddingComposite(structsampler, minorminer)

        dtest.assert_sampler_api(sampler)
