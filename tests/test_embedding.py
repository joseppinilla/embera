import unittest
import minorminer
import dimod.testing as dtest
from embedding_methods.architectures import generators
from dimod.reference.samplers.random_sampler import RandomSampler
from dimod.reference.composites.structure import StructureComposite
from embedding_methods.composites.embedding import EmbeddingComposite


class TestEmbeddingComposite(unittest.TestCase):

    def test_instantiate_pegasus(self):
        # Use the provided architectures
        target_graph = generators.p6_graph()

        structsampler = StructureComposite(RandomSampler(), target_graph.nodes, target_graph.edges)
        sampler = EmbeddingComposite(structsampler, minorminer)

        dtest.assert_sampler_api(sampler)

    def test_instantiate_chimera(self):
        # Use the provided architectures
        target_graph = generators.dw2x_graph()

        structsampler = StructureComposite(RandomSampler(), target_graph.nodes, target_graph.edges)
        sampler = EmbeddingComposite(structsampler, minorminer)

        dtest.assert_sampler_api(sampler)
