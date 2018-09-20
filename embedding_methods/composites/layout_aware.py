"""
A dimod composite_ to map unstructured problems to a structured_ sampler using
a layout-aware embedding algorithm.

A structured_ sampler can only solve problems that map to a specific graph: the
D-Wave system's architecture is represented by a Chimera_ graph.

.. _composite: http://dimod.readthedocs.io/en/latest/reference/samplers.html
.. _structured: http://dimod.readthedocs.io/en/latest/reference/samplers.html#module-dimod.core.structured
.. _Chimera: http://dwave-system.readthedocs.io/en/latest/reference/intro.html#chimera

"""
import dimod
from embedding_methods import disperse

class LayoutAwareEmbeddingComposite(dimod.ComposedSampler):

    def __init__(self, child_sampler, **embedding_parameters):
        if not isinstance(child_sampler, dimod.Structured):
            raise dimod.InvalidComposition("LayoutAwareEmbeddingComposite should only be applied to a Structured sampler")
        self._children = [child_sampler]
        self._embedding = None
        self._embedding_parameters = embedding_parameters
