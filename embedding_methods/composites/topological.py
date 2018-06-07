"""
A dimod composite_ to map unstructured problems to a structured_ sampler using
a topology-aware embedding algorithm.

A structured_ sampler can only solve problems that map to a specific graph: the
D-Wave system's architecture is represented by a Chimera_ graph.

.. _composite: http://dimod.readthedocs.io/en/latest/reference/samplers.html
.. _structured: http://dimod.readthedocs.io/en/latest/reference/samplers.html#module-dimod.core.structured
.. _Chimera: http://dwave-system.readthedocs.io/en/latest/reference/intro.html#chimera

"""
import dimod
from embedding_methods.topological import topological

class TopologicalEmbeddingComposite(dimod.ComposedSampler):

    def __init__(self, child_sampler, **embedding_parameters):
        if not isinstance(child_sampler, dimod.Structured):
            raise dimod.InvalidComposition("TopologicalEmbeddingComposite should only be applied to a Structured sampler")
        self._children = [child_sampler]
        self._embedding = None
        self._embedding_parameters = embedding_parameters

    @property
    def children(self):
        """list: Children property inherited from :class:`dimod.Composite` class.

        For an instantiated composed sampler, contains the single wrapped structured sampler.

        .. _configuration: http://dwave-cloud-client.readthedocs.io/en/latest/#module-dwave.cloud.config

        """
        return self._children

    @property
    def parameters(self):
        """dict[str, list]: Parameters in the form of a dict.

        For an instantiated composed sampler, keys are the keyword parameters accepted by the child sampler.

        .. _configuration: http://dwave-cloud-client.readthedocs.io/en/latest/#module-dwave.cloud.config

        """
        # does not add or remove any parameters
        param = self.child.parameters.copy()
        param['chain_strength'] = []
        return param

    @property
    def properties(self):
        """dict: Properties in the form of a dict.

        For an instantiated composed sampler, contains one key :code:`'child_properties'` that
        has a copy of the child sampler's properties.

        .. _configuration: http://dwave-cloud-client.readthedocs.io/en/latest/#module-dwave.cloud.config

        """
        return {'child_properties': self.child.properties.copy()}

    def get_embedding(self, S=None, T=None, get_new=False, **embedding_parameters):
        """

        """
        embedding_method = self._embedding_method
        self._embedding_parameters = embedding_parameters

        if get_new or not self._embedding:
            embedding = topological.find_embedding(S,T,**embedding_parameters)
            self._embedding = embedding

        return self._embedding

    def sample(self, bqm, chain_strength=1.0, **parameters):
        """Sample from the provided binary quadratic model.

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            chain_strength (float, optional, default=1.0):
                Magnitude of the quadratic bias (in SPIN-space) applied between variables to create
                chains. Note that the energy penalty of chain breaks is 2 * `chain_strength`.

            **parameters:
                Parameters for the sampling method, specified by the child sampler.

        Returns:
            :class:`dimod.Response`

        """

        # use the given embedding method with the given parameters
        embedding_method = self._embedding_method
        embedding_parameters = self._embedding_parameters

        # solve the problem on the child system
        child = self.child

        # apply the embedding to the given problem to map it to the child sampler
        __, target_edgelist, target_adjacency = child.structure

        # add self-loops to edgelist to handle singleton variables
        source_edgelist = list(bqm.quadratic) + [(v, v) for v in bqm.linear]

        # get the embedding
        embedding = topological.find_embedding(source_edgelist, target_edgelist, **embedding_parameters)
        self._embedding = embedding

        if bqm and not embedding:
            raise ValueError("no embedding found")

        bqm_embedded = dimod.embed_bqm(bqm, embedding, target_adjacency, chain_strength=chain_strength)

        response = child.sample(bqm_embedded, **parameters)

        return dimod.unembed_response(response, embedding, source_bqm=bqm)
