"""
A dimod composite_ to map unstructured problems to a structured_ sampler using
layout awareness.

A structured_ sampler can only solve problems that map to a specific graph: the
Ising solver architecture is represented by a graph.

.. _composite: http://dimod.readthedocs.io/en/latest/reference/samplers.html
.. _minorminer: https://github.com/dwavesystems/minorminer
.. _structured: http://dimod.readthedocs.io/en/latest/reference/samplers.html#module-dimod.core.structured
.. _Chimera: http://dwave-system.readthedocs.io/en/latest/reference/intro.html#chimera

"""
import dimod
import minorminer
from embedding_methods import disperse

from embedding_methods.preprocess import diffusion_placer
from embedding_methods.utilities.architectures.generators import dw2000q_graph

from dwave.embedding.transforms import embed_bqm, unembed_sampleset
from dimod.binary_quadratic_model import BinaryQuadraticModel

class LayoutAwareEmbeddingComposite(dimod.ComposedSampler):

    def __init__(self, child_sampler, layout = None,
                embedding_method=minorminer,
                candidates_method=diffusion_placer,
                architecture_method=dw2000q_graph,
                embedding_parameters={},
                candidates_parameters={} ):

        if not isinstance(child_sampler, dimod.Structured):
            raise dimod.InvalidComposition("LayoutAwareEmbeddingComposite should only be applied to a Structured sampler")
        self._children = [child_sampler]
        self._layout = layout
        self._embedding = None
        self._embedding_method = embedding_method
        self._candidates_method = candidates_method
        self._architecture_method = architecture_method
        self._embedding_parameters = embedding_parameters
        self._candidates_parameters = candidates_parameters

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
        param['force_embed'] = []
        #TODO: Find a way to display embedding_method.find_embedding parameters

        return param

    @property
    def properties(self):
        """dict: Properties in the form of a dict.

        For an instantiated composed sampler, contains one key :code:`'child_properties'` that
        has a copy of the child sampler's properties.

        .. _configuration: http://dwave-cloud-client.readthedocs.io/en/latest/#module-dwave.cloud.config

        """
        properties = {'child_properties': self.child.properties.copy()}
        properties['embedding_method'] = self._embedding_method.__name__
        return properties

    def get_ising_embedding(self, h, J, **parameters):
        """Retrieve or create a minor-embedding from Ising model
        """
        bqm = BinaryQuadraticModel.from_ising(h,J)
        embedding = self.get_embedding(bqm, **parameters)
        return embedding

    def get_qubo_embedding(self, Q, **parameters):
        """Retrieve or create a minor-embedding from QUBO
        """
        bqm = BinaryQuadraticModel.from_qubo(Q)
        embedding = self.get_embedding(bqm, **parameters)
        return embedding

    def set_embedding(self, embedding):
        """Write to the embedding parameter. Useful if embedding is taken from
        a file or independent method.
        Args:
            embedding (dict):
                Dictionary that maps labels in S_edgelist to lists of labels in the
                graph of the structured sampler.
        """
        self._embedding = embedding

    def get_embedding(self, bqm=None, target_edgelist=None, force_embed=False, embedding_parameters={}, candidates_parameters={}):
        """Retrieve or create a minor-embedding from BinaryQuadraticModel

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            target_edgelist (list, optional, default=<Child Structure>):
                An iterable of label pairs representing the edges in the target graph.

            force_embed (bool, optional, default=False):
                If the sampler has an embedding return it. Otherwise, embed problem.

            **parameters:
                Parameters for the embedding method.

        Returns:
            embedding (dict):
                Dictionary that maps labels in S_edgelist to lists of labels in the
                graph of the structured sampler.
        """
        child = self.child
        layout = self._layout
        embedding_method = self._embedding_method
        candidates_method = self._candidates_method
        architecture_method = self._architecture_method
        self._embedding_parameters = embedding_parameters
        self._candidates_parameters = embedding_parameters


        # add self-loops to edgelist to handle singleton variables
        source_edgelist = list(bqm.quadratic) + [(v, v) for v in bqm.linear]

        if target_edgelist is None:
            _, target_edgelist, _ = child.structure

        if force_embed or not self._embedding:
            Tg = architecture_method(edge_list=target_edgelist)

            candidates = candidates_method.find_candidates(source_edgelist, Tg,
                                        layout=layout,
                                        **candidates_parameters)

            embedding = embedding_method.find_embedding(source_edgelist, target_edgelist,
                                        initial_chains = candidates,
                                        **embedding_parameters)

            self._candidates = candidates
            self._embedding = embedding

        if bqm and not self._embedding:
            raise ValueError("no embedding found")

        return self._embedding

    def get_child_response(self):
        return self._child_response

    def sample(self, bqm, chain_strength=1.0, force_embed=False, chain_break_fraction=True, **parameters):
        """Sample from the provided binary quadratic model.

        Args:
            bqm (:obj:`dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

            chain_strength (float, optional, default=1.0):
                Magnitude of the quadratic bias (in SPIN-space) applied between variables to create
                chains. Note that the energy penalty of chain breaks is 2 * `chain_strength`.

            force_embed (bool, optional, default=False):
                If the sampler has an embedding return it. Otherwise, embed problem.

            chain_break_fraction (bool, optional, default=True):
                If True, a ‘chain_break_fraction’ field is added to the unembedded response which report
                what fraction of the chains were broken before unembedding.

            **parameters:
                Parameters for the sampling method, specified by the child sampler.

        Returns:
            :class:`dimod.Response`

        """

        # use the given embedding method with the given parameters
        embedding_parameters = self._embedding_parameters
        candidates_parameters = self._candidates_parameters

        # solve the problem on the child system
        child = self.child

        # apply the embedding to the given problem to map it to the child sampler
        __, target_edgelist, target_adjacency = child.structure

        # get the embedding
        embedding = self.get_embedding(bqm, target_edgelist=target_edgelist,
                                    force_embed=force_embed,
                                    candidates_parameters=candidates_parameters,
                                    embedding_parameters=embedding_parameters)

        if bqm and not embedding:
            raise ValueError("no embedding found")

        bqm_embedded = embed_bqm(bqm, embedding, target_adjacency, chain_strength=chain_strength)

        if 'initial_state' in parameters:
            parameters['initial_state'] = _embed_state(embedding, parameters['initial_state'])

        response = child.sample(bqm_embedded, **parameters)

        # Store embedded response
        self._child_response = response

        return unembed_sampleset(response, embedding, source_bqm=bqm,
                                    chain_break_fraction=chain_break_fraction)
