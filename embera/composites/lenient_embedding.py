"""
A generic dimod composite_ to map unstructured problems to a structured_ sampler
allowing missing couplers between chains and using chain break methods to
recover configurations.

A structured_ sampler can only solve problems that map to a specific graph: the
Ising solver architecture is represented by a graph.

.. _composite: http://dimod.readthedocs.io/en/latest/reference/samplers.html
.. _minorminer: https://github.com/dwavesystems/minorminer
.. _structured: http://dimod.readthedocs.io/en/latest/reference/samplers.html#module-dimod.core.structured
.. _Chimera: http://dwave-system.readthedocs.io/en/latest/reference/intro.html#chimera

"""
import dimod
import minorminer
from dwave.embedding.transforms import embed_bqm, unembed_sampleset
from dimod.binary_quadratic_model import BinaryQuadraticModel
from dwave.embedding.chain_breaks import majority_vote, broken_chains
from dwave.embedding.exceptions import MissingEdgeError, MissingChainError, InvalidNodeError

class LenientEmbeddingComposite(dimod.ComposedSampler):

    def __init__(self, child_sampler, embedding_method=minorminer, **embedding_parameters):
        if not isinstance(child_sampler, dimod.Structured):
            raise dimod.InvalidComposition("EmbeddingComposite should only be applied to a Structured sampler")
        self._children = [child_sampler]
        self._embedding = None
        self._child_response = None
        self._embedding_method = embedding_method
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

    def get_embedding(self, bqm, target_edgelist=None, force_embed=False, **embedding_parameters):
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
        embedding_method = self._embedding_method
        self._embedding_parameters = embedding_parameters

        # add self-loops to edgelist to handle singleton variables
        source_edgelist = list(bqm.quadratic) + [(v, v) for v in bqm.linear]

        if target_edgelist is None:
            _, target_edgelist, _ = child.structure

        if force_embed or not self._embedding:
            embedding = embedding_method.find_embedding(source_edgelist, target_edgelist,**embedding_parameters)
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

        # solve the problem on the child system
        child = self.child

        # apply the embedding to the given problem to map it to the child sampler
        __, target_edgelist, target_adjacency = child.structure

        # get the embedding
        embedding = self.get_embedding(bqm, target_edgelist=target_edgelist,
                                    force_embed=force_embed,
                                    **embedding_parameters)

        if bqm and not embedding:
            raise ValueError("no embedding found")

        bqm_embedded = lenient_embed_bqm(bqm, embedding, target_adjacency, chain_strength=chain_strength)

        response = child.sample(bqm_embedded, **parameters)

        # Store embedded response
        self._child_response = response

        return unembed_sampleset(response, embedding, source_bqm=bqm,
                                    chain_break_fraction=chain_break_fraction)


def lenient_embed_bqm(source_bqm, embedding, target_adjacency, chain_strength=1.0,
              smear_vartype=None):
    """Embed a binary quadratic model onto a target graph even if couplers are missing.

    Args:
        source_bqm (:obj:`.BinaryQuadraticModel`):
            Binary quadratic model to embed.

        embedding (dict):
            Mapping from source graph to target graph as a dict of form {s: {t, ...}, ...},
            where s is a source-model variable and t is a target-model variable.

        target_adjacency (dict/:class:`networkx.Graph`):
            Adjacency of the target graph as a dict of form {t: Nt, ...},
            where t is a variable in the target graph and Nt is its set of neighbours.

        chain_strength (float, optional):
            Magnitude of the quadratic bias (in SPIN-space) applied between variables to create chains. Note
            that the energy penalty of chain breaks is 2 * `chain_strength`.

        smear_vartype (:class:`.Vartype`, optional, default=None):
            When a single variable is embedded, it's linear bias is 'smeared' evenly over the
            chain. This parameter determines whether the variable is smeared in SPIN or BINARY
            space. By default the embedding is done according to the given source_bqm.

    Returns:
        :obj:`.BinaryQuadraticModel`: Target binary quadratic model.

    """
    if smear_vartype is dimod.SPIN and source_bqm.vartype is dimod.BINARY:
        return embed_bqm(source_bqm.spin, embedding, target_adjacency,
                         chain_strength=chain_strength, smear_vartype=None).binary
    elif smear_vartype is dimod.BINARY and source_bqm.vartype is dimod.SPIN:
        return embed_bqm(source_bqm.binary, embedding, target_adjacency,
                         chain_strength=chain_strength, smear_vartype=None).spin

    # create a new empty binary quadratic model with the same class as source_bqm
    target_bqm = source_bqm.empty(source_bqm.vartype)

    # add the offset
    target_bqm.add_offset(source_bqm.offset)

    # start with the linear biases, spreading the source bias equally over the target variables in
    # the chain
    for v, bias in source_bqm.linear.items():

        if v in embedding:
            chain = embedding[v]
        else:
            raise MissingChainError(v)

        if any(u not in target_adjacency for u in chain):
            raise InvalidNodeError(v, next(u not in target_adjacency for u in chain))

        b = bias / len(chain)

        target_bqm.add_variables_from({u: b for u in chain})

    # next up the quadratic biases, spread the quadratic biases evenly over the available
    # interactions
    for (u, v), bias in source_bqm.quadratic.items():
        available_interactions = {(s, t) for s in embedding[u] for t in embedding[v] if s in target_adjacency[t]}

        if not available_interactions:
            continue

        b = bias / len(available_interactions)

        target_bqm.add_interactions_from((u, v, b) for u, v in available_interactions)

    for chain in embedding.values():

        # in the case where the chain has length 1, there are no chain quadratic biases, but we
        # none-the-less want the chain variables to appear in the target_bqm
        if len(chain) == 1:
            v, = chain
            target_bqm.add_variable(v, 0.0)
            continue

        quadratic_chain_biases = lenient_chain_to_quadratic(chain, target_adjacency, chain_strength)
        target_bqm.add_interactions_from(quadratic_chain_biases, vartype=dimod.SPIN)  # these are spin

        # add the energy for satisfied chains to the offset
        energy_diff = -sum(quadratic_chain_biases.values())
        target_bqm.add_offset(energy_diff)

    return target_bqm

def lenient_chain_to_quadratic(chain, target_adjacency, chain_strength):
    """Determine the quadratic biases that induce the given chain.

    Args:
        chain (iterable):
            The variables that make up a chain.

        target_adjacency (dict/:class:`networkx.Graph`):
            Should be a dict of the form {s: Ns, ...} where s is a variable
            in the target graph and Ns is the set of neighbours of s.

        chain_strength (float):
            The magnitude of the quadratic bias that should be used to create chains.

    Returns:
        dict[edge, float]: The quadratic biases that induce the given chain.

    Raises:
        ValueError: If the variables in chain do not form a connected subgraph of target.

    Examples:
        >>> chain = {1, 2}
        >>> target_adjacency = {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}
        >>> dimod.embedding.chain_to_quadratic(chain, target_adjacency, 1)
        {(1, 2): -1}

    """
    quadratic = {}  # we will be adding the edges that make the chain here
    queue = set(chain)

    # For each connected component
    while queue:
        # do a breadth first search
        seen = set()
        next_level = {queue.pop()}
        while next_level:
            this_level = next_level
            next_level = set()
            for v in this_level:
                if v not in seen:
                    seen.add(v)

                    for u in target_adjacency[v]:
                        if u not in chain:
                            continue
                        next_level.add(u)
                        if u != v and (u, v) not in quadratic:
                            quadratic[(v, u)] = -chain_strength
                        queue.discard(u)

    return quadratic
