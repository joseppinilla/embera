.. index-start-marker

embedding-methods
=============

``embedding-methods`` offers a collection of minor-embedding methods and utilities. These can be used as composites making part of the `D-Wave Ocean <http://dw-docs.readthedocs.io/en/latest/overview/stack.html#stack>`_ software stack, or from the interface functions, such as ``find_embedding()``. Additional resources are provided to generate graphs for existing and conceptual architectures of Ising samplers (e.g. D-Wave's Quantum Annealers) and preprocessing of the source and/or target graphs.

**Definition:**

*A graph G is a minor of H if G is isomorphic to a graph obtained from a subgraph of H by successively contracting edges.*

.. index-end-marker

Installation
------------
.. installation-start-marker
To install from source:

.. code-block:: bash

  python setup.py install

.. installation-end-marker

Example Usage
-------------

**Using dimod:**

.. dimod-start-marker

When using along with ``dimod``, either use the method-specific composites (i.e. ``MinorMinerEmbeddingComposite``, ``LayoutAwareEmbeddingComposite``, ...):

.. code-block:: python

    from embedding_methods.utilities.architectures import generators
    from dimod.reference.composites.structure import StructureComposite
    from embedding_methods.composites.minorminer import MinorMinerEmbeddingComposite
    from dimod.reference.samplers.simulated_annealing import SimulatedAnnealingSampler

    # Use the provided architectures
    target_graph = generators.dw2x_graph()

    # Use any sampler and make structured (i.e. Simulated Annealing, Exact) or use structured sampler if available (i.e. D-Wave machine)
    structsampler = StructureComposite(SimulatedAnnealingSampler(), target_graph.nodes, target_graph.edges)
    sampler = MinorMinerEmbeddingComposite(structsampler)

or the generic ``EmbeddingComposite``:

.. code-block:: python

    import minorminer
    from embedding_methods.utilities.architectures import generators
    from dimod.reference.samplers.random_sampler import RandomSampler
    from dimod.reference.composites.structure import StructureComposite
    from embedding_methods.composites.embedding import EmbeddingComposite


    # Use the provided architectures
    target_graph = generators.p6_graph()

    structsampler = StructureComposite(RandomSampler(), target_graph.nodes, target_graph.edges)
    sampler = EmbeddingComposite(structsampler, minorminer)

The composite is then compatible with the use of the ``sample()`` method as any other sampler.
In addition, a method ``get_embedding()`` is provided as an interface for the user to obtain a new embedding or retrieve the resulting embedding from which the problem was sampled.

.. dimod-end-marker

**Using function interface:**

.. examples-start-marker

Example comparing the embeddings obtained from a Layout-Agnostic and a Layout-Aware embedding flow using minorminer.

.. code-block:: python

  import networkx as nx
  import dwave_networkx as dnx
  import matplotlib.pyplot as plt
  from minorminer import find_embedding
  from embedding_methods.utilities.architectures import generators
  from embedding_methods.utilities.architectures import drawing
  from embedding_methods.preprocess.diffusion_placer import find_candidates

  # A 16x16 grid problem graph
  Sg = nx.grid_2d_graph(16, 16)
  S_edgelist = list(Sg.edges())
  # Layout of the problem graph
  layout = {v:v for v in Sg}

  # The corresponding graph of the D-Wave 2000Q annealer
  Tg = generators.dw2000q_graph()
  T_edgelist = list(Tg.edges())

  print('Layout-Agnostic')
  # Find a minor-embedding
  embedding = find_embedding(S_edgelist, T_edgelist)
  print('sum: %s' % sum(len(v) for v in embedding.values()))
  print('max: %s' % max(len(v)for v in embedding.values()))
  plt.figure(1)
  plt.title('Layout-Agnostic')
  drawing.draw_architecture_embedding(Tg, embedding)
  plt.tight_layout()

  print('Layout-Aware (enable_migration=True)')
  # Find a global placement for problem graph
  candidates = find_candidates(S_edgelist, Tg, layout=layout)
  # Find a minor-embedding using the initial chains from global placement
  guided_embedding = find_embedding(S_edgelist, T_edgelist, initial_chains=candidates)
  print('sum: %s' % sum(len(v) for v in guided_embedding.values()))
  print('max: %s' % max(len(v)for v in guided_embedding.values()))
  plt.figure(3)
  plt.title('Layout-Aware (enable_migration=True)')
  drawing.draw_architecture_embedding(Tg, guided_embedding)
  plt.tight_layout()

  print('Layout-Aware (enable_migration=False)')
  # Find a global placement for problem graph
  candidates = find_candidates(S_edgelist, Tg, layout=layout, enable_migration=False)
  # Find a minor-embedding using the initial chains from global placement
  guided_embedding = find_embedding(S_edgelist, T_edgelist, initial_chains=candidates)
  print('sum: %s' % sum(len(v) for v in guided_embedding.values()))
  print('max: %s' % max(len(v)for v in guided_embedding.values()))
  plt.figure(2)
  plt.title('Layout-Aware (enable_migration=False)')
  drawing.draw_architecture_embedding(Tg, guided_embedding)

  plt.show()


+----------------------------------------+------------------------------------------------------+-------------------------------------------------------+
|                                        |                                                      |                                                       |
|   .. image:: ./docs/layout_agnostic.png|   .. image:: ./docs/layout_aware_enable_migration.png|   .. image:: ./docs/layout_aware_disable_migration.png|
|      :width: 30%                       |      :width: 30%                                     |      :width: 30%                                      |
|                                        |                                                      |                                                       |
+----------------------------------------+------------------------------------------------------+-------------------------------------------------------+
Example of a Layout-Aware embedding flow using disperse routing on a smaller target graph with 5% of the nodes removed.
This example uses the diffusion placer without migration to demonstrate the anchored nodes.

.. code-block:: python

  import networkx as nx
  import matplotlib.pyplot as plt
  from embedding_methods.disperse import find_embedding
  from embedding_methods.utilities.architectures import drawing, generators
  from embedding_methods.preprocess.diffusion_placer import find_candidates

  # A 2x2 grid problem graph
  p = 2
  Sg = nx.grid_2d_graph(p, p)
  S_edgelist = list(Sg.edges())
  # Layout of the problem graph
  layout = {v:v for v in Sg}

  # The corresponding graph of the D-Wave C4 annealer with 0.95 qubit yield
  Tg = generators.faulty(generators.rainier_graph, arch_yield=0.95)
  T_edgelist = list(Tg.edges())
  # Find a global placement for problem graph
  candidates = find_candidates(S_edgelist, Tg, layout=layout, enable_migration=False)
  # Find a minor-embedding using the disperse router method
  embedding = find_embedding(S_edgelist, T_edgelist, initial_chains=candidates)

  print('sum: %s' % sum(len(v) for v in embedding.values()))
  print('max: %s' % max(len(v)for v in embedding.values()))

  drawing.draw_architecture_embedding(Tg, embedding)
  plt.show()

.. examples-end-marker
