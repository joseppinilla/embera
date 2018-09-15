.. index-start-marker

embedding-methods
=============

`embedding-methods` offers a collection of minor-embedding methods and utilities.

A graph G is a minor of H if G is isomorphic to a graph obtained from a subgraph of H by successively contracting edges.

.. index-end-marker

Installation
------------
.. installation-start-marker
To install from source:

.. code-block:: bash

  python setup.py install
  
.. installation-end-marker

Examples
--------
.. examples-start-marker

Example comparing the embeddings obtained from a Layout-Agnostic and a Layout-Aware embedding flow.

.. code-block:: python

  import networkx as nx
  import dwave_networkx as dnx
  from minorminer import find_embedding
  from embedding_methods.global_placement.diffusion_based import find_candidates

  # A 16x16 grid problem graph
  Sg = nx.grid_2d_graph(16, 16)
  topology = {v:v for v in Sg}
  S_edgelist = list(Sg.edges())

  # A C16 chimera target graph
  Tg = dnx.chimera_graph(16, coordinates=True)
  T_edgelist = list(Tg.edges())

  print('Layout-Agnostic')
  # Find a minor-embedding
  embedding = find_embedding(S_edgelist, T_edgelist)
  print('sum: %s' % sum(len(v) for v in embedding.values()))
  print('max: %s' % max(len(v)for v in embedding.values()))

  print('Layout-Aware')
  # Find a global placement for problem graph
  candidates = find_candidates(S_edgelist, Tg, topology=topology)
  # Find a minor-embedding using the initial chains from global placement
  guided_embedding = find_embedding(S_edgelist, T_edgelist, initial_chains=candidates)
  print('sum: %s' % sum(len(v) for v in guided_embedding.values()))
  print('max: %s' % max(len(v)for v in guided_embedding.values()))

Example of a layout aware embedding flow.

.. code-block:: python

  import networkx as nx
  import dwave_networkx as dnx
  from embedding_methods.topological import find_embedding
  from embedding_methods.global_placement.diffusion_based import find_candidates

  # A 4x4 grid problem graph
  Sg = nx.grid_2d_graph(2, 2)
  topology = {v:v for v in Sg}
  S_edgelist = list(Sg.edges())

  # A C4 chimera target graph
  Tg = dnx.chimera_graph(4, coordinates=True)
  T_edgelist = list(Tg.edges())

  # Find a global placement for problem graph
  candidates = find_candidates(S_edgelist, Tg, topology=topology)
  # Find a minor-embedding using the topological method
  embedding = find_embedding(S_edgelist, T_edgelist, initial_chains=candidates)

  print(len(embedding))  # 3, one set for each variable in the triangle
  print(embedding)
  print('sum: %s' % sum(len(v) for v in embedding.values()))
  print('max: %s' % max(len(v)for v in embedding.values()))
  
.. examples-end-marker
