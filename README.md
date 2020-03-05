embera
======

[![Build Status](https://travis-ci.org/joseppinilla/embera.svg?branch=master)](https://travis-ci.org/joseppinilla/embera)
[![Coverage Status](https://coveralls.io/repos/github/joseppinilla/embera/badge.svg?branch=master)](https://coveralls.io/github/joseppinilla/embera?branch=master)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

`embera` offers a collection of minor-Embedding Resources and Algorithms for QUBO/Ising-model sampling applications.

Why `embera`?
------------
The name `embera` is a nod to the Emberá indigineous communities. Learn more about them [here](https://en.wikipedia.org/wiki/Ember%C3%A1) or [here](https://minorityrights.org/minorities/embera/).

The reason for `embera` as an open-source project lies in the limited  connectivity of practical quantum (or digital) annealing samplers, which means that a straightforward one-to-one mapping, from variables to qubits, is not likely to lead to a valid implementation, and may require the use of qubit _chains_. A _chain_ is an extension of a problem vertex over multiple connected qubits. Finding a good solution to the problem is vital for at
least two reasons: (a) the capabilities of the mapping algorithm
can determine the size (or complexity) of the problems that can be solved,
especially in the presence of defective qubits, and (b) the quality (energy levels) of the samples heavily depends on the structure of the mapping.

A mapping of the problem graph _G_={**P**,**E**}, with nodes **P** and edges **E**, to the sampler graph _H_={**Q**,**C**}, with qubits **Q** and couplers **C**, can be formulated as a _minor-embedding_ problem.

**Definition:**

*A graph G is a minor of H if G is isomorphic to a graph obtained from a
subgraph of H by successively contracting edges.*

**Resources:**
* Generate and draw network graphs of existing and conceptual `architectures` of QUBO/Ising-Model samplers; e.g. D-Wave's Quantum Annealers.
* Generate network graphs of real-world and arbitrary problem `topologies` to be embedded onto the samplers, as well as node and edge weight `parameters` assignment.
* Provide an easy to use `EmberaDataBase` to store and load problem
configurations and results; i.e. `BinaryQuadraticModel`, `Embedding`, `SampleSet`, `Report`.

**Algortihms:**
* [Layout-aware minor-embedding](https://doi.org/10.1007/978-3-030-20656-7_7) to take advantage of inherent topological information of the problem graph.
* Preprocessing of the source and/or target graphs for context-aware
embedding methods.
* Systematic minor-embedding methods for bipartite and other layered network graphs.
* Graph metric algorithms for benchmarking of the embedding results through proxy metrics and sample distribution evaluation.

Some of our methods can be used as composites making part of the
[D-Wave Ocean](https://ocean.dwavesys.com/) software stack or, otherwise,
from their interface functions, such as `find_embedding()`.

Installation
------------

To install from source:

``` bash
python setup.py install
```

Example Usage
-------------

**Using dimod:**

When using along with [dimod](https://github.com/dwavesystems/dimod), either use the method-specific composites
(i.e. `MinorMinerEmbeddingComposite`, `LayoutAwareEmbeddingComposite`,
...) with the corresponding method parameters:

``` python
from embera.architectures import generators
from dimod.reference.composites.structure import StructureComposite
from embera.composites.minorminer import MinorMinerEmbeddingComposite
from dimod.reference.samplers.simulated_annealing import SimulatedAnnealingSampler

# Use the provided architectures
target_graph = generators.dw2x_graph()

# Use any sampler and make structured (i.e. Simulated Annealing, Exact) or use structured sampler if available (i.e. D-Wave machine)
structsampler = StructureComposite(SimulatedAnnealingSampler(), target_graph.nodes, target_graph.edges)
sampler = MinorMinerEmbeddingComposite(structsampler, tries=20)
```

or the generic `EmbeddingComposite`:

``` python
import minorminer
from embera.architectures import generators
from dimod.reference.samplers.random_sampler import RandomSampler
from dimod.reference.composites.structure import StructureComposite
from embera.composites.embedding import EmbeddingComposite

# Use the provided architectures
target_graph = generators.p6_graph()

structsampler = StructureComposite(RandomSampler(), target_graph.nodes, target_graph.edges)
sampler = EmbeddingComposite(structsampler, minorminer, random_seed=42)
```

The composite is then compatible with the use of the `sample()` method
as any other sampler. In addition, a method `get_embedding()` is
provided as an interface for the user to obtain a new embedding or
retrieve the resulting embedding from which the problem was sampled.
Using `get_embedding(get_new=True)` forces a a new run of the chosen
embedding algorithm.

**Using function interface:**

Example comparing the embeddings obtained from a Layout-Agnostic and a
Layout-Aware embedding flow using minorminer.

``` python
import networkx as nx
import dwave_networkx as dnx
import matplotlib.pyplot as plt
from minorminer import find_embedding
from embera.architectures import generators
from embera.architectures import drawing
from embera.preprocess.diffusion_placer import find_candidates

# A 16x16 grid problem graph
Sg = nx.grid_2d_graph(16, 16)
S_edgelist = list(Sg.edges())
# Layout of the problem graph
layout = {v:v for v in Sg}

# The corresponding graph of the D-Wave 2000Q annealer
Tg = generators.dw2000q_graph()
# or other graph architectures
# Tg = generators.p16_graph()
T_edgelist = list(Tg.edges())

print('Layout-Agnostic')
# Find a minor-embedding
embedding = find_embedding(S_edgelist, T_edgelist)
print('sum: %s' % sum(len(v) for v in embedding.values()))
print('max: %s' % max(len(v) for v in embedding.values()))
plt.figure(1)
plt.title('Layout-Agnostic')
drawing.draw_architecture_embedding(Tg, embedding)
plt.tight_layout()

print('Layout-Aware (enable_migration=True)')
# Find a global placement for problem graph
candidates = find_candidates(S_edgelist, Tg, layout=layout)
# Find a minor-embedding using the initial chains from global placement
migrated_embedding = find_embedding(S_edgelist, T_edgelist, initial_chains=candidates)
print('sum: %s' % sum(len(v) for v in migrated_embedding.values()))
print('max: %s' % max(len(v) for v in migrated_embedding.values()))
plt.figure(2)
plt.title('Layout-Aware (enable_migration=True)')
drawing.draw_architecture_embedding(Tg, migrated_embedding)
plt.tight_layout()

print('Layout-Aware (enable_migration=False)')
# Find a global placement for problem graph
candidates = find_candidates(S_edgelist, Tg, layout=layout, enable_migration=False)
# Find a minor-embedding using the initial chains from global placement
guided_embedding = find_embedding(S_edgelist, T_edgelist, initial_chains=candidates)
print('sum: %s' % sum(len(v) for v in guided_embedding.values()))
print('max: %s' % max(len(v) for v in guided_embedding.values()))
plt.figure(3)
plt.title('Layout-Aware (enable_migration=False)')
drawing.draw_architecture_embedding(Tg, guided_embedding)

plt.show()
```


| Agnostic | Aware w/ Migration | Aware w/o Migration |
:----------:|:--------------:|:-----:|
| ![](/docs/layout_agnostic.png)| ![](/docs/layout_aware_enable_migration.png)|![](/docs/layout_aware_disable_migration.png)|

Example of a Layout-Aware embedding flow using disperse routing on a
smaller target graph with 5% of the nodes removed. This example uses the
diffusion placer without migration to demonstrate the anchored nodes.

``` python
import networkx as nx
import matplotlib.pyplot as plt
from embera.disperse import find_embedding
from embera.benchmark.topologies import pruned_graph_gen
from embera.architectures import drawing, generators
from embera.preprocess.diffusion_placer import find_candidates

# A 2x2 grid problem graph
p = 2
Sg = nx.grid_2d_graph(p, p)
S_edgelist = list(Sg.edges())
# Layout of the problem graph
layout = {v:v for v in Sg}

# The corresponding graph of the D-Wave C4 annealer with 0.95 qubit yield
Tg = pruned_graph_gen(generators.rainier_graph, node_yield=0.95)()
T_edgelist = list(Tg.edges())
# Find a global placement for problem graph
candidates = find_candidates(S_edgelist, Tg, layout=layout, enable_migration=False)

# Draw candidates as if embedded
plt.figure(0)
drawing.draw_architecture_embedding(Tg, candidates, show_labels=True)
plt.title('Candidates')

# Find a minor-embedding using the disperse router method
embedding = find_embedding(S_edgelist, T_edgelist, initial_chains=candidates)

print('sum: %s' % sum(len(v) for v in embedding.values()))
print('max: %s' % max(len(v)for v in embedding.values()))

# Draw embedding (colours may vary from candidates.)
plt.figure(1)
drawing.draw_architecture_embedding(Tg, embedding, show_labels=True)
plt.title('Disperse Router')
plt.show()
```

Example of tiling a Pegasus architecture graph.

``` python
import dwave_networkx as dnx
import matplotlib.pyplot as plt
from embera.architectures import drawing, generators
from embera.architectures.tiling import Tiling

# Pegasus graph
Tg = generators.p6_graph()

# Tile graph and gather qubit assignments
colours = {}
for tile, data in Tiling(Tg).tiles.items():
    if data.qubits:
        colours[tile] = data.qubits

# Use embedding drawing to show qubit assignments
drawing.draw_architecture_embedding(Tg, colours, show_labels=True)
plt.show()
```
