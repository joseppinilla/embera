embera
======

[![Build Status](https://travis-ci.org/joseppinilla/embera.svg?branch=master)](https://travis-ci.org/joseppinilla/embera)
[![Coverage Status](https://coveralls.io/repos/github/joseppinilla/embera/badge.svg?branch=master)](https://coveralls.io/github/joseppinilla/embera?branch=master)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

`embera` offers a collection of minor-Embedding Resources and Algorithms for QUBO/Ising-model sampling applications.

Why `embera`?
------------
The name `embera` is a nod to the [Emberá](https://minorityrights.org/minorities/embera/) indigeneous communities.

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

A set of examples are provided in the `examples` folder.
