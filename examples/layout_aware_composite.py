""" Example of a Layout-Aware embedding flow using a layout-aware flow on a
target graph with 5% of the nodes removed.
"""
import networkx as nx
import matplotlib.pyplot as plt

from random import uniform

from embera.benchmark.topologies import pruned_graph_gen

from embera.utilities.architectures import drawing, generators
from embera.composites.layout_aware import LayoutAwareEmbeddingComposite

from dimod.reference.composites.structure import StructureComposite
from dimod.reference.samplers.simulated_annealing import SimulatedAnnealingSampler

# The corresponding graph of the D-Wave annealer with 0.95 qubit yield
J_RANGE = [-2.0,2.0]
Tg = pruned_graph_gen(generators.rainier_graph, node_yield=0.95)()

# A 4x4 grid problem graph
p = 4
Sg = nx.grid_2d_graph(p, p)
S_edgelist = list(Sg.edges())
# Create problem random values
for (u, v, data) in Sg.edges(data=True):
    data['weight'] = uniform(*J_RANGE)
h = {v:0.0 for v in Sg}
J = {(u,v):data['weight'] for u,v,data in Sg.edges(data=True)}
# Layout of the problem graph
layout = {v:v for v in Sg}

# Setup Composite
candidates_parameters = {'vicinity':0,
                        'd_lim':0.125,
                        'delta_t':0.4,
                        'enable_migration':True}
embedding_parameters = {'tries':20}

# Use any sampler and make structured (i.e. Simulated Annealing, Exact) or use structured sampler if available (i.e. D-Wave machine)
structsampler = StructureComposite(SimulatedAnnealingSampler(), Tg.nodes, Tg.edges)
sampler = LayoutAwareEmbeddingComposite(structsampler, layout=layout,
                                        candidates_parameters=candidates_parameters,
                                        embedding_parameters=embedding_parameters)

# Get results
response = sampler.sample_ising(h, J, num_reads=100)
energies = [datum.energy for datum in response.data()]

plt.figure(1)
_ = plt.hist(energies, bins=100)
plt.show()

# Retrieve embedding
embedding = sampler.get_ising_embedding(h, J)

print('sum: %s' % sum(len(v) for v in embedding.values()))
print('max: %s' % max(len(v)for v in embedding.values()))

plt.figure(2)
drawing.draw_architecture_embedding(Tg, embedding)
plt.title('Disperse Router')
plt.show()
