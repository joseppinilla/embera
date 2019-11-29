""" Example of a Layout-Aware embedding flow using disperse routing on a smaller
target graph with 5% of the nodes removed. This example uses the diffusion placer
without migration to demonstrate the nodes anchored to their candidates.
"""

import networkx as nx
import matplotlib.pyplot as plt
from embera.disperse import find_embedding
from embera.utilities.architectures import drawing, generators
from embera.preprocess.diffusion_placer import find_candidates

# A 2x2 grid problem graph
p = 2
Sg = nx.grid_2d_graph(p, p)
S_edgelist = list(Sg.edges())
# Layout of the problem graph
layout = {v:v for v in Sg}

# The corresponding graph of the D-Wave C4 annealer with 0.95 qubit yield
Tg = generators.faulty_arch(generators.rainier_graph, node_yield=0.95)()
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
print('max: %s' % max(len(v) for v in embedding.values()))

# Draw embedding (colours may vary from candidates.)
plt.figure(1)
drawing.draw_architecture_embedding(Tg, embedding, show_labels=True)
plt.title('Disperse Router')
plt.show()
