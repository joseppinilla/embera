""" Example of a Layout-Aware embedding flow using disperse routing.
"""

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

# The corresponding graph of the D-Wave C4 annealer
Tg = generators.rainier_graph()
T_edgelist = list(Tg.edges())

# Find a global placement for problem graph
candidates = find_candidates(S_edgelist, Tg, layout=layout)
# Find a minor-embedding using the disperse router method
embedding = find_embedding(S_edgelist, T_edgelist, initial_chains=candidates)

print('sum: %s' % sum(len(v) for v in embedding.values()))
print('max: %s' % max(len(v)for v in embedding.values()))

drawing.draw_architecture_embedding(Tg, embedding)
plt.show()
