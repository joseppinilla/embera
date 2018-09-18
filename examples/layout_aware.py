""" Example of a layout aware embedding flow.
"""

import networkx as nx
import matplotlib.pyplot as plt
from embedding_methods.architectures import drawing, generators
from embedding_methods.topological import find_embedding
from embedding_methods.global_placement.diffusion_based import find_candidates

# A 3x3 grid problem graph
Sg = nx.grid_2d_graph(3, 3)
S_edgelist = list(Sg.edges())
# Layout of the problem graph
topology = {v:v for v in Sg}

# The corresponding graph of the D-Wave C4 annealer
Tg = generators.rainier_graph()
T_edgelist = list(Tg.edges())

# Find a global placement for problem graph
candidates = find_candidates(S_edgelist, Tg, topology=topology)
# Find a minor-embedding using the topological method
embedding = find_embedding(S_edgelist, T_edgelist, initial_chains=candidates)

print('sum: %s' % sum(len(v) for v in embedding.values()))
print('max: %s' % max(len(v)for v in embedding.values()))

drawing.draw_architecture_embedding(Tg, embedding)
plt.show()
