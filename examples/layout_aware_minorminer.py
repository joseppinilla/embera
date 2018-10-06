""" Example comparing the embeddings obtained from a Layout-Agnostic
and a Layout-Aware embedding flow using minorminer.
"""

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
