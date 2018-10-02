""" Example of a
"""

import networkx as nx
import matplotlib.pyplot as plt
from embedding_methods.utilities.architectures import drawing, generators
from embedding_methods.preprocess.complete_bipartite_placer import find_candidates

# A 2x2 grid problem graph
p, q = 4,3
Sg = nx.complete_bipartite_graph(p,q)
S_edgelist = list(Sg.edges())

# The corresponding graph of the D-Wave C4 annealer with 0.95 qubit yield
Tg = generators.faulty_arch(generators.rainier_graph, arch_yield=0.95)()
T_edgelist = list(Tg.edges())

# Systematically find the best candidates
candidates = find_candidates(S_edgelist, Tg)
#candidates = find_candidates(S_edgelist, Tg)

drawing.draw_architecture_embedding(Tg, candidates)
plt.title('Disperse Router')
plt.show()
