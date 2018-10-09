""" Example of a systematic embedding approach for complete bipartite graphs.
"""

import matplotlib.pyplot as plt
from embedding_methods.utilities.architectures import drawing, generators
from embedding_methods.preprocess.complete_bipartite_placer import find_candidates

# The corresponding graph of the D-Wave C4 annealer with 0.95 qubit yield
Tg = generators.faulty_arch(generators.rainier_graph, arch_yield=0.95)()

# Systematically find the best candidates
p, q = 10, 16
candidates = find_candidates((p, q), Tg)

drawing.draw_architecture_embedding(Tg, candidates)
plt.title('Bipartite Embedding')
plt.show()
