""" Example of a systematic embedding approach for complete bipartite graphs.
"""

import matplotlib.pyplot as plt
from embedding_methods.utilities.architectures import drawing, generators
from embedding_methods.preprocess.complete_bipartite_placer import CompleteBipartitePlacer, find_candidates

# Problem dimensions
p, q = 10, 6
# The corresponding graph of the D-Wave C4 (Rainier) annealer with 0.95 qubit yield
Tg = generators.faulty_arch(generators.rainier_graph, node_yield=0.95)(coordinates=True)
plt.figure(0)
drawing.draw_architecture_yield(Tg, node_size=20)

# Systematically find the best candidates for K_{10,16} starting at row 3, col 6
origin = (6,4)
(P, Q), faults = find_candidates((p, q), Tg, origin=origin, shores=True, show_faults=True)
print('Faults at: %s' % faults)
candidates = {**P, **Q}
plt.figure(1)
drawing.draw_architecture_embedding(Tg, candidates, node_size=20, show_labels=True)
plt.show()

# Create placer object from candidates and perform transformations
placer = CompleteBipartitePlacer.from_candidates((p,q), Tg, candidates)
assert(placer.origin==origin)
placer.rotate()
placer.shuffle()
assert(placer.origin==origin)
plt.figure(2)
drawing.draw_architecture_embedding(Tg, {**placer.P, **placer.Q}, node_size=20, show_labels=True)
plt.show()

plt.figure(3)
placer.sort()
drawing.draw_architecture_embedding(Tg, {**placer.P, **placer.Q}, node_size=20, show_labels=True)
