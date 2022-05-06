""" Example of a systematic embedding approach for complete bipartite graphs """

import matplotlib.pyplot as plt
from embera.algorithms.transform.graph import prune
from embera.architectures import drawing, generators
from embera.algorithms.embedding.complete_bipartite_placer import find_candidates

# Problem dimensions
p, q = 8, 6
# The corresponding graph of the D-Wave C4 (Rainier) annealer with 0.95 qubit yield
Tg = prune(generators.rainier_graph(coordinates=True), node_yield=0.95)
plt.subplot(2, 2, 1)
drawing.draw_architecture_yield(Tg, node_size=20)

# Systematically find the best candidates for K_{10,16} starting at row 3, col 6
origin = (3,6)
params = {'origin':origin, 'shores':True, 'orientation':0, 'show_faults':True}
(P, Q), faults = find_candidates((p, q), Tg, **params)
candidates = {**P, **Q}
plt.subplot(2, 2, 2)
drawing.draw_architecture_embedding(Tg, candidates, node_size=20, show_labels=True)

""" Example of a transformable embedding for complete bipartite graphs """
from embera.algorithms.embedding.complete_bipartite_placer import CompleteBipartitePlacer

# Create placer object from candidates and perform transformations
placer = CompleteBipartitePlacer.from_candidates((p,q), Tg, candidates)
assert(origin==placer.origin)

# ...or, create the placer object from scratch:
placer = CompleteBipartitePlacer((p,q), Tg)
(P, Q), faults = placer.run()

plt.subplot(2, 2, 3)
drawing.draw_architecture_embedding(Tg, {**P, **Q}, node_size=20, show_labels=True)

# and perform transformations:
placer.rotate()
placer.shuffle()

plt.subplot(2, 2, 4)
drawing.draw_architecture_embedding(Tg, {**P, **Q}, node_size=20, show_labels=True)
