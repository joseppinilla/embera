""" Example of preprocessing an input graph with layout information over a
pegasus architecture target graph.
"""

%matplotlib tk

import dwave_networkx as dnx
import matplotlib.pyplot as plt
from embedding_methods.utilities.architectures import drawing, generators
from embedding_methods.utilities.architectures.tiling import Tiling

Tg = dnx.generators.pegasus_graph(6)

colors = {}
for tile, data in Tiling(Tg).tiles.items():
    colors[tile] = data.qubits

# Plot the graph and color each tile independently
plt.clf()
drawing.draw_architecture_embedding(Tg, colors)
plt.show()
