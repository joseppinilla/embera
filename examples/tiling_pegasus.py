""" Example of tiling a Pegasus architecture graph.
"""
import dwave_networkx as dnx
import matplotlib.pyplot as plt
from embedding_methods.utilities.architectures import drawing, generators
from embedding_methods.utilities.architectures.tiling import Tiling

p=3
Tg = generators.p6_graph()
colours = {}
for tile, data in Tiling(Tg).tiles.items():
    if data.qubits:
        colours[tile] = data.qubits

drawing.draw_architecture_embedding(Tg, colours, show_labels=True)
plt.show()
