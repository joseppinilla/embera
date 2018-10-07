""" Example of preprocessing an input graph with layout information over a
pegasus architecture target graph.
"""
%matplotlib tk
import networkx as nx
import dwave_networkx as dnx
import matplotlib.pyplot as plt
from embedding_methods.utilities.architectures import drawing, generators
from embedding_methods.utilities.architectures.tiling import Tiling

p=3
Tg = dnx.generators.pegasus_graph(p,coordinates=True)
plt.figure(1)
#dnx.draw_pegasus(Tg, with_labels=True)

colors = {}
for tile, data in Tiling(Tg).tiles.items():
    colors[tile] = data.qubits
drawing.draw_architecture_embedding(Tg, colors)

Tg_int = dnx.generators.pegasus_graph(p)
plt.figure(2)
#dnx.draw_pegasus(Tg_int, with_labels=True)

colors = {}
for tile, data in Tiling(Tg_int).tiles.items():
    colors[tile] = data.qubits
drawing.draw_architecture_embedding(Tg_int, colors)

conv = dnx.generators.pegasus.pegasus_coordinates(p)
names = list(Tg_int.nodes)
new_names = { name:conv.tuple(name) for name in names }
Tg_int2c = nx.relabel_nodes(Tg_int, new_names, copy=True)


plt.figure(3)
#dnx.draw_pegasus(Tg_int2c, with_labels=True)
colors = {}
for tile, data in Tiling(Tg_int2c).tiles.items():
    colors[tile] = data.qubits
drawing.draw_architecture_embedding(Tg_int2c, colors)

nx.draw(Tg_int2c)



# colors = {}
# for tile, data in Tiling(Tg).tiles.items():
#     colors[tile] = data.qubits
#
# # Plot the graph and color each tile independently
# plt.clf()
# drawing.draw_architecture_embedding(Tg, colors)
# plt.show()

#
# %matplotlib tk
#
# import networkx as nx
# import dwave_networkx as dnx
# import matplotlib.pyplot as plt
# from minorminer import find_embedding
# #from embedding_methods.disperse import find_embedding
# from embedding_methods.utilities.architectures import drawing, generators
# from embedding_methods.preprocess.diffusion_placer import find_candidates
#
# from dwave_networkx.generators.pegasus import pegasus_coordinates
#
# from embedding_methods.utilities.architectures.tiling import Tiling
#
#
# # A 8x8 grid problem graph
# p = 2
# Sg = nx.grid_2d_graph(p, p)
# S_edgelist = list(Sg.edges())
# # Layout of the problem graph
# layout = {v:v for v in Sg}
#
#
# Tg = generators.faulty_arch(generators.p6_graph, arch_yield=1.0)(coordinates=True)
# T_edgelist = list(Tg.edges())
#
#
#
#
# colors = {}
# for tile, data in Tiling(Tg).tiles.items():
#     colors[tile] = data.qubits
#
# del colors[None]
#
# candidates = { v:qubits for v,qubits in colors.items() if qubits }
#
# candidates
#
# # Find a global placement for problem graph
# #candidates = find_candidates(S_edgelist, Tg, layout=layout, vicinity=0, enable_migration=True)
# # Find a minor-embedding using the disperse router method
# #embedding = find_embedding(S_edgelist, T_edgelist, initial_chains=candidates)
#
# # print('sum: %s' % sum(len(v) for v in embedding.values()))
# # print('max: %s' % max(len(v)for v in embedding.values()))
#
# plt.clf()
# drawing.draw_architecture_embedding(Tg, candidates, show_labels=True)
# plt.show()
#
#
# #drawing.draw_architecture_embedding(Tg, candidates, with_labels=True)
# #plt.figure(5)
#
# # pos = dnx.drawing.pegasus_layout(Tg)
# # for node, loc in pos.items():
# #     (u,w,k,z) = node
# #     if u==0:
# #         #print(pos)
# #         x,y = loc
# #         #pos[node] = (x+0.03,y-0.03)
# #         pos[node] = (x+0.055,y-0.055)
# # plt.figure(3)
# # plt.clf()
# # #nx.draw(Tg,node_size=20, pos=pos)
# # dnx.qubit_layout.draw_embedding(Tg, pos, candidates, node_size=30, show_labels=True)
# # #dnx.qubit_layout.draw_embedding(Tg, pos, candidates)
# # plt.gca().invert_yaxis()
# # plt.gca().invert_xaxis()
# # #plt.show()
# #
# #
# # # plt.title('Disperse Router')
# # # #plt.show()
