import embera
import networkx as nx
import matplotlib.pyplot as plt

bench_graph_list = embera.benchmark.topologies.embera_bench()

nrows = 7
ncols = 4
fig, axs = plt.subplots(nrows, ncols, subplot_kw={'aspect':'equal'})
fig.set_size_inches(ncols , nrows)
for i, ax in enumerate(axs.flat):
    if i>=len(bench_graph_list): fig.delaxes(ax); continue
    Sg = bench_graph_list[i]
    pos = Sg.graph.setdefault('pos', nx.spring_layout(Sg))
    nx.draw(Sg, pos=pos, ax=axs.flat[i], node_size=1, width=0.2, edge_color='grey')

plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)

plt.show()
    
