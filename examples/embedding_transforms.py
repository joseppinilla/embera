import embera
import minorminer
import networkx as nx
import dwave_networkx as dnx
import matplotlib.pyplot as plt

import pulp
pulp.__version__

################################################################################
S = nx.complete_graph(11)
T = dnx.chimera_graph(7)
embedding = minorminer.find_embedding(S,T)
_ = plt.figure(figsize=(4,4))
dnx.draw_chimera_embedding(T,embedding,node_size=10)

################################################################################
origin = (2,3)
tra_embedding = embera.transform.embedding.translate(T,embedding,origin)
_ = plt.figure(figsize=(4,4))
dnx.draw_chimera_embedding(T,tra_embedding,node_size=10)

################################################################################
rot_embedding = embera.transform.embedding.rotate(T,embedding)
_ = plt.figure(figsize=(4,4))
dnx.draw_chimera_embedding(T,rot_embedding,node_size=10)

################################################################################
mir_embedding = embera.transform.embedding.mirror(T,rot_embedding,0)
_ = plt.figure(figsize=(4,4))
dnx.draw_chimera_embedding(T,mir_embedding,node_size=10)

################################################################################
P4 = dnx.pegasus_graph(4)
Pembedding = minorminer.find_embedding(S,P4)
_ = plt.figure(figsize=(4,4))
dnx.draw_pegasus_embedding(P4,Pembedding,node_size=10)


P4frag = dnx.generators.pegasus.get_tuple_fragmentation_fn(P4)
P4defrag = dnx.generators.pegasus.get_tuple_defragmentation_fn(P4)

P4frag()


import minorminer.busclique
P4_busgraph = minorminer.busclique.busgraph_cache(P4)
m, n, t, nodes, edges = P4_busgraph._graph.fragment_graph_spec()

f = dnx.chimera_graph(m, n=n, t=t, node_list=nodes, edge_list=edges)
f_emb = {
    k : P4_busgraph._graph.fragment_nodes(c)
    for k, c in P4_busgraph._graph.delabel(Pembedding).items()
}
_ = plt.figure(figsize=(16,16))
dnx.draw_chimera_embedding(f, f_emb,node_size=10)

mir_f_emb = embera.transform.embedding.rotate(f,f_emb)
_ = plt.figure(figsize=(4,4))
dnx.draw_chimera_embedding(f,mir_f_emb,node_size=10)


P4_mir_embedding = {
            k : P4_busgraph._graph.fragment_nodes(c)
            for k, c in P4_busgraph._graph.delabel(mir_f_emb).items()
            }

mir_f_emb

_ = plt.figure(figsize=(4,4))
dnx.draw_pegasus_embedding(P4,P4_mir_embedding,node_size=10)



defrag(A)
