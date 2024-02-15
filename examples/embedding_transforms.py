import embera
import minorminer
import networkx as nx
import dwave_networkx as dnx
import matplotlib.pyplot as plt

S = nx.complete_graph(11)
T = dnx.chimera_graph(7)
embedding = minorminer.find_embedding(S,T)
_ = plt.figure(figsize=(4,4))
dnx.draw_chimera_embedding(T,embedding,node_size=10)
origin = (2,3)
new_embedding = embera.transform.embedding.translate(T,embedding,origin)
_ = plt.figure(figsize=(4,4))
dnx.draw_chimera_embedding(T,new_embedding,node_size=10)
