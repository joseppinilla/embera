""" Example comparing the embeddings obtained from a Layout-Agnostic
and a Layout-Aware embedding flow using minorminer.
"""
import time
import networkx as nx
import dwave_networkx as dnx
import matplotlib.pyplot as plt
from minorminer import find_embedding

from embedding_methods.utilities import graph_topologies
from embedding_methods.utilities.architectures import generators
from embedding_methods.utilities.architectures import drawing
from embedding_methods.preprocess.diffusion_placer import find_candidates

sizes = [50]
tries = 50

sum_results_agnostic = []
max_results_agnostic = []
time_results_agnostic = []
sum_results_aware = []
max_results_aware = []
time_results_aware = []

for size in sizes:
    print('>>SIZE: %s' % size)
    for i in range(tries):
        print('>>TRY: %s' % i)
        # Random graph of given size
        Sg = graph_topologies.random(size)
        S_edgelist = list(Sg.edges())

        # The corresponding graph of the D-Wave 2000Q annealer
        Tg = generators.dw2000q_graph()
        T_edgelist = list(Tg.edges())

        print('Layout-Agnostic')
        start_time = time.time()
        # Find a minor-embedding
        embedding = find_embedding(S_edgelist, T_edgelist)
        if embedding:
            agnostic_sum = sum(len(v) for v in embedding.values())
            agnostic_max = max(len(v)for v in embedding.values())
            agnostic_time = (time.time() - start_time)
            print('sum: %s' % agnostic_sum)
            print('max: %s' % agnostic_max)
            print("time: %ss" % agnostic_time)
            sum_results_agnostic.append(agnostic_sum)
            max_results_agnostic.append(agnostic_max)
            time_results_agnostic.append(agnostic_time)

        print('Layout-Aware')
        start_time = time.time()
        # Layout of the problem graph
        layout = nx.spring_layout(Sg)
        # Find a global placement for problem graph
        candidates = find_candidates(S_edgelist, Tg, layout=layout)
        # Find a minor-embedding using the initial chains from global placement
        guided_embedding = find_embedding(S_edgelist, T_edgelist, initial_chains=candidates)
        if guided_embedding:
            aware_sum = sum(len(v) for v in guided_embedding.values())
            aware_max = max(len(v)for v in guided_embedding.values())
            aware_time = (time.time() - start_time)
            print('sum: %s' % aware_sum)
            print('max: %s' % aware_max)
            print("time: %ss" % aware_time)
            sum_results_aware.append(aware_sum)
            max_results_aware.append(aware_max)
            time_results_aware.append(aware_time)

sum_mean_agnostic = sum(sum_results_agnostic)/len(sum_results_agnostic)
sum_mean_aware = sum(sum_results_aware)/len(sum_results_aware)
plt.plot(sum_results_agnostic, label='Agnostic %s' % sum_mean_agnostic)
plt.plot(sum_results_aware, label='Aware %s' % sum_mean_aware)
plt.legend()
plt.title('Sum')
plt.show()

max_mean_agnostic = sum(max_results_agnostic)/len(max_results_agnostic)
max_mean_aware = sum(max_results_aware)/len(max_results_aware)
plt.plot(max_results_agnostic, label='Agnostic %s' % max_mean_agnostic)
plt.plot(max_results_aware, label='Aware %s' % max_mean_aware)
plt.legend()
plt.title('Max')
plt.show()

time_mean_agnostic = sum(time_results_agnostic)/len(time_results_agnostic)
time_mean_aware = sum(time_results_aware)/len(time_results_aware)
plt.plot(time_results_agnostic, label='Agnostic %s' % time_mean_agnostic)
plt.plot(time_results_aware, label='Aware %s' % time_mean_aware)
plt.legend()
plt.title('Time')
plt.show()
