#TEMP: standalone test

import sys
import pulp
import random
import warnings

import networkx as nx
import dwave_networkx as dnx
import matplotlib.pyplot as plt

from math import floor, sqrt
from heapq import heappop, heappush

from embedding_methods.topological.topological import find_embedding, find_candidates

def get_stats(embedding):
    max_chain = 0
    min_chain = sys.maxsize
    total = 0

    N = len(embedding)
    for chain in embedding.values():
        chain_len = len(chain)
        total += chain_len
        if chain_len > max_chain:
            max_chain = chain_len
        if chain_len < min_chain:
            min_chain =  chain_len
    avg_chain = total/N
    sum_deviations = 0
    for chain in embedding.values():
        chain_len = len(chain)
        deviation = (chain_len - avg_chain)**2
        sum_deviations += deviation
    std_dev = sqrt(sum_deviations/N)

    return max_chain, min_chain, total, avg_chain, std_dev

if __name__ == "__main__":

    import time
    import traceback

    verbose = 2

    p = 2
    Sg = nx.grid_2d_graph(p, p)
    topology = {v:v for v in Sg}

    #S = nx.cycle_graph(p)
    #topology = nx.circular_layout(S)

    #S = nx.complete_graph(p)
    #topology = nx.spring_layout(S)

    m = 3
    Tg = dnx.chimera_graph(m, coordinates=True) #TODO: Needs coordinates?
    #Tg = dnx.pegasus_graph(m, coordinates=True)


    S_edgelist = list(Sg.edges())
    T_edgelist = list(Tg.edges())

    try:
        candidates = find_candidates(S_edgelist, Tg,
                                     topology=topology,
                                     enable_migration=True,
                                     verbose=verbose)
        embedding = find_embedding(S_edgelist, T_edgelist,
                                    initial_chains=candidates,
                                    verbose=verbose)
        print('Layout:\n%s' % str(get_stats(embedding)))
    except:
        traceback.print_exc()

    # import minorminer
    #
    # t_start = time.time()
    # mm_embedding = minorminer.find_embedding( S_edgelist, T_edgelist,
    #                                 initial_chains=candidates,
    #                                 verbose=verbose)
    # t_end = time.time()
    # t_elap = t_end-t_start
    # print('MinorMiner:\n%s in %s' % (str(get_stats(mm_embedding)), t_elap) )
    #
    # t_start = time.time()
    # mm_embedding = minorminer.find_embedding( S_edgelist, T_edgelist,
    #                                 #initial_chains=candidates,
    #                                 verbose=verbose)
    # t_end = time.time()
    # t_elap = t_end-t_start
    # print('MinorMiner:\n%s in %s' % (str(get_stats(mm_embedding)), t_elap) )




    plt.clf()
    dnx.draw_chimera_embedding(Tg, embedding)
    #dnx.draw_pegasus_embedding(Tg, embedding)
    plt.show()
