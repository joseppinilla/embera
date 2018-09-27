""" Example comparing the embeddings obtained from a Layout-Agnostic
and a Layout-Aware embedding flow using minorminer.
"""
import os
import sys
import time
import pickle
import random
import networkx as nx
import dwave_networkx as dnx

from embedding_methods.utilities.graph_topologies import *
from embedding_methods.utilities.architectures.generators import *

from minorminer import find_embedding
from embedding_methods import dense
from embedding_methods import disperse
from embedding_methods.preprocess.diffusion_placer import find_candidates

def layout_agnostic(Sg, Tg, **kwargs):
    """ Layout-Agnostic Embedding method using minorminer
    """
    S_edgelist = list(Sg.edges())
    T_edgelist = list(Tg.edges())
    # Find a minor-embedding
    embedding = find_embedding(S_edgelist, T_edgelist, **kwargs)

    return [S_edgelist, T_edgelist, embedding]

def layout_diffuse(Sg, Tg, **kwargs):
    """ Layout-Aware Embedding method using minorminer with migration
    """
    S_edgelist = list(Sg.edges())
    T_edgelist = list(Tg.edges())
    # Layout of the problem graph
    layout = Sg.graph['pos'] if 'pos' in Sg.graph else nx.spring_layout(Sg)
    # Find a global placement for problem graph
    candidates = find_candidates(S_edgelist, Tg,
                                enable_migration=True,
                                layout=layout)
    # Find a minor-embedding using the initial chains from global placement
    embedding = find_embedding(S_edgelist, T_edgelist,
                                initial_chains=candidates,
                                **kwargs)
    return [S_edgelist, T_edgelist, embedding]

def layout_spread(Sg, Tg, **kwargs):
    """ Layout-Aware Embedding method using minorminer without migration
    """
    S_edgelist = list(Sg.edges())
    T_edgelist = list(Tg.edges())
    # Layout of the problem graph
    layout = Sg.graph['pos'] if 'pos' in Sg.graph else nx.spring_layout(Sg)
    # Find a global placement for problem graph
    candidates = find_candidates(S_edgelist, Tg,
                                enable_migration=False,
                                layout=layout)
    # Find a minor-embedding using the initial chains from global placement
    embedding = find_embedding(S_edgelist, T_edgelist,
                                initial_chains=candidates,
                                **kwargs)
    return [S_edgelist, T_edgelist, embedding]

""" BENCHMARKS SETUP """
#    target_graphs (list): NetworkX graph generators of sampler architectures
target_archs = [#faulty_arch(rainier_graph), rainier_graph,
                #faulty_arch(vesuvius_graph), vesuvius_graph,
                #faulty_arch(dw2x_graph), dw2x_graph,
                faulty_arch(dw2000q_graph)#, dw2000q_graph,
                #faulty_arch(p6_graph), p6_graph,
                #faulty_arch(p16_graph), p16_graph
                ]
# source_graphs (list): NetworkX graph generators
source_graphs = [#prune_graph(complete_graph), complete_graph,
                #prune_graph(complete_bipartite_graph), complete_bipartite_graph,
                random_graph, #prune_graph(random_graph),
                #rooks_graph, #prune_graph(rooks_graph),
                #grid_2d_graph#, prune_graph(grid_2d_graph),
                ]
# source_sizes (list): Integer values for the number of vertices in the source graph
source_sizes = [256]
# embed_methods (list): Embedding methods with a "find_embedding()" function interface
embed_methods = [layout_agnostic,
                layout_diffuse,
                layout_spread]
# embed_tries (int): Multiple tries to account for minor-embedding heuristic noise
embed_tries = 100
timeout = 200

""" LOGGING SETUP """
# Chosen directory for result files
resultsdir = "./results/"
# Previous stored results
if not os.path.exists(resultsdir):
    os.makedirs(resultsdir)
results_db = set(os.listdir(resultsdir))
def log(obj, filename):
    filepath = resultsdir + filename
    # Pickle allows dumping non-string keyed dictionaries
    with open(filepath, 'wb') as fp:
        pickle.dump(obj, fp)

"""  BENCHMARK LOOP """
for i_arch in target_archs:
    for i_graph in source_graphs:
        for i_size in source_sizes:
            for i_method in embed_methods:
                data = [i_arch.__name__, i_graph.__name__,
                        str(i_size), i_method.__name__]
                base = '-'.join(data)
                filename = base + '.pkl'
                # Bypass if results exist
                if (filename in results_db): continue
                else: print(filename)
                # Create graphs and run embedding
                results = {}
                Sg = i_graph(i_size)
                Tg = i_arch()
                for i in range(embed_tries):
                    start_time = time.time()
                    # Find a minor-embedding
                    result = i_method(Sg, Tg, timeout=timeout, tries=1, random_seed=i)
                    t_elap = time.time() - start_time
                    results[i] = [t_elap] + result
                log(results, filename)
