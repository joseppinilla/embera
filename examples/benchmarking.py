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
import matplotlib.pyplot as plt
from minorminer import find_embedding

from embedding_methods.utilities.graph_topologies import *
from embedding_methods.utilities.architectures.generators import *

from embedding_methods import dense
from embedding_methods import disperse
from embedding_methods.preprocess.diffusion_placer import find_candidates

bar = progressbar.ProgressBar(maxval = )

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

""" BENCHMARKING SETUP """
#    target_graphs (list): NetworkX graph generators of sampler architectures
target_archs = [#faulty_arch(rainier_graph), rainier_graph,
                #faulty_arch(vesuvius_graph), vesuvius_graph,
                #faulty_arch(dw2x_graph), dw2x_graph,
                faulty_arch(dw2000q_graph), dw2000q_graph,
                #faulty_arch(p6_graph), p6_graph,
                faulty_arch(p16_graph), p16_graph
                ]
# source_graphs (list): NetworkX graph generators
source_graphs = [#prune_graph(complete_graph), complete_graph,
                #prune_graph(complete_bipartite_graph), complete_bipartite_graph,
                prune_graph(random_graph), random_graph,
                #prune_graph(rooks_graph), rooks_graph,
                #prune_graph(grid_2d_graph), grid_2d_graph
                ]
# source_sizes (list): Integer values for the number of vertices in the source graph
source_sizes = [64, 72, 80]
# embed_methods (list): Embedding methods with a "find_embedding()" function interface
embed_methods = [layout_agnostic,
                layout_diffuse,
                layout_spread]
# embed_tries (int): Multiple tries to account for minor-embedding heuristic noise
embed_tries = 200

"""  BENCHMARK LOOPS """
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
                    result = i_method(Sg, Tg, timeout=100, tries=1, random_seed=i)
                    t_elap = time.time() - start_time
                    results[i] = [t_elap] + result
                log(results, filename)

""" PROCESS DATA """
figsdir = "./figs/"
if not os.path.exists(figsdir):
    os.makedirs(figsdir)

def read_log_pickle(filename):
    fp = open(filename, 'rb')
    data = pickle.load(fp)
    fp.close()
    return data

def read_logs(filedir):
    for file in os.listdir(filedir):
        filename = os.path.join(filedir, file)
        base, _ = os.path.splitext(file)
        fp = open(filename, 'rb')
        results = pickle.load(fp)
        fp.close()
        arch, graph, size, method = base.split('-')
        yield file, arch, graph, size, method, results

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
    std_dev = math.sqrt(sum_deviations/N)

    return max_chain, min_chain, total, avg_chain, std_dev

# Create stats container
stats = {}
for log in read_logs(resultsdir):
    file, arch, graph, size, method, results = log
    if arch not in stats:
        stats[arch] = {}
    if graph not in stats[arch]:
        stats[arch][graph] = {}
    if size not in stats[arch][graph]:
        stats[arch][graph][size] = {}
    if method not in stats[arch][graph][size]:
        stats[arch][graph][size][method] = {}
        stats[arch][graph][size][method]['total'] = []
        stats[arch][graph][size][method]['time'] = []
        stats[arch][graph][size][method]['max'] = []
        stats[arch][graph][size][method]['avg'] = []
        stats[arch][graph][size][method]['stdev'] = []
        stats[arch][graph][size][method]['min'] = []
        stats[arch][graph][size][method]['valid'] = 0

    for key, result in results.items():
        t_elap, _, _, embedding = result
        if embedding:
            max_chain, min_chain, total, avg_chain, std_dev = get_stats(embedding)
            stats[arch][graph][size][method]['total'].append(total)
            stats[arch][graph][size][method]['time'].append(t_elap)
            stats[arch][graph][size][method]['max'].append(max_chain)
            stats[arch][graph][size][method]['avg'].append(avg_chain)
            stats[arch][graph][size][method]['stdev'].append(std_dev)
            stats[arch][graph][size][method]['min'].append(min_chain)
            stats[arch][graph][size][method]['valid'] += 1

for i_arch, stats_graph in stats.items():
    for i_graph, stats_size in stats_graph.items():
        for i_size, stats_method in stats_size.items():
            ticks = range(len(stats_method)+1)
            for i, metric in enumerate(['max','total','time','avg','stdev','min']):
                plt.clf()
                plt.figure(i)
                figname =   i_arch + '_' + i_graph + '_' + \
                            i_size + '_' + metric
                textstr =   'Arch: %s \n' % i_arch + \
                            'Graph: %s \n' % i_graph + \
                            'Size: %s \n' % i_size

                data_points = []
                method_labels = ['']
                for method_name, method_dict in stats_method.items():
                    data_points.append(method_dict[metric])
                    method_labels.append('%s_%s' % (method_name, method_dict['valid']) )
                wrapped_labels = [ '\n'.join(l.split('_')) for l in method_labels ]
                plt.boxplot(data_points)
                plt.xticks(ticks, wrapped_labels)
                side_text = plt.figtext(1, 0.5, textstr, fontsize=12)
                plt.tight_layout()
                ymin, ymax = plt.ylim()
                plt.ylim(ymin=0, ymax=ymax*1.1)
                plt.xlabel('Method')
                plt.ylabel(metric)
                plt.savefig(figsdir + figname + '.png', bbox_extra_artists=(side_text,),bbox_inches='tight')
                #plt.show()
