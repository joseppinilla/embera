import os
import sys
import json
import math
import pickle

import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['axes.formatter.useoffset'] = False

from collections import Counter, OrderedDict

from embedding_methods.architectures import *


resultsdir = "./results/"
profilesdir = "./profiles/"
figsdir = "./figs/"

""" Process Data """
def chain_length_histo(embedding):
    histo = Counter()
    for chain in embedding.values():
        key = len(chain)
        histo[key] += 1
    return histo

def read_log_json(filename):
    fp = open(filename, 'r')
    data = json.load(fp)
    fp.close()
    return data

def read_log_pickle(filename):
    fp = open(filename, 'rb')
    data = pickle.load(fp)
    fp.close()
    return data

def get_stats(embedding):
    max_chain = 0
    min_chain = sys.maxsize
    total = 0
    N = len(embedding)
    for node, chain in embedding.items():
        chain_len = len(chain)
        total += chain_len
        if chain_len > max_chain:
            max_chain = chain_len
        if chain_len < min_chain:
            min_chain =  chain_len
    avg_chain = total/N
    sum_deviations = 0
    for node, chain in embedding.items():
        chain_len = len(chain)
        deviation = (chain_len - avg_chain)**2
        sum_deviations += deviation
    std_dev = math.sqrt(sum_deviations/N)

    return max_chain, min_chain, total, avg_chain, std_dev

def plot_histo(title, result):
    for i, result in results.items():
        time, embedding = result
        if embedding:
            histo = chain_length_histo(embedding)
            plt.bar(list(histo.keys()), histo.values())
            plt.xticks(list(histo.keys()))
            plt.title(title)
            plt.show()
            #draw(T, embedding)
            #plt.show()

def read_logs(filedir):
    for file in os.listdir(filedir):
        filename = os.path.join(filedir, file)
        base, ext = os.path.splitext(file)
        results = read_log_pickle(filename)
        arch, fault, graph, size, prune, method = base.split('-')

        yield file, arch, fault, graph, size, prune, method, results

# for log in read_logs(resultsdir):
#     file, arch, fault, graph, size, prune, method, results = log
#     for i in range(200):
#         if str(i) not in results:
#             results[i] = [0.0, {}]
#             with open(file, 'wb') as fp:
#                 pickle.dump(results, fp)



if __name__== "__main__":

    if not os.path.exists(resultsdir):
        os.makedirs(resultsdir)
    if not os.path.exists(profilesdir):
        os.makedirs(profilesdir)
    if not os.path.exists(figsdir):
        os.makedirs(figsdir)


    stats = {}

    for log in read_logs(resultsdir):
        file, arch, fault, graph, size, prune, method, results = log
        if arch not in stats:
            stats[arch] = {}
        if graph not in stats[arch]:
            stats[arch][graph] = {}
        if size not in stats[arch][graph]:
            stats[arch][graph][size] = {}
        if prune not in stats[arch][graph][size]:
            stats[arch][graph][size][prune] = {}
            stats[arch][graph][size][prune]['total'] = []
            stats[arch][graph][size][prune]['time'] = []
            stats[arch][graph][size][prune]['max'] = []
            stats[arch][graph][size][prune]['avg'] = []
            stats[arch][graph][size][prune]['stdev'] = []
            stats[arch][graph][size][prune]['min'] = []

        valid = 0
        for key, result in results.items():
            t_elap, embedding = result
            if embedding:
                max_chain, min_chain, total, avg_chain, std_dev = get_stats(embedding)
                stats[arch][graph][size][prune]['total'].append(total)
                stats[arch][graph][size][prune]['time'].append(t_elap)
                stats[arch][graph][size][prune]['max'].append(max_chain)
                stats[arch][graph][size][prune]['avg'].append(avg_chain)
                stats[arch][graph][size][prune]['stdev'].append(std_dev)
                stats[arch][graph][size][prune]['min'].append(min_chain)
                valid+=1
        stats[arch][graph][size][prune]['success'] = valid/len(results)


    for i_arch, stats_graph in stats.items():
        for i_graph, stats_size in stats_graph.items():
            for i_size, stats_prune in stats_size.items():

                n = int(i_size)
                if i_graph=='complete':
                    edges = ((n**2)-n)/2
                if i_graph=='bipartite':
                    edges = (n**2)/4
                if i_graph=='grid2d':
                    edges = 2*n - 2*math.sqrt(n)

                prune_labels = ['']
                prune_ordered = ['']
                for i_prune in sorted(stats_prune.keys(), key=lambda x: int(x)):
                    prune_ordered.append(i_prune)
                    pct = int(i_prune)/edges
                    prune_labels.append("%.3f" % pct)

                ticks = list(range(len(prune_labels)+1))

                for i, metric in enumerate(['max','total','time','avg','stdev','min']):
                    plt.clf()
                    plt.figure(i)
                    figname = i_arch + '_' + i_graph + '_' + i_size + '_' + metric
                    data_points = []
                    for prune in prune_ordered[1:]:
                        data_points.append(stats_prune[prune][metric])
                    plt.boxplot(data_points)
                    plt.xticks(ticks, prune_labels, rotation='vertical')
                    plt.title(figname)
                    plt.tight_layout()
                    plt.savefig(figsdir + figname + '.png')
                    plt.show()
