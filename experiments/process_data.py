import os
import json
import math
import pickle
import matplotlib.pyplot as plt

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
    total = 0
    N = len(embedding)
    for node, chain in embedding.items():
        chain_len = len(chain)
        total += chain_len
        if chain_len > max_chain:
            max_chain = chain_len
    avg_chain = total/N
    sum_deviations = 0
    for node, chain in embedding.items():
        chain_len = len(chain)
        deviation = (chain_len - avg_chain)**2
        sum_deviations += deviation
    std_dev = math.sqrt(sum_deviations/N)



    return max_chain, total, avg_chain, std_dev

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
        result = read_log_pickle(filename)
        setup, experiment = base.split('_')
        arch, graph, size, method = setup.split('-')
        try:
            i, i_prune = experiment.split('-')
        except:
            continue

        yield base, arch, graph, size, method, i, int(i_prune), result

if __name__== "__main__":

    if not os.path.exists(resultsdir):
        os.makedirs(resultsdir)
    if not os.path.exists(profilesdir):
        os.makedirs(profilesdir)
    if not os.path.exists(figsdir):
        os.makedirs(figsdir)


    results = {}

    for log in read_logs(resultsdir):
        base, arch, graph, size, method, i, prune, result = log

        if arch not in results:
            results[arch] = {}
        if graph not in results[arch]:
            results[arch][graph] = {}
        if size not in results[arch][graph]:
            results[arch][graph][size] = {}
        if prune not in results[arch][graph][size]:
            results[arch][graph][size][prune] = {}
            results[arch][graph][size][prune]['total'] = []
            results[arch][graph][size][prune]['time'] = []
            results[arch][graph][size][prune]['max'] = []
            results[arch][graph][size][prune]['avg'] = []
            results[arch][graph][size][prune]['stdev'] = []

        t_elap, embedding = result
        max_chain, total, avg_chain, std_dev = get_stats(embedding)
        if total != 0:
            results[arch][graph][size][prune]['total'].append(total)
            results[arch][graph][size][prune]['time'].append(t_elap)
            results[arch][graph][size][prune]['max'].append(max_chain)
            results[arch][graph][size][prune]['avg'].append(avg_chain)
            results[arch][graph][size][prune]['stdev'].append(std_dev)


    for i_arch, results_graph in results.items():
        for i_graph, results_size in results_graph.items():
            for i_size, results_prune in results_size.items():
                prune_labels = ['']
                for i_prune in sorted(results_prune.keys()):
                    prune_labels.append(i_prune)

                ticks = list(range(len(x_labels)+1))

                for i, metric in enumerate(['max','total','time','avg','stdev']):
                    plt.clf()
                    plt.figure(i)
                    figname = i_arch + '_' + i_graph + '_' + i_size + '_' + metric
                    data_points = []
                    for prune in prune_labels[1:]:
                        data_points.append(results_prune[prune][metric])
                    plt.boxplot(data_points)
                    plt.xticks(ticks, prune_labels, rotation='vertical')
                    plt.title(figname)
                    plt.tight_layout()
                    plt.savefig(figsdir + figname + '.png')
                    plt.show()
