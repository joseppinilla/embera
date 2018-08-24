import os
import json
import pickle
import matplotlib.pyplot as plt

from collections import Counter

from embedding_methods.architectures import *


filedir = "./results/"
profilesdir = "./profiles/"

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
    for node, chain in embedding.items():
        chain_len = len(chain)
        total += chain_len
        if chain_len > max_chain:
            max_chain = chain_len
    return max_chain, total

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
        setup, experiment = base.split('_')
        arch, graph, size, method = setup.split('-')
        i, i_size, i_prune = experiment.splt('-')

        yield base, arch, graph, size, method, i, i_size, i_prune, results




if __name__== "__main__":
    total_results_arch = {}
    max_results_arch = {}
    for log in read_logs(profilesdir):
        base, arch, graph, size, method, i, i_size, i_prune, results = log

        if not max_results_arch[arch]:
            max_results_arch[arch] = {}
        if not total_results_arch[arch]:
            total_results_arch[arch] = {}

        if not max_results_arch[arch][size]:
            total_results_arch[arch][size] = {}
        if not total_results_arch[arch][size]:
            total_results_arch[arch][size] = {}

        if not max_results_arch[arch][size][i_size]:
            max_results_arch[arch][size][i_size] = {}
        if not total_results_arch[arch][size][i_size]:
            total_results_arch[arch][size][i_size] = {}

        if not max_results_arch[arch][size][i_size][i_prune]:
            max_results_arch[arch][size][i_size][i_prune] = []
        if not total_results_arch[arch][size][i_size][i_prune]:
            total_results_arch[arch][size][i_size][i_prune] = []

        for embedding in results:
            max_chain, total, get_stats(embedding)
            max_results_arch[arch][size][i_size][i_prune].append(max)
            total_results_arch[arch][size][i_size][i_prune].append(total)

    #for i_arch, results_size in results_arch.items():
    #    for i_size, results_i_size in results_size.items()
    #        plot_data = [results_prunes[]]


    #plt.boxplot()
