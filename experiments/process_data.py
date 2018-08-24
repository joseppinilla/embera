import os
import json
import pickle
import matplotlib.pyplot as plt

from collections import Counter

from embedding_methods.architectures import *


filedir = "./profiles/"

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

def plot_histo(title):
    for log in read_logs():
        arch, graph, size, method, results = log
        for i, result in results.items():
            time, embedding = result
            if embedding:
                histo = chain_length_histo(embedding)
                plt.bar(list(histo.keys()), histo.values())
                plt.xticks(list(histo.keys()))
                plt.show()
                #draw(T, embedding)
                #plt.show()

def read_logs():
    for file in os.listdir(filedir):
        filename = os.path.join(filedir, file)
        base, ext = os.path.splitext(file)
        if ext=='.pkl':
            results = read_log_pickle(filename)
        elif ext=='.json':
            results = read_log_json(filename)
        arch, graph, size, method = base.split('-')
        gen, draw, specs = ARCHS[arch]
        T = gen(*specs)
        yield arch, graph, size, method, results



if __name__== "__main__":
    plot_histo()
