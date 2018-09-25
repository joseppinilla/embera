import os
import sys
import json
import math
import pickle

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.artist as artists
matplotlib.rcParams['axes.formatter.useoffset'] = False

from collections import Counter, OrderedDict


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

def read_logs(filedir):
    for file in os.listdir(filedir):
        filename = os.path.join(filedir, file)
        base, _ = os.path.splitext(file)
        results = read_log_pickle(filename)
        arch, fault, graph, size, prune, method = base.split('-')

        yield file, arch, fault, graph, size, prune, method, results

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
        if fault not in stats[arch]:
            stats[arch][fault] = {}
        if graph not in stats[arch][fault]:
            stats[arch][fault][graph] = {}
        if size not in stats[arch][fault][graph]:
            stats[arch][fault][graph][size] = {}
        if prune not in stats[arch][fault][graph][size]:
            stats[arch][fault][graph][size][prune] = {}
            stats[arch][fault][graph][size][prune]['total'] = []
            stats[arch][fault][graph][size][prune]['time'] = []
            stats[arch][fault][graph][size][prune]['max'] = []
            stats[arch][fault][graph][size][prune]['avg'] = []
            stats[arch][fault][graph][size][prune]['stdev'] = []
            stats[arch][fault][graph][size][prune]['min'] = []

        valid = 0
        for key, result in results.items():
            t_elap, _, _, embedding = result
            if embedding:
                max_chain, min_chain, total, avg_chain, std_dev = get_stats(embedding)
                stats[arch][fault][graph][size][prune]['total'].append(total)
                stats[arch][fault][graph][size][prune]['time'].append(t_elap)
                stats[arch][fault][graph][size][prune]['max'].append(max_chain)
                stats[arch][fault][graph][size][prune]['avg'].append(avg_chain)
                stats[arch][fault][graph][size][prune]['stdev'].append(std_dev)
                stats[arch][fault][graph][size][prune]['min'].append(min_chain)
                valid+=1

        #stats[arch][fault][graph][size][prune]['success'] = valid/len(results)


    for i_arch, stats_fault in stats.items():
        for i_fault, stats_graph in stats_fault.items():
            for i_graph, stats_size in stats_graph.items():
                for i_size, stats_prune in stats_size.items():

                    n = int(i_size)
                    if i_graph=='complete':
                        edges = ((n**2)-n)/2
                    elif i_graph=='bipartite':
                        edges = (n**2)/4
                    elif i_graph=='grid2d':
                        edges = 2*n - 2*math.sqrt(n)
                    elif i_graph=='hypercube':
                        d = math.log(n,2)
                        edges = (2**(d-1))*d
                    elif i_graph=='rooks':
                        edges = n*math.sqrt(n) - n

                    prune_labels = ['']
                    prune_ordered = ['']
                    for i_prune in sorted(stats_prune.keys(), key=lambda x: int(x)):
                        pct = int(i_prune)/edges*100
                        if pct<100:
                            prune_ordered.append(i_prune)
                            prune_labels.append("%.2f%%" % pct)

                    ticks = list(range(len(prune_labels)+1))

                    for i, metric in enumerate(['max','total','time','avg','stdev','min']):
                    #for i, metric in enumerate(['time', 'total']):
                        plt.clf()
                        plt.figure(i)
                        figname =   i_arch + '_' + i_fault + '_' + \
                                    i_graph + '_' + i_size + '_' + metric
                        textstr =   'Arch: %s \n' % i_arch + \
                                    'Faults: %s%%\n' % i_fault + \
                                    'Graph: %s \n' % i_graph + \
                                    'Size: %s \n' % i_size + \
                                    'Edges: %d' %  edges

                        data_points = []
                        for prune in prune_ordered[1:]:
                            data_points.append(stats_prune[prune][metric])
                        plt.boxplot(data_points)
                        plt.xticks(ticks, prune_labels, rotation='vertical')
                        side_text = plt.figtext(1, 0.5, textstr, fontsize=12)
                        plt.tight_layout()
                        ymin, ymax = plt.ylim()
                        plt.ylim(ymin=0, ymax=ymax*1.1)
                        plt.xlabel('Prune')
                        plt.ylabel(metric)
                        plt.savefig(figsdir + figname + '.png', bbox_extra_artists=(side_text,),bbox_inches='tight')
                        #plt.show()
