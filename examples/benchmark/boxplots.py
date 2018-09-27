import os
import sys
import math
import pickle
import matplotlib.pyplot as plt

figsdir = "./figs/"
resultsdir = "./results/"

if not os.path.exists(figsdir):
    os.makedirs(figsdir)

def log(obj, filename):
    filepath = './' + filename
    # Pickle allows dumping non-string keyed dictionaries
    with open(filepath, 'wb') as fp:
        pickle.dump(obj, fp)

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

""" PROCESS DATA """
# Create and populate stats container
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
