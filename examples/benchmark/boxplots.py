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

def read_log(filename):
    filepath = os.path.join(resultsdir, filename)
    try:
        fp = open(filepath, 'rb')
        data = pickle.load(fp)
        fp.close()
        for i, result in data.items():
            yield result
    except:
        print('File %s not found.' % filepath)
        return None

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

    stats={'max':max_chain, 'total': total, 'min': min_chain,
            'avg': avg_chain, 'stdev': std_dev}

    return stats

""" PROCESS DATA """
# Create and populate stats container
archs = ['p16_graph0.95']
#archs = ['dw2000q_graph0.95', 'p6_graph0.95']
#archs = ['dw2000q_graph0.95']
methods = ['layout_agnostic']
graphs = ['complete_graph', 'complete_graph0.95', 'complete_graph0.9']
sizes = [140,141,142]
metrics = ['max','total','time','avg','stdev','min']
for arch in archs:
    for method in methods:
        ticks = range( len(sizes) * len(graphs) + 1)
        stats = {}

        box_data = {}
        for graph in graphs:
            for size in sizes:
                data_points = {'valid':0}
                for metric in metrics:
                    data_points[metric] = []
                filename = '-'.join([arch,graph,str(size),method]) + '.pkl'
                for result in read_log(filename):
                    stats['time'], _, _, embedding = result
                    if embedding:
                        stats.update(get_stats(embedding))
                        for metric in metrics:
                            data_points[metric].append(stats[metric])
                        data_points['valid'] += 1
                box_data[size, graph] = data_points

        for i, metric in enumerate(metrics):
            plt.clf()
            plt.figure(i)
            figname =   arch + '_' + method + '_' + metric + '_'.join([str(x) for x in sizes])
            textstr =   'Arch: %s \n' % arch + \
                        'Method: %s \n' % method

            labels = ['']
            box = []

            for (size, graph), data  in box_data.items():
                labels.append('%s_%s_%s' % (graph, size, data['valid']))
                box.append(data[metric])
            wrapped_labels = [ '\n'.join(l.split('_')) for l in labels ]



            plt.boxplot(box)
            plt.xticks(ticks, wrapped_labels)
            side_text = plt.figtext(1, 0.5, textstr, fontsize=12)
            plt.tight_layout()
            ymin, ymax = plt.ylim()
            plt.ylim(ymin=0, ymax=ymax*1.1)
            plt.xlabel('Graph')
            plt.ylabel(metric)
            #plt.savefig(figsdir + figname + '.png', bbox_extra_artists=(side_text,),bbox_inches='tight')
            plt.savefig(figsdir + figname + '.png')
            #plt.show()
