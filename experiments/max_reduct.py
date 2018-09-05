import os
import sys
import math
import json
import time
import numpy
import pickle
import random
import unittest
import traceback
import minorminer
import networkx as nx
import dwave_networkx as dnx
import matplotlib.pyplot as plt

from collections import Counter

from embedding_methods.architectures import *

from embedding_methods.dense import dense
from embedding_methods.topological import topological
from embedding_methods.composites.embedding import EmbeddingComposite

from dimod.reference.samplers.exact_solver import ExactSolver
from dimod.reference.samplers.random_sampler import RandomSampler
from dimod.reference.composites.structure import StructureComposite
from dimod.reference.samplers.simulated_annealing import SimulatedAnnealingSampler


profilesdir = "./profiles/"
resultsdir = "./results/"

verbose = 1

def read_log_pickle(filename):
    fp = open(filename, 'rb')
    data = pickle.load(fp)
    fp.close()
    return data

def log_pickle(obj, filename):
    """ Pickle allows dumping non-string keyed dictionaries """
    with open(filename, 'wb') as fp:
        pickle.dump(obj, fp)

def log(obj, filename):
    if not os.path.exists(resultsdir):
        os.makedirs(resultsdir)
    filepath = resultsdir + filename
    if verbose: print('File: %s' % filepath)
    log_pickle(obj, filepath)

if __name__== "__main__":

    if not os.path.exists(resultsdir):
        os.makedirs(resultsdir)

    results_db = set(os.listdir(resultsdir))

    archs = ['DW2X', 'P6'] #['C4', 'DW2X', 'DW2000Q', 'P6', 'P16']
    fault_pcts = [0, 2, 5, 10]
    methods = [minorminer]
    prunes = [ int(x) for x in sys.argv[1:] ]
    #prunes = [0,8,16] #[0,1,2,4,8,16,32,64,128,256]
    graphs = ['grid2d'] #['complete', 'bipartite', 'grid2d', 'hypercube', 'rooks']
    s_sizes = [] #[50]
    tries = 200

    mocksampler = SimulatedAnnealingSampler()

    try:
        ############################################################################
        for i_method in methods:
        ############################################################################
            for i_arch in archs:
                gen, draw, specs, profile = ARCHS[i_arch]
        ############################################################################
                for i_fault_pct in fault_pcts:
                    T = gen(*specs)
                    faults = int( i_fault_pct/100 * len(T))
                    for val in range(faults):
                        fault = random.choice(list(T.nodes))
                        T.remove_node(fault)
                    structsampler = StructureComposite(mocksampler, T.nodes, T.edges)
                    sampler = EmbeddingComposite(structsampler, i_method)
        ############################################################################
                    for i_graph in graphs:
                        sizes = [ profile[i_graph] ] + s_sizes
        ############################################################################
                        for i_size in sizes:
                            if i_graph == 'complete':
                                S = nx.complete_graph(i_size)
                            elif i_graph == 'bipartite':
                                p = int(i_size/2)
                                S = nx.complete_bipartite_graph(p,p)
                            elif i_graph == 'grid2d':
                                p = int(math.sqrt(i_size))
                                S = nx.grid_2d_graph(p,p)
                            elif i_graph == 'hypercube':
                                p = int( math.log(i_size,2))
                                S = nx.hypercube_graph(p)
                            elif i_graph == 'rooks':
                                p =int( math.sqrt(i_size))
                                G = nx.complete_graph(p)
                                H = nx.complete_graph(p)
                                S = nx.cartesian_product(G,H)
                            else:
                                print('FAILED!')
        ############################################################################
                            for i_prune in prunes:
                                if verbose: print('\n\nPrune %s' % i_prune)
                                # Run experiment once
                                data = [i_arch,str(i_fault_pct),
                                        i_graph,str(i_size),
                                        str(i_prune),i_method.__name__]
                                print(data)
                                base = '-'.join(data)
                                filename = base + '.pkl'
                                if (filename in results_db): continue
                                results = {}
        ############################################################################
                                for i in range(tries):
                                    S_edgelist = list(S.edges)
                                    S_edges = len(S_edgelist)
                                    if i_prune >= S_edges: break
                                    for val in range(i_prune):
                                        S_edgelist.pop( random.randrange(S_edges) )
                                        S_edges-=1
                                    if verbose:
                                        print('-------------------------Iteration %s' % i)

                                    t_start = time.time()
                                    embedding = sampler.get_embedding(S_edgelist,
                                                        get_new = True,
                                                        timeout = 400,
                                                        tries=1,
                                                        verbose=verbose)
                                    t_end = time.time()
                                    t_elap = t_end-t_start
                                    results[i] = [t_elap, S_edgelist, T.edges, embedding]
                                    if verbose:
                                        print('--len %s' % len(embedding))
        ############################################################################
                                log(results,filename)
    except:
        traceback.print_exc()
        print('Current log %s may be corrupted' % filename)
        log(results,filename)
