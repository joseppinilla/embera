import os
import sys
import math
import json
import time
import numpy
import pickle
import random
import unittest
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

    results_db = set(os.listdir(resultsdir))

    for file in os.listdir(profilesdir):
        filename = os.path.join(profilesdir, file)
        base, ext = os.path.splitext(file)
        if ext=='.pkl':
            results = read_log_pickle(filename)
        elif ext=='.json':
            results = read_log_json(filename)
        arch, graph, size_str, _ = base.split('-')
        size = int(size_str)
        gen, draw, specs = ARCHS[arch]
        T = gen(*specs)

        tries = 200
        method = minorminer
        gen, _, specs = ARCHS[arch]
        T = gen(*specs)
        structsampler = StructureComposite(SimulatedAnnealingSampler(), T.nodes, T.edges)
        sampler = EmbeddingComposite(structsampler, method)


        if verbose: print('\n\nSize %s' % size)
        if graph == 'complete':
            S = nx.complete_graph(size)
        elif graph == 'bipartite':
            p = int(size/2)
            S = nx.complete_bipartite_graph(p,p)
        elif graph == 'grid2d':
            #TEMP: Not testing grids
            continue
            p = int(math.sqrt(size))
            S = nx.grid_2d_graph(p,p)
        else:
            print('FAILED!')


        prunes = [0,1,2,4,8,16,32,64,128,256]

        for i_prune in prunes:
            if verbose: print('\n\nPrune %s' % i_prune)
            for i in range(tries):
                i_str = "_%s" % i
                filename = base + i_str + '-' + str(i_prune)

                # Pruning
                S_edgelist = list(S.edges())
                edges = len(S_edgelist)
                if i_prune >= edges: continue
                for val in range(i_prune):
                    S_edgelist.pop( random.randrange(edges) )
                    edges-=1
                if verbose:
                    print('-------------------------Iteration %s' % i)

                if (filename in results_db):
                    continue

                t_start = time.time()
                embedding = sampler.get_embedding(S_edgelist,
                                    get_new = True,
                                    verbose=verbose)
                t_end = time.time()
                t_elap = t_end-t_start
                if not embedding: continue
                if verbose:
                    print('--len %s' % len(embedding))

                log([t_elap, embedding],filename)
