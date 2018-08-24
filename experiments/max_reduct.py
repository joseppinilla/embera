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

verbose = 0

def embed(S):
    sampler = self.sampler
    S_edgelist = list(S.edges())

    results = {}
    valid = False
    for i in range(self.tries):
        t_start = time.time()
        embedding = sampler.get_embedding(S_edgelist,
                            get_new = True,
                            verbose=verbose)
        t_end = time.time()
        t_elap = t_end-t_start

        if bool(embedding): valid = True
        results[i] = [ t_elap, embedding ]

    return valid, results

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

def get_max_chain(embedding):
    max_chain = 0
    total = 0
    for node, chain in embedding.items():
        chain_len = len(chain)
        total += chain_len
        if chain_len > max_chain:
            max_chain = chain_len
    return max_chain, total

def read_pickle():
    filename = resultsdir + "DW2000Q-bipartite-128-minorminer_best"
    filename = resultsdir + "C4-complete-17-minorminer_best"
    filename = resultsdir + "C4-bipartite-32-minorminer_best"
    filename = resultsdir + "C4-grid2d-49-minorminer_best"
    fp = open(filename, 'rb')
    data = pickle.load(fp)
    print(data)
    fp.close()
    return data

if __name__== "__main__":

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
        #TODO: Test
        # Find best embedding (min qubits, min max chain)
            # remove one edge at random find best embedding
        # remove one node at random
            #Find best embedding (min qubits, min max chain)
                # remove one edge at random find best embedding

        tries = 200
        method = minorminer
        gen, _, specs = ARCHS[arch]
        T = gen(*specs)
        structsampler = StructureComposite(SimulatedAnnealingSampler(), T.nodes, T.edges)
        sampler = EmbeddingComposite(structsampler, method)

        sizes = [size, size-1]

        for i_size in sizes:

            best_embed = sys.maxsize
            i_best_embed = None
            best_chain = sys.maxsize
            i_best_chain = None
            if verbose: print('\n\nSize %s' % i_size)
            if graph == 'complete':
                S = nx.complete_graph(i_size)
            elif graph == 'bipartite':
                p = int(i_size/2)
                S = nx.complete_bipartite_graph(p,p)
            elif graph == 'grid2d':
                p = int(math.sqrt(i_size))
                S = nx.grid_2d_graph(p,p)
            else:
                print('FAILED!')


            prunes = [0,1]

            for i_prune in prunes:
                if verbose: print('\n\nPrune %s' % i_prune)
                for i in range(tries):
                    #Pruning
                    S_edgelist = list(S.edges())
                    edges = len(S_edgelist)
                    for val in range(i_prune):
                        S_edgelist.pop( random.randrange(edges) )
                        edges-=1
                    if verbose:
                        print('-------------------------Iteration %s' % i)
                    t_start = time.time()
                    embedding = sampler.get_embedding(S_edgelist,
                                        get_new = True,
                                        verbose=verbose)
                    t_end = time.time()
                    t_elap = t_end-t_start
                    if not embedding: continue
                    if verbose:
                        print('--len %s' % len(embedding))

                    # Total qubits stats
                    # Max chain stats
                    max_chain, total = get_max_chain(embedding)

                    if total < best_embed:
                        i_best_embed = i
                        best_embed = total
                    if max_chain < best_chain:
                        i_best_chain = i
                        best_chain = max_chain

                    i_str = "_%s" % i
                    filename = base + i_str + '-' + str(i_size) + '-' + str(i_prune)
                    result = [t_elap, embedding]
                    log(results,filename)
                    if verbose:
                        print('max %s' % max_chain)
                        print('total %s' % total)

                best_filename = base + '-best-' + str(i_size) + '-' + str(i_prune)
                best = [i_best_embed, best_embed, i_best_chain, best_chain]
                log(best,best_filename)
                print(best_filename)
                print(best)
