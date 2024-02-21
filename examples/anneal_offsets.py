%matplotlib qt
# Import packages
import math
import qaml
import torch
import dwave
import dimod
import embera
import random
import minorminer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create RBM
SEED = 42
random.seed(SEED)
SHAPE = (16,128)

Knm = embera.benchmarks.generators.topologies.complete_bipartite_graph(128,16)
rbm = qaml.nn.RestrictedBoltzmannMachine(*SHAPE)

# qa_sampler = qaml.sampler.QASampler(rbm,solver="DW_2000Q_6")
qa_sampler = qaml.sampler.QASampler(rbm,solver="Advantage_system4.1")

# Create imbalance
interaction_edges = []
for e in Knm.edges:
    for i in qa_sampler.embedding.interaction_edges(e):
        interaction_edges.append(i)
q_set = set([q for edge in interaction_edges for q in edge])
embedding = {k:[q for q in chain if q in q_set] for k,chain in qa_sampler.embedding.items()}


T = qa_sampler.to_networkx_graph()
miner = minorminer.miner(Knm,T)
miner.quality_key(qa_sampler.embedding)
miner.quality_key(embedding)
orig_embedding = qa_sampler.embedding.copy()
qa_sampler.embedding = dwave.embedding.EmbeddedStructure(T.edges,embedding)

import dwave_networkx as dnx
dnx.draw_pegasus_embedding(T,orig_embedding,Knm,node_size=10)
dnx.draw_pegasus_embedding(T,embedding,Knm,node_size=10)

init_fcl = False
if init_fcl:
    sol = {k:(1 if random.random() < 0.5 else -1) for k in Knm}
    bqm = dimod.generators.fcl.frustrated_loop(Knm,10000,seed=SEED,planted_solution=sol)
    plt.hist(bqm.quadratic.values())

    _ = qa_sampler.embed_bqm(auto_scale=True)

    qubo = bqm.change_vartype("BINARY",inplace=False)
    ldata,(rind,cind,biases),offset = qubo.to_numpy_vectors()
    biasesTensor = torch.Tensor(biases).reshape(tuple(reversed(SHAPE)))
    rbm.b.data = torch.zeros(rbm.V)
    rbm.c.data = torch.zeros(rbm.H)
    rbm.W.data = biasesTensor*qa_sampler.scalar*4
else:
    _ = torch.nn.init.uniform_(rbm.b,-0.1,0.1)
    _ = torch.nn.init.uniform_(rbm.c,-0.1,0.1)
    _ = torch.nn.init.uniform_(rbm.W,-0.1,0.1)


def chain_delay(sampler, embedding, alpha=0.04, beta=0.69):

    minarr,maxarr = zip(*sampler.properties['anneal_offset_ranges'])
    minarr = np.asarray(minarr)
    minoff = max(minarr[minarr.nonzero()])

    maxarr = np.asarray(maxarr)
    maxoff = min(maxarr[maxarr.nonzero()])

    offset = [0]*sampler.properties['num_qubits']
    for _,chain in embedding.items():
        k = len(chain)
        for q in chain:
            delay = alpha*beta**((1-k)/k)-1
            if delay < minoff:
                delay = minoff
            if delay > maxoff:
                delay = maxoff
            offset[q] = delay
    return offset

offsets = chain_delay(qa_sampler.sampler,embedding)



gibbs_sampler = qaml.sampler.GibbsNetworkSampler(rbm)
vg,hg = gibbs_sampler(torch.rand((10000,rbm.V)),k=100)

sample_kwargs = {'auto_scale':True,"num_spin_reversal_transforms":4}
vs,hs = qa_sampler(10000,**sample_kwargs)
vo,ho = qa_sampler(10000,anneal_offsets=offsets,**sample_kwargs)

df_s = pd.DataFrame(rbm.free_energy(vs),columns=['QAP Energy'])
df_o = pd.DataFrame(rbm.free_energy(vo),columns=['Offset Energy'])
df_g = pd.DataFrame(rbm.free_energy(vg.bernoulli()),columns=['Gibbs Energy'])

df = pd.concat([df_s,df_g,df_o],axis=1)
df.plot.hist(y=['QAP Energy','Gibbs Energy','Offset Energy'],bins=200,alpha=0.5)

df_s = pd.DataFrame(rbm.energy(vs,hs),columns=['QAP Energy'])
df_o = pd.DataFrame(rbm.energy(vo,ho),columns=['Offset Energy'])
df_g = pd.DataFrame(rbm.energy(vg.bernoulli(),hg.bernoulli()),columns=['Gibbs Energy'])

df = pd.concat([df_s,df_g,df_o],axis=1)
df.plot.hist(y=['QAP Energy','Gibbs Energy','Offset Energy'],bins=200,alpha=0.5)
