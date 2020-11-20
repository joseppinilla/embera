import time
import dimod
import scipy
import embera
import dwave.cloud
import dwave.embedding

import numpy as np
import dwave_networkx as dnx

from collections import OrderedDict

class StructuredMockSampler(dimod.Structured):

    def __init__(self, failover=None,**config):
        super(StructuredMockSampler, self).__init__()
        self.client = dwave.cloud.Client.from_config(**config)
        solver = self.client.get_solver()

        self.properties.update(solver.properties)
        self.properties['category'] = 'software'

        self._nodelist = self.properties['qubits']
        self._edgelist = self.properties['couplers']

    @property
    def edgelist(self):
        # Assumption: cloud client nodes are always integer-labelled
        try:
            edgelist = self._edgelist
        except AttributeError:
            self._edgelist = edgelist = sorted(set((u, v) if u < v else (v, u)
                                                   for u, v in self.solver.edges))
        return edgelist

    @property
    def nodelist(self):
        # Assumption: cloud client nodes are always integer-labelled
        try:
            nodelist = self._nodelist
        except AttributeError:
            self._nodelist = nodelist = sorted(self.solver.nodes)
        return nodelist

    def to_networkx_graph(self):
        """Converts DWaveSampler's structure to a Chimera or Pegasus NetworkX graph.

        Returns:
            :class:`networkx.Graph`:
                Either an (m, n, t) Chimera lattice or a Pegasus lattice of size m.

        """
        topology_type = self.properties['topology']['type']
        shape = self.properties['topology']['shape']

        if topology_type == 'chimera':
            G = dnx.chimera_graph(*shape,
                                  node_list=self.nodelist,
                                  edge_list=self.edgelist)

        elif topology_type == 'pegasus':
            G = dnx.pegasus_graph(shape[0],
                                  node_list=self.nodelist,
                                  edge_list=self.edgelist)

        return G

    def validate_anneal_schedule(self,arg):
        pass

class StructuredSASampler(StructuredMockSampler,dimod.SimulatedAnnealingSampler):
    def __init__(self, failover=None,**config):
        dimod.SimulatedAnnealingSampler.__init__(self)
        StructuredMockSampler.__init__(self,failover=failover,**config)

class StructuredRandomSampler(StructuredMockSampler,dimod.RandomSampler):
    def __init__(self, failover=None,**config):
        dimod.RandomSampler.__init__(self)
        StructuredMockSampler.__init__(self,failover=failover,**config)


def embed_and_report(method, *args, **kwargs):
    report = {}

    start = time.time()
    embedding = method(*args,**kwargs)
    end = time.time()

    report["valid"] = bool(embedding) # TODO: more elaborate, for incomplete embeddings
    report['embedding_runtime'] = end-start
    report['embedding_method'] = method.__name__

    embedding_obj = embera.Embedding(embedding,**report)

    return embedding_obj

def sample_and_report(method, bqm, embedding, target):
    solver = target.name
    target_adj = target.adj

    embed_args = {'chain_strength':1.0}
    emb_bqm = dwave.embedding.embed_bqm(bqm,embedding,target_adj,**embed_args)
    sampleset = method(emb_bqm,solver)

    sampleset.info.update(embed_args)
    sampleset.info.update(embedding.properties)
    sampleset.info.update({'sampler_method':method.__name__})

    return sampleset

def measure_and_report(method, embeddings, samplesets, **kwargs):
    report = {}

    metric = method.__name__
    for emb,samp in zip(embeddings,samplesets):
        index =  emb.id
        kwargs.update({'sampleset':samp})
        report[index] = method(emb,**kwargs)

    return report

def relative_k_hamming_trench(samplesets, hamm_k, norm=False, info_key=None):
    # TODO: Avoid calculating hamming distance twice
    pockets = []
    energies = OrderedDict()
    union = dimod.concatenate(samplesets)
    for data in union.data(sorted_by='energy'):
        value = tuple(data.sample.values())
        local = (scipy.spatial.distance.hamming(value,p)<=hamm_k for p in pockets)
        if not any(local):
            pockets.append(value)
            energies[value] = data.energy

    pockets_i = OrderedDict()
    pockets_union = OrderedDict([(k,0) for k in pockets])
    for i,sampleset in enumerate(samplesets):
        bins = OrderedDict([(k,0) for k in pockets])
        for sample in sampleset.samples(sorted_by='energy'):
            value = tuple(sample.values())
            hits = []
            k = hamm_k
            for pocket in pockets:
                hamm_dist = scipy.spatial.distance.hamming(value,pocket)
                if hamm_dist<k:
                    hits = [pocket]
                    k = hamm_dist
                elif hamm_dist==k:
                    hits += [pocket]
            split = 1/len(hits)
            for hit in hits:
                bins[hit] = split + bins.get(hit,0)
                pockets_union[hit] += split
        name = sampleset.info.get(info_key,str(i))
        pockets_i[str(name)] = bins

    if norm:
        norm_pockets = OrderedDict()
        for samples,(name,pockets) in zip(samplesets,pockets_i.items()):
            nsamples = len(samples)
            norm_pocket = OrderedDict([(k,v/nsamples) for k,v in pockets.items()])
            norm_pockets[name] = norm_pocket
        norm_union = OrderedDict([(k,v/len(union)) for k,v in pockets_union.items()])
        return norm_union, energies, norm_pockets
    else:
        return pockets_union, energies, pockets_i

def absolute_k_hamming_trench(sampleset, hamm_k, norm=False, info_key=None):
    pockets = OrderedDict()
    energies = OrderedDict()
    for data in sampleset.data(sorted_by='energy'):
        value = tuple(data.sample.values())

        hits = []
        k = hamm_k
        for pocket in pockets:
            hamm_dist = scipy.spatial.distance.hamming(value,pocket)
            if hamm_dist<k:
                hits = [pocket]
                k = hamm_dist
            elif hamm_dist==k:
                hits += [pocket]
        if hits:
            split = 1/len(hits)
            for hit in hits:
                pockets[hit] = split + pockets.get(hit,0.0)
        else:
            pockets[value] = 0.0
            energies[value] = data.energy

    if norm:
        nsamples = len(sampleset)
        norm_pockets = OrderedDict([(k,v/nsamples) for k,v in pockets.items()])
        return energies, norm_pockets

    return energies, pockets

def figure_of_merit(energies, pockets, E0=None):
    accum = 1.0
    minE = next(iter(energies.values())) if E0 is None else E0
    for pocket,prob in pockets.items():
        energy = energies[pocket]
        accum*=(energy/minE)*(1.0 + prob)
    return accum/2.0
