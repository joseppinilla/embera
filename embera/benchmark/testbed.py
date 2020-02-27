import time
import dimod
import embera
import dwave.cloud
import dwave.embedding

class StructuredSASampler(dimod.SimulatedAnnealingSampler,dimod.Structured):
    nodelist = None
    edgelist = None
    def __init__(self, failover=None,**config):
        super(StructuredSASampler, self).__init__()
        self.client = dwave.cloud.Client.from_config(**config)
        solver = self.client.get_solver()

        self.properties.update(solver.properties)
        self.properties['category'] = 'software'
        self.properties['chip_id'] = "SIM_"+solver.name

        G = embera.architectures.graph_from_solver(solver)
        self.nodelist = list(G.nodes())
        self.edgelist = list(G.edges())

class StructuredRandomSampler(dimod.RandomSampler,dimod.Structured):
    nodelist = None
    edgelist = None
    def __init__(self, failover=None,**config):
        super(StructuredRandomSampler, self).__init__()
        self.client = dwave.cloud.Client.from_config(**config)
        solver = self.client.get_solver()

        self.properties.update(solver.properties)
        self.properties['category'] = 'software'
        self.properties['chip_id'] = "RND_"+solver.name

        G = embera.architectures.graph_from_solver(solver)
        self.nodelist = list(G.nodes())
        self.edgelist = list(G.edges())

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
