import time
import dimod
import embera
import dwave.cloud

class SAnnealSampler(dimod.SimulatedAnnealingSampler,dimod.Structured):
    nodelist = None
    edgelist = None
    def __init__(self, failover=None,**config):
        super(SAnnealSampler, self).__init__()
        self.client = dwave.cloud.Client.from_config(**config)
        solver = self.client.get_solver()

        self.properties.update(solver.properties)
        self.properties['category'] = 'software'
        self.properties['chip_id'] = "SIM_"+solver.name

        G = embera.architectures.graph_from_solver(solver)
        self.nodelist = list(G.nodes())
        self.edgelist = list(G.edges())

def embed_and_report(method, *args, **kwargs):
    report = {}
    embedding = {}

    start = time.time()
    embedding = method(*args,**kwargs)
    end = time.time()

    report = {"valid":bool(embedding)} # This can be more elaborate, for incomplete embeddings
    report['embedding_runtime'] = end-start
    report['embedding_method'] = method.__name__

    embedding_obj = embera.Embedding(embedding,**report)

    return embedding_obj

def sample_and_report(bqm, sampler, **kwargs):
    report = {}

    sampleset = sampler.sample(bqm,**kwargs)

    report['bqm_info'] = bqm.info
    report['embedding_method'] = embedding.properties['embedding_method']
    report['solver'] = sampler.properties
    report['parameters'] = sampler.parameters

    sampleset.info.update(report)

    return sampleset
