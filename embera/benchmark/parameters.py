import dimod
import embera

def init_bm(G, RNG_SEED=None):
    embera.random.seed(RNG_SEED)
    """ Simulated Ising parameters of initialization of a Boltzmann Machine. """
    N = len(G.nodes)
    node_biases = embera.random.bimodal(N,loc1=.0,scale1=.25,size1=N//2,
                              loc2=.75,scale2=.075,size2=N-N//2)
    edge_biases = embera.random.uniform(low=-2.0,high=2.0,size=len(G.edges))
    h = {v:b for v,b in zip(G.nodes,node_biases)}
    J = {(u,v):b for (u,v),b in zip(G.edges,edge_biases) if u!=v}
    return h,J

def trained_bm(G, RNG_SEED=None):
    embera.random.seed(RNG_SEED)
    """ Simulated Ising parameters of a trained Boltzmann Machine. """
    node_biases = embera.random.uniform(low=-1.0,high=1.0,size=len(G.nodes))
    edge_biases = embera.random.normal(scale=0.5,size=len(G.edges))
    h = {v:b for v,b in zip(G.nodes,node_biases)}
    J = {(u,v):b for (u,v),b in zip(G.edges,edge_biases) if u!=v}
    return h,J

def csp(G, RNG_SEED=None):
    embera.random.seed(RNG_SEED)
    """ Simulated Ising parameters of a Constrained Satisfaction Problem. """
    node_vals = [-1.0,-0.8,-0.4,-0.2,0.0,0.2,0.4,0.8,1.0]
    node_biases = embera.random.categorical(len(G.nodes), node_vals)
    edge_vals = [-2.0,-1.0,1.0,2.0]
    edge_biases = embera.random.categorical(len(G.edges), edge_vals)
    h = {v:b for v,b in zip(G.nodes,node_biases)}
    J = {(u,v):b for (u,v),b in zip(G.edges,edge_biases) if u!=v}
    return h,J
