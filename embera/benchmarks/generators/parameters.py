import dimod
import numpy as np

""" ======================== BinaryQuadraticModel ========================== """
def init_bm(G, RNG_SEED=None):
    np.random.seed(RNG_SEED)
    """ Simulated Ising parameters of initialization of a Boltzmann Machine. """
    A = (.0,.25,len(G.nodes)//2)
    B = (.75,.05,len(G.nodes) - len(G.nodes)//2)
    node_biases = np.concatenate((np.random.normal(*A), np.random.normal(*B)))
    edge_biases = np.random.uniform(low=-2.0,high=2.0,size=len(G.edges))
    h = {v:b for v,b in zip(G.nodes,node_biases)}
    J = {(u,v):b for (u,v),b in zip(G.edges,edge_biases) if u!=v}

    return dimod.BinaryQuadraticModel.from_ising(h,J)

def trained_bm(G, RNG_SEED=None):
    np.random.seed(RNG_SEED)
    """ Simulated Ising parameters of a trained Boltzmann Machine. """
    node_biases = np.random.uniform(low=-1.0,high=1.0,size=len(G.nodes))
    edge_biases = np.random.normal(scale=0.5,size=len(G.edges))
    h = {v:b for v,b in zip(G.nodes,node_biases)}
    J = {(u,v):b for (u,v),b in zip(G.edges,edge_biases) if u!=v}

    return dimod.BinaryQuadraticModel.from_ising(h,J)

def csp(G, RNG_SEED=None):
    np.random.seed(RNG_SEED)
    """ Simulated Ising parameters of a Constrained Satisfaction Problem. """
    node_vals = [-1.0,-0.8,-0.4,-0.2,0.0,0.2,0.4,0.8,1.0]
    node_biases = categorical(len(G.nodes), node_vals)
    edge_vals = [-2.0,-1.0,1.0,2.0]
    edge_biases = categorical(len(G.edges), edge_vals)
    h = {v:b for v,b in zip(G.nodes,node_biases)}
    J = {(u,v):b for (u,v),b in zip(G.edges,edge_biases) if u!=v}

    return dimod.BinaryQuadraticModel.from_ising(h,J)

""" ================================= Values =============================== """

def categorical(N, vals):
    """ Helper function to generate a categorical distribution of N samples.
        i.e. A multinomial distribution of normally distributed probabilities
        over the possible values.

        Args:

            N (int):

            vals (list of objects):
                Any suppor

        Returns:

            samples (numpy.array)

    """

    def prob_vector(N):
        vec = [np.random.normal(0, 1) for i in range(N)]
        mag = sum(x**2 for x in vec) ** .5
        return [(x/mag)**2 for x in vec]

    bins = np.random.multinomial(n=N, pvals=prob_vector(len(vals)))
    samples = np.array([vals[i] for i,b in enumerate(bins) for _ in range(b)])
    np.random.shuffle(samples)
    return samples
