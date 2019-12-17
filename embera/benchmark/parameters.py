import embera

def init_bm(G):
    """ Simualted Ising parameters of initialization of a Boltzmann Machine. """
    N = len(G.nodes)
    h = embera.random.bimodal(N,loc1=.0,scale1=.25,size1=N//2,
                              loc2=.75,scale2=.075,size2=N-N//2)
    J = embera.random.uniform(low=-2.0,high=2.0,size=len(G.edges))
    return h,J

def trained_bm(G):
    """ Simualted Ising parameters of a trained Boltzmann Machine. """
    h = embera.random.uniform(low=-1.0,high=1.0,size=len(G.nodes))
    J = embera.random.normal(scale=0.5,size=len(G.edges))
    return h,J

def csp(G):
    """ Simualted Ising parameters of a Constrained Satisfaction Problem. """
    h_vals = [-1.0,-0.8,-0.4,-0.2,0.0,0.2,0.4,0.8,1.0]
    h = embera.random.categorical(len(G.nodes), h_vals)
    J_vals = [-2.0,-1.0,1.0,2.0]
    J = embera.random.categorical(len(G.edges), J_vals)
    return h,J
