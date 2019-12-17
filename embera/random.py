from numpy import concatenate
from numpy.random import multinomial, normal, uniform, seed, shuffle

""" Probability functions and distributions useful in embedding and testing
    Ising models.
"""

def prob_vector(N):
    vec = [normal(0, 1) for i in range(N)]
    mag = sum(x**2 for x in vec) ** .5
    return [(x/mag)**2 for x in vec]

def bimodal(N,loc1,scale1,size1,loc2,scale2,size2):
    samples1 = normal(loc1,scale1,size1)
    samples2 = normal(loc2,scale2,size2)
    samples = concatenate([samples1,samples2])
    shuffle(samples)
    return samples

def categorical(N, vals):
    bins = multinomial(n=N, pvals=prob_vector(len(vals)))
    samples = [vals[i] for i,b in enumerate(bins) for _ in range(b)]
    shuffle(samples)
    return samples
