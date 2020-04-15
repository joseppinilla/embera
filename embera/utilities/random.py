import numpy as np

from random import sample
from random import seed as _py_seed

from numpy.random import choice, shuffle
from numpy.random import multinomial, normal, uniform
from numpy.random import seed as _np_seed


__all__ = ["choice","sample","seed","shuffle",
           "prob_vector","bimodal","categorical"]

""" Probability functions and distributions useful in embedding and testing
    Ising models.
"""

def seed(a):
    _py_seed(a)
    _np_seed(a)

def prob_vector(N):
    vec = [normal(0, 1) for i in range(N)]
    mag = sum(x**2 for x in vec) ** .5
    return [(x/mag)**2 for x in vec]

def bimodal(N, loc1=-1.0,scale1=.25,size1=None,
               loc2=+1.0,scale2=.25,size2=None):
    if size1 is None:
        size1=N//2
    if size2 is None:
        size2=N-size1
    samples1 = normal(loc1,scale1,size1)
    samples2 = normal(loc2,scale2,size2)
    samples = np.concatenate([samples1,samples2])
    shuffle(samples)
    return samples

def categorical(N, vals):
    bins = multinomial(n=N, pvals=prob_vector(len(vals)))
    samples = np.array([vals[i] for i,b in enumerate(bins) for _ in range(b)])
    shuffle(samples)
    return samples
