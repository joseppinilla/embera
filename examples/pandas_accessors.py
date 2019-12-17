import embera
from pandas import DataFrame, SparseDataFrame

# Ising SampleSet Example
df = DataFrame({'qubit1':[-1.0,1.0],'energy':[1.0,2.0],'num_occurrences':[2,1]})
df.sampleset.energy
df.sampleset.first()

# BQM Example
# No labels
df = DataFrame({0: [1.1, 0.0, 0.0],
                   1: [0.5, -1.0,  0.0],
                   2: [0.0, 1.5,  0.5]})

# With labels
df = DataFrame({'a': {'a': 1.1, 'b': 0.0, 'c': 0.0},
                'b': {'a': 0.5, 'b': -1.0, 'c': 0.0},
                'c': {'a': 0.0, 'b': 1.5, 'c': 0.5}})

df.bqm.embedding = {'0':[1,2,3,45],'1':[4,7],'2':[5,6]}

df
# dimod Examples
import dimod
model = dimod.BinaryQuadraticModel({'a': 1.1, 'b': -1., 'c': .5},
                                    {('a', 'b'): .5, ('b', 'c'): 1.5},
                                    1.4, # offset
                                    dimod.BINARY)
dimoddf = model.to_pandas_dataframe()
dimoddf
dimoddf.to_dict()

import networkx as nx
import dwave_networkx as dnx
G = dnx.chimera_graph(1)
nx.to_pandas_adjacency(G)


#embedding
