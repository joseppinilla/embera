import embera
import minorminer
import networkx as nx
import dwave_networkx as dnx

S = nx.complete_graph(11)
T = dnx.chimera_graph(7)
embedding = minorminer.find_embedding(S,T)
# dnx.draw_chimera_embedding(T,embedding,node_size=10)
new_embedding = embera.transform.embedding.greedy_fit(S,T,embedding)
dnx.draw_chimera_embedding(T,new_embedding,node_size=10)


coords = embera.dwave_coordinates.from_dwave_networkx(T)
nice_embedding = {k:[coords.linear_to_nice(q) for q in v] for k,v in new_embedding.items()}



P = dnx.pegasus_graph(16,nice_coordinates=True)
dnx.draw_pegasus(P,node_size=10,crosses=True)
dnx.draw_pegasus_embedding(P,nice_embedding,node_size=10)


iter_emb = list(iter_sliding_window(P,nice_embedding))

len(iter_emb)
dnx.draw_pegasus_embedding(P,iter_emb[228],node_size=2,crosses=True)

import matplotlib.pyplot as plt


list(iter_sliding_window(T,embedding))

list(iter_sliding_window(P,nice_embedding))

for emb in iter_sliding_window(T,embedding):
    dnx.draw_chimera_embedding(T,emb,node_size=2)
    plt.pause(0.2)

for emb in iter_sliding_window(P,nice_embedding):
    dnx.draw_pegasus_embedding(P,emb,node_size=2,crosses=True)
    plt.pause(0.2)


import numpy as np
from embera.preprocess.tiling_parser import DWaveNetworkXTiling


def iter_sliding_window(T, embedding):
    """ Use a sliding window approach to iteratively transport the embedding
        from one region of the Chimera graph to another.

        Example:
            >>> import embera
            >>> import networkx as nx
            >>> import dwave_networkx as dnx
            >>> import matplotlib.pyplot as plt
            >>> S = nx.complete_graph(11)
            >>> T = dnx.chimera_graph(7)
            >>> embedding = minorminer.find_embedding(S,T)
            >>> dnx.draw_chimera_embedding(T,embedding,node_size=10)
            >>> slide = embera.transform.embedding.sliding_window(T,embedding)
            >>> for new_embedding in slide:
            ...     dnx.draw_chimera_embedding(T,new_embedding,node_size=10)
            ...     plt.pause(0.2)
    """
    tiling = DWaveNetworkXTiling(T)
    shape = np.array(tiling.shape)
    # Initialize edges
    origin = shape
    end = (0,)*len(origin)
    # Find edges
    for v,chain in embedding.items():
        for q in chain:
            tile = np.array(tiling.get_tile(q))
            origin = [min(t,o) for t,o in zip(tile,origin)]
            end = [max(t,e) for t,e in zip(tile,end)]

    # Move tiles to origin and translate to try and find valid embedding
    size = np.array(end) - np.array(origin)
    interactions = lambda u,v,E:((s,t) for s in E[u] for t in E[v])
    is_connected = lambda edges: any(T.has_edge(s,t) for s,t in edges)

    dims = list((sh-sz) for sh,sz in zip(shape,size))
    depth,height,width = np.pad(dims,(3-len(dims),0),constant_values=1)
    print(depth,height,width)
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                slide = {}
                offset = np.array([z,y,x])[-len(shape):]
                # Translate all qubits
                for v,chain in embedding.items():
                    new_chain = []
                    for q in chain:
                        tile = np.array(tiling.get_tile(q))
                        new_tile = tuple(tile - np.array(origin) + offset)
                        new_q = tiling.set_tile(q,new_tile)
                        new_chain.append(new_q)
                    slide[v] = new_chain
                yield slide
