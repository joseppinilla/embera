import minorminer

import numpy as np
import networkx as nx
import dwave_networkx as dnx

from embera.utilities.decorators import nx_graph, dnx_graph, dnx_graph_embedding
from embera.preprocess.tiling_parser import DWaveNetworkXTiling

def translate(S, T, embedding, origin=(0,0)):
    """ Transport the embedding on the same graph to re-distribute qubit
        assignments.

        Example:

            >>> import embera
            >>> import matplotlib.pyplot as plt
            >>> S = nx.complete_graph(11)
            >>> T = dnx.chimera_graph(7)
            >>> embedding = minorminer.find_embedding(S,T)
            >>> dnx.draw_chimera_embedding(T,embedding,node_size=10)
            >>> offset = (2,3)
            >>> new_embedding = embera.transform.embedding.translate(S,T,embedding,offset)
            >>> dnx.draw_chimera_embedding(T,new_embedding,node_size=10)
    """

    tiling = DWaveNetworkXTiling(T)
    shape = tiling.shape
    # Initialize offset
    offset = shape
    # Find margins
    for v,chain in embedding.items():
        for q in chain:
            tile = np.array(tiling.get_tile(q))
            offset = [min(t,o) for t,o in zip(tile,offset)]
    # Define flips
    m,n = tiling.shape
    t = tiling.graph['tile']
    new_embedding = {}
    for v,chain in embedding.items():
        new_chain = []
        for q in chain:
            k = tiling.get_k(q)
            tile = tiling.get_tile(q)
            shore = tiling.get_shore(q)
            new_tile = tuple(np.array(tile) - np.array(offset) + np.array(origin))
            new_q = tiling.set_tile(q,new_tile)
            new_chain.append(new_q)
        new_embedding[v] = new_chain

    return new_embedding


def mirror(S, T, embedding, axis=0):
    """ Flip the embedding on the same graph to re-distribute qubit
        assignments. If a perfect fit isn't found, due to disabled qubits,
        the invalid embedding is still returned.

        Example:
            >>> import embera
            >>> import matplotlib.pyplot as plt
            >>> S = nx.complete_graph(11)
            >>> T = dnx.chimera_graph(7)
            >>> embedding = minorminer.find_embedding(S,T)
            >>> dnx.draw_chimera_embedding(T,embedding,node_size=10)
            >>> axis = 1
            >>> new_embedding = embera.transform.embedding.mirror(S,T,embedding,axis)
            >>> dnx.draw_chimera_embedding(T,new_embedding,node_size=10)
    """
    tiling = DWaveNetworkXTiling(T)
    shape = np.array(tiling.shape)
    # Define flips
    m,n = tiling.shape
    t = tiling.graph['tile']
    if axis is 0:
        new_tile = lambda i,j: (i,n-j-1)
        new_k = lambda k,shore: k if shore else t-k-1
    elif axis is 1:
        new_tile = lambda i,j: (m-i-1,j)
        new_k = lambda k,shore: t-k-1 if shore else k
    else:
        raise ValueError("Value of axis not supported")
    # Rotate all qubits by chain
    new_embedding = {}
    for v,chain in embedding.items():
        new_chain = []
        for q in chain:
            k = tiling.get_k(q)
            tile = tiling.get_tile(q)
            shore = tiling.get_shore(q)
            new_coordinates = (new_tile(*tile),shore,new_k(k,shore))
            new_chain.append(next(tiling.get_qubits(*new_coordinates)))
        new_embedding[v] = new_chain

    return new_embedding

def rotate(S, T, embedding, theta=90):
    """ Rotate the embedding on the same graph to re-distribute qubit
        assignments. If a perfect fit isn't found, due to disabled qubits,
        the invalid embedding is still returned.

        Example:
            >>> import embera
            >>> import matplotlib.pyplot as plt
            >>> S = nx.complete_graph(11)
            >>> T = dnx.chimera_graph(7)
            >>> embedding = minorminer.find_embedding(S,T)
            >>> dnx.draw_chimera_embedding(T,embedding,node_size=10)
            >>> theta = 270
            >>> new_embedding = embera.transform.embedding.rotate(S,T,embedding,theta)
            >>> dnx.draw_chimera_embedding(T,new_embedding,node_size=10)
    """
    tiling = DWaveNetworkXTiling(T)
    shape = np.array(tiling.shape)
    # Define rotations
    m,n = tiling.shape
    t = tiling.graph['tile']
    if theta in [90,-270]:
        new_tile = lambda i,j: (j, m-i-1)
        new_shore = lambda shore: 0 if shore else 1
        new_k = lambda k,shore: t-k-1 if shore else k
    elif theta in [180,-180]:
        new_tile = lambda i,j: (m-i-1,n-j-1)
        new_shore = lambda shore: shore
        new_k = lambda k,shore: t-k-1
    elif theta in [-90,270]:
        new_tile = lambda i,j: (n-j-1, i)
        new_shore = lambda shore: 0 if shore else 1
        new_k = lambda k,shore: k if shore else t-k-1
    elif theta in [0,360]:
        return embedding
    else:
        raise ValueError("Value of theta not supported")
    # Rotate all qubits by chain
    new_embedding = {}
    for v,chain in embedding.items():
        new_chain = []
        for q in chain:
            k = tiling.get_k(q)
            tile = tiling.get_tile(q)
            shore = tiling.get_shore(q)
            new_coordinates = (new_tile(*tile),new_shore(shore),new_k(k,shore))
            new_chain.append(next(tiling.get_qubits(*new_coordinates)))
        new_embedding[v] = new_chain

    return new_embedding

def spread_out(S, T, embedding):
    """ Alter the embedding to add qubit chains by moving qubit
        assignments onto qubit in tiles farther from the center of the device
        graph.
            1) Use Chimera/Pegasus index to determine tile of each node
            2) Transform the tile assignment to spread out the embedding
            3) Assign nodes to corresponding qubit in new tiles
            4) Perform an "embedding pass" or path search to reconnect all nodes

        Example:
            >>> import embera
            >>> S = nx.complete_graph(10)
            >>> T = dnx.chimera_graph(8)
            >>> embedding = minorminer.find_embedding(S,T)
            >>> dnx.draw_chimera_embedding(T,embedding)
            >>> new_embedding = embera.transform.embedding.spread_out(S,T,embedding)
            >>> dnx.draw_chimera_embedding(T,new_embedding,node_size=10)
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
    # Make sure it fits
    if tuple((np.array(end)-np.array(origin))*2) > tiling.shape:
        raise RuntimeError("Can't spread out")
    # Spread out all qubits by chain
    new_embedding = {}
    for v,chain in embedding.items():
        new_chain = []
        for q in chain:
            tile = np.array(tiling.get_tile(q))
            new_tile = tuple((tile - np.array(origin))*2)
            new_q = tiling.set_tile(q,new_tile)
            new_chain.append(new_q)
        new_embedding[v] = new_chain

    return new_embedding

def open_seam(S, T, embedding, seam, direction=None):
    """
        Args:
            seam
            direction (str:{'left','right','up','down'})

        Example:
            >>> import embera
            >>> S = nx.complete_graph(10)
            >>> T = dnx.chimera_graph(8)
            >>> embedding = minorminer.find_embedding(S,T,random_seed=10)
            >>> dnx.draw_chimera_embedding(T,embedding,node_size=10)
            >>> seam = 2
            >>> direction = 'right'
            >>> new_embedding = embera.transform.embedding.open_seam(S,T,embedding,seam,direction)
            >>> dnx.draw_chimera_embedding(T,new_embedding,node_size=10)

    """
    tiling = DWaveNetworkXTiling(T)

    if direction is 'left':
        shift = lambda tile: tile[1]<=seam
        offset = np.array([0,-1])
    elif direction is 'right':
        shift = lambda tile: tile[1]>=seam
        offset = np.array([0,+1])
    elif direction is 'up':
        shift = lambda tile: tile[0]<=seam
        offset = np.array([-1,0])
    elif direction is 'down':
        shift = lambda tile: tile[0]>=seam
        offset = np.array([+1,0])
    else:
        raise ValueError("Direction not in {'left','right','up','down'}")

    new_embedding = {}
    for v,chain in embedding.items():
        new_chain = []
        for q in chain:
            tile = np.array(tiling.get_tile(q))
            new_tile = tuple(tile + offset) if shift(tile) else tuple(tile)
            new_q = tiling.set_tile(q,new_tile)
            new_chain.append(new_q)
        new_embedding[v] = new_chain

    return new_embedding

def lp_chain_reduce(S, T, embedding):
    """ TODO: Use a linear programming formulation to resolve shorter chains
        from a given embedding.
            1) Turn chains into shared qubits
            2) Create LP formulation
            3) Resolve chains
    """
    return embedding

def greedy_fit(S, T, embedding):
    """ Using a sling window approach, transform the embedding from one region
        of the Chimera graph to another. This is useful when an embedding is
        done for a D-Wave machine and it's necessary to find an identical
        embedding on another D-Wave machine with different yield.

        Algorithm:
            1) Parse embedding and target graph to find margins.
            2) Move qubit to window i and check if nodes are available
            3) If all nodes are available, go to 4, else go to 3
            4) Check if edges are available, if not, return to 2.
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
    # Define flips
    m,n = tiling.shape
    t = tiling.graph['tile']
    #


def reconnect(S, T, embedding, return_overlap=False):
    """ Perform a short run of minorminer to find a valid embedding """
    # Assign current embedding to suspend_chains to preserve the layout
    suspend_chains = {k:[[q] for q in chain] for k,chain in embedding.items()}
    # Run minorminer as a short run without chainlength optimization
    miner_params = {'suspend_chains':suspend_chains,
                    'chainlength_patience':0,
                    'return_overlap':return_overlap}
    return minorminer.find_embedding(S,T,**miner_params)
