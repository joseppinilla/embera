import minorminer

import numpy as np

from embera.utilities.decorators import nx_graph
from embera.architectures.tiling import DWaveNetworkXTiling

__all__ = ['translate','mirror','rotate','spread_out','open_seam',
           'iter_sliding_window', 'greedy_fit','reconnect']

""" ################### Naive Embedding Transformations ####################
    Transformation methods for embeddings onto Tiled D-Wave Architectures

    Arguments:
        T: (networkx.Graph)
            A NetworkX Graph with the construction parameters generated using
            `dwave_networkx`:
                    family : {'chimera','pegasus', ...}
                    rows : (int)
                    columns : (int)
                    labels : {'coordinate', 'int', 'nice'}

        embedding: (dict)
            A dictionary mapping variable names to lists of labels in T

    Note:
        A valid embedding is not guaranteed from these transformations. To
        generate a valid embedding from the result of this transformation, use
        `embera.transform.embedding.reconnect(S,T,new_embedding)` or similar.

"""

def translate(T, embedding, origin=(0,0)):
    """ Transport the embedding on the same graph to re-distribute qubit
        assignments.

        Optional arguments:
            origin: (tuple)
                A tuple of tile coordinates pointing to where the left-uppermost
                occupied tile in the embedding should move to. All other tiles
                are moved relative to the origin.

        Example:
            >>> import embera
            >>> import minorminer
            >>> import networkx as nx
            >>> import dwave_networkx as dnx
            >>> S = nx.complete_graph(11)
            >>> T = dnx.chimera_graph(7)
            >>> embedding = minorminer.find_embedding(S,T)
            >>> dnx.draw_chimera_embedding(T,embedding,node_size=10)
            >>> origin = (2,3)
            >>> new_embedding = embera.transform.embedding.translate(T,embedding,origin)
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


def mirror(T, embedding, axis=0):
    """ Flip the embedding on the same graph to re-distribute qubit
        assignments.

        Optional arguments:

            axis: {0,1}
                0 toflip on horizontal and 1 to flip on vertical

        Example:
            >>> import embera
            >>> import minorminer
            >>> import networkx as nx
            >>> import dwave_networkx as dnx
            >>> S = nx.complete_graph(11)
            >>> T = dnx.chimera_graph(7)
            >>> embedding = minorminer.find_embedding(S,T)
            >>> dnx.draw_chimera_embedding(T,embedding,node_size=10)
            >>> axis = 1
            >>> new_embedding = embera.transform.embedding.mirror(T,embedding,axis)
            >>> dnx.draw_chimera_embedding(T,new_embedding,node_size=10)
    """
    #TODO: not supported for Pegasus yet
    if T.graph['family']=='pegasus': return {k:[] for k in embedding}
    tiling = DWaveNetworkXTiling(T)
    shape = np.array(tiling.shape)
    # Define flips
    m,n = tiling.shape
    t = tiling.graph['tile']
    if axis == 0:
        new_tile = lambda i,j: (i,n-j-1)
        new_k = lambda k,shore: k if shore else t-k-1
    elif axis == 1:
        new_tile = lambda i,j: (m-i-1,j)
        new_k = lambda k,shore: t-k-1 if shore else k
    else:
        raise ValueError("Value of axis not supported")
    # Mirror all qubits by chain
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

def rotate(T, embedding, theta=90):
    """ Rotate the embedding on the same graph to re-distribute qubit
        assignments. If a perfect fit isn't found, due to disabled qubits,
        the invalid embedding is still returned.

        Optional arguments:

            theta: ({0,90,180,270,360,-90,-180,-270})
                Rotation angle.

        Example:
            >>> import embera
            >>> import minorminer
            >>> import networkx as nx
            >>> import dwave_networkx as dnx
            >>> S = nx.complete_graph(11)
            >>> T = dnx.chimera_graph(7)
            >>> embedding = minorminer.find_embedding(S,T)
            >>> dnx.draw_chimera_embedding(T,embedding,node_size=10)
            >>> theta = 270
            >>> new_embedding = embera.transform.embedding.rotate(T,embedding,theta)
            >>> dnx.draw_chimera_embedding(T,new_embedding,node_size=10)
    """
    #TODO: not supported for Pegasus yet
    if T.graph['family']=='pegasus': return {k:[] for k in embedding}
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

def spread_out(T, embedding, sheer=None):
    """ Transform the tile assignment to spread out the embedding starting from
        tile (0,0) and placing originally adjacent tiles 1 extra tile away.

        Optional arguments:

            sheer: {None,0, 1}
                Perform a translation of every odd column (sheer=0) or row
                (sheer=1)

        Example:
            >>> import embera
            >>> import minorminer
            >>> import networkx as nx
            >>> import dwave_networkx as dnx
            >>> S = nx.complete_graph(17)
            >>> T = dnx.chimera_graph(8)
            >>> embedding = minorminer.find_embedding(S,T)
            >>> dnx.draw_chimera_embedding(T,embedding, node_size=10)
            >>> new_embedding = embera.transform.embedding.spread_out(T,embedding)
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
    if sheer == None:
        shift = lambda tile,origin: (tile-origin)*2
    elif sheer == 0:
        shift = lambda tile,origin: (tile-origin)*2+np.flip((tile-origin)%[2,1])
    elif sheer == 1:
        shift = lambda tile,origin: (tile-origin)*2+np.flip((tile-origin)%[1,2])

    for v,chain in embedding.items():
        new_chain = []
        for q in chain:
            tile = np.array(tiling.get_tile(q))
            new_tile = tuple(shift(tile,origin))
            new_q = tiling.set_tile(q,new_tile)
            new_chain.append(new_q)
        new_embedding[v] = new_chain

    return new_embedding

def open_seam(T, embedding, seam, direction):
    """
        Arguments (continued):
            seam: (int)
                If direction is 'left' or 'right', seam corresponds to the
                column number. If direction is 'up' or 'down', seam corresponds
                to the row number.

            direction: (None or str:{'left','right','up','down'})
                Given a seam index, that column/row is cleared and all utilized
                qubits in the embedding are shifted in this direction.

        Example:
            >>> import embera
            >>> import minorminer
            >>> import networkx as nx
            >>> import dwave_networkx as dnx
            >>> S = nx.complete_graph(10)
            >>> T = dnx.chimera_graph(8)
            >>> embedding = minorminer.find_embedding(S,T,random_seed=10)
            >>> dnx.draw_chimera_embedding(T,embedding,node_size=10)
            >>> seam = 2
            >>> direction = 'right'
            >>> new_embedding = embera.transform.embedding.open_seam(T,embedding,seam,direction)
            >>> dnx.draw_chimera_embedding(T,new_embedding,node_size=10)
    """
    tiling = DWaveNetworkXTiling(T)

    if direction == 'left':
        shift = lambda tile: tile[1]<=seam
        offset = np.array([0,-1])
    elif direction == 'right':
        shift = lambda tile: tile[1]>=seam
        offset = np.array([0,+1])
    elif direction == 'up':
        shift = lambda tile: tile[0]<=seam
        offset = np.array([-1,0])
    elif direction == 'down':
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

def iter_sliding_window(T, embedding):
    """ Use a sliding window approach to iteratively transport the embedding
        from one region of the Chimera graph to another.

        Example:
            >>> import embera
            >>> import minorminer
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


""" ########################### Optimize Embedding #########################
    Transformation methods to try and find a valid embedding from an invalid one

    Arguments:
        S: (networkx.Graph or list of 2-tuples)
            A NetworkX Graph with the adjacency of the embedded graph, or a list
            of edges.

        T: (networkx.Graph)
            A NetworkX Graph with the construction parameters generated using
            `dwave_networkx`:
                    family : {'chimera','pegasus', ...}
                    rows : (int)
                    columns : (int)
                    labels : {'coordinate', 'int', 'nice'}

        embedding: (dict)
            A dictionary mapping variable names to lists of labels in T
"""
@nx_graph(0)
def lp_chain_reduce(S, T, embedding):
    """ TODO: Use a linear programming formulation to resolve shorter chains
        from a given embedding.
            1) Turn chains into shared qubits
            2) Create LP formulation
            3) Resolve chains
    """
    import warnings
    warnings.warn("WIP: Not implemented yet")
    return embedding

@nx_graph(0)
def greedy_fit(S, T, embedding):
    """ Using a sling window approach, transform the embedding from one region
        of the Chimera graph to another. This is useful when an embedding is
        done for a D-Wave machine and it's necessary to find an identical
        embedding on another D-Wave machine with different yield.

        Algorithm:
            1) Parse embedding and target graph to find margins.
            2) Move qubit to window i and check if nodes are available
            3) If all edges are available, return embedding, else go to 4
            4) Test same window with 90, 180, and 270 rotations.

        Example:
            >>> import embera
            >>> import minorminer
            >>> import networkx as nx
            >>> import dwave_networkx as dnx
            >>> S = nx.complete_graph(11)
            >>> T = dnx.chimera_graph(7)
            >>> embedding = minorminer.find_embedding(S,T)
            >>> dnx.draw_chimera_embedding(T,embedding,node_size=10)
            >>> new_embedding = embera.transform.embedding.greedy_fit(S,T,embedding)
            >>> dnx.draw_chimera_embedding(T,new_embedding,node_size=10)
    """
    interactions = lambda u,v,E:((s,t) for s in E[u] for t in E[v])
    is_connected = lambda edges: any(T.has_edge(s,t) for s,t in edges)
    for emb in iter_sliding_window(T,embedding):
        if all(is_connected(interactions(u,v,emb)) for u,v in S.edges):
            return emb
        mir = mirror(T,emb)
        if all(is_connected(interactions(u,v,mir)) for u,v in S.edges):
            return mir
        e90 = rotate(T,emb,90)
        if all(is_connected(interactions(u,v,mir)) for u,v in S.edges):
            return e90
        e180 = rotate(T,emb,180)
        if all(is_connected(interactions(u,v,mir)) for u,v in S.edges):
            return e180
        e270 = rotate(T,emb,270)
        if all(is_connected(interactions(u,v,mir)) for u,v in S.edges):
            return e270
    return {}

def reconnect(S, T, embedding, **miner_params):
    """ Perform a short run of minorminer to find a valid embedding """
    # Assign current embedding to suspend_chains to preserve the layout
    suspend_chains = {k:[q for q in chain if q in T] for k,chain in embedding.items()}
    # Run minorminer as a short run without chainlength optimization
    miner_params['suspend_chains'] = suspend_chains
    return minorminer.find_embedding(S,T,**miner_params)
