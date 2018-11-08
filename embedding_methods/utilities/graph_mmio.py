""" Methods to interface with the Sparse Matrix Collection.

T. A. Davis, The University of Florida Sparse Matrix Collection,
ACM Transactions on Mathematical Software (submitted, 2009),
also available as a tech report at:
http://www.cise.ufl.edu/~davis/techreports/matrices.pdf

"""
import os
import numpy as np
import networkx as nx

from os.path import isfile

from scipy.sparse import coo_matrix
from scipy.io import mmread, mminfo, mmwrite


__all__ = ['read', 'read_networkx', 'write_networkx']

MM_DIR = './graphs/'
TXT_EXT = '.txt'
MM_EXT = '.mtx'


def read(mtx_name, mm_dir=MM_DIR, data=True):
    """ Read from a Matrix Market file and return a NumPy array with
    optional auxiliary attributes found in the same directory.
    Args:
        mtx_name (str): base name of '*.mtx' matrix file

    Optional Parameters:
        mm_dir (str): directory with Matrix Market files.
        data (bool): if True include name, coord, nodename as attributes
        in the output.

    Return:
        mtx (ndarray): array recovered from given file with or w/o auxiliary data.

    """
    base, prefix, seq = mtx_name.partition('_G')

    mtx_filepath = mm_dir + mtx_name + MM_EXT
    mtx = mmread(mtx_filepath)

    if data:
        Gname = base + prefix + 'name' + seq
        name_filepath = mm_dir + Gname + TXT_EXT
        name =  open(name_filepath) if isfile(name_filepath) else mtx_name
        mtx.name =  name

        Gcoord = base + prefix + 'coord' + seq
        coord_filepath = mm_dir + Gcoord + MM_EXT
        coord = mmread(coord_filepath) if isfile(coord_filepath) else None
        mtx.coord =  coord

        Gnodename = base + prefix + 'nodename' + seq
        nodename_filepath = mm_dir + Gnodename + MM_EXT
        nodename = mmread(nodename_filepath) if isfile(nodename_filepath) else []
        mtx.nodename =  nodename

    return mtx

def read_networkx(mtx_name, mm_dir=MM_DIR, data=True):
    """ Read from a Matrix Market file and return a NetworkX Graph
    with auxiliary attributes found in the same directory.
    Args:
        mtx_name (str): base name of '*.mtx' matrix file

    Optional Parameters:
        mm_dir (str): directory with Matrix Market files.
        data (bool): if True include name, coord, nodename as attributes
        in the output.

    Return:
        Gnx (Graph): NetworkX Graph recovered from given file with
        or w/o auxiliary data.
    """
    mtx = read(mtx_name, mm_dir, data)
    Gnx = nx.Graph(mtx)

    if data:
        Gnx.name = mtx.name

        labels = { i:int(label) for i, label in enumerate(mtx.nodename) }
        if labels: nx.relabel_nodes(Gnx, labels, copy=False)
        pos = {}
        if mtx.coord is not None:
            _, dim = mtx.coord.shape
            for i, v in labels.items():
                if dim==3:
                    x,y,z = mtx.coord[i]
                    pos[v] = x+z, y+z
                elif dim==2:
                    x,y = mtx.coord[i]
                    pos[v] = x, y
                else:
                    raise ValueError("Invalid XY or XYZ coordinate dimensions")
        Gnx.graph['pos'] = pos

    return Gnx

def read_mapping(mtx_a_name, mtx_b_name, mm_dir=MM_DIR, data=True):
    """

    """
    #TODO: Read out embedding from Matrix Market files
    embedding = {}
    return embedding

def write_networkx(Gnx, pos=None, mtx_name=None, mm_dir=MM_DIR, data=True, **params):
    """ Write to a Matrix Market file from a NetworkX Graph
    with auxiliary attributes.
    Args:
        Gnx (Graph): NetworkX Graph recovered from given file with
        or w/o auxiliary data.
        pos (dict): optional dictionary of XY/XYZ locations of nodes
        mtx_name (str): base name of '*.mtx' matrix file. If not given, use
        name in Graph.name

    Optional Parameters:
        mm_dir (str): directory with Matrix Market files.
        data (bool): if True write files for name, coord, nodename
    """
    if mtx_name is None:
        if Gnx.name:
            mtx_name = Gnx.name
        else:
            raise RuntimeError("Name of Graph not found.")

    mtx_filepath = mm_dir + mtx_name + MM_EXT

    mtx = coo_matrix(nx.to_numpy_matrix(Gnx))
    mmwrite(mtx_filepath, mtx, **params)

    if data:
        base, prefix, seq = mtx_name.partition('_G')

        if Gnx.name:
            Gname = base + prefix + 'name' + seq
            name_filepath = mm_dir + Gname + TXT_EXT
            with open(name_filepath, 'w') as fp: fp.write(Gname)

        if pos:
            Gcoord = base + prefix + 'coord' + seq
            coord_filepath = mm_dir + Gcoord + MM_EXT
            coord = np.array( list(pos.values()) )
            mmwrite(coord_filepath, coord)

        Gnodename = base + prefix + 'nodename' + seq
        nodename_filepath = mm_dir + Gnodename + MM_EXT
        Gnx_ints = np.array(Gnx.nodes())
        nodename = np.ndarray(  shape=(len(Gnx),1),
                                dtype=int,
                                buffer=Gnx_ints)
        mmwrite(nodename_filepath, nodename)



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    TEST_GRAPH='grid_2d_graph'

    # Write graph to Matrix Market file
    Gnx = nx.grid_2d_graph(4,4)
    Gnx.name = TEST_GRAPH
    pos = {v:v for v in Gnx}
    write_networkx(Gnx, pos=pos)

    # Read Graph from Matrix Market file
    Gnx=read_networkx(TEST_GRAPH)
    nx.draw(Gnx, pos=Gnx.graph['pos'])
    plt.show()
