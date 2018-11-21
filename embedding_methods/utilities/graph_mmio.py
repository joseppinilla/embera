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
        mtx.name =  name.readline()

        Gcoord = base + prefix + 'coord' + seq
        coord_filepath = mm_dir + Gcoord + MM_EXT
        coord = mmread(coord_filepath) if isfile(coord_filepath) else None
        mtx.coord =  coord

        Gnodename = base + prefix + 'nodename' + seq
        nodename_fp = mm_dir + Gnodename + TXT_EXT
        nodename = open(nodename_fp).readlines() if isfile(nodename_fp) else []
        if all(line[:-1].isdigit() for line in nodename):
            mtx.nodename = [int(line[:-1]) for line in nodename]
        else:
            mtx.nodename = [line[:-1] for line in nodename]

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
        pos = {}
        if mtx.coord is not None:
            _, dim = mtx.coord.shape
            for i, coord in enumerate(mtx.coord):
                if dim==3:
                    x,y,z = coord
                    pos[i] = x+z, y+z
                elif dim==2:
                    x,y = coord
                    pos[i] = x, y
                else:
                    raise ValueError("Invalid XY or XYZ coordinate dimensions")

        labels = { i:label for i, label in enumerate(mtx.nodename) }
        if labels:
            Gnx = nx.relabel_nodes(Gnx, labels, copy=True)
            if pos: pos = {labels[v]:pos[v] for v in labels}

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

        Gname = base + prefix + 'name' + seq
        name_filepath = mm_dir + Gname + TXT_EXT
        with open(name_filepath, 'w') as fp: fp.write(mtx_name)

        if pos:
            coord = [ pos[v] for v in Gnx.nodes ]
            Gcoord = base + prefix + 'coord' + seq
            coord_filepath = mm_dir + Gcoord + MM_EXT
            mmwrite(coord_filepath, np.array(coord))

        Gnodename = base + prefix + 'nodename' + seq
        nodename_filepath = mm_dir + Gnodename + TXT_EXT
        with open(nodename_filepath, 'w') as fp:
            for nodename in Gnx.nodes:
                fp.write('%s\n' % str(nodename))



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    TEST_GRAPH='grid_2d_graph'

    # Write graph to Matrix Market file
    Gnx = nx.grid_2d_graph(4,4)
    Gnx.name = TEST_GRAPH
    pos = {v:v for v in Gnx}
    write_networkx(Gnx, pos=pos)

    # Read Graph from Matrix Market file
    G = read_networkx(TEST_GRAPH)
    nx.draw(G, pos=G.graph['pos'], with_labels=True)
    plt.show()
