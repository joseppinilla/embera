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

MM_DIR = './graphs'
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

    if not os.path.isdir(mm_dir):
        raise ValueError("Given directory name does not exist.")
    dir_abspath = os.path.abspath(mm_dir)

    mtx_filepath = os.path.join(dir_abspath, mtx_name + MM_EXT)
    mtx = mmread(mtx_filepath)

    if data:
        Gname = base + prefix + 'name' + seq
        name_filepath = os.path.join(dir_abspath, Gname + TXT_EXT)
        name =  open(name_filepath) if isfile(name_filepath) else mtx_name
        mtx.name =  name.readline()

        Gcoord = base + prefix + 'coord' + seq
        coord_filepath = os.path.join(dir_abspath, Gcoord + MM_EXT)
        coord = mmread(coord_filepath) if isfile(coord_filepath) else None
        mtx.coord =  coord

        Gnodename = base + prefix + 'nodename' + seq
        nodename_fp = os.path.join(dir_abspath, Gnodename + TXT_EXT)
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

    if not os.path.isdir(mm_dir):
        os.makedirs(mm_dir)
    dir_abspath = os.path.abspath(mm_dir)
    mtx_filepath = os.path.join(dir_abspath, mtx_name + MM_EXT)

    mtx = coo_matrix(nx.to_numpy_matrix(Gnx))
    mmwrite(mtx_filepath, mtx, **params)

    if data:
        base, prefix, seq = mtx_name.partition('_G')

        Gname = base + prefix + 'name' + seq
        name_filepath = os.path.join(dir_abspath, Gname + TXT_EXT)
        with open(name_filepath, 'w') as fp: fp.write(mtx_name)

        if pos:
            coord = [ pos[v] for v in Gnx.nodes ]
            Gcoord = base + prefix + 'coord' + seq
            coord_filepath = os.path.join(dir_abspath, Gcoord + MM_EXT)
            mmwrite(coord_filepath, np.array(coord))

        Gnodename = base + prefix + 'nodename' + seq
        nodename_filepath = os.path.join(dir_abspath, Gnodename + TXT_EXT)
        with open(nodename_filepath, 'w') as fp:
            for nodename in Gnx.nodes:
                fp.write('%s\n' % str(nodename))



if __name__ == "__main__":
    import random
    import matplotlib.pyplot as plt

    J_RANGE = [-2.0,2.0]
    name ='GRID_2D_16X16'
    mm_dir = './graphs'

    # Write graph to Matrix Market file
    Gnx = nx.grid_2d_graph(16,16)
    Gnx.name = name
    pos = {v:v for v in Gnx}
    for (u, v, data) in Gnx.edges(data=True):
        data['weight'] = random.uniform(*J_RANGE)

    comments = "2D Grid 16x16"
    write_networkx(Gnx, pos=pos, mtx_name=name, mm_dir=mm_dir, comment=comments)

    # Read Graph from Matrix Market file to visually verify pos
    G = read_networkx(name, mm_dir=mm_dir)
    nx.draw(G, pos=G.graph['pos'], with_labels=True)
    plt.show()
