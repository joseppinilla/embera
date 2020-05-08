import os
import json
import dimod
import embera
import tarfile
import requests

def init_bm(G, RNG_SEED=None):
    embera.random.seed(RNG_SEED)
    """ Simulated Ising parameters of initialization of a Boltzmann Machine. """
    N = len(G.nodes)
    node_biases = embera.random.bimodal(N,loc1=.0,scale1=.25,size1=N//2,
                              loc2=.75,scale2=.075,size2=N-N//2)
    edge_biases = embera.random.uniform(low=-2.0,high=2.0,size=len(G.edges))
    h = {v:b for v,b in zip(G.nodes,node_biases)}
    J = {(u,v):b for (u,v),b in zip(G.edges,edge_biases) if u!=v}

    return dimod.BinaryQuadraticModel.from_ising(h,J)

def trained_bm(G, RNG_SEED=None):
    embera.random.seed(RNG_SEED)
    """ Simulated Ising parameters of a trained Boltzmann Machine. """
    node_biases = embera.random.uniform(low=-1.0,high=1.0,size=len(G.nodes))
    edge_biases = embera.random.normal(scale=0.5,size=len(G.edges))
    h = {v:b for v,b in zip(G.nodes,node_biases)}
    J = {(u,v):b for (u,v),b in zip(G.edges,edge_biases) if u!=v}

    return dimod.BinaryQuadraticModel.from_ising(h,J)

def csp(G, RNG_SEED=None):
    embera.random.seed(RNG_SEED)
    """ Simulated Ising parameters of a Constrained Satisfaction Problem. """
    node_vals = [-1.0,-0.8,-0.4,-0.2,0.0,0.2,0.4,0.8,1.0]
    node_biases = embera.random.categorical(len(G.nodes), node_vals)
    edge_vals = [-2.0,-1.0,1.0,2.0]
    edge_biases = embera.random.categorical(len(G.edges), edge_vals)
    h = {v:b for v,b in zip(G.nodes,node_biases)}
    J = {(u,v):b for (u,v),b in zip(G.edges,edge_biases) if u!=v}

    return dimod.BinaryQuadraticModel.from_ising(h,J)

def frust_loops_bench():
    """" Frustrated Ising problems with planted solutions used by Marshall et
         al. in [1]. Linear biases are set to 0, and planted
         solutions are created using [2].

        Each benchmark is a dimod.BinaryQuadraticModel with added information:
            'E0' : <float>
                Ground State if known
            'energy' : list
                Sorted list of known energies
            'degeneracy' : list
                Sorted list of known degeneracies

         [1] Marshall, J., Venturelli, D., Hen, I., & Rieffel, E. G. (2019).
         Power of Pausing: Advancing Understanding of Thermalization in
         Experimental Quantum Annealers. Physical Review Applied, 11(4).
         https://doi.org/10.1103/PhysRevApplied.11.044083
         [2] Hen, I., Job, J., Albash, T., Rønnow, T. F., Troyer, M., & Lidar,
         D. A. (2015). Probing for quantum speedup in spin-glass problems with
         planted solutions. Physical Review A - Atomic, Molecular, and Optical
         Physics, 92(4). https://doi.org/10.1103/PhysRevA.92.042325
    """
    benchmark_set = []
    path = "./frust_loops.tar.gz"
    url = "http://www.ece.ubc.ca/~jpinilla/resources/embera/frust_loops/frust_loops.tar.gz"

    # Download
    if not os.path.isfile(path):
        print(f"-> Downloading Frustrated Loops benchmarks to {path}")
        with open(path, 'wb') as f:
            response = requests.get(url)
            f.write(response.content)
    # Unzip, untar, unpickle
    with tarfile.open(path) as contents:
        for member in contents.getmembers():
            f = contents.extractfile(member)
            bqm_ser = json.load(f)
            bqm = dimod.BinaryQuadraticModel.from_serializable(bqm_ser)
            benchmark_set.append(bqm)

    return benchmark_set
