import os
import re
import json
import dimod
import tarfile
import requests

def frust_loops_bench(Nq=(0,512), Nsg=(0,3000), s=(0,100),
                      with_degeneracy=False):
    """ Frustrated Ising problems with planted solutions used by Marshall et
        al. in [1]. Linear biases are set to 0, and planted solutions are
        created using [2]. All linear biases h=0. Problem sizes are limited to
        a 512 qubit Chimera graph with imperfections. Benchmark names follow:

            Nq<Number of qubits>_Nsg<Number of loops>_s<Instance number>

        Each benchmark is a dimod.BinaryQuadraticModel with additional key:value
        entries in bqm.info:

            'E0' : <float>
                Ground State if known
            'energy' : list
                Sorted list of known energies
            'degeneracy' : list
                Sorted list of known degeneracies calculated by [3]

        Arguments:

            Nq: (int or tuple(int,int), default=(0,512))
                If int, returns benchmark parameters with size Nq, if found.
                If tuple(a,b), returns benchmark parameters within a <= Nq <= b

            Nsg: (int or tuple(int,int), default=(0,3000))
                If int, returns benchmark parameters with value Nsg, if found.
                If tuple(a,b), returns benchmark parameters within a <= Nsg <= b

            s: (int or tuple(int,int), default=(0,100))
                If int, returns benchmark parameters instance s, if found.
                If tuple, returns benchmark parameters instances a <= s <= b

            with_degeneracy: (bool, default=False)
                If True, only returns instances with degeneracy solutions [3].
                If False, returns all instances regardless of information.

         [1] Marshall, J., Venturelli, D., Hen, I., & Rieffel, E. G. (2019).
         Power of Pausing: Advancing Understanding of Thermalization in
         Experimental Quantum Annealers. Physical Review Applied, 11(4).
         https://doi.org/10.1103/PhysRevApplied.11.044083
         [2] Hen, I., Job, J., Albash, T., Rønnow, T. F., Troyer, M., & Lidar,
         D. A. (2015). Probing for quantum speedup in spin-glass problems with
         planted solutions. Physical Review A - Atomic, Molecular, and Optical
         Physics, 92(4). https://doi.org/10.1103/PhysRevA.92.042325
         [3] Barash, L., Marshall, J., Weigel, M., & Hen, I. (2019). Estimating
         the density of states of frustrated spin systems. New Journal of
         Physics, 21(7). https://doi.org/10.1088/1367-2630/ab2e39
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

    if isinstance(Nq,int):
        Nq_a,Nq_b = Nq,Nq
    elif isinstance(Nq,tuple):
        Nq_a,Nq_b = Nq

    if isinstance(Nsg,int):
        Nsg_a,Nsg_b = Nsg,Nsg
    elif isinstance(Nsg,tuple):
        Nsg_a,Nsg_b = Nsg

    if isinstance(s,int):
        s_a,s_b = s,s
    elif isinstance(s,tuple):
        s_a,s_b = s

    pattern = re.compile('Nq(?P<Nq>\d+)_Nsg(?P<Nsg>\d+)_s(?P<s>\d+)')
    # Unzip, untar, parse
    with tarfile.open(path) as contents:
        for member in contents.getmembers():
            params = pattern.search(member.name)
            if Nq_a<=int(params.group('Nq'))<=Nq_b:
                if Nsg_a<=int(params.group('Nsg'))<=Nsg_b:
                    if s_a<=int(params.group('s'))<=s_b:
                        f = contents.extractfile(member)
                        bqm_ser = json.load(f)
                        bqm = dimod.BinaryQuadraticModel.from_serializable(bqm_ser)
                        if with_degeneracy:
                            if bqm.info['degeneracy']:
                                benchmark_set.append(bqm)
                        else:
                            benchmark_set.append(bqm)

    return benchmark_set
