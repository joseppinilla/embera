'''
Test the embedding methods provided in...

Using example from
http://dw-docs.readthedocs.io/en/latest/examples/multi_gate.html

Modified:
Added EmbeddingComposite to bypass need for D-Wave connection

'''
from embedding_methods.embedding import MinorEmbeddingComposite

import dwave_networkx as dnx
import dwavebinarycsp as dcsp
from dwave.system.composites.embedding import EmbeddingComposite
from dimod.reference.composites.structure import StructureComposite
from dimod.reference.samplers.exact_solver import ExactSolver
from dimod.reference.samplers.random_sampler import RandomSampler
from dimod.reference.samplers.simulated_annealing import SimulatedAnnealingSampler

def logic_circuit(a, b, c, d, z):
    not1 = not b
    or2 = b or c
    and3 = a and not1
    or4 = or2 or d
    and5 = and3 and or4
    not6 = not or4
    or7 = and5 or not6
    return (z == or7)

# Size of Chimera Graph
m,n,t = 16,16,4
#strucsampler = StructureComposite(ExactSolver(), node_list, edge_list)
#strucsampler = StructureComposite(RandomSampler(), node_list, edge_list)
chimera = dnx.generators.chimera.chimera_graph(m,n,t)
strucsampler = StructureComposite(SimulatedAnnealingSampler(), chimera.nodes, chimera.edges)
sampler = MinorEmbeddingComposite(strucsampler)

# Problem setup
csp = dcsp.ConstraintSatisfactionProblem(dcsp.BINARY)
csp.add_constraint(logic_circuit, ['a', 'b', 'c', 'd', 'z'])

# Convert the binary constraint satisfaction problem to a binary quadratic model
bqm = dcsp.stitch(csp)

# Set up a solver using the structured sampler
response = sampler.sample(bqm)

# Check how many solutions meet the constraints (are valid)
valid, invalid, data = 0, 0, []

for datum in response.data():
    sample, energy = datum
    if (csp.check(sample)):
        valid = valid+1
        data.append((sample, energy, '1'))
    else:
        invalid = invalid+0
        data.append((sample, energy, '0'))

print(valid, invalid)
