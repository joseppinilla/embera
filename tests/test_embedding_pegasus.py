'''
Test the embedding methods provided in...

Using example from
http://dw-docs.readthedocs.io/en/latest/examples/multi_gate.html

Modified:
Added EmbeddingComposite to bypass need for D-Wave connection

'''
import operator
import minorminer
import dwave_networkx as dnx
import dwavebinarycsp as dcsp
import matplotlib.pyplot as plt
import dwavebinarycsp.factories.constraint.gates as gates
from dimod.reference.samplers.exact_solver import ExactSolver
from dimod.reference.samplers.random_sampler import RandomSampler
from dimod.reference.composites.structure import StructureComposite
from embedding_methods.composites.embedding import EmbeddingComposite
from dimod.reference.samplers.simulated_annealing import SimulatedAnnealingSampler


csp = dcsp.ConstraintSatisfactionProblem(dcsp.BINARY)
csp.add_constraint(operator.ne, ['b', 'not1'])  # add NOT 1 gate
csp.add_constraint(gates.or_gate(['b', 'c', 'or2']))  # add OR 2 gate
csp.add_constraint(gates.and_gate(['a', 'not1', 'and3']))  # add AND 3 gate
csp.add_constraint(gates.or_gate(['d', 'or2', 'or4']))  # add OR 4 gate
csp.add_constraint(gates.and_gate(['and3', 'or4', 'and5']))  # add AND 5 gate
csp.add_constraint(operator.ne, ['or4', 'not6'])  # add NOT 6 gate
csp.add_constraint(gates.or_gate(['and5', 'not6', 'z']))  # add OR 7 gate

# Size of Pegasus Graph
m = 2
pegasus = dnx.generators.pegasus.pegasus_graph(m)

#strucsampler = StructureComposite(ExactSolver(), chimera.nodes, chimera.edges)
#strucsampler = StructureComposite(RandomSampler(), chimera.nodes, chimera.edges)
strucsampler = StructureComposite(SimulatedAnnealingSampler(), pegasus.nodes, pegasus.edges)
sampler = EmbeddingComposite(strucsampler, minorminer)

# Convert the binary constraint satisfaction problem to a binary quadratic model
bqm = dcsp.stitch(csp)

# Set up a solver using the structured sampler
response = sampler.sample(bqm, num_reads=100)

embedding = sampler.get_embedding()

# Check how many solutions meet the constraints (are valid)
valid, invalid, data = 0, 0, []

for datum in response.data():
    sample, energy = datum
    if (csp.check(sample)):
        valid = valid+1
        data.append((sample, energy, '1'))
    else:
        invalid = invalid+1
        data.append((sample, energy, '0'))

print(valid, invalid)

dnx.draw_chimera_embedding(chimera,embedding)
plt.show()

plt.ion()
plt.scatter(range(len(data)), [x[1] for x in data], c=['y' if (x[2] == '1') else 'r' for x in data],marker='.')
