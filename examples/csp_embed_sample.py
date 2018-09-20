'''
Test the embedding methods.

Using example from
http://dw-docs.readthedocs.io/en/latest/examples/multi_gate.html

Modified:
Added StructureComposite to bypass need for D-Wave connection

'''
import minorminer
import dwave_networkx as dnx
import matplotlib.pyplot as plt

import operator
import dwavebinarycsp as dcsp
import dwavebinarycsp.factories.constraint.gates as gates

from dimod.reference.composites.structure import StructureComposite
from dimod.reference.samplers.exact_solver import ExactSolver
from dimod.reference.samplers.random_sampler import RandomSampler

from embedding_methods.composites.embedding import EmbeddingComposite

csp = dcsp.ConstraintSatisfactionProblem(dcsp.BINARY)
csp.add_constraint(operator.ne, ['b', 'not1'])  # add NOT 1 gate
csp.add_constraint(gates.or_gate(['b', 'c', 'or2']))  # add OR 2 gate
csp.add_constraint(gates.and_gate(['a', 'not1', 'and3']))  # add AND 3 gate
csp.add_constraint(gates.or_gate(['d', 'or2', 'or4']))  # add OR 4 gate
csp.add_constraint(gates.and_gate(['and3', 'or4', 'and5']))  # add AND 5 gate
csp.add_constraint(operator.ne, ['or4', 'not6'])  # add NOT 6 gate
csp.add_constraint(gates.or_gate(['and5', 'not6', 'z']))  # add OR 7 gate

# Convert the binary constraint satisfaction problem to a binary quadratic model
bqm = dcsp.stitch(csp)

# Size of Chimera Graph
m,n,t = 4,4,4
chimera = dnx.generators.chimera.chimera_graph(m,n,t)

# Select underlying sampler
structsampler = StructureComposite(ExactSolver(), chimera.nodes, chimera.edges)
#structsampler = StructureComposite(RandomSampler(), chimera.nodes, chimera.edges)

# Select embedding method
sampler = EmbeddingComposite(structsampler, minorminer)

# Sample BQM from the structured sampler
embedding = sampler.get_bqm_embedding(bqm)
response = sampler.sample(bqm)

# Sample Ising from the structured sampler
#h, J = bqm.linear, bqm.quadratic
#embedding = sampler.get_ising_embedding(h, J)
#response =  sampler.sample_ising(h,J)

plt.figure(1)
dnx.draw_chimera_embedding(chimera, embedding, node_size=50)
print('sum: %s' % sum(len(v) for v in embedding.values()))
print('max: %s' % max(len(v)for v in embedding.values()))

# Check how many solutions meet the constraints (are valid)
valid, invalid, data = 0, 0, []

for datum in response.data():
    sample, energy = datum.sample, datum.energy
    if (csp.check(sample)):
        valid = valid+1
        data.append((sample, energy, '1'))
    else:
        invalid = invalid+1
        data.append((sample, energy, '0'))
print(valid, invalid)
plt.figure(2)
plt.scatter(range(len(data)), [x[1] for x in data], c=['y' if (x[2] == '1') else 'r' for x in data],marker='.')

plt.show()
