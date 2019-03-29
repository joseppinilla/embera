""" Example of using the checkerboard transform. This code compares the
exaxt results from running with and without the checkeboard transformation.
"""
import networkx as nx
import matplotlib.pyplot as plt

from embedding_methods import dw2000q_graph
from embedding_methods import CompleteBipartitePlacer
from embedding_methods import EmbeddingComposite
from embedding_methods import CheckerboardTransformComposite

from dimod import ExactSolver
from dimod import BinaryQuadraticModel, StructureComposite

# Create trivial Ising
Sg = nx.complete_bipartite_graph(4,4)
Sg.vartype = 'SPIN'
for v, data in Sg.nodes(data=True):
    data['bias'] = -2
for (u, v, data) in Sg.edges(data=True):
    data['bias'] = -1

# Create BQM
bqm = BinaryQuadraticModel.from_networkx_graph(Sg)

# Create target graph
Tg = dw2000q_graph()
nodelist = list(Tg.nodes())
edgelist = list(Tg.edges())

############# Create Composite Solvers

# Exact solver for this example size (8 nodes)
exact_solver = ExactSolver()
# Add structure of Chimera
struct_solver = StructureComposite(exact_solver, nodelist, edgelist)

# Solver without transformations
solver = EmbeddingComposite(struct_solver)

# Solver with transformations. By default uses chimera.
checkered_solver = CheckerboardTransformComposite(struct_solver)
transform_solver = EmbeddingComposite(checkered_solver)

# Create node placement
placer = CompleteBipartitePlacer(Sg, Tg)
candidates = placer.get_candidates()

solver.set_embedding(candidates)
response = solver.sample(bqm)
response.first

transform_solver.set_embedding(candidates)
transform_response = transform_solver.sample(bqm)

# Compare ground state
assert(response.first==transform_response.first)

# Compore distributions
energies = [datum.energy for datum in response.data()
                for _ in range(datum.num_occurrences)]
transform_energies = [datum.energy for datum in transform_response.data()
                for _ in range(datum.num_occurrences)]

# Plot of distributions.
# NOTE: The frequencies in the transformed responses should be 4X those
# of the one without transformations.

plt.subplot(1,2,1)
_ = plt.hist(energies)
plt.title('ExactSolver')
plt.subplot(1,2,2)
_ = plt.hist(transform_energies)
plt.title('Checkerboard\n 4X ExactSolver')
plt.show()
