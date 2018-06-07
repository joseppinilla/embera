'''
Test the embedding methods provided in...

Using example from
http://dw-docs.readthedocs.io/en/latest/examples/multi_gate.html

Modified:
Added EmbeddingComposite to bypass need for D-Wave connection

'''

# Size of Pegasus Graph
m = 2
#strucsampler = StructureComposite(ExactSolver(), node_list, edge_list)
#strucsampler = StructureComposite(RandomSampler(), node_list, edge_list)
pegasus = dnx.generators.pegasus.pegasus_graph(m)

strucsampler = StructureComposite(SimulatedAnnealingSampler(), chimera.nodes, chimera.edges)
sampler = EmbeddingComposite(strucsampler)
