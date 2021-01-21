import embera
db = embera.emberaDB()
# Define Embedding methods
method_A = ...
method_B = ...
...
# Select a set of benchmarks.
# e.g. D-Wave tests, NASA, LANL, QCA, Misc.
graph_set = embera.benchmark.topologies.embera_bench()
# Obtain D-Wave device collection (online and offline)
dw_collection = embera.architectures.dwave_collection()
# Embed graphs using different methods
for G in graph_set:
   for H in dw_collection:
      for i in range(N):
         db.dump_embedding(method_A(G,H),labels=['A',i])
         db.dump_embedding(method_B(G,H),labels=['B',i])
         ...
# Quantify results
for G in graph_set:
   for l in ['A','B']:
      embeddings = db.load_embeddings(G,H,labels=[l]):
      embera.evaluation.embeddings(embeddings) # embedding_<metric>
