import embera
db = embera.EmberaDataBase()

# Iterations
N = 2

# Define Embedding methods
import minorminer
def method_A(S,T): minorminer.find_embedding(S,T,random_seed=42)
def method_B(S,T): minorminer.find_embedding(S,T,random_seed=24)
# Select a set of benchmarks.
# e.g. D-Wave tests, NASA, LANL, QCA, Misc.
graph_set = embera.benchmarks.topologies.embera_bench()
# Obtain D-Wave device collection (online and offline)
dw_collection = embera.architectures.generators.dwave_collection()
# Embed graphs using different methods
for G in graph_set:
   for H in dw_collection:
      for i in range(N):
         db.dump_embedding(G,H,method_A(G,H),tags=['A',str(i)])
         db.dump_embedding(G,H,method_B(G,H),tags=['B',str(i)])

# Quantify results
for G in graph_set:
   for l in ['A','B']:
      embeddings = db.load_embeddings(G,H,tags=[l])
      for embedding in embeddings:
          if embedding:
              print(l,embera.evaluation.get_chain_stats(embedding))
