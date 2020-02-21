""" Embera Embedding Class """

class Embedding(dict):

    properties = {}
    def __init__(self, embedding, **properties):
        super(Embedding,self).__init__(embedding)
        self.properties = properties

    def qubit_labels(self):
        """ Inverse mapping of qubits to source labels """
        return {s:v for v,chain in self.items() for s in chain}

    """ ############################ Histograms ############################ """
    def chain_histogram(self):
        # Based on dwavesystems/minorminer quality_key by Boothby, K.
        hist = {}
        for s in map(len,self.values()):
            hist[s] = 1 + hist.get(s, 0)

        return hist

    def interactions_histogram(self, source_edgelist, target_edgelist):
        interactions = self.edge_interactions(source_edgelist,target_edgelist)

        hist = {}
        for size in map(len,interactions.values()):
            hist[size] = 1 + hist.get(size, 0)

        return hist

    """ ########################## Source Metrics ########################## """
    def node_interactions(self, source_edgelist, target_edgelist):
        """ List of qubit interactions for each source node """
        edge_inters = self.edge_interactions(source_edgelist,target_edgelist)

        node_inters = {}
        for (u,v),ie in edge_inters.items():
            for (s,t) in ie:
                node_inters[u] = [(s,t)] + node_inters.get(u,[])
                node_inters[v] = [(t,s)] + node_inters.get(v,[])

        return node_inters

    def edge_interactions(self, source_edgelist, target_edgelist):
        """ List of qubit interactions for each source edge """
        if not self: return {}
        target_adj = {}
        for s,t in target_edgelist:
            if (s==t): continue
            target_adj[s] = [t] + target_adj.get(s,[])
            target_adj[t] = [s] + target_adj.get(t,[])

        edge_inters = {}
        for u,v in source_edgelist:
            if (u==v): continue
            for s in self[u]:
                for t in self[v]:
                    if t in target_adj[s]:
                        edge_inters[(u,v)] = [(s,t)] + edge_inters.get((u,v),[])

        return edge_inters

    """ ########################## Target Metrics ########################## """
    def qubit_connections(self, source_edgelist, target_edgelist):
        """ List of source node edges for each qubit """
        if not self: return {}
        interactions = self.edge_interactions(source_edgelist,target_edgelist)

        connections = {}
        for (u,v),edge_interactions in interactions.items():
            for s,t in edge_interactions:
                connections[s] = [(u,v)] + connections.get(s,[])
                connections[t] = [(v,u)] + connections.get(t,[])

        return connections

    def qubit_connectivity(self, source_edgelist, target_edgelist):
        """ Fraction of unique qubit interactions over degree of source node """
        q_labels = self.qubit_labels()
        connections = self.qubit_connections(source_edgelist,target_edgelist)

        source_degree = {}
        for u,v in source_edgelist:
            if (u==v): continue
            source_degree[u] = 1 + source_degree.get(u,0)
            source_degree[v] = 1 + source_degree.get(v,0)

        connectivity = {}
        for s,edges in connections.items():
            u = q_labels[s]
            edgeset = set(edges)
            connectivity[s] = len(edgeset)/source_degree[u]

        return connectivity

    def qubit_interactions(self, source_edgelist, target_edgelist, active=True):
        """ List of active (or inactive) qubit interactions for each qubit """
        q_labels = self.qubit_labels()

        source_adj = {}
        for u,v in source_edgelist:
            if (u==v): continue
            source_adj[u] = [v] + source_adj.get(u,[])
            source_adj[v] = [u] + source_adj.get(v,[])

        inters = {}
        for s,t in target_edgelist:
            u = q_labels[s]
            v = q_labels[t]
            if u==v: continue
            if (v in source_adj[u]) ^ (not active):
                inters[s] = [t] + inters.get(s,[])
                inters[t] = [s] + inters.get(t,[])

        return inters

    """ ############################# Quality ############################# """
    @property
    def max_chain(self):
        hist = self.chain_histogram()
        return max(hist)

    @property
    def total_qubits(self):
        hist = self.chain_histogram()
        return sum([bin*count for bin,count in hist.items()])

    @property
    def quality_key(self):
        hist = self.chain_histogram()
        qkey = []
        for bin in sorted(hist.items(), reverse=True):
            for c in bin:
                qkey.append(c)

        return qkey

    """ ############################# SampleSet ############################ """
    def chain_breaks(self, sampleset):
        """ Fraction of chain breaks averaged over all samples """

        import numpy as np
        source_nodes = list(self)
        target_nodes = list(sampleset.variables)

        target_relabel = {q: idx for idx, q in enumerate(target_nodes)}
        chains = [[target_relabel[q] for q in self[v]] for v in source_nodes]

        samples = sampleset.record.sample
        values = list(sampleset.vartype.value)

        ratio = np.zeros(len(self), dtype=float, order='F')

        for cidx, chain in enumerate(chains):
            chain = np.asarray(chain)
            if len(chain) <= 1: continue
            np_params = {'assume_unique':True,'invert':True}
            # If the average for each chain is not one of the possible values
            # from vartype then there was a break
            broken = np.isin(samples[:,chain].mean(axis=1), values, **np_params)
            ratio[cidx] = broken.mean()

        return dict(zip(source_nodes,ratio))

    """ ############################# Interface ############################ """
    @property
    def id(self):
        """ The Embedding IDentifier is made up of two parts:
                Quality ID: String of the Chain Histogram
                Chains ID: String of the 8 last digits of the chain key.
        """
        chains_id = f"{self.__hash__():08}"
        quality_id = "".join(map(str,self.quality_key))
        if not self:
            return "XXXXXXXX_XXXXXXXX"
        else:
            return quality_id[:8] + "_" + chains_id[:8]

    def __key(self):
        embedding_key = []
        for v,chain in self.items():
            chain_key = ""
            for t in sorted(chain):
                chain_key+=str(t)
            embedding_key.append(int(chain_key))

        return embedding_key

    def __hash__(self):
        return sum(self.__key())

    def __eq__(self, other):
        return self.__key() == other.__key()
    def __ne__(self, other):
        return self.__key() != other.__key()
    def __lt__(self, other):
        return self.quality_key < other.quality_key
    def __le__(self, other):
        return self.quality_key <= other.quality_key
    def __gt__(self, other):
        return self.quality_key > other.quality_key
    def __ge__(self, other):
        return self.quality_key >= other.quality_key
