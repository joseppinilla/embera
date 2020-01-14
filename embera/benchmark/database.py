import os
import json

from json import load as _load
from json import dump as _dump

from networkx.readwrite.json_graph import node_link_data as _serialize_graph
from networkx.readwrite.json_graph import node_link_graph as _deserialize_graph

__all__ = ["EmberaDataBase","Embedding","EmberaEncoder","EmberaEncoder"]

def validate_graphs(func):
    def func_wrapper(self, source, target, *args, **kwargs):
        source_id = hash(tuple(source))
        assert(self.source_id==source_id), "Source ID does not match."

        target_id = hash(tuple(target))
        assert(self.target_id==target_id), "Target ID does not match."

        res = func(self, source, target, *args, **kwargs)
        return res

    return func_wrapper

def parse_edges(func):
    def func_wrapper(self, source, target, *args, **kwargs):
        if hasattr(source, 'edges'): source_edgelist = source.edges()
        else: source_edgelist = sorted((tuple(sorted(e)) for e in source))

        if hasattr(target, 'edges'): target_edgelist = target.edges()
        else: target_edgelist = sorted((tuple(sorted(e)) for e in target))

        res = func(self, source_edgelist, target_edgelist, *args, **kwargs)
        return res
    return func_wrapper

class Embedding(dict):

    source_id = None
    target_id = None

    @parse_edges
    def __init__(self, source, target, embedding):
        self.update(embedding)
        self.source_id = hash(tuple(source))
        self.target_id = hash(tuple(target))

    def chain_histogram(self):
        # Based on dwavesystems/minorminer quality_key by Boothby, K.
        sizes = [len(c) for c in self.values()]
        hist = {}
        for s in sizes:
            hist[s] = 1 + hist.get(s, 0)
        return hist

    @parse_edges
    @validate_graphs
    def interactions_histogram(self, source, target):
        interactions = {}
        for u, v in source:
            source_edge = tuple(sorted((u,v)))
            edge_interactions = []
            for s in self[u]:
                for t in self[v]:
                    target_edge = tuple(sorted((s,t)))
                    if target_edge in target:
                        edge_interactions.append(target_edge)
            interactions[(u,v)] = edge_interactions

        sizes = [len(i) for i in interactions.values()]
        hist = {}
        for s in sizes:
            hist[s] = 1 + hist.get(s, 0)
        return hist

    def quality_key(self):
        #TEMP: Can be better
        hist = self.chain_histogram()
        return [value for item in sorted(hist.items(), reverse=True) for value in item]

    def id(self):
        # To create a unique ID we use the quality key as an ID string ...
        quality_id = "".join([str(v) for v in self.quality_key()])
        # ...and the last 8 digits of this object's hash.
        hash_id = f"{self.__hash__():08}"[-8:]
        return f"{quality_id}_{hash_id}"

    def __embedding_key(self):
        return tuple((k,tuple(self[k])) for k in sorted(self))

    def __key(self):
        return (self.source_id, self.target_id, self.__embedding_key())

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return self.__key() == other.__key()
    def __ne__(self, other):
        return self.__key() != other.__key()
    def __lt__(self, other):
        return self.quality_key() < other.quality_key()
    def __le__(self, other):
        return self.quality_key() <= other.quality_key()
    def __gt__(self, other):
        return self.quality_key() > other.quality_key()
    def __ge__(self, other):
        return self.quality_key() >= other.quality_key()

def parse_graph_names(func):
    def func_wrapper(self, source, target, *args, **kwargs):
        if hasattr(source, 'name'): source_name = source.name
        elif isinstance(source,str): source_name = source
        else: source_name = str(hash(tuple(sorted((tuple(sorted(e)) for e in source)))))

        if hasattr(target, 'name'): target_name = target.name
        elif isinstance(target,str): target_name = target
        else: target_name = str(hash(tuple(sorted((tuple(sorted(e)) for e in target)))))

        res = func(self, source_name, target_name, *args, **kwargs)
        return res
    return func_wrapper

def parse_embedding(func):
    def func_wrapper(self, source, target, embedding, *args, **kwargs):
        if isinstance(embedding,Embedding): embedding_obj = embedding
        else: embedding_obj = Embedding(source,target,embedding)

        res = func(self, source, target, embedding_obj, *args, **kwargs)
        return res
    return func_wrapper

class EmberaEncoder(json.JSONEncoder):
    def iterencode(self, obj):
        if isinstance(obj, dict):
            s = [f"\"{k}\":\"{v}\"" for k,v in obj.items()]
            formatted = ",".join(s)
            return f"{{{formatted}}}"
        return super(EmberaEncoder, self).iterencode(obj)

class EmberaDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if isinstance(obj,dict):
            format_eval = lambda k: k if k.isalpha() else eval(k)
            return {format_eval(key):eval(value) for key, value in obj.items()}
        return obj

class EmberaDataBase:
    """ DataBase class to store embeddings, and samplesets. """

    def __init__(self, path="./EmberaDB/"):

        self.path = path
        if not os.path.isdir(self.path):
            os.mkdir(self.path)

        self.embeddings_path = os.path.join(self.path,'embeddings')
        if not os.path.isdir(self.embeddings_path):
            os.mkdir(self.embeddings_path)

        self.samplesets_path =  os.path.join(self.path,'samplesets')
        if not os.path.isdir(self.samplesets_path):
            os.mkdir(self.samplesets_path)


    """ Inputs are stored as graphs and filed by name """
    def _list_inputs(self):
        inputs_map = map(os.path.splitext,os.listdir(self.inputs_path))
        return [name for name,ext in inputs_map if ext=='.json']

    def load_input(self, name):
        if name in self.inputs:
            input_path = os.path.join(inputs_path,input_name + '.json')
            with open(input_path,'r') as fp:
                return _deserialize_graph(_load(fp))
        print(f"Input {name} not found")

    def dump_input(self, input, force=False):
        name = input.name
        if (name in self.inputs) and not force:
            print("Input already in DB. Use `force=True` to rewrite.")
        else:
            input_path = os.path.join(inputs_path,input.name + '.json')
            with open(input_path,'w') as fp:
                return _dump(_deserialize_graph(input), fp)
            self.inputs = self._list_inputs()

    def load_inputs(self):
        return [load_input(name) for name in self.inputs]

    def dump_inputs(self, inputs, force=False):
        for input in inputs:
            dump_input(input,force)
        self.inputs = self._list_inputs()

    """ Samplers are stored as graphs and filed by name """
    def _list_samplers(self):
        samplers_map = map(os.path.splitext,os.listdir(self.samplers_path))
        return [name for name,ext in samplers_map if ext=='.json']

    def load_sampler(self, name):
        if name in self.samplers:
            sampler_path = os.path.join(samplers_path,name + '.json')
            with open(sampler_path,'r') as fp:
                return _deserialize_graph(_load(fp))
        print(f"Sampler {name} not found")

    def dump_sampler(self, sampler, force=False):
        name = sampler.name
        if (name in self.samplers and not force):
            print("Sampler already in DB. Use `force=True` to rewrite.")
        else:
            sampler_path = os.path.join(self.samplers_path,name + '.json')
            with open(sampler_path,'w') as fp:
                _dump(_serialize_graph(sampler), fp)
            self.samplers = self._list_samplers()

    def load_samplers(self, names=[]):
        if names:
            return [load_sampler(name) for name in names]
        else:
            return [load_sampler(name) for name in self.samplers]

    def dump_samplers(self, samplers, force=False):
        for sampler in samplers:
            dump_sampler(sampler,force)
        self.samplers = self._list_samplers()

    """ Samples are stored as Dimod SampleSets and filed under
        sampler/input/emmbedding """
    def load_sampleset(self,input_name,sampler_name,embedding):
        pass

    def dump_sampleset(self,input,sampler,embedding,samples):
        pass

    def load_samplessets(self,input_name,sampler_name,embedding):
        pass

    def dump_samplesets(self,input_name,sampler_name,embedding):
        pass

    """ ######################### Embeddings ########################## """
    @parse_graph_names
    def load_embedding(self, source_name, target_name, embedding_tag="", rank=0):
        """ Load an embedding object from JSON format, filed under:
            <EmberaDB>/<target_name>/<source_name>/<embedding_id>.json
            or, if tag is provided:
            <EmberaDB>/<target_name>/<source_name>/<tag>/<embedding_id>.json


        """
        if not target_name: raise ValueError("Target graph name is empty")
        target_path = os.path.join(self.embeddings_path,target_name)
        if not os.path.isdir(target_path):
            os.mkdir(target_path)

        if not source_name: raise ValueError("Source graph name is empty")
        source_path = os.path.join(target_path,source_name)
        if not os.path.isdir(source_path):
            os.mkdir(source_path)

        tag_path = os.path.join(source_path, embedding_tag)
        if not os.path.isdir(tag_path):
            os.mkdir(tag_path)

        embeddings = os.listdir(tag_path)
        embedding_filename = embeddings.pop(0)
        embedding_path = os.path.join(tag_path, embedding_filename)

        with open(embedding_path,'r') as fp:
            embedding = _load(fp,cls=EmberaDecoder)

        return embedding

    @parse_embedding
    @parse_graph_names
    def dump_embedding(self, source_name, target_name, embedding_obj, embedding_tag=""):
        """ Store an embedding object in JSON format, filed under:
            <EmberaDB>/<target_name>/<source_name>/<embedding_id>.json
            or, if tag is provided:
            <EmberaDB>/<target_name>/<source_name>/<tag>/<embedding_id>.json

            `source_name` and `target_name` can be provided in `source` and
            `target`, in NetworkX graph, or will be automatically generated
            using `hash` as decribed below.

            Arguments:
                source: (str, list of edges, or NetworkX graph)
                    If `str`, embedding must be an Embedding object.
                    If list of edges, name will be taken from hash of sorted
                    tuple of sorted tuples.

                target: (str, list of edges, or NetworkX graph)
                    If `str`, embedding must be an Embedding object.
                    If list of edges, name will be taken from hash of sorted
                    tuple of sorted tuples.

                embedding: (dictionary, or Embedding object)
                    If `Embedding`, source and targets can be `str`.
                    Otherwise, source and targets must be list of edges or NetworkX

            Optional Arguments:
                embedding_tag: (str)
                    If provided, embedding is stored under a directory ./<tag>/
        """
        if not target_name: raise ValueError("Target graph name is empty")
        target_path = os.path.join(self.embeddings_path,target_name)
        if not os.path.isdir(target_path):
            os.mkdir(target_path)

        if not source_name: raise ValueError("Source graph name is empty")
        source_path = os.path.join(target_path,source_name)
        if not os.path.isdir(source_path):
            os.mkdir(source_path)

        tag_path = os.path.join(source_path, embedding_tag)
        if not os.path.isdir(tag_path):
            os.mkdir(tag_path)

        embedding_filename = embedding_obj.id() + ".json"
        embedding_path = os.path.join(tag_path, embedding_filename)

        with open(embedding_path,'w+') as fp:
            _dump(embedding_obj,fp,cls=EmberaEncoder)


    def load_embeddings(self,input_name,sampler_name):
        pass

    def dump_embeddings(self,input_name,sampler_name,embeddings):
        pass
