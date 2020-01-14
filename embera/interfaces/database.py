import os
import json

from json import load as _load
from json import dump as _dump

from embera.interfaces.embedding import Embedding
from embera.interfaces.json import EmberaEncoder, EmberaDecoder

from dimod.serialization.json import DimodEncoder, DimodDecoder

from networkx.readwrite.json_graph import node_link_data as _serialize_graph
from networkx.readwrite.json_graph import node_link_graph as _deserialize_graph

__all__ = ["EmberaDataBase"]


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

            `source_name` and `target_name` can be provided in `source` and
            `target`, in NetworkX graph, or will be automatically generated
            using `hash` as decribed below.

            Arguments:
                source_name: (str, list of edges, or NetworkX graph)
                    If `str`, embedding must be an Embedding object.
                    If list of edges, name will be taken from hash of sorted
                    tuple of sorted tuples.

                target_name: (str, list of edges, or NetworkX graph)
                    If `str`, embedding must be an Embedding object.
                    If list of edges, name will be taken from hash of sorted
                    tuple of sorted tuples.

            Optional Arguments:
                embedding_tag: (str, default="")
                    If provided, embedding is stored under a directory ./<tag>/

                rank: (int, default=0)
                    Due to the naming convention used in the Embedding class,
                    embeddings are stored sorted by `quality_key`. Therefore,
                    `rank==0` is the "best" embedding of that source, in that
                    target, with that `tag`.

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
        if not embeddings: return {}

        embedding_filename = embeddings.pop(rank)
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
                embedding_tag: (str, default="")
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
