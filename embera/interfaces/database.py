import os
import json

from json import load as _load
from json import dump as _dump

from embera.interfaces.embedding import Embedding
from embera.interfaces.json import EmberaEncoder, EmberaDecoder

from dimod import BinaryQuadraticModel
from dimod.serialization.json import DimodEncoder, DimodDecoder

from networkx import Graph
from networkx.readwrite.json_graph import node_link_data as _serialize_graph
from networkx.readwrite.json_graph import node_link_graph as _deserialize_graph

__all__ = ["EmberaDataBase"]


class EmberaDataBase:
    """ DataBase class to store embeddings, reports, and samplesets. """
    bqm_aliases = {}
    source_aliases = {}
    target_aliases = {}

    def __init__(self, path="./EmberaDB/"):

        self.path = path
        if not os.path.isdir(self.path):
            os.mkdir(self.path)

        self.bqms_path = os.path.join(self.path,'bqms')
        if not os.path.isdir(self.bqms_path):
            os.mkdir(self.bqms_path)

        self.embeddings_path = os.path.join(self.path,'embeddings')
        if not os.path.isdir(self.embeddings_path):
            os.mkdir(self.embeddings_path)

        self.samplesets_path =  os.path.join(self.path,'samplesets')
        if not os.path.isdir(self.samplesets_path):
            os.mkdir(self.samplesets_path)



    def update_alias(self):
        for id,name in self.bqm_aliases.items():
            pass
        for id,name in self.source_aliases.items():
            pass
        for id,name in self.target_aliases.items():
            pass
        # TODO: checks if directories have duplicates, merges them

    def set_bqm_alias(self, bqm, alias):
        id = self.id_bqm(bqm)
        self.bqm_aliases[id] = alias

    def set_source_alias(self, source, alias):
        id = self.id_source(source)
        self.source_aliases[id] = alias

    def set_target_alias(self, target, alias):
        id = self.id_target(target)
        self.target_aliases[id] = alias

    def id_bqm(self, bqm):
        if isinstance(bqm,BinaryQuadraticModel):
            id = str(hash(json.dumps(bqm,cls=DimodEncoder)))
        elif isinstance(bqm,Graph):
            id = str(hash(json.dumps(BinaryQuadraticModel(bqm),cls=DimodEncoder)))
        elif isinstance(bqm,str):
            id = bqm
        else:
            raise ValueError("BQM must be dimod.BinaryQuadraticModel, networkx.Graph, or str")
        return id

    def id_source(self, source):
        if isinstance(source,BinaryQuadraticModel):
            id = str(hash(tuple(sorted((tuple(sorted(e)) for e in source.quadratic)))))
        elif isinstance(source,Graph):
            id = str(hash(tuple(sorted((tuple(sorted(e)) for e in source.edges)))))
        elif isinstance(source,list):
            id = str(hash(tuple(sorted((tuple(sorted(e)) for e in source)))))
        elif isinstance(source,str):
            id = source
        else:
            raise ValueError("Source must be dimod.BinaryQuadraticModel, networkx.Graph, list of tuples or str")
        return id

    def id_target(self, target):
        if isinstance(target,list):
            id = str(hash(tuple(sorted((tuple(sorted(e)) for e in target)))))
        elif isinstance(target,Graph):
            id = str(hash(tuple(target.edges())))
        elif isinstance(target,str):
            id = target
        else:
            raise ValueError("Target must be list of tuples, networkx.Graph, or str")
        return id

    def id_embedding(self, source, target, embedding):
        if isinstance(embedding,Embedding):
            id = embedding.id
        elif source and target:
            id = Embedding(source,target,embedding).id
        elif isinstance(embedding,str):
            id = embedding
        else:
            raise ValueError("Embedding must be embera.Embedding, dict, or str")
        return id

    def get_path(self, dir_path, filename=None):
        path = "./"
        for dir in dir_path:
            path = os.path.join(path,dir)
            if not os.path.isdir(path):
                os.mkdir(path)
        if filename is not None:
            path = os.path.join(path,filename+'.json')
        return path

    """ ######################## BinaryQuadraticModels ##################### """
    def load_bqm(self, bqm_id, tag=""):
        bqm_filename = bqm_id + ".json"
        bqm_path = os.path.join(self.bqms_path,tag,bqm_filename)

        with open(bqm_path,'r') as fp:
            bqm = _load(fp,cls=DimodDecoder)

        return bqm

    def dump_bqm(self, bqm, tag=""):
        bqm_id = self.id_bqm(bqm)
        bqm_filename = embedding_id + ".json"

        bqm_path = os.path.join(self.bqms_path,tag,bqm_filename)

        with open(bqm_path,'w+') as fp:
            _dump(bqm,fp,cls=DimodEncoder)


    """ ############################# SampleSets ########################### """
    def load_sampleset(self, bqm, target, tag="", embedding={}):
        """ Load a sampleset object from JSON format, filed under:
            <EmberaDB>/<bqm_id>/<target_id>/<tag>/<embedding_id>.json
            If tag and/or embedding are not provided, returns the concatenation
            of all samples found under the given criteria.

        Optional Arguments:
            If none of the optional arguments are given, all samplesets under
            that path are concatenated.

            embedding: (embera.Embedding, dict, or str)
                List is converted to Embedding, Embedding ID is used.
                String is taken literally as path.

            tag: (str, default="")
                If provided, sampleset is read from directory ./<tag>/

        """
        bqm_id = self.id_bqm(bqm)
        target_id = self.id_target(target)
        samplesets_path = os.path.join(self.samplesets_path,bqm_id,target_id,tag)

        embedding_id = self.id_embedding(bqm,target,embedding)
        sampleset_filename = embedding_id + ".json"
        samplesets = []
        for root, dirs, files in os.walk(samplesets_path):
            if sampleset_filename in files:
                sampleset_path = os.path.join(root,sampleset_filename)
                with open(embedding_path,'r') as fp:
                    sampleset = _load(fp,cls=DimodDecoder)
                samplesets.append(sampleset)

        if not samplesets:
            return {}
        else:
            return dimod.sampleset.concatenate(samplesets)

    def dump_sampleset(self, bqm, target, sampleset, tag="", embedding={}):
        bqm_id = self.id_bqm(bqm)
        target_id = self.id_target(target)
        samplesets_path = [self.samplesets_path,bqm_id,target_id,tag]

        embedding_id = self.id_embedding(bqm,target,embedding)
        sampleset_filename = embedding_id + ".json"

        sampleset_path = self.get_path(samplesets_path, sampleset_filename)

        with open(sampleset_path,'w+') as fp:
            _dump(sampleset,fp,cls=DimodEncoder)

    """ ############################ Embeddings ############################ """
    def load_embedding(self, source, target, tag="", rank=0):
        """ Load an embedding object from JSON format, filed under:
            <EmberaDB>/<source_id>/<target_id>/<embedding_id>.json
            or, if tag is provided:
            <EmberaDB>/<source_id>/<target_id>/<tag>/<embedding_id>.json

            Arguments:
                source: (dimod.BinaryQuadraticModel, networkx.Graph w/ biases, list of tuples, or str)
                    BQM, Graph, and list of edge tuples are hashed.
                    String is taken literally as path.
                target: (list of tuples, networkx.Graph, or str)
                    List and Graph are hashed. String is taken literally as path.

            Optional Arguments:
                tag: (str, default="")
                    If provided, embedding is read from directory ./<tag>/
                rank: (int, default=0)
                    Embeddings are stored sorted by `quality_key`. Therefore,
                    `rank==0` is the "best" embedding found for that bqm, onto
                    that target, with that `tag`.

            Returns:
                embedding: (embera.Embedding)
                    Embedding at given rank or empty dictionary if none found.

        """
        source_id = self.id_source(source)
        target_id = self.id_target(target)

        embeddings_path = self.get_path([self.embeddings_path,source_id,target_id,tag])
        embeddings = os.listdir(embeddings_path) #TODO: if no tag is given, also search within all tags for best embedding
        if not embeddings: return {}

        embedding_filename = embeddings.pop(rank)
        embedding_path = os.path.join(embeddings_path, embedding_filename)

        with open(embedding_path,'r') as fp:
            embedding = _load(fp,cls=EmberaDecoder)

        return embedding


    def dump_embedding(self, source, target, embedding, tag=""):
        """ Store an embedding object in JSON format, filed under:
            <EmberaDB>/<source_id>/<target_id>/<embedding_id>.json
            or, if tag is provided:
            <EmberaDB>/<source_id>/<target_id>/<tag>/<embedding_id>.json

            Arguments:
                source: (dimod.BinaryQuadraticModel, networkx.Graph w/ biases, list of tuples, or str)
                    BQM, Graph, and list of edge tuples are hashed.
                    String is taken literally as path.

                target: (list of edge tuples, networkx.Graph, or str)
                    List and Graph are hashed. String is taken literally as path.

                embedding: (embera.Embedding, dict, or str)
                    List is converted to Embedding, Embedding ID is used.
                    String is taken literally as path.

            Optional Arguments:
                tag: (str, default="")
                    If provided, embedding is stored under a directory ./<tag>/
                    Useful to identify method used for embedding.
        """
        source_id = self.id_source(source)
        target_id = self.id_target(target)
        embeddings_path = [self.embeddings_path,source_id,target_id,tag]

        if isinstance(embedding,Embedding): embedding_obj = embedding
        else: embedding_obj = Embedding(source,target,embedding)

        embedding_id = embedding_obj.id
        embedding_filename = embedding_id + ".json"

        embedding_path = self.get_path(embeddings_path, embedding_filename)

        with open(embedding_path,'w+') as fp:
            _dump(embedding_obj,fp,cls=EmberaEncoder)
