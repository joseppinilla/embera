import os
import json
import time
import dimod
import numpy
import pandas as pd
import networkx as nx

from hashlib import md5

from json import load as _load
from json import dump as _dump

from embera.interfaces.graph import Graph
from embera.interfaces.embedding import Embedding
from embera.interfaces.json import EmberaEncoder, EmberaDecoder

from dimod.variables import iter_serialize_variables
from dimod.serialization.json import DimodEncoder, DimodDecoder

from dwave.embedding import unembed_sampleset

from networkx.readwrite.json_graph import node_link_data as _serialize_graph
from networkx.readwrite.json_graph import node_link_graph as _deserialize_graph

__all__ = ["EmberaDataBase"]

class EmberaDataBase:
    """ DataBase class to store bqms, embeddings, samplesets, and reports """
    path = None
    aliases = {}

    def __init__(self, path="./EmberaDB/", hash_method=md5):
        # WIP
        import warnings
        warnings.warn("EmberaDataBase is a Work In Progress. All file formats and indexing is subject to change.")

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

        self.reports_path = os.path.join(self.path,'reports')
        if not os.path.isdir(self.reports_path):
            os.mkdir(self.reports_path)

        self.aliases_path = os.path.join(self.path,'aliases.json')
        if os.path.exists(self.aliases_path):
            with open(self.aliases_path,'r') as fp:
                self.aliases = _load(fp)

        self.hash = lambda ser: hash_method(ser).hexdigest()

    def update_aliases(self):
        with open(self.aliases_path,'w+') as fp:
            _dump(self.aliases,fp)

    def set_bqm_alias(self, bqm, alias):
        id = self.id_bqm(bqm)
        bqm_aliases = self.aliases.get('bqm',{})
        bqm_aliases[alias] = id
        self.aliases['bqm'] = bqm_aliases
        self.update_aliases()

    def set_source_alias(self, source, alias):
        id = self.id_source(source)
        source_aliases = self.aliases.get('source',{})
        source_aliases[alias] = id
        self.aliases['source'] = source_aliases
        self.update_aliases()

    def set_target_alias(self, target, alias):
        id = self.id_target(target)
        target_aliases = self.aliases.get('target',{})
        target_aliases[alias] = id
        self.aliases['target'] = target_aliases
        self.update_aliases()

    """ ############################### Hashing ############################ """
    def id_bqm(self, bqm, alias=None):
        if isinstance(bqm,str):
            return self.aliases.get('bqm',{}).get(bqm,bqm)

        if isinstance(bqm,dimod.BinaryQuadraticModel):
            ser = bqm.to_serializable()
        elif isinstance(bqm,nx.Graph):
            ser = dimod.BinaryQuadraticModel.from_networkx_graph(bqm).to_serializable()
        else:
            raise ValueError("BQM must be dimod.BinaryQuadraticModel, networkx.Graph, or str")

        try: # TMP: Check if BQM is sortable in Python>3
            list(bqm.variables).sort()
        except:
            import warnings
            warnings.warn("BQM variables labels cannot be sorted, hashing may not work")

        keys = ['variable_labels','linear_biases','quadratic_biases']
        filtered = {k:v for k,v in ser.items() if k in keys}
        json_bqm = json.dumps(filtered,sort_keys=True)
        id = self.hash(json_bqm.encode("utf-8"))

        if not alias is None:
            self.set_bqm_alias(id,alias)
        return id

    def id_source(self, source, alias=None):
        if isinstance(source,str):
            return self.aliases.get('source',{}).get(source,source)

        if isinstance(source,dimod.BinaryQuadraticModel):
            ser = Graph(source.quadratic).to_serializable()
        elif isinstance(source,nx.Graph):
            ser = Graph(source.edges).to_serializable()
        elif isinstance(source,list):
            ser = Graph(source).to_serializable()
        else:
            raise ValueError("Source must be dimod.BinaryQuadraticModel, networkx.Graph, list of tuples or str")

        json_source = json.dumps(ser)
        id = self.hash(json_source.encode("utf-8"))

        if alias!=None:
            self.set_source_alias(id,alias)

        return id

    def id_target(self, target, alias=None):
        if isinstance(target,str):
            return self.aliases.get('target',{}).get(target,target)

        if isinstance(target,nx.Graph):
            ser = Graph(target.edges).to_serializable()
        elif isinstance(target,list):
            ser = Graph(target).to_serializable()
        else:
            raise ValueError("Target must be networkx.Graph, list of tuples or str")

        json_target = json.dumps(ser)
        id = self.hash(json_target.encode("utf-8"))

        if alias!=None:
            self.set_target_alias(id,alias)

        return id

    def id_embedding(self, embedding):
        if isinstance(embedding,str):
            return self.aliases.get('embedding',{}).get(embedding,embedding)

        if isinstance(embedding,Embedding):
            ser = embedding.to_serializable()
        elif isinstance(embedding,dict):
            ser = Embedding(embedding).to_serializable()
        else:
            raise ValueError("Embedding must be embera.Embedding, dict, or str")

        keys = ['variable_labels','chains']
        json_embedding = json.dumps({k:v for k,v in ser.items() if k in keys})
        return self.hash(json_embedding.encode("utf-8"))

    def get_path(self, dir_path, filename=None):
        path = ""
        for dir in dir_path:
            path = os.path.join(path,dir)
            if not os.path.isdir(path):
                os.mkdir(path)
        if filename is not None:
            path = os.path.join(path,filename+'.json')
        return path

    """ ######################## BinaryQuadraticModels ##################### """
    def load_bqms(self, source, tags=[]):
        source_id = self.id_source(source)
        bqms_path = os.path.join(self.bqms_path,source_id)

        bqms = []
        for root, dirs, files in os.walk(bqms_path):
            root_dirs = os.path.normpath(root).split(os.path.sep)
            if all(tag in root_dirs for tag in tags):
                for file in files:
                    bqm_path = os.path.join(root,file)
                    with open(bqm_path,'r') as fp:
                        bqm = json.load(fp,cls=EmberaDecoder)
                    bqms.append(bqm)
        return bqms

    def load_bqm(self, source, tags=[], index=0):
        bqms = self.load_bqms(source,tags)

        if not bqms:
            raise ValueError("No BQMs found")

        return bqms.pop(index)

    def dump_bqm(self, bqm, tags=[], alias=None):
        source_id = self.id_source(bqm)
        bqms_path = [self.bqms_path,source_id]+tags

        bqm_ser = json.dumps(bqm,cls=EmberaEncoder)

        bqm_id = self.id_bqm(bqm, alias=alias)
        bqm_path = self.get_path(bqms_path, bqm_id)
        with open(bqm_path,'w+') as fp:
            fp.write(bqm_ser)

        return bqm_id

    def dump_ising(self, h, J, tags=[], alias=None, return_bqm=False):
        """ Convert Ising parameters into BQM and dump to JSON file

            Arguments:
                h: (dict)
                    Dictionary of nodes with Ising parameters.
                J: (dict)
                    Dictionary of edges with Ising parameters.

            Optional Arguments:
                return_bqm: (bool, default=False)
                    If True, return a tuple of (BQM, ID(BQM)).
        """
        bqm = dimod.BinaryQuadraticModel(h,J,0.0,'SPIN',tags=tags)
        bqm_id = self.dump_bqm(bqm,tags,alias=alias)
        if return_bqm:
            return bqm, bqm_id
        else:
            return bqm_id

    """ ############################# SampleSets ########################### """
    def load_samplesets(self, bqm, target, embedding, tags=[], unembed_args=None):
        bqm_id = self.id_bqm(bqm)
        target_id = self.id_target(target)
        embedding_id = self.id_embedding(embedding)

        dir_path = [self.samplesets_path,bqm_id,target_id,embedding_id]
        samplesets_path = os.path.join(*dir_path)

        samplesets = []
        for root, dirs, files in os.walk(samplesets_path):
            root_dirs = os.path.normpath(root).split(os.path.sep)
            if all(tag in root_dirs for tag in tags):
                for file in files:
                    sampleset_path = os.path.join(root,file)
                    with open(sampleset_path,'r') as fp:
                        sampleset = _load(fp,cls=DimodDecoder)
                    samplesets.append(sampleset)

        if embedding == "":
            return samplesets
        elif not isinstance(embedding,(Embedding,dict)):
            raise ValueError("Embedding alias or id cannot be used to unembed")

        if unembed_args is None:
            return samplesets
        else:
            return [unembed_sampleset(s,embedding,bqm,**unembed_args) for s in samplesets]

    def load_sampleset(self, bqm, target, embedding, tags=[], unembed_args=None, index=None):
        """ Load a sampleset object from JSON format, filed under:
            <EmberaDB>/<bqm_id>/<target_id>/<embedding_id>/<tags>/<sampleset_id>.json
            If more than one sampleset is found, returns the concatenation
            of all samples found under the given criteria.

        Arguments:
            bqm: (BinaryQuadraticModel or networkx.Graph)

            target: (networkx.Graph, list, or str)

        Optional Arguments:
            If none of the optional arguments are given, all samplesets under
            that path are concatenated.

            tag: (str, default=[])
                If provided, sampleset is read from directory ./<tag>/

            embedding: (embera.Embedding, dict, or str, default=None)
                Dictionary is converted to Embedding, Embedding ID is used.
                String is taken literally as path.
                If "", concatenate all under <EmberaDB>/<bqm_id>/<target_id>/<tag>
                If {}, return `native` sampleset. i.e. <Embedding({}).id>.json

        """
        samplesets = self.load_samplesets(bqm,target,embedding,tags,unembed_args)

        if not samplesets:
            return dimod.SampleSet.from_samples([],bqm.vartype,None)

        if index is not None:
            return samplesets.pop(index)

        try:
            sampleset = dimod.concatenate(samplesets)
            for s in samplesets: sampleset.info.update(s.info)
            return sampleset
        except KeyError:
            raise RuntimeError("Samplesets don't share the same embedding")

    def dump_sampleset(self, bqm, target, embedding, sampleset, tags=[]):
        bqm_id = self.id_bqm(bqm)
        target_id = self.id_target(target)
        embedding_id = self.id_embedding(embedding)
        samplesets_path = [self.samplesets_path,bqm_id,target_id,embedding_id]+tags

        sampleset_ser = json.dumps(sampleset,cls=EmberaEncoder)

        sampleset_id = self.hash(sampleset_ser.encode("utf-8"))
        sampleset_path = self.get_path(samplesets_path, sampleset_id)
        with open(sampleset_path, 'w+') as fp:
                fp.write(sampleset_ser)
        return sampleset_id


    """ ############################ Embeddings ############################ """
    def load_embeddings(self, source, target, tags=[]):
        source_id = self.id_source(source)
        target_id = self.id_target(target)

        embeddings_path = os.path.join(self.embeddings_path,source_id,target_id)

        embedding_filenames = []
        for root, dirs, files in os.walk(embeddings_path):
            root_dirs = os.path.normpath(root).split(os.path.sep)
            if all(tag in root_dirs for tag in tags):
                for file in files:
                    embedding_filenames.append((root,file))

        embeddings = []
        for embedding_filename in embedding_filenames:
            embedding_path = os.path.join(*embedding_filename)

            with open(embedding_path,'r') as fp:
                embedding = _load(fp,cls=EmberaDecoder)
            embeddings.append(embedding)

        return embeddings

    def load_embedding(self, source, target, tags=[], index=0):
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
                index: (int, default=0)
                    Embeddings are stored sorted by `quality_key`. Therefore,
                    `index==0` is the "best" embedding found for that bqm, onto
                    that target, with that `tag`.

            Returns:
                embedding: (embera.Embedding)
                    Embedding at given rank or empty dictionary if none found.

        """
        embeddings = self.load_embeddings(source,target,tags)

        if not embeddings:
            return Embedding({})
        else:
            return embeddings.pop(index)


    def dump_embedding(self, source, target, embedding, tags=[]):
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
        if not isinstance(embedding,Embedding):
            embedding = Embedding(embedding)

        source_id = self.id_source(source)
        target_id = self.id_target(target)

        embeddings_path = [self.embeddings_path,source_id,target_id] + tags

        embedding_ser = json.dumps(embedding.to_serializable(),cls=EmberaEncoder)
        embedding_id = self.hash(embedding_ser.encode("utf-8"))
        embedding_path = self.get_path(embeddings_path, embedding_id)
        with open(embedding_path, 'w+') as fp:
            fp.write(embedding_ser)
        return embedding_id

    """ ############################# Reports ############################# """
    def load_reports(self, bqm, target, tags=[], dataframe=False):
        bqm_id = self.id_bqm(bqm)
        target_id = self.id_target(target)

        dir_path = [self.reports_path,bqm_id,target_id]
        reports_path = os.path.join(*dir_path)

        reports = {}
        for root, dirs, files in os.walk(reports_path):
            root_dirs = os.path.normpath(root).split(os.path.sep)
            if all(tag in root_dirs for tag in tags):
                for file in files:
                    report_path = os.path.join(root,file)
                    with open(report_path,'r') as fp:
                        report = _load(fp,cls=EmberaDecoder)
                    metric, ext =  os.path.splitext(file)
                    if not dataframe:
                        reports[metric] = report
                    else:
                        kwargs = {'columns':list(bqm),'orient':'index'}
                        reports[metric] = pd.DataFrame.from_dict(report,**kwargs)
        return reports

    def load_report(self, bqm, target, metric, tags=[], dataframe=False):
        reports = self.load_reports(bqm,target,tags,dataframe)
        report = reports.get(metric,{})
        return report

    def dump_report(self, bqm, target, report, metric, tags=[]):
        bqm_id = self.id_bqm(bqm)
        target_id = self.id_target(target)

        reports_path = [self.reports_path,bqm_id,target_id]+tags

        report_filename = metric
        report_path = self.get_path(reports_path, report_filename)

        with open(report_path,'w+') as fp:
            _dump(report,fp,cls=EmberaEncoder)
        return report_filename
