import os

from json import load as _load
from json import dump as _dump

from dimod.serialization.json import DimodEncoder, DimodDecoder

from networkx.readwrite.json_graph import node_link_data as _serialize_graph
from networkx.readwrite.json_graph import node_link_graph as _deserialize_graph


class EmberaDataBase:
    """ DataBase class to store embeddings, embedding reports and samples
        from structured Ising solvers such as D-Wave's Quantum Annealers.

        Every load/dump function takes as arguments names of graphs.

    """

    def __init__(self, path="./EmberaDB/"):

        self.path = path
        if not os.path.isdir(self.path):
            os.mkdir(self.path)

        self.inputs_path = os.path.join(self.path,'inputs')
        if not os.path.isdir(self.inputs_path):
            os.mkdir(self.inputs_path)
        self.inputs = self._list_inputs()

        self.samplers_path = os.path.join(self.path,'samplers')
        if not os.path.isdir(self.samplers_path):
            os.mkdir(self.samplers_path)
        self.samplers = self._list_samplers()

        self.embeddings_path = os.path.join(self.path,'embeddings')
        if not os.path.isdir(self.embeddings_path):
            os.mkdir(self.embeddings_path)

        self.samplesets_path =  os.path.join(self.path,'samplesets')
        if not os.path.isdir(self.samplesets_path):
            os.mkdir(self.samplesets_path)

    """ Inputs are stored as Dimod Binary Quadratic Models and filed by name """
    def _list_inputs(self):
        inputs_map = map(os.path.splitext,os.listdir(self.inputs_path))
        return [name for name,ext in inputs_map if ext=='.json']

    def load_input(self, input_name):
        if input_name in self.inputs:
            input_path = os.path.join(inputs_path,input_name)
            with open(input_path,'r') as fp:
                return _load(fp,cls=DimodDecoder)

    def dump_input(self, input, force=False):
        if (input.name in self.inputs) and not force:
            print("Input already in DB. Use `force=True` to rewrite.")
        else:
            input_path = os.path.join(inputs_path,input.name)
            with open(input_path,'w') as fp:
                _dump(input,fp,cls=DimodEncoder)
            self.inputs = self._list_inputs()

    def load_inputs(self, input_names):
        return [load_input(name) for name in input_names]

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
        print(f"Sampler {sampler} not found")

    def dump_sampler(self, sampler, force=False):
        name = sampler.name
        if (name in self.samplers and not force):
            print("Sampler already in DB. Use `force=True` to rewrite.")
        else:
            sampler_path = os.path.join(self.samplers_path,name + '.json')
            with open(sampler_path,'w') as fp:
                _dump(_serialize_graph(sampler), fp)
            self.samplers = self._list_samplers()

    def load_samplers(self):
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

    """ Embeddings named are stored in a sorted list under sampler_name directories
    """
    def load_embedding(self,input_name,sampler_name,index=0):
        embeddings_path = os.path.join(self.embeddings_path,sampler_name)



    def dump_embedding(self,input_name,sampler_name,embedding):
        sampler_path = os.path.join(self.embeddings_path,sampler_name)
        if os.path.isdir(sampler_path):
            embeddings_path = os.path.join(sampler_path,input_name)
            with open(embeddings_path,'w+') as fp:
                embeddings = _load(fp)
                embeddings += [embedding]
                _dump(embeddings,fp)


    def load_embeddings(self,input_name,sampler_name):


    def dump_embeddings(self,input_name,sampler_name,embeddings):


import os
a = ['a.json','b.json','c.json']
list(map(os.path.splitext, a))

db = EmberaDataBase()
