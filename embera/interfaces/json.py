import json
import dimod
import numpy
import embera


class EmberaEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (dimod.SampleSet, embera.Embedding, embera.Graph)):
            return obj.to_serializable()
        elif isinstance(obj,dimod.BQM):
            return obj.to_serializable(bias_dtype=numpy.float64)
        return json.JSONEncoder.default(self, obj)

class EmberaDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if obj.get("type","") == "SampleSet":
            return dimod.SampleSet.from_serializable(obj)
        elif obj.get("type","") == "BinaryQuadraticModel":
            return dimod.BinaryQuadraticModel.from_serializable(obj)
        elif obj.get("type","") == "Embedding":
            return embera.Embedding.from_serializable(obj)
        elif obj.get("type","") == "Graph":
            return embera.Graph.from_serializable(obj)
        return obj
