import json
import embera

class EmberaEncoder(json.JSONEncoder):
    def iterencode(self, obj):
        if isinstance(obj,embera.Embedding):
            embedding = obj.items()
            items_list = [f"\"{k}\":\"{v}\"" for k,v in embedding]
            serialized = ",".join(items_list)
            return f"{{{serialized}}}"
        return super(EmberaEncoder, self).iterencode(obj)

class EmberaDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if isinstance(obj,dict):
            format_eval = lambda k: k if k.isalpha() else eval(k)
            return {format_eval(key):eval(value) for key, value in obj.items()}
        return obj
