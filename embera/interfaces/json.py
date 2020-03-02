import json
import embera

class EmberaEncoder(json.JSONEncoder):
    def iterencode(self, obj,_one_shot=False):
        if isinstance(obj,embera.Embedding):
            embedding = obj.items()
            embedding_serial = [f"{repr(k)}:{repr(v)}" for k,v in obj.items()]
            embedding_dict = {"type": "Embedding",
                              "embedding": f"{{{','.join(embedding_serial)}}}",
                              "properties": obj.properties}
            return super(EmberaEncoder, self).iterencode(embedding_dict,_one_shot)
        elif isinstance(obj,dict):
            serial = [f"{repr(k)}:{repr(v)}" for k,v in obj.items()]
            obj_dict = {"type": "dict",
                        "items": f"{{{','.join(serial)}}}"}
            return super(EmberaEncoder, self).iterencode(obj_dict,_one_shot)

        return super(EmberaEncoder, self).iterencode(obj,_one_shot)

class EmberaDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if isinstance(obj,dict):
            if obj.get("type")=="Embedding":
                embedding = eval(obj["embedding"])
                properties = obj["properties"]
                embedding_obj = embera.Embedding(embedding,**properties)
                return embedding_obj
            elif obj.get("type")=="dict":
                return eval(obj.get("items","{}"))
        return obj
