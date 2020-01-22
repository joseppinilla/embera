import json
import embera

class EmberaEncoder(json.JSONEncoder):
    def embera_format(self,obj):
        if isinstance(obj,str):
            return f"\'{obj}\'"
        else:
            return obj

    def iterencode(self, obj,_one_shot=False):
        if isinstance(obj,embera.Embedding):
            embedding = obj.items()
            embedding_serial = [f"{self.embera_format(k)}:{self.embera_format(v)}" for k,v in embedding]
            serial = ",".join(embedding_serial)
            serial_brackets = f"{{{serial}}}"
            embedding_dict = {"type": "Embedding",
                              "embedding": serial_brackets,
                              "properties": obj.properties}
            return super(EmberaEncoder, self).iterencode(embedding_dict,_one_shot)

        return super(EmberaEncoder, self).iterencode(obj,_one_shot)

class EmberaDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, obj):
        if isinstance(obj,dict):
            if obj.get("type",None)=="Embedding":
                embedding = eval(obj["embedding"])
                properties = obj["properties"]
                embedding_obj = embera.Embedding(embedding,**properties)
                return embedding_obj
        return obj
