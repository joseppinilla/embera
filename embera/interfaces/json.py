import json

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
