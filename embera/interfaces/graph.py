""" Embera Graph Class """
class Graph(list):
    def to_serializable(self):
        edges = []
        for u,v in self:
            edge = sorted([u,v],key=lambda x:str(x))

        doc = {# metadata
               "type": 'Graph',
               # graph
               "edgelist": sorted(edges,key=lambda x:str(x))}

        return doc

    @classmethod
    def from_serializable(cls, obj):
        edgelist = obj["edgelist"]
        return edgelist
