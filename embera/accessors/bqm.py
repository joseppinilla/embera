import pandas as pd

try: # Allows plotting D-Wave topologies
    import dwave_networkx as dnx
    _dnx = True
except ImportError:
    _dnx = False

try: # Allows plotting joint grids
    import seaborn as sns
    _sns = True
except ImportError:
    _sns = False

""" Pandas DataFrame accessor for dimod.BinaryQuadraticModel type """
@pd.api.extensions.register_dataframe_accessor("bqm")
class BinaryQuadraticModelAccessor:
    def __init__(self, pandas_obj):
         self._validate(pandas_obj)
         self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        m,n = obj.shape
        if n != m:
            raise AttributeError("Must be square matrix")

    @classmethod
    def from_dict(cls,h,J):
        # TODO:
        pass

    """ Properties """

    @property
    def embedding(self):
        return self._obj['chain']

    @embedding.setter
    def embedding(self, dict):
        self._obj['chain'] = dict.values()

    @property
    def degree_tally(self):
        pass

    @property
    def max_possible_clique(self):
        pass

    def get_chain(self,label):
        return self._obj.loc[label]['chain']

    def embed(self,target_graph):
        pass
