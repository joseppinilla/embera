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

from dwave.embedding import majority_vote

""" Pandas DataFrame accessor for dimod.SampleSet type """
@pd.api.extensions.register_dataframe_accessor("sampleset")
class SampleSetAccessor:

    def __init__(self, pandas_obj):
         self._validate(pandas_obj)
         self._obj = pandas_obj

    @staticmethod
    def _validate(obj):
        if 'energy' not in obj.columns:
            raise AttributeError("Must have 'energy' column")
        if 'num_occurrences' not in obj.columns:
             raise AttributeError("Must have 'num_occurrences' column")
        m,n = obj.shape
        if n < 3:
            raise AttributeError("Must have at least one label column")
        if m < 2:
            raise AttributeError("Must have at least one sample row")

    @classmethod
    def from_samples(cls):
        # TODO:
        pass

    @classmethod
    def from_samples_bqm(cls):
        # TODO:
        pass

    """ Properties """

    @property
    def energy(self, bqm=None, chain_break_method=None, **kwargs):
        """ Given a different BinaryQuadraticModel, calculate energies for
        all configurations.
        """
        return False

    """ Export """

    def to_dwave_sampleset(self):
        # TODO:
        pass

    def first(self):
        # TODO: check if sorted?
        idx = self._obj['energy'].idxmin()
        return self._obj.iloc[idx]


    """ Plotting """
    def plot_joint(self,x=None,y=None,
                   axis_labels=None,
                   **marginal_kws):
        """ Plots a 2D Joint Plot """
        if not _sns:
            raise ExecutionError("This method requires the 'seaborn' package")

        g = sns.jointplot(x,y,data=self._obj)
        g = sns.JointGrid(x=x,y=y, data=self._obj)


    def unembed(self, embedding, source_bqm,
                chain_break_method=None, chain_break_fraction=False,
                return_embedding=False):
        """ Returns a DataFrame using the source BinaryQuadraticModel to
        calculate energies and number of occurences.
        """
        pass
