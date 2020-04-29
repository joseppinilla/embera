""" ============== Architecture agnostic coordinate converter ============== """

import dwave_networkx as dnx

__all__ = ['dwave_coordinates']

class dwave_coordinates:

    def __init__(self):
        raise RuntimeError("Use classmethods {from_dwave_networkx, from_graph_dict}")

    @classmethod
    def from_graph_dict(cls, graph):
        try:
            family = graph["family"]
            m = graph["columns"]
            n = graph["rows"]
            t = graph["tile"]
        except:
            raise ValueError("Target graph needs to have family, columns, rows,\
            and tile attributes.")

        if family is 'chimera':
            return chimera_coordinates(m,n,t)
        elif family is 'pegasus':
            return pegasus_coordinates(m)
        else:
            raise ValueError('Graph family not supported')

    @classmethod
    def from_dwave_networkx(cls, T):
        return cls.from_graph_dict(T.graph)

class agnostic_coordinates:
    """ To be inherited by <architecture>_coordinates """
    def __init__(self, family):
        self.family = family

    def coordinate_to_linear(self, q):
        method = getattr(self, f'{self.family}_to_linear')
        return method(q)

    def coordinate_to_nice(self, q):
        method = getattr(self, f'{self.family}_to_nice')
        return method(q)

    def linear_to_coordinate(self, r):
        method = getattr(self, f'linear_to_{self.family}')
        return method(q)

    def nice_to_coordinate(self, n):
        method = getattr(self, f'nice_to_{self.family}')
        return method(q)

    def iter_coordinate_to_linear(self, qlist):
        method = getattr(self, f'iter_{self.family}_to_linear')
        return method(qlist)

    def iter_coordinate_to_nice(self, qlist):
        method = getattr(self, f'iter_{self.family}_to_nice')
        return method(qlist)

    def iter_linear_to_coordinate(self, rlist):
        method = getattr(self, f'iter_linear_to_{self.family}')
        return method(rlist)

    def iter_nice_to_coordinate(self, nlist):
        method = getattr(self, f'iter_nice_to_{self.family}')
        return method(nlist)

    def iter_coordinate_to_linear_pairs(self, qlist):
        method = getattr(self, f'iter_{self.family}_to_linear_pairs')
        return method(qlist)

    def iter_coordinate_to_nice_pairs(self, qlist):
        method = getattr(self, f'iter_{self.family}_to_nice_pairs')
        return method(qlist)

    def iter_linear_to_coordinate_pairs(self, rlist):
        method = getattr(self, f'iter_linear_to_{self.family}_pairs')
        return method(rlist)

    def iter_nice_to_coordinate_pairs(self, nlist):
        method = getattr(self, f'iter_nice_to_{self.family}_pairs')
        return method(nlist)

class chimera_coordinates(dnx.chimera_coordinates, agnostic_coordinates):
    """ Augmented chimera_coordinates class """
    def __init__(self, m, n=None, t=None):
        agnostic_coordinates.__init__(self, 'chimera')
        dnx.chimera_coordinates.__init__(self, m, n, t)

    def linear_to_nice(self, r):
        return self.chimera_to_nice(self.linear_to_chimera(r))

    def nice_to_linear(self, n):
        return self.chimera_to_linear(self.nice_to_chimera(n))

    @staticmethod
    def chimera_to_nice(q):
        return (0,) + q

    @staticmethod
    def nice_to_chimera(n):
        t,i,j,u,k = n
        return (i,j,u,k)

    def iter_linear_to_nice(self, rlist):
        for r in rlist:
            yield self.linear_to_nice(r)

    def iter_nice_to_linear(self, nlist):
        for n in nlist:
            yield self.nice_to_linear(n)

    @classmethod
    def iter_chimera_to_nice(cls, qlist):
        for q in qlist:
            yield (0,) + q

    @classmethod
    def iter_nice_to_chimera(cls, nlist):
        for n in nlist:
            t,i,j,u,k = n
            yield (i,j,u,k)

class pegasus_coordinates(dnx.pegasus_coordinates, agnostic_coordinates):
    """ Augmented chimera_coordinates class """
    def __init__(self, m, n=None, t=None):
        agnostic_coordinates.__init__(self, 'pegasus')
        dnx.pegasus_coordinates.__init__(self, m)
