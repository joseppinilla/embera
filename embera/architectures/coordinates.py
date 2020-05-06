import dwave_networkx as dnx

__all__ = ['dwave_coordinates']

class dwave_coordinates:
    """ Architecture agnostic coordinate converter
            The notation (i, j, u, k) is called the chimera coordinates index:
                i : indexes the row of the Chimera tile from 0 to m inclusive
                j : indexes the column of the Chimera tile from 0 to n inclusive
                u : qubit orientation (0 = left-hand nodes, 1 = right-hand nodes)
                k : indexes the qubit within either the left- or right-hand shore
                    from 0 to t inclusive
            The notation (u, w, k, z) is called the pegasus coordinates index:
                u : qubit orientation (0 = vertical, 1 = horizontal)
                w : orthogonal major offset
                k : orthogonal minor offset
                z : parallel offset
            The notation (t, i, j, u, k) is called the nice coordinates index:
                t : indexes the Chimera subgraph. Chimera t=0. Pegasus 0 <= t < 3
                i : indexes the row of the Chimera tile from 0 to m inclusive
                j : indexes the column of the Chimera tile from 0 to n inclusive
                u : qubit orientation (0 = left-hand nodes, 1 = right-hand nodes)
                k : indexes the qubit within either the left- or right-hand shore
                    from 0 to t inclusive
            The notation `int` is called the linear index:
                A single positive integer indexes the whole graph. Chimera and
                Pegasus linear indices may coincide differ:
                    i.e. linear_to_chimera(int) != linear_to_pegasus(int)
    """
    def __init__(self,*args,**kwargs):
        raise RuntimeError("Use classmethods {from_dwave_networkx, from_graph_dict}")

    @classmethod
    def from_graph_dict(cls, graph):
        try:
            family = graph["family"]
            m = graph["rows"]
            n = graph["columns"]
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
        return method(r)

    def nice_to_coordinate(self, n):
        method = getattr(self, f'nice_to_{self.family}')
        return method(n)

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

class chimera_coordinates(agnostic_coordinates, dnx.chimera_coordinates):
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

class pegasus_coordinates(agnostic_coordinates, dnx.pegasus_coordinates):
    """ Augmented chimera_coordinates class """
    def __init__(self, m):
        agnostic_coordinates.__init__(self, 'pegasus')
        dnx.pegasus_coordinates.__init__(self, m)
