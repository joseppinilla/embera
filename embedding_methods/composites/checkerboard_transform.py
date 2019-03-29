# Copyright 2018 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# ================================================================================================
"""
The D-Wave machine exhibits intrinsic control errors such that the effective
values of the final Hamiltonian coefficients may deviate slightly from their
programmed values.
A gauge transformation on the qubits induces a transformation on the Ising
model coefficients :math: `H_j` and :math:`J_{ij}`. One interesting property
of gauge transformations is that they transforms ferromagnetic couplings into
antiferromagnetic couplings, and vice versa. This is especially useful in
mitigating the coupler/bias leakage effect in chained qubits.
In the checkerboard transformation, the qubits on one shore of each tile are
flipped while the qubits on the other shore are unchanged. Because of the way
this pattern alternates from one unit cell to the next, this is called a
basket weave or checkerboard gauge transformation.

WARNING!: This composite uses 4 gauge transformations, and uses the kwarg
num_reads for each gauge. If num_reads=1000, then len(response)=4000

The gauge transformations are:
     I: "Identity" transformation (no spins flipped)
     G: Checkered 0 transformation (spins flipped on shore 0)
    -I: Inverse transformation     (spins flipped on shores 1 and 0)
    -G: Checkered 1 transformation (spins flipped on shore 1)
"""
from random import random
import itertools

import numpy as np

from dwave_networkx.generators import chimera, pegasus

from dimod.core.composite import Composite
from dimod.core.sampler import Sampler
from dimod.core.structured import Structured
from dimod.sampleset import SampleSet, concatenate
from dimod.vartypes import Vartype
from dimod.exceptions import InvalidComposition

__all__ = ['CheckerboardTransformComposite']


class CheckerboardTransformComposite(Sampler, Composite, Structured):
    """Composite for applying a checkerboard spin reversal transform.

    The Checkerboard Spin reversal transform [#ah]_ is applied by flipping the
    spin of variables on one of the shores of each tile in the Chimera target
    graph. After sampling the transformed Ising problem, the same bits are
    flipped in the resulting sample [#km]_.

    Args:
        sampler: A `dimod` sampler object.

    Examples:
        This example composes a dimod ExactSolver sampler with spin transforms then
        uses it to sample an Ising problem.

        >>> # Compose the sampler
        >>> base_sampler = dimod.ExactSolver()
        >>> composed_sampler = dimod.SpinReversalTransformComposite(base_sampler)
        >>> base_sampler in composed_sampler.children
        True
        >>> # Sample an Ising problem
        >>> response = composed_sampler.sample_ising({'a': -0.5, 'b': 1.0}, {('a', 'b'): -1})
        >>> print(next(response.data()))           # doctest: +SKIP
        Sample(sample={'a': 1, 'b': 1}, energy=-1.5)

    References
    ----------
    .. [#ah] Adachi, S. H., & Henderson, M. P. Application of Quantum Annealing
        to Training of Deep Neural Networks. https://doi.org/10.1038/nature10012,
        2015.
    .. [#km] Andrew D. King and Catherine C. McGeoch. Algorithm engineering
        for a quantum annealing platform. https://arxiv.org/abs/1410.2628,
        2014.

    """
    children = None
    nodelist = None
    edgelist = None
    parameters = None
    properties = None
    coordinates = None

    def __init__(self, child, family='chimera', m=None):
        self.children = [child]
        self.nodelist = child.nodelist
        self.edgelist = child.edgelist
        self.parameters = child.parameters

        if not isinstance(child, Structured):
            raise InvalidComposition("Checkerboard transformations should only be applied to a Structured sampler")

        if family is not 'chimera':
            raise NotImplementedError("Only Chimera family devices are supported.")

        if family=='chimera':
            t = 8
            if m==None:
                m = round((len(self.nodelist)/t)**(1/2))
            self.coordinates = chimera.chimera_coordinates(m)
        elif family=='pegasus':
            self.coordinates = pegasus.pegasus_coordinates()
        else:
            raise ValueError("Sampler family can only be Chimera or Pegasus.")

        self.family = family
        self.properties = {'child_properties': child.properties}

    def sample(self, bqm, **kwargs):
        """Sample from the binary quadratic model.

        Args:
            bqm (:obj:`~dimod.BinaryQuadraticModel`):
                Binary quadratic model to be sampled from.

        Returns:
            :obj:`.SampleSet`

        """

        # make a main response
        responses = []

        flipped_bqm = bqm.copy()
        transform = {v: False for v in bqm.variables}

        # "Identity" transformation (no spins flipped)
        response = self.child.sample(bqm, **kwargs)
        responses.append(response)

        # Checkered 0 transformation (spins flipped on shore 0)
        # Inverse transformation     (spins flipped on shores 1 and 0)
        # Checkered 1 transformation (spins flipped on shore 1)
        for flip_shore in [0,1,0]:
            # Create flipped BQM
            try:
                for v in bqm:
                    (i,j,u,k) = v if isinstance(v,tuple) else self.coordinates.tuple(v)
                    if u is flip_shore:
                        transform[v] = not transform
                        flipped_bqm.flip_variable(v)

                # Sample flipped BQM
                flipped_response = self.child.sample(flipped_bqm, **kwargs)
                tf_idxs = [flipped_response.variables.index(v) for v, flip in transform.items() if flip]

                if bqm.vartype is Vartype.SPIN:
                    flipped_response.record.sample[:, tf_idxs] = -1 * flipped_response.record.sample[:, tf_idxs]
                else:
                    flipped_response.record.sample[:, tf_idxs] = 1 - flipped_response.record.sample[:, tf_idxs]

                responses.append(flipped_response)

            except:
                raise ValueError('Not the right coordinates of node in structure.')

        # Merge all gauge transformation responses
        return concatenate(responses)
