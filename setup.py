#!/usr/bin/env python

from setuptools import setup

packages = ['embera',
            'embera.composites',
            'embera.preprocess',
            'embera.utilities',
            'embera.utilities.architectures'
            ]

install_requires = ['networkx>=2.0,<3.0',
                    'decorator>=4.1.0,<5.0.0',
                    'dimod>=0.8.2,<0.9.0',
                    'pulp>=1.6.0,<2.0.0',
                    'minorminer>=0.1.5,<0.2.0',
                    'dwave-networkx>=0.6.4,<0.8.0']


setup(name='embera',
      version='0.0.1',
      description='Embedding Resources and Algorithms',
      long_description="Collection of minor-embedding methods and utilities "
      "to map unstructured binary quadratic problems to a structured sampler "
      "such as a D-Wave system.",
      author='Jose Pinilla',
      author_email='jpinilla@ece.ubc.ca',
      url='https://github.com/joseppinilla/embera',
      packages=packages,
      platforms='any',
      install_requires=install_requires,
      license='MIT'
     )
