#!/usr/bin/env python

from setuptools import setup

packages = ['embedding_methods',
            'embedding_methods.composites',
            'embedding_methods.preprocess',
            'embedding_methods.utilities',
            'embedding_methods.utilities.architectures',
            ]

install_requires = ['networkx>=2.0,<3.0',
                    'decorator>=4.1.0,<5.0.0',
                    'dimod>=0.6.8,<0.8.0',
                    'pulp>=1.6.0,<2.0.0',
                    'minorminer>=0.1.5,<0.2.0',
                    'dwave-networkx>=0.6.4,<0.7.0']


setup(name='embedding_methods',
      version='0.0.1',
      description='Minor-Embedding Methods',
      long_description="Collection of minor-embedding methods and utilities "
      "to map unstructured binary quadratic problems to a structured sampler "
      "such as a D-Wave system.",
      author='Jose Pinilla',
      author_email='jpinilla@ece.ubc.ca',
      url='https://github.com/joseppinilla/embedding-methods',
      packages=packages,
      platforms='any',
      install_requires=install_requires,
      license='TO-DO' #TODO: Choose License
     )
