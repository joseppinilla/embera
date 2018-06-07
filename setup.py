#!/usr/bin/env python

from distutils.core import setup

packages = ['embedding_methods',
            'embedding_methods.dense',
            'embedding_methods.topological']

install_requires = ['networkx>=2.0,<3.0',
                    'decorator>=4.1.0,<5.0.0',
                    'dimod>=0.6.8,<0.7.0']


setup(name='embedding_methods',
      version='0.0.1',
      description='Minor-Embedding Methods',
      long_description="Collection of minor-embedding methods. A graph G is a minor of H if G is isomorphic to a graph obtained from a subgraph of H by succesively contracting edges.",
      author='Jose Pinilla',
      author_email='jpinilla@ece.ubc.ca',
      url='https://github.com/joseppinilla/embedding-methods',
      packages=packages,
     )
