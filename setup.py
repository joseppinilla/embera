#!/usr/bin/env python

from distutils.core import setup

setup(name='minoremb',
      version='1.0',
      description='Minor-Embedding Methods',
      long_description="Collection of minor-embedding methods. A graph G is a minor of H if G is isomorphic to a graph obtained from a subgraph of H by succesively contracting edges.",
      author='Jose Pinilla',
      author_email='jpinilla@ece.ubc.ca',
      url='https://github.com/joseppinilla/embedding-methods',
      packages=['topological', 'dense'],
     )
