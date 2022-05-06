#!/usr/bin/env python

from setuptools import setup

packages = ['embera',
            'embera.algorithms',
            'embera.architectures',
            'embera.benchmarks',
            'embera.composites',
            'embera.evaluation',
            'embera.interfaces',
            'embera.utilities',
            ]

install_requires = ['dimod',
                    'dwave_networkx',
                    'dwave-system',
                    'matplotlib',
                    'minorminer',
                    'networkx',
                    'numpy',
                    'pulp',
                    'scipy',
                    ]

with open("README.md", "r", encoding='utf-8') as fh:
    long_description = fh.read()

setup(name='embera',
      version='0.0.1a',
      description='Embedding Resources and Algorithms',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Jose Pinilla',
      author_email='jpinilla@ece.ubc.ca',
      url='https://github.com/joseppinilla/embera',
      packages=packages,
      platforms='any',
      install_requires=install_requires,
      python_requires='>=3.6', # f-string support
      license='MIT'
     )
