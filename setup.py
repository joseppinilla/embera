#!/usr/bin/env python

from setuptools import setup

packages = ['embera',
            'embera.architectures',
            'embera.benchmark',
            'embera.composites',
            'embera.interfaces',
            'embera.preprocess',
            'embera.transform',
            'embera.utilities',
            ]

install_requires = ['dimod>=0.12.14',
                    'dwave_networkx>=0.8.14',
                    'dwave-system>=1.23.0',
                    'matplotlib>=3.8.2',
                    'minorminer>=0.2.13',
                    'networkx>=3.2.1',
                    'numpy>=1.26.4',
                    'pulp>=2.8.0',
                    'scipy>=1.12.0']

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
